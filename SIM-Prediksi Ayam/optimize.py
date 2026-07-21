"""
optimize.py
=========================================================
Optimasi model LSTM harga ayam:
1. TimeSeriesSplit evaluation -> cek konsistensi performa di beberapa
   potongan waktu berbeda (bukan cuma 1x split train/val/test).
2. Grid search WINDOW size (7/14/21/30) -> cari lookback terbaik.

PIPELINE DISAMAKAN PERSIS DENGAN train.py (revisi):
----------------------------------------------------------
Sebelumnya file ini punya feature engineering, skema split, dan
callback training sendiri yang BERBEDA dari train.py -- akibatnya
hasil "window terbaik" di sini tidak apple-to-apple dengan model
produksi yang sungguh-sungguh dipakai app.py. Diperbaiki dengan:

  1. Fitur & load_and_prepare() diimpor LANGSUNG dari
     feature_engineering.py (satu sumber kebenaran yang sama dipakai
     train.py/predict.py/app.py) -- bukan reimplementasi lokal dengan
     fitur tambahan (Lag14, RollStd7, DayOfWeek, Hari_Ke_Lebaran, dst).
  2. Skema split train/val/test (70%/15%/15%, kronologis) dibuat
     IDENTIK dengan train.py -- bukan lagi split 85%/15% ad-hoc di
     dalam train_and_eval().
  3. Callback & hyperparameter default (EarlyStopping patience=10,
     TANPA ReduceLROnPlateau, TANPA sample-weight Lebaran) disamakan
     dengan build_model()/run_training() di train.py.
  4. Pengaturan determinisme TensorFlow (env var oneDNN, single-thread,
     enable_op_determinism) disalin dari train.py -- mengurangi noise
     run-to-run yang sebelumnya membuat "window terbaik" berpindah-pindah
     antar run (lihat catatan di train.py soal ini).

Dengan perubahan ini, angka RMSE/MAE/MAPE dari grid search window di
sini seharusnya mendekati apa yang akan didapat train.py jika dilatih
ulang dengan window yang sama -- meski TIDAK akan 100% identik bit-per-
bit, karena grid search ini melatih model HANYA pada potongan data
70%+15% (bukan seluruh train.py yang juga sekaligus menyimpan model
final ke models/).

CARA MENJALANKAN (CLI, standalone):
    python optimize.py
Akan mencetak tabel hasil ke terminal, dan menyimpan
models/optimize_results.json

CARA MENJALANKAN (diimpor, mis. dari app.py):
    from optimize import run_full_optimization
    hasil = run_full_optimization(progress_callback=lambda msg: print(msg))
    # hasil = {"timeseries_cv": [...], "window_grid_search": [...]}
Semua logic dibungkus fungsi -- TIDAK ADA kode training yang berjalan
otomatis saat file ini di-import (aman dipakai di dalam app Streamlit).
"""

import os

# Sama seperti train.py: WAJIB di-set SEBELUM import tensorflow, supaya
# hasil training deterministik antar run (mengurangi noise yang membuat
# "window terbaik" berpindah-pindah tiap kali grid search dijalankan
# ulang pada data yang sama).
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "42"

import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from feature_engineering import FEATURE_COLUMNS, TARGET_ENGINEERED, load_and_prepare

# Sama seperti train.py -- paksa operasi TensorFlow deterministik &
# kunci ke 1 thread, supaya hasil training identik bit-per-bit antar run
# dengan seed yang sama.
tf.config.experimental.enable_op_determinism()
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

DATA_PATH = "dataset_bersih_lstm_final.csv"
MODEL_DIR = "models"
EPOCHS = 60
BATCH_SIZE = 16
PATIENCE = 10  # disamakan dengan EarlyStopping patience di train.py (sebelumnya 6)

TRAIN_FRAC = 0.70  # disamakan persis dengan train.py
VAL_FRAC = 0.85    # disamakan persis dengan train.py


def create_sequences(features: np.ndarray, target: np.ndarray, window: int):
    """Identik dengan create_sequences di train.py."""
    X, y = [], []
    for i in range(window, len(features)):
        X.append(features[i - window:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def build_model(window: int, n_features: int, units1: int = 64, units2: int = 32,
                 dropout: float = 0.3, learning_rate: float = 0.001) -> Sequential:
    """Identik dengan build_model di train.py, hanya diparameterisasi
    supaya bisa dipakai untuk grid search hyperparameter oleh tune.py."""
    model = Sequential([
        Input(shape=(window, n_features)),
        LSTM(units1, return_sequences=True, recurrent_dropout=0.1),
        Dropout(dropout),
        LSTM(units2, recurrent_dropout=0.1),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="huber")
    return model


def hitung_metrik_return(model, X_set, y_set_return, harga_prev_arr, harga_actual_arr):
    if len(X_set) == 0:
        return {"mae": None, "rmse": None, "mape": None, "n_samples": 0}
    pred_return = model.predict(X_set, verbose=0).flatten()
    harga_pred = harga_prev_arr * (1 + pred_return)

    mae = float(np.mean(np.abs(harga_actual_arr - harga_pred)))
    rmse = float(np.sqrt(np.mean((harga_actual_arr - harga_pred) ** 2)))
    mape = float(np.mean(np.abs((harga_actual_arr - harga_pred) / np.maximum(harga_actual_arr, 1))) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape, "n_samples": int(len(X_set))}


def train_and_eval(train_df, val_df, test_df, full_df, window, verbose=0,
                    units1=64, units2=32, dropout=0.3, learning_rate=0.001,
                    batch_size=None):
    """Latih model pada train_df, validasi pada val_df, evaluasi pada test_df.
    full_df dipakai untuk membangun sequence yang butuh histori sebelum awal
    val_df/test_df (sama seperti train.py -- scaler & windowing dihitung dari
    seluruh df, lalu diiris sesuai posisi train/val/test).

    Skema ini SEKARANG IDENTIK dengan run_training() di train.py: split
    3 bagian kronologis (train/val/test), scaler HANYA di-fit pada train_df,
    tanpa sample-weighting apapun (dulu ada lebaran_weight, sekarang
    dihapus supaya sama persis dengan model produksi).

    units1/units2/dropout/learning_rate/batch_size: hyperparameter model,
    dipakai oleh tune.py untuk grid search (default = sama seperti train.py)."""
    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])

    scaled_all = scaler.transform(full_df[FEATURE_COLUMNS])
    target_all = full_df[TARGET_ENGINEERED].values

    X_all, y_all = create_sequences(scaled_all, target_all, window)

    train_size = len(train_df)
    val_size = len(train_df) + len(val_df)

    train_end = train_size - window
    val_end = val_size - window

    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:val_end + len(test_df)], y_all[val_end:val_end + len(test_df)]

    if len(X_train) < 20 or len(X_test) == 0:
        return None

    model = build_model(window, len(FEATURE_COLUMNS), units1=units1, units2=units2,
                         dropout=dropout, learning_rate=learning_rate)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    ]
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=batch_size or BATCH_SIZE, shuffle=False,
        callbacks=callbacks, verbose=verbose,
    )

    harga_full = full_df["Harga_Ayam"].values
    harga_prev = harga_full[window + val_end - 1: window + val_end - 1 + len(X_test)]
    harga_actual = harga_full[window + val_end: window + val_end + len(X_test)]

    metrics = hitung_metrik_return(model, X_test, y_test, harga_prev, harga_actual)
    return metrics


def run_timeseries_cv(df: pd.DataFrame, window_fixed: int = 14, n_splits: int = 5,
                       progress_callback=None):
    """5-fold TimeSeriesSplit evaluation dengan window tetap. Di dalam tiap
    fold, 15% ekor dari train_idx dipisah jadi val (menjaga rasio train:val
    ~85:15 yang sama seperti split utama train.py) supaya EarlyStopping tetap
    berbasis val_loss yang genuine, bukan bocor dari test.
    progress_callback(str) dipanggil setiap fold selesai (opsional, untuk UI)."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(df) * 0.10))
    cv_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        full_fold = df.iloc[: test_idx[-1] + 1].reset_index(drop=True)

        val_cut = int(len(train_idx) * 0.85)
        train_fold = df.iloc[train_idx[:val_cut]]
        val_fold = df.iloc[train_idx[val_cut:]]
        test_fold = df.iloc[test_idx]

        t0 = time.time()
        m = train_and_eval(train_fold, val_fold, test_fold, full_fold, window_fixed)
        dt = time.time() - t0

        if m is None:
            msg = f"Fold {fold}: dilewati (data tidak cukup)"
            if progress_callback:
                progress_callback(msg)
            continue

        tgl_awal = df["Tanggal"].iloc[test_idx[0]].date()
        tgl_akhir = df["Tanggal"].iloc[test_idx[-1]].date()
        msg = (f"Fold {fold} [{tgl_awal} s.d {tgl_akhir}] "
               f"MAE=Rp{m['mae']:,.0f} RMSE=Rp{m['rmse']:,.0f} MAPE={m['mape']:.2f}% ({dt:.0f}s)")
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
        cv_results.append({"fold": fold, "test_start": str(tgl_awal), "test_end": str(tgl_akhir), **m})

    return cv_results


def run_window_grid_search(df: pd.DataFrame, windows=(7, 14, 21, 30),
                            train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC,
                            progress_callback=None):
    """Grid search window size, dengan split 70/15/15 -- IDENTIK dengan
    skema di train.py (train_size = 70% data, val_size = 85% data,
    sisanya test)."""
    n_total = len(df)
    train_size = int(n_total * train_frac)
    val_size = int(n_total * val_frac)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:val_size]
    test_df = df.iloc[val_size:]

    window_results = []
    for w in windows:
        t0 = time.time()
        m = train_and_eval(train_df, val_df, test_df, df, w)
        dt = time.time() - t0
        if m is None:
            msg = f"WINDOW={w}: dilewati (data tidak cukup)"
            if progress_callback:
                progress_callback(msg)
            continue
        msg = (f"WINDOW={w:>2} -> MAE=Rp{m['mae']:,.0f} RMSE=Rp{m['rmse']:,.0f} "
               f"MAPE={m['mape']:.2f}% ({dt:.0f}s)")
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
        window_results.append({"window": w, **m})

    return window_results


def run_full_optimization(data_path: str = DATA_PATH, model_dir: str = MODEL_DIR,
                           progress_callback=None, seed: int = 42):
    """Jalankan TimeSeriesCV + grid search window, simpan hasil ke
    <model_dir>/optimize_results.json, dan kembalikan hasilnya sebagai dict.

    progress_callback: fungsi opsional dipanggil dengan 1 argumen string
    setiap ada progres baru (dipakai app.py untuk update UI live)."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    df = load_and_prepare(data_path)
    if progress_callback:
        progress_callback(
            f"Jumlah baris data: {len(df)}, "
            f"{df['Tanggal'].min().date()} s/d {df['Tanggal'].max().date()}"
        )

    if progress_callback:
        progress_callback("=== 1. TIME SERIES CROSS VALIDATION (5 fold, WINDOW=14) ===")
    cv_results = run_timeseries_cv(df, window_fixed=14, n_splits=5,
                                    progress_callback=progress_callback)

    if cv_results:
        mae_list = [r["mae"] for r in cv_results]
        rmse_list = [r["rmse"] for r in cv_results]
        if progress_callback:
            progress_callback(
                f"Rata-rata MAE : Rp{np.mean(mae_list):,.0f}  (std: Rp{np.std(mae_list):,.0f}) | "
                f"Rata-rata RMSE: Rp{np.mean(rmse_list):,.0f}  (std: Rp{np.std(rmse_list):,.0f})"
            )

    if progress_callback:
        progress_callback("=== 2. GRID SEARCH WINDOW SIZE ===")
    window_results = run_window_grid_search(df, windows=(7, 14, 21, 30),
                                             progress_callback=progress_callback)

    best = None
    if window_results:
        best = min(window_results, key=lambda r: r["rmse"])
        if progress_callback:
            progress_callback(
                f"Window terbaik (RMSE terkecil): WINDOW={best['window']} "
                f"(RMSE=Rp{best['rmse']:,.0f}, MAE=Rp{best['mae']:,.0f})"
            )

    os.makedirs(model_dir, exist_ok=True)
    hasil = {
        "timeseries_cv": cv_results,
        "window_grid_search": window_results,
        "best_window": best["window"] if best else None,
    }
    with open(os.path.join(model_dir, "optimize_results.json"), "w") as f:
        json.dump(hasil, f, indent=2)

    if progress_callback:
        progress_callback(f"✅ Hasil optimasi disimpan ke {model_dir}/optimize_results.json")

    return hasil


if __name__ == "__main__":
    run_full_optimization(progress_callback=print)