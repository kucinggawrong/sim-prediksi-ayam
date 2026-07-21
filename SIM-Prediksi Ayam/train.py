"""
train.py
=========================================================
Melatih model LSTM multivariate untuk memprediksi Harga Ayam Ras
di Kota Bogor, menggunakan fitur harga pakan (jagung pipilan),
flag hari raya, dan efek lebaran.

PERUBAHAN vs versi sebelumnya:
----------------------------------------------------------
Logika training (arsitektur, target return, scaler, dsb -- TIDAK ADA
YANG BERUBAH) dibungkus dalam fungsi run_training(data_path, model_dir)
supaya bisa dipanggil berkali-kali dengan dataset & folder output
berbeda -- dulu dipakai untuk eksperimen "data asli" vs "data setelah
deteksi-LOCF & interpolasi ulang" (lihat bandingkan_model_a_b.py).
Hasil eksperimen itu (perbandingan_model_a_b.csv) membuktikan data
yang sudah di-interpolasi ulang menghasilkan RMSE test yang lebih
kecil (~6-9% lebih rendah) dibanding data asli -- karena itu
DATA_PATH default sekarang mengarah ke dataset_bersih_lstm_final.csv
(hasil preprocess.py -> locf_reinterpolasi.py), bukan lagi ke dataset
mentah hasil interpolasi tahap 1 saja.

Menjalankan file ini langsung (python train.py) tetap berperilaku
sederhana: melatih 1 model dari DATA_PATH default, simpan ke MODEL_DIR
default ("models").

UPDATE WINDOW SIZE (revisi terbaru -- kembali ke 14):
----------------------------------------------------------
Sempat diubah ke WINDOW=30 berdasarkan grid search pada dataset lama
(2.014 baris, s.d. 7 Juli 2026), di mana window 30 menang tipis
(RMSE=Rp663 vs Rp664 -- selisih ~Rp1, jelas dalam batas noise).
Setelah dataset bertambah menjadi 2.024 baris (s.d. 17 Juli 2026),
grid search window diulang dan window=14 kembali menang dengan
margin yang lebih meyakinkan (RMSE=Rp449 vs Rp451/452/454 untuk
window 7/21/30 -- selisih Rp3-5, bukan cuma Rp1). WINDOW dikembalikan
ke 14 mengikuti hasil ini. Catatan: RMSE keseluruhan juga turun jauh
(dari kisaran Rp660an ke Rp450an) karena test set kini mencakup
rentang tanggal yang berbeda (data 10 hari lebih baru) -- bukan
berarti model tiba-tiba jauh lebih akurat dari sebelumnya, melainkan
efek dari pergeseran komposisi data uji.

CARA MENJALANKAN:
    python train.py

Output di folder models/ (atau folder lain jika dipanggil sebagai
modul dengan model_dir berbeda):
    - lstm_model.keras   (model terlatih, memprediksi RETURN)
    - scaler.pkl          (scaler untuk fitur input)
    - metadata.json       (window_size, daftar fitur, info rekonstruksi)
    - metrics.json        (MAE/RMSE/MAPE dalam Rupiah, train/val/test)
    - history.json        (riwayat loss per epoch)
"""

import os

# Wajib di-set SEBELUM import tensorflow, supaya oneDNN tidak reorder
# operasi floating-point secara berbeda tiap run (penyebab metrics.json
# berubah-ubah tiap re-run walau seed sudah di-fix).
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = "42"

import json

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from feature_engineering import FEATURE_COLUMNS, TARGET_ENGINEERED, load_and_prepare

# Paksa TensorFlow memakai operasi yang deterministik (hasil identik
# bit-per-bit tiap run dengan seed yang sama) -- tanpa ini, set_seed()
# saja TIDAK menjamin hasil training identik antar run.
tf.config.experimental.enable_op_determinism()

# Kunci ke 1 thread -- tanpa ini, urutan penjumlahan floating-point di
# dalam operasi paralel (mis. reduce/sum pada LSTM) masih bisa berbeda
# tiap run tergantung thread mana yang selesai duluan, walau
# enable_op_determinism() sudah aktif. Ini penyebab metrics.json masih
# berubah tipis (Rp101->Rp89) meski fix sebelumnya sudah diterapkan.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# =========================================================
# KONFIGURASI DEFAULT (dipakai kalau file ini dijalankan langsung)
# =========================================================
DATA_PATH = "dataset_bersih_lstm_final.csv"
MODEL_DIR = "models"
# WINDOW=14 dipilih berdasarkan grid search terbaru pada dataset
# 2.024 baris (s.d. 17 Juli 2026): WINDOW=14 -> RMSE=Rp449, lebih baik
# dari window 7/21/30 (RMSE=Rp451/452/454, selisih Rp3-5). Sebelumnya
# sempat WINDOW=30 berdasarkan grid search pada dataset lama (2.014
# baris), tapi selisihnya waktu itu cuma ~Rp1 (noise) -- dengan data
# lebih banyak, window=14 kini menang dengan margin lebih jelas.
WINDOW = 14
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85
EPOCHS = 60
BATCH_SIZE = 32  # direvisi dari 16 -- lihat catatan di build_model()
TARGET_COL = "Harga_Ayam"
SEED = 42


def create_sequences(features: np.ndarray, target: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(features)):
        X.append(features[i - window:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def build_model(window: int, n_features: int) -> Sequential:
    # Arsitektur (units1=32, units2=16, dropout=0.3), learning_rate=0.001,
    # dan batch_size=32 berasal dari grid search hyperparameter tune.py
    # YANG DIJALANKAN PADA WINDOW=14 (window default saat ini) --
    # RMSE=Rp441, MAE=Rp174 -- lihat models/tune_results.json.
    # Revisi dari kombinasi sebelumnya (units1=64, units2=32, lr=0.0005,
    # batch=16) yang ternyata berasal dari tuning window=30 (sudah usang
    # sejak WINDOW dikembalikan ke 14) -- kombinasi lama itu RMSE=Rp454
    # kalau dites ulang di window=14, ~Rp13 (~3%) lebih buruk dari
    # kombinasi baru ini. Beda ini lebih besar dari noise run-to-run
    # yang biasa terlihat (~Rp1-5), jadi kemungkinan perbaikan genuine.
    model = Sequential([
        Input(shape=(window, n_features)),
        LSTM(32, return_sequences=True, recurrent_dropout=0.1),
        Dropout(0.3),
        LSTM(16, recurrent_dropout=0.1),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="huber")
    return model


def run_training(
    data_path: str = DATA_PATH,
    model_dir: str = MODEL_DIR,
    window: int = WINDOW,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    verbose: int = 1,
    label: str = "",
) -> dict:
    """Latih 1 model LSTM dari data_path, simpan artefak ke model_dir,
    kembalikan dict metrics {train, val, test}. Fungsi ini adalah SATU
    SUMBER KEBENARAN untuk proses training -- dipanggil baik dari
    __main__ di bawah, maupun dari bandingkan_model_a_b.py."""

    np.random.seed(seed)
    tf.random.set_seed(seed)

    prefix = f"[{label}] " if label else ""

    # ---- 1. Load data + feature engineering ----
    df = load_and_prepare(data_path)
    print(f"{prefix}Jumlah baris data: {len(df)}")
    print(f"{prefix}Rentang tanggal  : {df['Tanggal'].min().date()} s/d {df['Tanggal'].max().date()}")

    n_total = len(df)
    min_required = window + 10
    if n_total < min_required:
        raise ValueError(
            f"Dataset terlalu kecil ({n_total} baris) untuk WINDOW={window}. "
            f"Minimal dibutuhkan {min_required} baris."
        )

    # ---- 2. Split time series (urut waktu, tidak di-shuffle) ----
    train_size = int(n_total * train_frac)
    val_size = int(n_total * val_frac)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:val_size]
    test_df = df.iloc[val_size:]

    print(f"{prefix}Train: {len(train_df)} baris | Val: {len(val_df)} baris | Test: {len(test_df)} baris")
    print(f"{prefix}Train Harga_Ayam range: {train_df['Harga_Ayam'].min():.0f} - {train_df['Harga_Ayam'].max():.0f}")
    print(f"{prefix}Test  Harga_Ayam range: {test_df['Harga_Ayam'].min():.0f} - {test_df['Harga_Ayam'].max():.0f}")

    # ---- 3. Scaler -- hanya di-fit pada data train (anti data leakage) ----
    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])

    scaled_features = scaler.transform(df[FEATURE_COLUMNS])
    target_values = df[TARGET_ENGINEERED].values

    # ---- 4. Windowing ----
    X, y = create_sequences(scaled_features, target_values, window)

    train_end = train_size - window
    val_end = val_size - window

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"{prefix}Shape X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

    # ---- 5. Arsitektur & training ----
    model = build_model(window, len(FEATURE_COLUMNS))
    if verbose:
        model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop],
        verbose=verbose,
    )

    # ---- 6. Evaluasi: rekonstruksi harga dari return, hitung MAE/RMSE/MAPE ----
    def hitung_metrik(X_set, y_set_return, start_idx_in_df: int, nama_set: str) -> dict:
        if len(X_set) == 0:
            print(f"{prefix}[WARNING] Set '{nama_set}' kosong, dilewati.")
            return {}

        pred_return = model.predict(X_set, verbose=0).flatten()

        harga_prev = df["Harga_Ayam"].values[start_idx_in_df - 1: start_idx_in_df - 1 + len(X_set)]
        harga_actual = df["Harga_Ayam"].values[start_idx_in_df: start_idx_in_df + len(X_set)]
        harga_pred = harga_prev * (1 + pred_return)

        mae = float(np.mean(np.abs(harga_actual - harga_pred)))
        rmse = float(np.sqrt(np.mean((harga_actual - harga_pred) ** 2)))
        mape = float(np.mean(np.abs((harga_actual - harga_pred) / np.maximum(harga_actual, 1))) * 100)

        print(f"{prefix}[{nama_set.upper()}] MAE=Rp{mae:,.0f} | RMSE=Rp{rmse:,.0f} | MAPE={mape:.2f}%")
        return {"mae": mae, "rmse": rmse, "mape": mape, "n_samples": int(len(X_set))}

    metrics = {
        "train": hitung_metrik(X_train, y_train, window, "train"),
        "val": hitung_metrik(X_val, y_val, window + train_end, "val"),
        "test": hitung_metrik(X_test, y_test, window + val_end, "test"),
    }

    # ---- 7. Simpan artefak ----
    os.makedirs(model_dir, exist_ok=True)

    model.save(os.path.join(model_dir, "lstm_model.keras"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump({
            "data_path": data_path,
            "window_size": window,
            "features": FEATURE_COLUMNS,
            "target_engineered": TARGET_ENGINEERED,
            "target_asli": TARGET_COL,
            "catatan_rekonstruksi": "harga_t = harga_aktual_(t-1) * (1 + return_prediksi_t)",
            "train_frac": train_frac,
            "val_frac": val_frac,
            "n_total_rows": n_total,
            "tanggal_data_terakhir": df["Tanggal"].max().strftime("%Y-%m-%d"),
        }, f, indent=2)

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(model_dir, "history.json"), "w") as f:
        json.dump({
            "loss": [float(v) for v in history.history.get("loss", [])],
            "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        }, f, indent=2)

    print(f"{prefix}✅ TRAINING SELESAI - Model & artefak tersimpan di folder '{model_dir}/'")
    return metrics


if __name__ == "__main__":
    run_training(DATA_PATH, MODEL_DIR)