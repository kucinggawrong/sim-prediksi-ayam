"""
tune.py
=========================================================
Grid search hyperparameter model LSTM harga ayam: jumlah unit LSTM,
dropout rate, learning rate, dan batch size.

Berbeda dari optimize.py (yang fokus ke WINDOW size & konsistensi
antar-waktu via TimeSeriesCV), script ini fokus ke ARSITEKTUR &
proses training model itu sendiri -- karena hasil grid search window
di optimize.py menunjukkan window size TIDAK banyak berpengaruh ke
RMSE, sehingga hyperparameter lain kemungkinan lebih berpengaruh.

Menggunakan skema split yang SAMA PERSIS dengan train.py (70% train /
15% val / 15% test, kronologis) -- disesuaikan dari versi sebelumnya
yang memakai split 85%/15% ad-hoc, supaya hasil tuning ini benar-benar
mencerminkan performa model produksi (lihat catatan di optimize.py).

CARA MENJALANKAN (CLI, standalone):
    python tune.py
Akan mencetak tabel hasil ke terminal, dan menyimpan
models/tune_results.json

CARA MENJALANKAN (diimpor, mis. dari app.py):
    from tune import run_hyperparameter_tuning
    hasil = run_hyperparameter_tuning(progress_callback=lambda msg: print(msg))
"""

import os
import json
import time
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf

from optimize import (
    load_and_prepare, train_and_eval, FEATURE_COLUMNS, DATA_PATH, MODEL_DIR,
)

# Window dipakai tetap (bukan yang di-tuning di sini -- lihat optimize.py
# untuk pencarian window). Default disamakan dengan WINDOW default di
# train.py (14, direvisi dari 30 setelah dataset bertambah -- lihat
# catatan di train.py) -- ganti sesuai kebutuhan.
WINDOW_TETAP = 14
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85

# =========================================================
# GRID HYPERPARAMETER
# =========================================================
# Dijaga tetap kecil (2-3 opsi per parameter) supaya total kombinasi
# tidak meledak -- combined grid search bisa sangat lama karena tiap
# kombinasi = 1x training LSTM dari nol.
PARAM_GRID = {
    "units1": [32, 64],
    "units2": [16, 32],
    "dropout": [0.2, 0.3],
    "learning_rate": [0.001, 0.0005],
    "batch_size": [16, 32],
}


def run_hyperparameter_tuning(data_path: str = DATA_PATH, model_dir: str = MODEL_DIR,
                               window: int = WINDOW_TETAP, param_grid: dict = None,
                               max_combinations: int = 16, progress_callback=None,
                               seed: int = 42):
    """Grid search hyperparameter (units1, units2, dropout, learning_rate,
    batch_size) pada 1 split waktu (70% train / 15% val / 15% test, sama
    persis seperti train.py).

    max_combinations: batasi jumlah kombinasi yang benar-benar dicoba (grid
    penuh bisa puluhan kombinasi = lama sekali) -- kombinasi diambil acak
    tapi dengan seed tetap supaya hasilnya reproducible.

    progress_callback(str) dipanggil setiap kombinasi selesai (opsional,
    untuk update UI live di app.py).

    Mengembalikan dict {"results": [...], "best": {...}} dan menyimpan ke
    <model_dir>/tune_results.json."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    grid = param_grid or PARAM_GRID
    df = load_and_prepare(data_path)

    if progress_callback:
        progress_callback(
            f"Jumlah baris data: {len(df)}, "
            f"{df['Tanggal'].min().date()} s/d {df['Tanggal'].max().date()} | window tetap={window}"
        )

    n_total = len(df)
    train_size = int(n_total * TRAIN_FRAC)
    val_size = int(n_total * VAL_FRAC)
    train_df_w = df.iloc[:train_size]
    val_df_w = df.iloc[train_size:val_size]
    test_df_w = df.iloc[val_size:]

    keys = list(grid.keys())
    all_combos = list(itertools.product(*[grid[k] for k in keys]))

    rng = np.random.RandomState(seed)
    if len(all_combos) > max_combinations:
        idx = rng.choice(len(all_combos), size=max_combinations, replace=False)
        combos = [all_combos[i] for i in sorted(idx)]
        if progress_callback:
            progress_callback(
                f"Grid penuh ada {len(all_combos)} kombinasi -- dibatasi ke "
                f"{max_combinations} kombinasi (sampel acak, seed={seed}) supaya tidak terlalu lama."
            )
    else:
        combos = all_combos

    if progress_callback:
        progress_callback(f"Total kombinasi yang akan dicoba: {len(combos)}")

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        t0 = time.time()
        m = train_and_eval(
            train_df_w, val_df_w, test_df_w, df, window,
            units1=params["units1"], units2=params["units2"],
            dropout=params["dropout"], learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
        )
        dt = time.time() - t0

        if m is None:
            msg = f"[{i+1}/{len(combos)}] {params}: dilewati (data tidak cukup)"
            if progress_callback:
                progress_callback(msg)
            continue

        msg = (f"[{i+1}/{len(combos)}] {params} -> "
               f"MAE=Rp{m['mae']:,.0f} RMSE=Rp{m['rmse']:,.0f} MAPE={m['mape']:.2f}% ({dt:.0f}s)")
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
        results.append({**params, **m})

    best = None
    if results:
        best = min(results, key=lambda r: r["rmse"])
        if progress_callback:
            progress_callback(
                f"🏆 Kombinasi terbaik (RMSE terkecil): {({k: best[k] for k in keys})} "
                f"-> RMSE=Rp{best['rmse']:,.0f}, MAE=Rp{best['mae']:,.0f}"
            )

    os.makedirs(model_dir, exist_ok=True)
    hasil = {"window": window, "results": results, "best": best}
    with open(os.path.join(model_dir, "tune_results.json"), "w") as f:
        json.dump(hasil, f, indent=2)

    if progress_callback:
        progress_callback(f"✅ Hasil tuning disimpan ke {model_dir}/tune_results.json")

    return hasil


if __name__ == "__main__":
    run_hyperparameter_tuning(progress_callback=print)