"""
feature_engineering.py
=========================================================
Modul SATU SUMBER KEBENARAN untuk seluruh proses rekayasa fitur
(feature engineering) yang dipakai bersama oleh train.py,
predict.py, dan app.py.

Mengapa modul ini dibutuhkan?
---------------------------------------------------------
Pada kode versi awal, logika feature engineering DITULIS ULANG secara
manual di beberapa file berbeda (train.py sempat punya versi sendiri
yang berbeda dari predict.py/app.py). Ini berisiko tinggi: kolom fitur
yang dipakai saat training bisa tidak ada lagi di dataframe yang dipakai
saat inference -- menyebabkan KeyError atau (lebih berbahaya) prediksi
yang salah tanpa error yang jelas.

Dengan modul ini, formula fitur HANYA ditulis SATU KALI di sini,
lalu di-import oleh train.py, predict.py, dan app.py.
"""

import pandas as pd

# =========================================================
# DAFTAR KOLOM FITUR YANG DIGUNAKAN MODEL (input LSTM)
# =========================================================
# Urutan ini PENTING karena harus sama persis dengan urutan saat
# scaler di-fit pada train.py. Index ke-0 (Harga_Ayam) adalah level
# harga terakini yang dipakai model sebagai referensi, BUKAN target
# yang diprediksi -- target sebenarnya adalah RETURN (lihat
# TARGET_ENGINEERED di bawah), supaya model tidak terganggu tren naik
# jangka panjang harga ayam (2021: ~13rb -> 2026: ~50rb).
FEATURE_COLUMNS = [
    "Harga_Ayam",
    "Harga_Jagung_Pipilan_Kg",
    "Harga_Ayam_Lag1",
    "Harga_Ayam_Lag7",
    "Harga_Ayam_RollMean7",
    "Is_Hari_Raya",
    "Efek_Lebaran",
]

# Target yang benar-benar diprediksi model (return harian, TIDAK di-scale
# -- lihat train.py bagian SCALER). Harga hasil prediksi direkonstruksi
# dengan reconstruct_harga() di bawah, BUKAN lewat scaler.inverse_transform.
TARGET_ENGINEERED = "Return"
TARGET_ASLI = "Harga_Ayam"

# Kolom mentah (raw) minimal yang wajib ada di CSV input
RAW_REQUIRED_COLUMNS = [
    "Tanggal",
    "Harga_Ayam",
    "Harga_Jagung_Pipilan_Kg",
    "Is_Hari_Raya",
    "Efek_Lebaran",
]

ROLLING_WINDOW_PAKAN = 7


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan kolom turunan (derived features) ke DataFrame:

    Fitur input model (dipakai FEATURE_COLUMNS):
    - Harga_Ayam_Lag1/Lag7  : harga ayam 1 & 7 hari sebelumnya.
    - Harga_Ayam_RollMean7  : rata-rata bergerak 7 hari harga ayam.
    - Return                : target model -- persentase perubahan
                               harga ayam terhadap hari sebelumnya.

    Fitur tambahan untuk tampilan dashboard (BUKAN input model):
    - Delta_Pakan      : selisih harga pakan hari ini vs hari sebelumnya.
    - Rolling_Pakan_7  : rata-rata bergerak 7 hari harga pakan.
    - Ratio_Pakan_Ayam : rasio harga pakan terhadap harga ayam
                         (mengukur tekanan margin peternak).

    PENTING: fungsi ini hanya menggunakan data MASA LALU & SAAT INI
    (diff, shift, rolling mean ke belakang) -- tidak ada look-ahead ke
    masa depan, sehingga aman dipakai untuk time series tanpa data leakage.

    df wajib sudah terurut berdasarkan Tanggal (ascending) sebelum
    fungsi ini dipanggil.
    """
    df = df.copy()

    # --- Fitur input model ---
    df["Harga_Ayam_Lag1"] = df["Harga_Ayam"].shift(1)
    df["Harga_Ayam_Lag7"] = df["Harga_Ayam"].shift(7)
    df["Harga_Ayam_RollMean7"] = df["Harga_Ayam"].rolling(ROLLING_WINDOW_PAKAN).mean()
    df["Return"] = df["Harga_Ayam"].pct_change()

    # --- Fitur tambahan untuk dashboard (Data Historis, metrik Ratio) ---
    df["Delta_Pakan"] = df["Harga_Jagung_Pipilan_Kg"].diff().fillna(0)
    df["Rolling_Pakan_7"] = (
        df["Harga_Jagung_Pipilan_Kg"]
        .rolling(ROLLING_WINDOW_PAKAN)
        .mean()
        .bfill()
    )
    df["Ratio_Pakan_Ayam"] = df["Harga_Jagung_Pipilan_Kg"] / (df["Harga_Ayam"] + 1)

    # Isi baris awal yang kosong akibat shift/rolling/pct_change dengan
    # backward-fill (sama seperti train.py) supaya tidak ada NaN yang lolos
    # ke scaler/model.
    kolom_model = ["Harga_Ayam_Lag1", "Harga_Ayam_Lag7", "Harga_Ayam_RollMean7", "Return"]
    df[kolom_model] = df[kolom_model].bfill()

    return df


def reconstruct_harga(harga_prev, pred_return):
    """Rekonstruksi harga ayam dari prediksi RETURN model:
        harga_t = harga_(t-1) * (1 + return_prediksi_t)
    Ini BUKAN scaler.inverse_transform -- Return tidak pernah di-scale
    (lihat train.py), jadi output model dipakai langsung di sini."""
    return harga_prev * (1.0 + pred_return)


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """
    Memuat CSV dataset, memvalidasi kolom yang wajib ada, mengurutkan
    berdasarkan tanggal, dan menambahkan seluruh fitur turunan.
    Mengembalikan DataFrame yang siap di-scale & di-windowing.
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di {csv_path}: {missing}. "
            f"Pastikan dataset memiliki kolom: {RAW_REQUIRED_COLUMNS}"
        )

    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values("Tanggal").reset_index(drop=True)
    df = add_features(df)

    return df


def create_sequences(scaled_array, window: int, target_col_idx: int = 0):
    """
    Mengubah array 2D (n_baris, n_fitur) yang sudah di-scale menjadi
    sequence 3D (n_sample, window, n_fitur) untuk input LSTM, beserta
    target y (harga ayam pada t+window).

    Dipakai konsisten oleh train.py (untuk membuat X_train/y_train)
    dan dapat dipakai ulang untuk evaluasi di app.py.
    """
    import numpy as np

    X, y = [], []
    for i in range(len(scaled_array) - window):
        X.append(scaled_array[i:i + window])
        y.append(scaled_array[i + window, target_col_idx])
    return np.array(X), np.array(y)
