"""
predict.py
=========================================================
Modul forecasting (prediksi multi-hari ke depan) untuk Harga Ayam
Ras Kota Bogor, secara AUTOREGRESSIVE: prediksi hari ke-N dipakai
sebagai input untuk memprediksi hari ke-(N+1), dan seterusnya.

PERBAIKAN dari versi sebelumnya:
1. Is_Hari_Raya & Efek_Lebaran untuk tanggal forecast SEKARANG
   dihitung otomatis dari kalender nasional (holiday_calendar.py),
   BUKAN disalin dari baris terakhir data historis. Bug lama
   menyebabkan flag hari raya "macet" di nilai yang sama untuk
   semua hari forecast, walau tanggalnya sudah berubah.
2. Tanggal forecast ikut dilacak (kolom 'Tanggal' bertambah +1 hari
   setiap iterasi) -- versi lama tidak meng-update tanggal sama sekali.
3. Harga pakan (Harga_Jagung_Pipilan_Kg) untuk hari forecast kini
   bisa diproyeksikan dengan 3 skenario (flat / tren rata-rata 30 hari
   terakhir / custom growth rate per hari), bukan hanya dibekukan flat
   selamanya seperti sebelumnya -- lebih realistis untuk forecast >7 hari.
4. Feature engineering memakai modul bersama feature_engineering.py
   agar konsisten dengan train.py.
5. Hasil forecast dikembalikan sebagai DataFrame (tanggal + harga)
   agar mudah ditampilkan/diunduh, bukan list angka polos.

CARA MENJALANKAN (setelah train.py selesai & folder models/ terisi):
    python predict.py
"""

import json

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from feature_engineering import FEATURE_COLUMNS, add_features, load_and_prepare, reconstruct_harga
from holiday_calendar import is_hari_raya, efek_lebaran

MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"
METADATA_PATH = "models/metadata.json"
DATA_PATH = "dataset_bersih_lstm_final.csv"


def load_artifacts():
    """Memuat model, scaler, dan metadata yang dihasilkan train.py."""
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def _proyeksi_harga_pakan(df_history: pd.DataFrame, skenario: str, custom_rate_persen: float = 0.0) -> float:
    """
    Memproyeksikan harga pakan (jagung) untuk SATU hari ke depan,
    berdasarkan skenario yang dipilih pengguna:

    - "flat"  : harga pakan diasumsikan SAMA dengan hari terakhir
                yang diketahui (asumsi paling konservatif/aman).
    - "tren"  : mengikuti rata-rata perubahan harian pakan dalam
                30 hari terakhir data historis/forecast (menangkap
                tren naik/turun yang sedang terjadi).
    - "custom": memakai growth rate harian (%) yang ditentukan
                pengguna secara manual (custom_rate_persen).
    """
    harga_terakhir = df_history["Harga_Jagung_Pipilan_Kg"].iloc[-1]

    if skenario == "flat":
        return harga_terakhir

    if skenario == "tren":
        lookback = df_history["Harga_Jagung_Pipilan_Kg"].tail(30)
        delta_harian = lookback.diff().dropna().mean()
        if pd.isna(delta_harian):
            delta_harian = 0.0
        return harga_terakhir + delta_harian

    if skenario == "custom":
        return harga_terakhir * (1 + custom_rate_persen / 100.0)

    raise ValueError(f"Skenario pakan tidak dikenal: {skenario}")


def predict_future(
    days: int = 7,
    skenario_pakan: str = "flat",
    custom_rate_persen: float = 0.0,
) -> pd.DataFrame:
    """
    Melakukan forecast harga ayam untuk `days` hari ke depan secara
    autoregressive.

    Parameters
    ----------
    days : int
        Jumlah hari ke depan yang ingin diprediksi.
    skenario_pakan : str
        "flat", "tren", atau "custom" -- lihat _proyeksi_harga_pakan().
    custom_rate_persen : float
        Hanya dipakai jika skenario_pakan == "custom". Growth rate
        harian harga pakan dalam persen (boleh negatif).

    Returns
    -------
    pd.DataFrame dengan kolom: Tanggal, Harga_Ayam_Prediksi,
    Harga_Jagung_Pipilan_Kg (asumsi), Is_Hari_Raya, Efek_Lebaran
    """
    model, scaler, metadata = load_artifacts()
    window = metadata["window_size"]
    features = metadata["features"]

    temp_df = load_and_prepare(DATA_PATH)

    hasil_rows = []

    for _ in range(days):
        # --- 1. Prediksi RETURN harian (BUKAN harga langsung) ---
        # Model dilatih untuk memprediksi persentase perubahan harga
        # (Return), bukan level harga -- lihat train.py. Return TIDAK
        # di-scale, jadi output model dipakai langsung (tidak lewat
        # scaler.inverse_transform).
        window_scaled = scaler.transform(temp_df[features])[-window:]
        window_scaled = window_scaled.reshape(1, window, len(features))

        pred_return = float(model.predict(window_scaled, verbose=0)[0][0])

        harga_sebelumnya = float(temp_df["Harga_Ayam"].iloc[-1])
        pred_real = reconstruct_harga(harga_sebelumnya, pred_return)

        # --- 2. Tentukan tanggal hari forecast ini ---
        tanggal_baru = temp_df["Tanggal"].iloc[-1] + pd.Timedelta(days=1)

        # --- 3. Proyeksikan harga pakan untuk hari ini ---
        harga_pakan_baru = _proyeksi_harga_pakan(temp_df, skenario_pakan, custom_rate_persen)

        # --- 4. Hitung flag kalender SECARA OTOMATIS untuk tanggal baru ---
        # (bug lama: flag ini disalin dari hari sebelumnya, sekarang
        # dihitung ulang sesuai kalender nasional yang sesungguhnya)
        flag_raya = is_hari_raya(tanggal_baru)
        flag_lebaran = efek_lebaran(tanggal_baru)

        new_row = {
            "Tanggal": tanggal_baru,
            "Wilayah": temp_df["Wilayah"].iloc[-1] if "Wilayah" in temp_df.columns else "Kota Bogor",
            "Harga_Ayam": pred_real,
            "Harga_Jagung_Pipilan_Kg": harga_pakan_baru,
            "Is_Hari_Raya": flag_raya,
            "Efek_Lebaran": flag_lebaran,
        }

        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

        # --- 5. Hitung ulang fitur turunan (Delta_Pakan, Rolling_Pakan_7, Ratio) ---
        temp_df = add_features(temp_df)

        hasil_rows.append({
            "Tanggal": tanggal_baru,
            "Harga_Ayam_Prediksi": round(pred_real, 2),
            "Harga_Jagung_Pipilan_Kg_Asumsi": round(harga_pakan_baru, 2),
            "Is_Hari_Raya": flag_raya,
            "Efek_Lebaran": flag_lebaran,
        })

    return pd.DataFrame(hasil_rows)


if __name__ == "__main__":
    print("\n=== FORECAST TEST (skenario pakan: flat) ===")
    hasil = predict_future(days=7, skenario_pakan="flat")
    print(hasil.to_string(index=False))

    print("\n=== FORECAST TEST (skenario pakan: tren 30 hari) ===")
    hasil_tren = predict_future(days=7, skenario_pakan="tren")
    print(hasil_tren.to_string(index=False))
