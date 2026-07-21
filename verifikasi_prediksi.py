"""
verifikasi_prediksi.py
=========================================================
Script SATU-KALI-PAKAI untuk MEMBUKTIKAN bahwa nilai "Prediksi"
di data_uji_wilcoxon.csv memang benar-benar keluaran model LSTM
(bukan asumsi/karangan) -- dijalankan dengan cara PERSIS SAMA
seperti yang dilakukan app.py di halaman "Evaluasi Model".

Ini BUKAN predict.py (yang autoregressive, untuk forecast ke
depan) -- ini meniru cara EVALUASI di app.py, yaitu memprediksi
seluruh baris di data TEST sekaligus, satu per satu dibandingkan
dengan harga aktualnya.

CARA MENJALANKAN:
    python verifikasi_prediksi.py

Pastikan file ini diletakkan di folder yang sama dengan app.py
(folder "SIM-Prediksi Ayam"), karena butuh feature_engineering.py,
dataset_bersih_lstm_final.csv, dan folder models/.
"""

import json
import os

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from feature_engineering import load_and_prepare, reconstruct_harga

DATA_PATH = "dataset_bersih_lstm_final.csv"
MODEL_DIR = "models"


def main():
    # =====================================================
    # 1. Muat model, scaler, metadata -- PERSIS seperti app.py
    # =====================================================
    print("Memuat model, scaler, metadata ...")
    model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "lstm_model.keras"), compile=False
    )
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
        meta = json.load(f)

    WINDOW = meta["window_size"]
    FEATURES = meta["features"]
    print(f"Window size = {WINDOW}, Fitur = {FEATURES}")

    # =====================================================
    # 2. Muat & siapkan data -- PERSIS seperti app.py
    # =====================================================
    df = load_and_prepare(DATA_PATH)
    print(f"Total baris dataset = {len(df)}")

    # =====================================================
    # 3. Split data test -- PERSIS seperti app.py (baris ~263-265)
    # =====================================================
    split_val = meta.get("val_frac", 0.85)
    split_idx = int(len(df) * split_val)
    test_df = df.iloc[split_idx:].copy()
    print(f"Split index = {split_idx}, jumlah baris test_df = {len(test_df)}")

    # =====================================================
    # 4. Scale fitur test set -- PERSIS seperti app.py
    # =====================================================
    test_scaled = scaler.transform(test_df[FEATURES])

    # =====================================================
    # 5. Bentuk sequence window 14 hari untuk SETIAP hari test
    #    -- PERSIS seperti app.py (baris ~272-275)
    # =====================================================
    X_test_eval = []
    for i in range(len(test_scaled) - WINDOW):
        X_test_eval.append(test_scaled[i:i + WINDOW])
    X_test_eval = np.array(X_test_eval)
    print(f"Jumlah sequence yang akan diprediksi = {len(X_test_eval)}  (harus = 290)")

    # =====================================================
    # 6. INI BAGIAN UTAMANYA: model.predict() -- LSTM menebak
    #    Return untuk SETIAP hari test sekaligus
    # =====================================================
    print("\nMenjalankan model.predict() ... (mungkin makan waktu beberapa detik)")
    pred_return = model.predict(X_test_eval, verbose=0).flatten()

    # =====================================================
    # 7. Rekonstruksi Return -> Harga Rupiah -- PERSIS seperti app.py
    # =====================================================
    harga_prev = test_df["Harga_Ayam"].values[WINDOW - 1: WINDOW - 1 + len(pred_return)]
    actual_real = test_df["Harga_Ayam"].values[WINDOW:WINDOW + len(pred_return)]
    pred_real = reconstruct_harga(harga_prev, pred_return)
    tanggal_plot = test_df["Tanggal"].values[WINDOW:WINDOW + len(pred_return)]

    hasil = pd.DataFrame({
        "Tanggal": tanggal_plot,
        "Aktual": actual_real,
        "Prediksi": np.round(pred_real, 2),
        "Return_Tebakan_Model": np.round(pred_return, 6),
    })

    # =====================================================
    # 8. Tampilkan 10 baris pertama untuk dicocokkan manual
    #    dengan data_uji_wilcoxon.csv Anda
    # =====================================================
    print("\n" + "=" * 70)
    print("HASIL VERIFIKASI (10 baris pertama data test):")
    print("=" * 70)
    print(hasil.head(10).to_string(index=False))

    # =====================================================
    # 9. Cek khusus baris 1 Oktober 2025 (contoh yang kita
    #    bahas di percakapan sebelumnya)
    # =====================================================
    cek = hasil[hasil["Tanggal"] == np.datetime64("2025-10-01")]
    if not cek.empty:
        print("\n" + "=" * 70)
        print("CEK KHUSUS TANGGAL 1 OKTOBER 2025:")
        print("=" * 70)
        print(cek.to_string(index=False))
        print(
            "\nBandingkan kolom 'Prediksi' di atas dengan nilai 39412.0 "
            "di data_uji_wilcoxon.csv -- kalau sama persis, berarti TERBUKTI "
            "model LSTM sungguhan yang menghasilkan angka itu, bukan asumsi."
        )
        print(
            "Kolom 'Return_Tebakan_Model' adalah angka Return ASLI langsung "
            "dari model.predict() -- bandingkan dengan -0.002228 yang "
            "dihitung mundur sebelumnya."
        )

    # =====================================================
    # 10. Simpan hasil lengkap (290 baris) untuk verifikasi total
    # =====================================================
    hasil.to_csv("verifikasi_prediksi_lengkap.csv", index=False)
    print(f"\nHasil lengkap ({len(hasil)} baris) disimpan ke: verifikasi_prediksi_lengkap.csv")


if __name__ == "__main__":
    main()