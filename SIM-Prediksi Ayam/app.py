"""
app.py
=========================================================
Dashboard Streamlit untuk Sistem Prediksi Harga Ayam Ras Kota
Bogor menggunakan model LSTM multivariate (harga pakan jagung,
flag hari raya, dan efek lebaran).

PERBAIKAN dari versi sebelumnya:
1. Feature engineering & flag kalender memakai modul bersama
   (feature_engineering.py & holiday_calendar.py) -- konsisten
   dengan train.py & predict.py, dan flag hari raya/lebaran pada
   hasil forecast kini dihitung otomatis per tanggal (bug lama:
   nilai disalin statis dari hari terakhir data historis).
2. Menu "Forecast": sumbu-x plot sekarang memakai TANGGAL asli
   (bukan index 0,1,2,...), pengguna bisa memilih skenario tren
   harga pakan, dan hasil bisa diunduh sebagai CSV.
3. Menu "Evaluasi Model": menambah metrik MAPE, menampilkan
   learning curve (train vs val loss) untuk membantu mengecek
   overfitting, dan membaca metrik dari models/metrics.json (hasil
   train.py) supaya konsisten dengan angka yang dipakai untuk
   melaporkan performa model -- bukan dihitung ulang dengan cara
   yang berbeda di setiap tempat.
4. Menambah menu "Tentang" yang menjelaskan fitur, arsitektur
   model, dan keterbatasan (penting secara etis: prediksi harga
   pangan punya konsekuensi nyata bagi peternak/pedagang, sehingga
   pengguna perlu tahu ini adalah ESTIMASI, bukan kepastian).
5. Penanganan error yang lebih ramah jika folder models/ belum ada
   (mengarahkan pengguna untuk menjalankan train.py dahulu) -- versi
   lama akan crash dengan traceback teknis yang membingungkan.

CARA MENJALANKAN:
    pip install streamlit tensorflow scikit-learn pandas numpy joblib matplotlib
    python train.py              # hanya sekali, atau setiap kali data diperbarui
    streamlit run app.py
"""

import os
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon

from feature_engineering import FEATURE_COLUMNS, add_features, load_and_prepare, reconstruct_harga
from holiday_calendar import is_hari_raya, efek_lebaran
from optimize import run_full_optimization
from tune import run_hyperparameter_tuning

# =========================================================
# 1. KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="SIM-Prediksi Harga Ayam Bogor",
    page_icon="🐓",
    layout="wide",
)

DATA_PATH = "dataset_bersih_lstm_final.csv"
MODEL_DIR = "models"

st.title("🐓 Sistem Prediksi Harga Ayam Ras — Kota Bogor")
st.caption("Model LSTM Multivariate · Harga Pakan, Hari Raya & Efek Lebaran")

# =========================================================
# 2. CEK KETERSEDIAAN MODEL (penanganan error yang ramah)
# =========================================================
model_files_required = [
    os.path.join(MODEL_DIR, "lstm_model.keras"),
    os.path.join(MODEL_DIR, "scaler.pkl"),
    os.path.join(MODEL_DIR, "metadata.json"),
]
model_belum_ada = any(not os.path.exists(p) for p in model_files_required)

if model_belum_ada:
    st.error(
        "⚠️ Model belum ditemukan di folder `models/`. "
        "Jalankan terlebih dahulu perintah berikut di terminal, lalu refresh halaman ini:"
    )
    st.code("python train.py", language="bash")
    st.stop()

# =========================================================
# 3. LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return load_and_prepare(DATA_PATH)

df = load_data()

# =========================================================
# 4. LOAD MODEL & ARTEFAK
# =========================================================
@st.cache_resource
def load_model_artifacts():
    model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "lstm_model.keras"), compile=False
    )
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_model_artifacts()
WINDOW = meta["window_size"]
FEATURES = meta["features"]

# Muat metrik & history training jika ada (dihasilkan train.py)
metrics_path = os.path.join(MODEL_DIR, "metrics.json")
history_path = os.path.join(MODEL_DIR, "history.json")
metrics_data = json.load(open(metrics_path)) if os.path.exists(metrics_path) else None
history_data = json.load(open(history_path)) if os.path.exists(history_path) else None

# =========================================================
# 5. SIDEBAR MENU
# =========================================================
menu = st.sidebar.radio(
    "Menu",
    ["Dashboard", "Data Historis", "Evaluasi Model", "Forecast", "Optimasi", "Tentang"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"📅 Data terakhir: **{df['Tanggal'].max().strftime('%d %B %Y')}**\n\n"
    f"📊 Total baris data: **{len(df):,}**"
)

# =========================================================
# 6. HALAMAN: DASHBOARD
# =========================================================
if menu == "Dashboard":

    st.subheader("📊 Ringkasan Harga Terbaru")

    harga_terakhir = df["Harga_Ayam"].iloc[-1]
    harga_sebelumnya = df["Harga_Ayam"].iloc[-2] if len(df) > 1 else harga_terakhir
    delta_ayam = harga_terakhir - harga_sebelumnya

    pakan_terakhir = df["Harga_Jagung_Pipilan_Kg"].iloc[-1]
    pakan_sebelumnya = df["Harga_Jagung_Pipilan_Kg"].iloc[-2] if len(df) > 1 else pakan_terakhir
    delta_pakan = pakan_terakhir - pakan_sebelumnya

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Harga Ayam Terakhir",
        f"Rp {int(harga_terakhir):,}",
        delta=f"{delta_ayam:+,.0f}",
    )
    col2.metric(
        "Harga Pakan (Jagung Pipilan)",
        f"Rp {int(pakan_terakhir):,}",
        delta=f"{delta_pakan:+,.0f}",
    )
    col3.metric(
        "Rasio Pakan : Ayam",
        f"{df['Ratio_Pakan_Ayam'].iloc[-1]:.3f}",
        help="Harga pakan dibagi harga ayam. Semakin tinggi rasio, semakin besar tekanan biaya pakan terhadap margin peternak.",
    )

    if bool(df["Is_Hari_Raya"].iloc[-1]):
        st.info("📌 Data terakhir bertepatan dengan **hari libur nasional**.")
    if bool(df["Efek_Lebaran"].iloc[-1]):
        st.info("📌 Data terakhir berada dalam periode **efek Lebaran**.")

    st.markdown("---")
    st.subheader("📈 Tren Harga Ayam vs Harga Pakan")

    rentang = st.select_slider(
        "Rentang waktu yang ditampilkan",
        options=["30 Hari", "90 Hari", "1 Tahun", "Semua Data"],
        value="90 Hari",
    )
    n_map = {"30 Hari": 30, "90 Hari": 90, "1 Tahun": 365, "Semua Data": len(df)}
    plot_df = df.tail(n_map[rentang])

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(plot_df["Tanggal"], plot_df["Harga_Ayam"], color="#d1495b", label="Harga Ayam")
    ax2.plot(plot_df["Tanggal"], plot_df["Harga_Jagung_Pipilan_Kg"], color="#3f7d20", label="Harga Pakan", alpha=0.7)

    # Tandai periode hari raya/lebaran pada grafik
    raya_dates = plot_df.loc[plot_df["Is_Hari_Raya"] == 1, "Tanggal"]
    for d in raya_dates:
        ax1.axvline(d, color="gray", alpha=0.15, linewidth=2)

    ax1.set_ylabel("Harga Ayam (Rp)", color="#d1495b")
    ax2.set_ylabel("Harga Pakan (Rp/Kg)", color="#3f7d20")
    ax1.set_title(f"Tren Harga Ayam vs Pakan — {rentang}")
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.98))
    fig.tight_layout()

    st.pyplot(fig)
    st.caption("Garis abu-abu vertikal menandai tanggal hari libur nasional pada rentang yang ditampilkan.")

# =========================================================
# 7. HALAMAN: DATA HISTORIS
# =========================================================
elif menu == "Data Historis":
    st.subheader("📋 Dataset Historis")

    col1, col2 = st.columns([1, 1])
    with col1:
        tampilkan_fitur_turunan = st.checkbox("Tampilkan kolom fitur turunan (Delta, Rolling, Ratio)", value=False)
    with col2:
        urutan = st.radio("Urutan", ["Terbaru dulu", "Terlama dulu"], horizontal=True)

    kolom_dasar = ["Tanggal", "Wilayah", "Harga_Ayam", "Harga_Jagung_Pipilan_Kg", "Is_Hari_Raya", "Efek_Lebaran"]
    kolom_tampil = kolom_dasar + (["Delta_Pakan", "Rolling_Pakan_7", "Ratio_Pakan_Ayam"] if tampilkan_fitur_turunan else [])
    kolom_tampil = [c for c in kolom_tampil if c in df.columns]

    tampil_df = df[kolom_tampil].copy()
    if urutan == "Terbaru dulu":
        tampil_df = tampil_df.sort_values("Tanggal", ascending=False)

    st.dataframe(tampil_df, width="stretch", height=480)

    st.download_button(
        "⬇️ Unduh dataset (CSV)",
        data=tampil_df.to_csv(index=False).encode("utf-8"),
        file_name="data_historis_harga_ayam_bogor.csv",
        mime="text/csv",
    )

# =========================================================
# 8. HALAMAN: EVALUASI MODEL
# =========================================================
elif menu == "Evaluasi Model":

    st.subheader("📈 Evaluasi Performa Model")

    if metrics_data:
        st.markdown("#### Ringkasan Metrik (dihitung sekali saat training, lihat `train.py`)")
        col_train, col_val, col_test = st.columns(3)

        for col, nama_set, label in [
            (col_train, "train", "Train"),
            (col_val, "val", "Validation"),
            (col_test, "test", "Test"),
        ]:
            data_set = metrics_data.get(nama_set, {})
            with col:
                st.markdown(f"**{label}** ({data_set.get('n_samples', '-')} sampel)")
                st.metric("MAE", f"Rp {data_set.get('mae', 0):,.0f}")
                st.metric("RMSE", f"Rp {data_set.get('rmse', 0):,.0f}")
                st.metric("MAPE", f"{data_set.get('mape', 0):.2f}%")

        st.caption(
            "MAE & RMSE dalam Rupiah (semakin kecil semakin baik). "
            "MAPE adalah rata-rata persentase kesalahan terhadap harga aktual — "
            "berguna untuk menilai akurasi relatif tanpa terpengaruh skala harga."
        )
    else:
        st.warning("File `models/metrics.json` tidak ditemukan. Jalankan ulang `train.py` versi terbaru untuk menghasilkannya.")

    st.markdown("---")
    st.markdown("#### Grafik Actual vs Prediksi (Data Test)")

    split_val = meta.get("val_frac", 0.85)
    split_idx = int(len(df) * split_val)
    test_df = df.iloc[split_idx:].copy()

    if len(test_df) <= WINDOW:
        st.warning("Data test terlalu sedikit untuk divisualisasikan (lebih kecil dari window model).")
    else:
        test_scaled = scaler.transform(test_df[FEATURES])

        X_test_eval = []
        for i in range(len(test_scaled) - WINDOW):
            X_test_eval.append(test_scaled[i:i + WINDOW])
        X_test_eval = np.array(X_test_eval)

        # Model memprediksi RETURN (bukan harga langsung) -- lihat train.py.
        # Return TIDAK di-scale, jadi output model dipakai langsung, direkonstruksi
        # jadi harga lewat reconstruct_harga() (harga_t = harga_(t-1) * (1+return)),
        # BUKAN scaler.inverse_transform (itu hanya valid untuk fitur yang di-scale).
        pred_return = model.predict(X_test_eval, verbose=0).flatten()

        harga_prev = test_df["Harga_Ayam"].values[WINDOW - 1: WINDOW - 1 + len(pred_return)]
        actual_real = test_df["Harga_Ayam"].values[WINDOW:WINDOW + len(pred_return)]
        pred_real = reconstruct_harga(harga_prev, pred_return)

        tanggal_plot = test_df["Tanggal"].values[WINDOW:WINDOW + len(pred_return)]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(tanggal_plot, actual_real, label="Aktual", color="#1d4e89", linewidth=1.8)
        ax.plot(tanggal_plot, pred_real, label="Prediksi", color="#d1495b", linewidth=1.8, linestyle="--")
        ax.set_title("Harga Ayam Aktual vs Prediksi (Data Test, di luar masa training)")
        ax.set_ylabel("Harga (Rp)")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()

        st.pyplot(fig)

        st.markdown("---")
        st.markdown("#### 🧪 Uji Statistik: Wilcoxon Signed-Rank Test (Aktual vs Prediksi)")
        st.caption(
            "Uji non-parametrik untuk data berpasangan (harga aktual vs prediksi pada "
            "tanggal yang sama) -- dipakai sebagai alternatif uji-t berpasangan karena "
            "tidak mengasumsikan selisihnya berdistribusi normal (cocok untuk data harga "
            "yang biasanya tidak simetris/normal). "
            "**H0:** median selisih (Aktual − Prediksi) = 0, artinya model **tidak bias** "
            "secara sistematis (tidak konsisten under/over-estimate)."
        )

        selisih = actual_real - pred_real
        n_efektif = int(np.sum(selisih != 0))

        if n_efektif < 10:
            st.warning(
                "Data test terlalu sedikit untuk uji Wilcoxon yang andal "
                "(idealnya minimal 10-20 pasangan dengan selisih tidak nol)."
            )
        else:
            try:
                w_stat, p_val_wilcoxon = wilcoxon(actual_real, pred_real, zero_method="wilcox")

                col_w1, col_w2, col_w3 = st.columns(3)
                col_w1.metric("N Pasangan (efektif)", f"{n_efektif}")
                col_w2.metric("Statistik W", f"{w_stat:.2f}")
                col_w3.metric(
                    "P-Value",
                    f"{p_val_wilcoxon:.4f}" if p_val_wilcoxon >= 0.0001 else "< 0.0001",
                )

                st.markdown(f"**Median selisih (Aktual − Prediksi):** Rp {np.median(selisih):,.0f}")

                alpha = 0.05
                if p_val_wilcoxon < alpha:
                    st.error(
                        f"❌ **Signifikan** (p = {p_val_wilcoxon:.4f} < {alpha}). "
                        "H0 ditolak — terdapat perbedaan sistematis antara harga aktual "
                        "dan hasil prediksi model (ada indikasi bias, entah cenderung "
                        "under-estimate atau over-estimate)."
                    )
                else:
                    st.success(
                        f"✅ **Tidak Signifikan** (p = {p_val_wilcoxon:.4f} ≥ {alpha}). "
                        "H0 gagal ditolak — tidak cukup bukti statistik bahwa prediksi "
                        "model berbeda secara sistematis dari harga aktual."
                    )

                fig_wx, ax_wx = plt.subplots(figsize=(9, 3))
                ax_wx.hist(selisih, bins=30, color="#3f7d20", edgecolor="white")
                ax_wx.axvline(0, color="#d1495b", linestyle="--", linewidth=2, label="Selisih = 0")
                ax_wx.axvline(np.median(selisih), color="#1d4e89", linestyle="-", linewidth=2, label="Median")
                ax_wx.set_xlabel("Selisih (Aktual − Prediksi) dalam Rupiah")
                ax_wx.set_ylabel("Frekuensi")
                ax_wx.set_title("Distribusi Selisih Aktual vs Prediksi (Data Test)")
                ax_wx.legend()
                fig_wx.tight_layout()
                st.pyplot(fig_wx)

                with st.expander("📥 Lihat & unduh data pasangan (Aktual vs Prediksi)"):
                    df_pairs = pd.DataFrame({
                        "Tanggal": tanggal_plot,
                        "Aktual": actual_real,
                        "Prediksi": np.round(pred_real, 0),
                        "Selisih": np.round(selisih, 0),
                    })
                    st.dataframe(df_pairs, width="stretch")
                    st.download_button(
                        "⬇️ Unduh data uji Wilcoxon (CSV)",
                        data=df_pairs.to_csv(index=False).encode("utf-8"),
                        file_name="data_uji_wilcoxon.csv",
                        mime="text/csv",
                    )
            except ValueError as e:
                st.warning(f"Uji Wilcoxon gagal dijalankan: {e}")

    if history_data and history_data.get("loss"):
        st.markdown("---")
        st.markdown("#### Learning Curve (Riwayat Training)")
        st.caption(
            "Jika garis val_loss terus naik sementara loss (train) terus turun, "
            "ini tanda model mulai overfitting."
        )

        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        ax2.plot(history_data["loss"], label="Train Loss")
        if history_data.get("val_loss"):
            ax2.plot(history_data["val_loss"], label="Validation Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE (skala ternormalisasi)")
        ax2.legend()
        fig2.tight_layout()

        st.pyplot(fig2)

# =========================================================
# 9. HALAMAN: FORECAST
# =========================================================
elif menu == "Forecast":

    st.subheader("🔮 Simulasi Forecast Harga Ayam")

    st.markdown(
        "Forecast dilakukan secara **autoregressive**: prediksi hari ke-N "
        "dipakai sebagai bagian input untuk memprediksi hari ke-(N+1), dan "
        "seterusnya. Semakin jauh horizon forecast, semakin besar potensi "
        "akumulasi ketidakpastian — gunakan hasil ini sebagai **estimasi**, bukan kepastian."
    )

    col1, col2 = st.columns(2)
    with col1:
        days = st.slider("Jumlah hari ke depan yang diprediksi", 1, 30, 7)
    with col2:
        skenario_pakan = st.selectbox(
            "Asumsi harga pakan ke depan",
            options=["flat", "tren", "custom"],
            format_func=lambda x: {
                "flat": "Tetap (flat) — harga pakan sama seperti hari terakhir",
                "tren": "Mengikuti tren 30 hari terakhir",
                "custom": "Custom — tentukan growth rate harian sendiri",
            }[x],
        )

    custom_rate = 0.0
    if skenario_pakan == "custom":
        custom_rate = st.slider(
            "Growth rate harga pakan per hari (%)", -2.0, 2.0, 0.0, step=0.1,
            help="Nilai positif = harga pakan diasumsikan naik setiap hari. Nilai negatif = turun.",
        )

    if st.button("🚀 Jalankan Forecast", type="primary"):

        with st.spinner("Menjalankan prediksi autoregressive..."):

            def proyeksi_harga_pakan(df_history, skenario, rate_persen=0.0):
                harga_terakhir = df_history["Harga_Jagung_Pipilan_Kg"].iloc[-1]
                if skenario == "flat":
                    return harga_terakhir
                if skenario == "tren":
                    lookback = df_history["Harga_Jagung_Pipilan_Kg"].tail(30)
                    delta_harian = lookback.diff().dropna().mean()
                    return harga_terakhir + (0.0 if pd.isna(delta_harian) else delta_harian)
                if skenario == "custom":
                    return harga_terakhir * (1 + rate_persen / 100.0)
                raise ValueError(f"Skenario tidak dikenal: {skenario}")

            temp_df = df.copy()
            hasil_rows = []

            for _ in range(days):
                window_scaled = scaler.transform(temp_df[FEATURES])[-WINDOW:]
                batch = window_scaled.reshape(1, WINDOW, len(FEATURES))

                # Model memprediksi RETURN (persentase perubahan), bukan harga
                # langsung -- lihat train.py. Return tidak di-scale, jadi output
                # model dipakai langsung, direkonstruksi jadi harga lewat
                # reconstruct_harga(), BUKAN scaler.inverse_transform.
                pred_return = float(model.predict(batch, verbose=0)[0][0])
                harga_sebelumnya = float(temp_df["Harga_Ayam"].iloc[-1])
                pred_real = reconstruct_harga(harga_sebelumnya, pred_return)

                tanggal_baru = temp_df["Tanggal"].iloc[-1] + pd.Timedelta(days=1)
                harga_pakan_baru = proyeksi_harga_pakan(temp_df, skenario_pakan, custom_rate)

                # Flag kalender dihitung otomatis berdasarkan tanggal -- BUKAN
                # disalin dari hari sebelumnya (perbaikan dari bug versi awal).
                flag_raya = is_hari_raya(tanggal_baru)
                flag_lebaran = efek_lebaran(tanggal_baru)

                new_row = temp_df.iloc[-1].copy()
                new_row["Tanggal"] = tanggal_baru
                new_row["Harga_Ayam"] = pred_real
                new_row["Harga_Jagung_Pipilan_Kg"] = harga_pakan_baru
                new_row["Is_Hari_Raya"] = flag_raya
                new_row["Efek_Lebaran"] = flag_lebaran

                temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
                temp_df = add_features(temp_df)

                hasil_rows.append({
                    "Tanggal": tanggal_baru,
                    "Harga_Ayam_Prediksi": round(pred_real, 0),
                    "Harga_Pakan_Asumsi": round(harga_pakan_baru, 0),
                    "Hari_Raya": "Ya" if flag_raya else "-",
                    "Efek_Lebaran": "Ya" if flag_lebaran else "-",
                })

            hasil_df = pd.DataFrame(hasil_rows)

            st.success(f"✅ Forecast {days} hari ke depan selesai!")

            fig, ax = plt.subplots(figsize=(11, 4.5))

            # Sertakan 14 hari terakhir data historis sebagai konteks
            konteks = df.tail(14)
            ax.plot(konteks["Tanggal"], konteks["Harga_Ayam"], marker="o", color="#1d4e89", label="Data Historis")
            ax.plot(hasil_df["Tanggal"], hasil_df["Harga_Ayam_Prediksi"], marker="o", color="#d1495b", linestyle="--", label="Forecast")

            for _, row in hasil_df.iterrows():
                if row["Hari_Raya"] == "Ya":
                    ax.axvline(row["Tanggal"], color="gray", alpha=0.2, linewidth=2)

            ax.set_title(f"Forecast Harga Ayam — {days} Hari ke Depan")
            ax.set_ylabel("Harga (Rp)")
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()

            st.pyplot(fig)

            st.markdown("#### Tabel Hasil Forecast")
            st.dataframe(hasil_df, width="stretch")

            st.download_button(
                "⬇️ Unduh hasil forecast (CSV)",
                data=hasil_df.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_harga_ayam_{days}hari.csv",
                mime="text/csv",
            )

            if hasil_df["Hari_Raya"].eq("Ya").any() or hasil_df["Efek_Lebaran"].eq("Ya").any():
                st.info(
                    "📌 Periode forecast ini melewati hari libur nasional dan/atau efek Lebaran "
                    "— histori menunjukkan harga ayam & pakan sering melonjak pada periode ini."
                )

# =========================================================
# 10. HALAMAN: OPTIMASI
# =========================================================
elif menu == "Optimasi":

    st.subheader("🔧 Optimasi Model (TimeSeriesCV & Window Size)")

    st.markdown(
        "Halaman ini menjalankan dua evaluasi tambahan di luar training utama:\n\n"
        "1. **TimeSeriesSplit Cross-Validation** (5-fold, window=14) — cek apakah "
        "performa model konsisten di beberapa potongan waktu berbeda, bukan cuma "
        "1x split train/val/test.\n"
        "2. **Grid Search Window Size** (7/14/21/30 hari) — cari panjang lookback "
        "yang menghasilkan RMSE terkecil.\n\n"
        "⚠️ **Proses ini melatih ulang beberapa model LSTM dari nol** (total 9 kali "
        "training: 5 fold CV + 4 window), jadi **bisa memakan waktu beberapa menit** "
        "tergantung spesifikasi komputer. Jalankan saat Anda punya waktu luang, "
        "bukan untuk dicoba berkali-kali secara cepat."
    )

    optimize_results_path = os.path.join(MODEL_DIR, "optimize_results.json")
    hasil_lama = json.load(open(optimize_results_path)) if os.path.exists(optimize_results_path) else None

    if hasil_lama:
        mtime = os.path.getmtime(optimize_results_path)
        st.caption(f"📄 Menampilkan hasil optimasi terakhir (disimpan: "
                   f"{pd.Timestamp(mtime, unit='s').strftime('%d %B %Y, %H:%M')})")

    jalankan = st.button("🚀 Jalankan Optimasi Baru", type="primary")

    if jalankan:
        log_box = st.empty()
        log_lines = []

        def update_log(msg: str):
            log_lines.append(msg)
            log_box.code("\n".join(log_lines), language=None)

        with st.spinner("Menjalankan TimeSeriesCV + grid search window... (bisa beberapa menit)"):
            hasil_lama = run_full_optimization(
                data_path=DATA_PATH, model_dir=MODEL_DIR, progress_callback=update_log,
            )
        st.success("✅ Optimasi selesai! Hasil di bawah sudah yang paling baru.")

    if not hasil_lama:
        st.info(
            "Belum ada hasil optimasi tersimpan. Klik tombol **Jalankan Optimasi Baru** "
            "di atas untuk memulai."
        )
    else:
        st.markdown("---")
        st.markdown("#### 1. Hasil TimeSeriesSplit Cross-Validation (5-fold, WINDOW=14)")

        cv_results = hasil_lama.get("timeseries_cv", [])
        if cv_results:
            cv_df = pd.DataFrame(cv_results).rename(columns={
                "fold": "Fold", "test_start": "Mulai Test", "test_end": "Akhir Test",
                "mae": "MAE (Rp)", "rmse": "RMSE (Rp)", "mape": "MAPE (%)", "n_samples": "N Sampel",
            })
            st.dataframe(cv_df, width="stretch")

            mae_list = [r["mae"] for r in cv_results]
            rmse_list = [r["rmse"] for r in cv_results]
            col_a, col_b = st.columns(2)
            col_a.metric("Rata-rata MAE (5 fold)", f"Rp {np.mean(mae_list):,.0f}",
                          help=f"Std antar fold: Rp {np.std(mae_list):,.0f}")
            col_b.metric("Rata-rata RMSE (5 fold)", f"Rp {np.mean(rmse_list):,.0f}",
                          help=f"Std antar fold: Rp {np.std(rmse_list):,.0f}")
            st.caption(
                "Std (standar deviasi) yang besar dibanding rata-rata berarti performa model "
                "tidak konsisten antar periode waktu — kemungkinan ada fold yang bertepatan "
                "dengan lonjakan harga/Lebaran."
            )

            fig_cv, ax_cv = plt.subplots(figsize=(9, 3.5))
            ax_cv.bar([f"Fold {r['fold']}" for r in cv_results], [r["rmse"] for r in cv_results],
                      color="#3f7d20")
            ax_cv.axhline(np.mean(rmse_list), color="#d1495b", linestyle="--", label="Rata-rata RMSE")
            ax_cv.set_ylabel("RMSE (Rp)")
            ax_cv.set_title("RMSE per Fold")
            ax_cv.legend()
            fig_cv.tight_layout()
            st.pyplot(fig_cv)
        else:
            st.warning("Tidak ada hasil TimeSeriesCV (kemungkinan data terlalu sedikit untuk 5 fold).")

        st.markdown("---")
        st.markdown("#### 2. Grid Search Window Size (7 / 14 / 21 / 30 hari)")

        window_results = hasil_lama.get("window_grid_search", [])
        if window_results:
            win_df = pd.DataFrame(window_results).rename(columns={
                "window": "Window (hari)", "mae": "MAE (Rp)", "rmse": "RMSE (Rp)",
                "mape": "MAPE (%)", "n_samples": "N Sampel",
            })
            st.dataframe(win_df, width="stretch")

            best_window = hasil_lama.get("best_window")
            if best_window is not None:
                st.success(f"🏆 Window terbaik (RMSE terkecil): **{best_window} hari**")
                if best_window != WINDOW:
                    st.warning(
                        f"⚠️ Model yang sedang dipakai di menu Forecast & Evaluasi Model saat ini "
                        f"memakai window **{WINDOW} hari**, berbeda dari window terbaik hasil "
                        f"optimasi ini ({best_window} hari). Untuk memakai window terbaik, "
                        f"latih ulang model lewat `train.py` dengan window={best_window}, lalu refresh halaman ini."
                    )

            windows_plot = [r["window"] for r in window_results]
            mape_plot = [r["mape"] for r in window_results]
            rmse_plot = [r["rmse"] for r in window_results]

            COLOR_MAPE = "#1F4E79"
            COLOR_RMSE = "#C00000"
            COLOR_TERPILIH = "#D4A017"

            fig_w, ax1 = plt.subplots(figsize=(8, 5.5), dpi=200)

            ax1.plot(windows_plot, mape_plot, marker="o", color=COLOR_MAPE,
                     linewidth=2, label="MAPE (%)")
            ax1.set_xlabel("Window Size (Hari)")
            ax1.set_ylabel("MAPE (%)", color=COLOR_MAPE, fontweight="bold")
            ax1.tick_params(axis="y", labelcolor=COLOR_MAPE)
            for w, v in zip(windows_plot, mape_plot):
                ax1.annotate(f"{v:.2f}%", (w, v), textcoords="offset points", xytext=(0, 10),
                             ha="center", color=COLOR_MAPE, fontsize=9)

            ax2 = ax1.twinx()
            ax2.plot(windows_plot, rmse_plot, marker="s", linestyle="--", color=COLOR_RMSE,
                      linewidth=2, label="RMSE (Rp)")
            ax2.set_ylabel("RMSE (Rp)", color=COLOR_RMSE, fontweight="bold")
            ax2.tick_params(axis="y", labelcolor=COLOR_RMSE)
            for w, v in zip(windows_plot, rmse_plot):
                ax2.annotate(f"Rp{v:.0f}", (w, v), textcoords="offset points", xytext=(0, -15),
                             ha="center", color=COLOR_RMSE, fontsize=9)

            mape_pad = max((max(mape_plot) - min(mape_plot)) * 0.25, 0.005)
            rmse_pad = max((max(rmse_plot) - min(rmse_plot)) * 0.25, 1.0)
            ax1.set_ylim(min(mape_plot) - mape_pad, max(mape_plot) + mape_pad)
            ax2.set_ylim(min(rmse_plot) - rmse_pad, max(rmse_plot) + rmse_pad)

            if best_window is not None and best_window in windows_plot:
                ax1.axvline(best_window, color=COLOR_TERPILIH, linestyle=":", linewidth=2, alpha=0.7)
                ax1.annotate("Window terpilih (RMSE terkecil)",
                             xy=(best_window, max(mape_plot) + mape_pad * 0.7),
                             ha="center", fontsize=9, color="#8a6d0a", fontweight="bold")

            ax1.set_xticks(windows_plot)
            ax1.set_title("Hubungan Window Size terhadap MAPE dan RMSE", fontsize=12)
            ax1.grid(axis="both", linestyle="--", alpha=0.3)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
                       bbox_to_anchor=(0.5, -0.12), ncol=2)

            fig_w.tight_layout()
            st.pyplot(fig_w)
        else:
            st.warning("Tidak ada hasil grid search window (kemungkinan data terlalu sedikit).")

        st.download_button(
            "⬇️ Unduh hasil optimasi (JSON)",
            data=json.dumps(hasil_lama, indent=2).encode("utf-8"),
            file_name="optimize_results.json",
            mime="application/json",
        )

    st.markdown("---")
    st.markdown("---")
    st.markdown("### 🎛️ Grid Search Hyperparameter (units, dropout, learning rate, batch size)")
    st.markdown(
        "Hasil grid search **window size** di atas menunjukkan window nyaris "
        "tidak berpengaruh ke RMSE — jadi bagian ini mencoba parameter lain "
        "yang lebih mungkin berdampak: jumlah unit LSTM, dropout rate, "
        "learning rate, dan batch size."
    )
    st.caption(
        "⚠️ Tiap kombinasi = 1x training LSTM dari nol. Dibatasi maksimal "
        "16 kombinasi (dari grid penuh) supaya tidak berjalan berjam-jam."
    )

    tune_results_path = os.path.join(MODEL_DIR, "tune_results.json")
    tune_hasil_lama = json.load(open(tune_results_path)) if os.path.exists(tune_results_path) else None

    if tune_hasil_lama:
        mtime_t = os.path.getmtime(tune_results_path)
        st.caption(f"📄 Menampilkan hasil tuning terakhir (disimpan: "
                   f"{pd.Timestamp(mtime_t, unit='s').strftime('%d %B %Y, %H:%M')})")

    jalankan_tune = st.button("🚀 Jalankan Hyperparameter Tuning")

    if jalankan_tune:
        log_box_tune = st.empty()
        log_lines_tune = []

        def update_log_tune(msg: str):
            log_lines_tune.append(msg)
            log_box_tune.code("\n".join(log_lines_tune), language=None)

        with st.spinner("Menjalankan grid search hyperparameter... (bisa cukup lama)"):
            tune_hasil_lama = run_hyperparameter_tuning(
                data_path=DATA_PATH, model_dir=MODEL_DIR, progress_callback=update_log_tune,
            )
        st.success("✅ Tuning selesai! Hasil di bawah sudah yang paling baru.")

    if not tune_hasil_lama:
        st.info(
            "Belum ada hasil tuning tersimpan. Klik tombol **Jalankan Hyperparameter Tuning** "
            "di atas untuk memulai."
        )
    else:
        tune_res = tune_hasil_lama.get("results", [])
        if tune_res:
            tune_df = pd.DataFrame(tune_res).sort_values("rmse").rename(columns={
                "units1": "Unit LSTM-1", "units2": "Unit LSTM-2", "dropout": "Dropout",
                "learning_rate": "Learning Rate", "batch_size": "Batch Size",
                "mae": "MAE (Rp)", "rmse": "RMSE (Rp)", "mape": "MAPE (%)", "n_samples": "N Sampel",
            })
            st.dataframe(tune_df, width="stretch")

            best_combo = tune_hasil_lama.get("best")
            if best_combo:
                st.success(
                    f"🏆 Kombinasi terbaik: unit LSTM ({best_combo['units1']}, {best_combo['units2']}), "
                    f"dropout {best_combo['dropout']}, learning rate {best_combo['learning_rate']}, "
                    f"batch size {best_combo['batch_size']} → RMSE=Rp{best_combo['rmse']:,.0f}"
                )
                st.info(
                    "Kalau kombinasi ini lebih baik dari model produksi saat ini, salin nilai "
                    "parameter di atas ke `train.py` (fungsi `build_model`), lalu jalankan ulang "
                    "`python train.py` supaya model produksi memakainya."
                )
        else:
            st.warning("Tidak ada hasil tuning (kemungkinan data terlalu sedikit).")

# =========================================================
# 11. HALAMAN: TENTANG
# =========================================================
elif menu == "Tentang":
    st.subheader("ℹ️ Tentang Sistem Ini")

    st.markdown(
        f"""
**Tujuan**
Memprediksi harga ayam ras harian di Kota Bogor menggunakan model
**LSTM (Long Short-Term Memory) multivariate**, dengan mempertimbangkan
harga pakan (jagung pipilan), serta pola musiman hari raya/Lebaran.

**Fitur yang digunakan model**
| Fitur | Keterangan |
|---|---|
| `Harga_Ayam` | Target yang diprediksi (juga dipakai sebagai fitur lag) |
| `Harga_Jagung_Pipilan_Kg` | Harga pakan utama ayam ras |
| `Ratio_Pakan_Ayam` | Rasio harga pakan terhadap harga ayam |
| `Delta_Pakan` | Perubahan harga pakan harian |
| `Rolling_Pakan_7` | Rata-rata bergerak 7 hari harga pakan |
| `Is_Hari_Raya` | Penanda hari libur nasional |
| `Efek_Lebaran` | Penanda periode di sekitar Idul Fitri (window ±beberapa hari) |

**Arsitektur model**: LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → Dense(1),
dilatih dengan window lookback **{WINDOW} hari** untuk memprediksi 1 hari ke depan.

**Pembagian data**: 70% data terlama untuk training, 15% berikutnya untuk
validasi, 15% terakhir untuk pengujian (test) — dibagi berurutan sesuai
waktu (time-based split), bukan acak, agar tidak ada kebocoran informasi
dari masa depan ke masa lalu.

---
### ⚠️ Keterbatasan Penting
- Forecast bersifat **autoregressive**: prediksi yang meleset di hari
  awal akan ikut mempengaruhi prediksi hari-hari berikutnya. Akurasi
  cenderung **menurun untuk horizon yang lebih jauh** (di atas ~7-14 hari).
- Proyeksi harga pakan untuk hari-hari forecast adalah **asumsi**
  (flat/tren/custom), bukan data aktual — pilih skenario sesuai
  pengetahuan/sumber Anda tentang kondisi pasar pakan terkini.
- Flag hari raya & efek Lebaran untuk tanggal forecast dihitung dari
  tabel kalender nasional yang sudah ditetapkan; jika ada perubahan
  penetapan resmi (mis. hasil sidang isbat berbeda dari estimasi),
  hasil forecast pada tanggal tersebut dapat sedikit bergeser akurasinya.
- Model ini **tidak memperhitungkan** faktor eksternal mendadak seperti
  kebijakan pemerintah, wabah penyakit unggas, atau gangguan distribusi
  yang tidak tercermin dalam data historis.
- Gunakan hasil sistem ini sebagai **salah satu bahan pertimbangan**,
  bukan satu-satunya dasar pengambilan keputusan bisnis/finansial.
        """
    )