# Sistem Prediksi Harga Ayam Ras — Kota Bogor (LSTM Multivariate)

Model LSTM untuk memprediksi harga ayam ras harian, menggunakan fitur
harga pakan jagung pipilan, flag hari raya, dan efek Lebaran.

## Struktur File (setelah dirapikan)

| File | Fungsi |
|---|---|
| `preprocess.py` | **Tahap 1 preprocessing** — gabungkan data mentah (excel harga ayam + pakan), reindex ke kalender harian, isi tanggal bolong dengan **interpolasi linear** → `dataset_bersih_lstm.csv` |
| `locf_reinterpolasi.py` | **Tahap 2 preprocessing** — deteksi runtun nilai "macet" (stale/LOCF, ≥60 hari identik) yang lolos dari tahap 1, NaN-kan, lalu **interpolasi linear ulang** → `dataset_bersih_lstm_final.csv` |
| `feature_engineering.py` | Satu sumber kebenaran untuk fitur turunan (lag, rolling mean, return) — dipakai train/predict/app |
| `holiday_calendar.py` | Kalender hari libur nasional Indonesia — untuk flag `Is_Hari_Raya`/`Efek_Lebaran` pada tanggal forecast |
| `train.py` | Melatih model LSTM dan menyimpan artefak ke folder `models/` |
| `predict.py` | Forecast autoregressive lewat command line (tanpa UI) |
| `app.py` | Dashboard web Streamlit (Dashboard, Data Historis, Evaluasi Model, Optimasi, Tuning, Forecast, Tentang) |
| `optimize.py` / `tune.py` | Modul opsional untuk TimeSeriesCV, grid search window, dan grid search hyperparameter — dipanggil dari menu "Optimasi"/"Tuning" di dashboard |
| `bandingkan_model_a_b.py` | **Script validasi opsional** (bukan langkah wajib) — bukti empiris kenapa Tahap 2 dipertahankan (lihat bagian RMSE di bawah) |
| `dataset_bersih_lstm.csv` | Output Tahap 1 (interpolasi linear atas tanggal bolong) |
| `dataset_bersih_lstm_final.csv` | Output Tahap 2 — **dataset final yang dipakai train.py/predict.py/app.py/optimize.py** |
| `perbandingan_locf.csv` | Tabel transparansi: titik mana saja yang diinterpolasi ulang di Tahap 2 |
| `models/` | Artefak model terlatih (`lstm_model.keras`, `scaler.pkl`, `metadata.json`, `metrics.json`, `history.json`) |

## Cara Menjalankan

```bash
# 1. Install dependency
pip install -r requirements.txt

# 2. (Hanya perlu diulang jika data mentah excel/csv berubah)
python preprocess.py            # Tahap 1: gabung data + interpolasi linear
python locf_reinterpolasi.py    # Tahap 2: perbaiki runtun stale + interpolasi ulang

# 3. Latih model (pakai dataset_bersih_lstm_final.csv secara otomatis)
python train.py

# 4. (Opsional) Tes forecast lewat terminal
python predict.py

# 5. Jalankan dashboard web
streamlit run app.py
```

## Kenapa Preprocessing 2 Tahap? (Bukti RMSE Lebih Kecil)

Data harga ayam mentah ternyata punya banyak runtun nilai yang "macet"
(nilai sama persis selama berminggu-minggu) — ciri khas data stale/LOCF
dari sumbernya, BUKAN interpolasi linear tahap 1 yang hanya mengisi
tanggal yang benar-benar kosong. Sekitar **25% titik data harga ayam**
ternyata adalah runtun stale seperti ini.

Ketika runtun tsb dideteksi dan diinterpolasi ulang (Tahap 2), model
LSTM yang dilatih pada data hasil Tahap 2 terbukti lebih baik di ketiga
subset dibanding data Tahap 1 saja (lihat `perbandingan_model_a_b.csv`):

| Subset | RMSE (Tahap 1 saja) | RMSE (Tahap 1+2) | Perbaikan |
|---|---|---|---|
| Train | Rp413 | Rp390 | -5.7% |
| Test  | Rp707 | Rp664 | -6.1% |

MAE dan MAPE membaik lebih drastis lagi (MAE test -40%, MAPE test -40%).
Karena itu Tahap 2 **bukan lagi eksperimen opsional** — sudah dijadikan
default di seluruh aplikasi.

Catatan: percobaan grid search window size (7/14/21/30) dan tuning
hyperparameter (lihat `models/optimize_results.json`, `models/tune_results.json`)
menunjukkan RMSE relatif tidak banyak berubah dari perlakuan tsb — bukti
lebih lanjut bahwa **kualitas data (preprocessing) adalah pengaruh
terbesar terhadap RMSE**, jauh lebih besar dari tuning arsitektur.

## Bagian yang Dihapus Saat Perapian

File berikut dihapus karena duplikat/usang dan tidak lagi relevan untuk
alur produksi:
- `preprocess _lama.py`, `trainlama.py` — versi lama `preprocess.py`/`train.py` yang sudah digantikan sepenuhnya
- `experiment_lstm.py` — prototipe awal yang berdiri sendiri (tidak pakai `feature_engineering.py`, sumber data beda: `Dataset_Harian_Kota Bogor.csv`)
- `dataset_bersih_lstm datalama.csv`, `Dataset_Harian_Kota Bogor.csv`, `matriks_laporan_tuning_lstm.csv` — dataset/laporan yang hanya dipakai `experiment_lstm.py`
- `files (1)/` dan `files (1).zip` — duplikat persis dari file excel harga ayam yang sudah ada di folder utama
- `models_A/`, `models_B/` — folder hasil eksperimen A/B; `models_B` (RMSE lebih kecil) sudah **dipromosikan menjadi `models/`** sehingga tidak perlu 2 salinan model
- `__pycache__/` — file cache Python, dibuat ulang otomatis saat dijalankan

## Keterbatasan yang Perlu Diketahui

- Forecast bersifat **autoregressive** — semakin jauh horizon (>14 hari),
  semakin besar potensi akumulasi kesalahan.
- Proyeksi harga pakan untuk hari-hari ke depan adalah **asumsi**, bukan
  data aktual.
- Jika Anda melatih ulang dengan data yang menjangkau tahun setelah 2026,
  **tambahkan dulu** tanggal hari libur tahun tersebut ke
  `holiday_calendar.py` (lihat komentar di dalam file) — tanggal Idul
  Fitri/Idul Adha tidak bisa dihitung dengan rumus, harus dirujuk dari
  penetapan resmi Kemenag RI setiap tahun.
- `optimize.py`/`tune.py` memakai daftar fitur turunan yang sedikit
  lebih kaya (lag 14 hari, volatilitas return, dsb.) dibanding
  `feature_engineering.py` yang dipakai `train.py` — ini bawaan desain
  lama untuk eksplorasi tuning, belum disatukan. Model produksi di
  `models/` tetap konsisten pakai `feature_engineering.py`.
- Gunakan hasil sistem ini sebagai salah satu bahan pertimbangan, bukan
  satu-satunya dasar keputusan bisnis/finansial.
