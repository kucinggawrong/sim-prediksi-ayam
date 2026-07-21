"""
locf_reinterpolasi.py
=========================================================
TAHAP 2 PREPROCESSING: deteksi runtun nilai identik yang tidak wajar
(kemungkinan LOCF/stale data bawaan sumber) pada dataset_bersih_lstm.csv
(hasil preprocess.py tahap 1), ubah jadi NaN, lalu INTERPOLASI ULANG
secara linear -- memakai mekanisme interpolasi yang sama seperti di
preprocess.py, hanya targetnya beda: bukan mengisi tanggal yang bolong,
tapi mengisi ulang titik yang "macet" (nilai sama berhari-hari).

MENGAPA TAHAP INI DIPERTAHANKAN (bukan opsional lagi):
  Pengujian empiris (lihat perbandingan_model_a_b.csv, dihasilkan dari
  bandingkan_model_a_b.py) membuktikan dataset yang sudah melalui tahap
  ini menghasilkan RMSE model LSTM yang LEBIH KECIL di semua subset
  (train/val/test) dibanding dataset tahap 1 saja. Karena itu output
  tahap ini (dataset_bersih_lstm_final.csv) yang dipakai sebagai
  DATA_PATH default di train.py, predict.py, app.py, dan optimize.py.

LOGIKA DETEKSI:
  Untuk tiap runtun nilai yang sama persis secara berurutan dengan
  panjang >= MIN_RUN_LENGTH:
    - Titik PERTAMA dalam runtun tetap dipertahankan (dianggap
      observasi asli/anchor)
    - Titik-titik SETELAHNYA dalam runtun yang sama diubah jadi NaN
    - NaN tsb diisi ulang dengan interpolasi linear antara anchor
      awal runtun dan titik pertama runtun BERIKUTNYA yang beda nilai

CARA MENJALANKAN:
    python locf_reinterpolasi.py
Membutuhkan dataset_bersih_lstm.csv di folder yang sama (jalankan
preprocess.py terlebih dahulu jika belum ada).

Output:
    - dataset_bersih_lstm_final.csv    (dataset final, dipakai train.py
                                         /predict.py/app.py/optimize.py)
    - perbandingan_locf.csv            (tabel transparansi semua titik
                                         yang berubah: Tanggal, Kolom,
                                         Sebelum, Sesudah, Delta, Delta_persen)
    - ringkasan tercetak ke terminal
"""

import pandas as pd
import numpy as np

INPUT_PATH = "dataset_bersih_lstm.csv"
OUTPUT_FINAL = "dataset_bersih_lstm_final.csv"
OUTPUT_PERBANDINGAN = "perbandingan_locf.csv"

MIN_RUN_LENGTH = 60  # hanya tangkap runtun >= 60 hari (~2 bulan) identik --
                      # jauh di luar pola mingguan/bulanan wajar, kandidat kuat stale/LOCF
KOLOM_DIPROSES = ["Harga_Ayam", "Harga_Jagung_Pipilan_Kg"]


def deteksi_dan_interpolasi_ulang(s: pd.Series, min_run: int) -> tuple[pd.Series, pd.Series]:
    """Kembalikan (series_hasil, mask_yang_diubah)."""
    run_id = (s != s.shift()).cumsum()
    run_len = s.groupby(run_id).transform("size")
    is_first_in_run = s != s.shift()

    # NaN-kan titik ke-2 dst dalam runtun panjang (titik pertama = anchor)
    mask_to_nan = (run_len >= min_run) & (~is_first_in_run)

    s_masked = s.mask(mask_to_nan)
    s_interp = s_masked.interpolate(method="linear", limit_direction="both").round(0)
    return s_interp, mask_to_nan


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["Tanggal"])
    df = df.sort_values("Tanggal").reset_index(drop=True)

    df_final = df.copy()
    semua_perbandingan = []

    for kolom in KOLOM_DIPROSES:
        hasil, mask = deteksi_dan_interpolasi_ulang(df[kolom], MIN_RUN_LENGTH)
        df_final[kolom] = hasil

        n_diubah = int(mask.sum())
        print(f"[{kolom}] Titik dianggap LOCF & diinterpolasi ulang: {n_diubah} "
              f"dari {len(df)} ({n_diubah/len(df)*100:.1f}%)")

        sub = df.loc[mask, ["Tanggal"]].copy()
        sub["Kolom"] = kolom
        sub["Sebelum"] = df.loc[mask, kolom].values
        sub["Sesudah"] = df_final.loc[mask, kolom].values
        sub["Delta"] = sub["Sesudah"] - sub["Sebelum"]
        sub["Delta_persen"] = (sub["Delta"] / sub["Sebelum"] * 100).round(3)
        semua_perbandingan.append(sub)

    perbandingan = pd.concat(semua_perbandingan, ignore_index=True)
    perbandingan.to_csv(OUTPUT_PERBANDINGAN, index=False)

    df_final["Tanggal"] = df_final["Tanggal"].dt.strftime("%Y-%m-%d")
    for k in KOLOM_DIPROSES:
        df_final[k] = df_final[k].astype(int)
    df_final.to_csv(OUTPUT_FINAL, index=False)

    print(f"\n=== RINGKASAN PERBANDINGAN (semua kolom) ===")
    for kolom in KOLOM_DIPROSES:
        s = perbandingan[perbandingan["Kolom"] == kolom]
        if len(s) == 0:
            continue
        print(f"\n{kolom}:")
        print(f"  Jumlah titik berubah : {len(s)}")
        print(f"  Rata-rata |delta|    : Rp{s['Delta'].abs().mean():,.0f}")
        print(f"  Delta maksimum       : Rp{s['Delta'].max():,.0f} "
              f"(pada {s.loc[s['Delta'].idxmax(), 'Tanggal'].date()})")
        print(f"  Delta minimum        : Rp{s['Delta'].min():,.0f} "
              f"(pada {s.loc[s['Delta'].idxmin(), 'Tanggal'].date()})")
        print(f"  Rata-rata delta %    : {s['Delta_persen'].mean():.3f}%")

    print(f"\n✅ Dataset final disimpan ke: {OUTPUT_FINAL}")
    print(f"✅ Tabel transparansi perubahan disimpan ke: {OUTPUT_PERBANDINGAN}")
    print(f"\nSelanjutnya: jalankan 'python train.py' -- otomatis memakai "
          f"'{OUTPUT_FINAL}' sebagai data training (RMSE lebih kecil, lihat "
          f"perbandingan_model_a_b.csv untuk buktinya).")


if __name__ == "__main__":
    main()
