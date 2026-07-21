"""
Praprocessing dataset harga ayam (Kota Bogor, 2021-2026) + harga pakan
jagung pipilan (data resmi Badan Pangan Nasional, Jawa Barat, bulanan)
menjadi satu dataset time-series harian bersih untuk LSTM.

Tahapan:
1. TRANSFORMASI  : baca 6 file excel ayam (format lebar -> long format)
                    dan file resmi pakan (filter komoditas + provinsi)
2. CLEANING      : bersihkan string harga, parsing tanggal & nama bulan
3. PENGGABUNGAN  : gabung semua tahun ayam; gabung pakan ke tanggal ayam
4. IMPUTASI      : reindex ke kalender harian penuh, isi missing value
                    dengan interpolasi linear (limit_direction='both')
5. FEATURE       : Is_Hari_Raya, Efek_Lebaran, dan Pakan_Flat_Fill (flag
                    transparansi: 1 = di luar rentang resmi data pakan,
                    nilainya hasil flat-fill dari bulan resmi terdekat)
6. OUTPUT        : dataset_bersih_lstm.csv
"""

import pandas as pd
import numpy as np
import glob
import os
import re

# ==== PATH (relatif terhadap folder file ini -- portable di Windows/Mac/Linux) ====
INPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(INPUT_DIR, "dataset_bersih_lstm.csv")

WILAYAH_TARGET = "Kota Bogor"
NAMA_KOLOM_HARGA = "Harga_Ayam"

FILE_PAKAN_XLSX = os.path.join(INPUT_DIR, "Dataset_Harga_Pakan_Jagung_2021_2026.xlsx")
FILE_PAKAN_CSV = os.path.join(INPUT_DIR, "Pakan_Baru.csv")
NAMA_KOLOM_PAKAN = "Harga_Jagung_Pipilan_Kg"

# Tanggal Idul Fitri (hari pertama, resmi pemerintah RI) 2021-2026 - untuk Efek_Lebaran
TANGGAL_LEBARAN = {
    2021: "2021-05-13", 2022: "2022-05-02", 2023: "2023-04-22",
    2024: "2024-04-10", 2025: "2025-03-31", 2026: "2026-03-21",
}
EFEK_LEBARAN_SEBELUM = 7   # H-7
EFEK_LEBARAN_SESUDAH = 3   # H+3 (termasuk hari-H)

# Seluruh hari libur nasional resmi RI (SKB 3 Menteri) 2021-2026
HARI_LIBUR_NASIONAL = [
    # 2021
    "2021-01-01", "2021-02-12", "2021-03-11", "2021-03-14", "2021-04-02",
    "2021-05-01", "2021-05-13", "2021-05-14", "2021-05-26", "2021-06-01",
    "2021-07-20", "2021-08-11", "2021-08-17", "2021-10-20", "2021-12-25",
    # 2022
    "2022-01-01", "2022-02-01", "2022-02-28", "2022-03-03", "2022-04-15",
    "2022-05-01", "2022-05-02", "2022-05-03", "2022-05-16", "2022-05-26",
    "2022-06-01", "2022-07-09", "2022-07-30", "2022-08-17", "2022-10-08",
    "2022-12-25",
    # 2023
    "2023-01-01", "2023-01-22", "2023-02-18", "2023-03-22", "2023-04-07",
    "2023-04-21", "2023-04-22", "2023-04-23", "2023-05-01", "2023-05-18",
    "2023-06-01", "2023-06-04", "2023-06-29", "2023-07-19", "2023-08-17",
    "2023-09-28", "2023-12-25",
    # 2024
    "2024-01-01", "2024-02-08", "2024-02-10", "2024-03-11", "2024-03-29",
    "2024-04-10", "2024-04-11", "2024-05-01", "2024-05-09", "2024-05-23",
    "2024-06-01", "2024-06-17", "2024-07-07", "2024-08-17", "2024-09-16",
    "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-27", "2025-01-29", "2025-03-27", "2025-03-28",
    "2025-03-31", "2025-04-01", "2025-05-01", "2025-05-12", "2025-05-29",
    "2025-06-01", "2025-06-06", "2025-06-27", "2025-08-17", "2025-09-05",
    "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-16", "2026-02-16", "2026-02-17", "2026-03-18",
    "2026-03-19", "2026-03-20", "2026-03-21", "2026-03-22", "2026-03-23",
    "2026-03-24", "2026-04-03", "2026-04-05", "2026-05-01", "2026-05-14",
    "2026-05-15", "2026-05-27", "2026-05-28", "2026-05-31", "2026-06-01",
    "2026-06-16",
]


def parse_harga(x):
    """Bersihkan string harga seperti '13,900' -> 13900.0. Nilai kosong/'-' -> NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = re.sub(r"[^\d]", "", str(x))
    return float(s) if s else np.nan


def parse_tanggal_kolom(col_name):
    """Ubah header kolom tanggal seperti '01/ 01/ 2021' -> Timestamp."""
    s = re.sub(r"\s+", "", str(col_name))
    try:
        return pd.to_datetime(s, format="%d/%m/%Y")
    except ValueError:
        return None


def load_satu_file_ayam(path):
    """Baca 1 file excel ayam -> long format untuk Kota Bogor saja."""
    df = pd.read_excel(path)
    kolom_wilayah = df.columns[1]
    baris = df[df[kolom_wilayah].astype(str).str.strip() == WILAYAH_TARGET]
    if baris.empty:
        raise ValueError(f"Baris '{WILAYAH_TARGET}' tidak ditemukan di {path}")
    baris = baris.iloc[0]

    records = []
    for col in df.columns[2:]:
        tgl = parse_tanggal_kolom(col)
        if tgl is None:
            continue
        records.append({"Tanggal": tgl, NAMA_KOLOM_HARGA: parse_harga(baris[col])})
    return pd.DataFrame(records)


def load_semua_ayam():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "Tabel_Harga_Berdasarkan_Komoditas_*.xlsx")))
    if not files:
        raise FileNotFoundError(
            f"File excel harga ayam tidak ditemukan di '{INPUT_DIR}'. "
            f"Pastikan file bernama 'Tabel_Harga_Berdasarkan_Komoditas_*.xlsx' "
            f"ada langsung di folder tersebut."
        )

    print(f"Ditemukan {len(files)} file harga ayam:")
    all_df = []
    for f in files:
        d = load_satu_file_ayam(f)
        all_df.append(d)
        print(f"  - {os.path.basename(f)}: {len(d)} baris, {d[NAMA_KOLOM_HARGA].isna().sum()} kosong")

    df = pd.concat(all_df, ignore_index=True)
    df = df.drop_duplicates(subset="Tanggal", keep="last").sort_values("Tanggal")
    return df.reset_index(drop=True)


def load_pakan():
    """Gabung 2 sumber data pakan jagung awal (harian), isi celah tanggal dengan interpolasi linear."""
    if not os.path.exists(FILE_PAKAN_XLSX):
        raise FileNotFoundError(f"File pakan tidak ditemukan: {FILE_PAKAN_XLSX}")
    if not os.path.exists(FILE_PAKAN_CSV):
        raise FileNotFoundError(f"File pakan tidak ditemukan: {FILE_PAKAN_CSV}")

    df_x = pd.read_excel(FILE_PAKAN_XLSX)
    df_c = pd.read_csv(FILE_PAKAN_CSV)
    for d in (df_x, df_c):
        d["Tanggal"] = pd.to_datetime(d["Tanggal"])
        d[NAMA_KOLOM_PAKAN] = pd.to_numeric(d[NAMA_KOLOM_PAKAN], errors="coerce")

    df = pd.concat([df_x, df_c], ignore_index=True)
    df = df.drop_duplicates(subset="Tanggal", keep="last").sort_values("Tanggal")
    print(f"Data pakan (awal): {len(df)} baris, "
          f"{df['Tanggal'].min().strftime('%Y-%m-%d')} s.d. {df['Tanggal'].max().strftime('%Y-%m-%d')}")
    return df[["Tanggal", NAMA_KOLOM_PAKAN]].reset_index(drop=True)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # ---- 1-3. Ayam ----
    df = load_semua_ayam()
    full_range = pd.date_range(df["Tanggal"].min(), df["Tanggal"].max(), freq="D")
    df = df.set_index("Tanggal").reindex(full_range)
    df.index.name = "Tanggal"

    n_before = df[NAMA_KOLOM_HARGA].isna().sum()
    df[NAMA_KOLOM_HARGA] = df[NAMA_KOLOM_HARGA].interpolate(method="linear", limit_direction="both")
    df[NAMA_KOLOM_HARGA] = df[NAMA_KOLOM_HARGA].round(0)  # bulatkan ke rupiah utuh, hindari desimal panjang hasil interpolasi
    print(f"\nAyam - missing sebelum interpolasi: {n_before}, sesudah: {df[NAMA_KOLOM_HARGA].isna().sum()}")

    df = df.reset_index()
    df["Wilayah"] = WILAYAH_TARGET

    # ---- Pakan (harian, gabungan 2 file, interpolasi linear untuk celah) ----
    df_pakan = load_pakan()
    df_pakan = df_pakan.set_index("Tanggal").reindex(full_range)
    df_pakan.index.name = "Tanggal"

    n_pakan_before = df_pakan[NAMA_KOLOM_PAKAN].isna().sum()
    df_pakan[NAMA_KOLOM_PAKAN] = df_pakan[NAMA_KOLOM_PAKAN].interpolate(
        method="linear", limit_direction="both"
    )
    df_pakan[NAMA_KOLOM_PAKAN] = df_pakan[NAMA_KOLOM_PAKAN].round(0)  # bulatkan ke rupiah utuh
    print(f"Pakan - missing sebelum interpolasi: {n_pakan_before}, sesudah: {df_pakan[NAMA_KOLOM_PAKAN].isna().sum()}")

    df_pakan = df_pakan.reset_index()
    df = df.merge(df_pakan, on="Tanggal", how="left")

    # ---- Fitur Hari Raya (semua libur nasional) / Efek Lebaran (H-7 s.d H+3) ----
    libur_dates = pd.to_datetime(HARI_LIBUR_NASIONAL)
    df["Is_Hari_Raya"] = df["Tanggal"].isin(libur_dates).astype(int)

    lebaran_dates = pd.to_datetime(list(TANGGAL_LEBARAN.values()))

    def efek_lebaran(tgl):
        for ld in lebaran_dates:
            delta = (tgl - ld).days
            if -EFEK_LEBARAN_SEBELUM <= delta <= EFEK_LEBARAN_SESUDAH:
                return 1
        return 0

    df["Efek_Lebaran"] = df["Tanggal"].apply(efek_lebaran)

    # ---- Susun kolom akhir ----
    df["Tanggal"] = df["Tanggal"].dt.strftime("%Y-%m-%d")
    df[NAMA_KOLOM_HARGA] = df[NAMA_KOLOM_HARGA].astype(int)
    df[NAMA_KOLOM_PAKAN] = df[NAMA_KOLOM_PAKAN].astype(int)
    df = df[["Tanggal", "Wilayah", NAMA_KOLOM_HARGA, NAMA_KOLOM_PAKAN,
              "Is_Hari_Raya", "Efek_Lebaran"]]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset akhir: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"Rentang tanggal: {df['Tanggal'].min()} s.d. {df['Tanggal'].max()}")
    print(f"Disimpan ke: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
