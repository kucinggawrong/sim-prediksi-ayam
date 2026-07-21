"""
holiday_calendar.py
=========================================================
Modul untuk menentukan flag 'Is_Hari_Raya' dan 'Efek_Lebaran'
untuk TANGGAL MASA DEPAN (di luar rentang dataset).

Mengapa modul ini dibutuhkan?
---------------------------------------------------------
Di dataset historis, kolom Is_Hari_Raya & Efek_Lebaran sudah
tersedia (hasil pencatatan manual / sumber resmi). Tapi saat
melakukan FORECAST ke hari-hari yang belum ada datanya, kita
butuh cara untuk menebak apakah tanggal forecast tersebut jatuh
pada hari libur nasional atau masuk periode "efek Lebaran".

Bug pada kode lama (app.py / predict.py versi awal):
    new_row["Is_Hari_Raya"]  = <disalin dari hari sebelumnya>
    new_row["Efek_Lebaran"]  = <disalin dari hari sebelumnya>
Akibatnya forecast 7-30 hari ke depan akan terus memakai flag
yang SAMA persis dari hari terakhir di data historis -- padahal
flag ini seharusnya berubah sesuai tanggal kalender yang sesungguhnya.

Modul ini memperbaikinya dengan tabel referensi hari libur
nasional Indonesia (fixed-date + tanggal hijriyah yang sudah
diketahui/ditetapkan pemerintah) dan dapat diperluas tahun-tahun
selanjutnya cukup dengan menambah baris pada HARI_LIBUR_HIJRIYAH.

CATATAN PENTING:
Tanggal Idul Fitri, Idul Adha, Isra Mikraj, Maulid Nabi, dst
mengikuti kalender Hijriyah dan TIDAK bisa dihitung dengan rumus
matematis sederhana -- harus dirujuk dari penetapan resmi
Kementerian Agama RI (Sidang Isbat) setiap tahun. Daftar di bawah
ini sudah mencakup tahun-tahun yang ada di dataset (2021-2026).
Jika Anda melatih ulang model dengan data yang lebih baru / ingin
forecast jauh ke tahun-tahun berikutnya, TAMBAHKAN dulu tanggalnya
di HARI_LIBUR_HIJRIYAH supaya fitur ini tetap akurat.
"""

import pandas as pd

# =========================================================
# 1. HARI LIBUR NASIONAL DENGAN TANGGAL TETAP (MASEHI)
# =========================================================
# Berlaku sama setiap tahun (bulan, hari)
HARI_LIBUR_TETAP = [
    (1, 1),    # Tahun Baru Masehi
    (5, 1),    # Hari Buruh Internasional
    (6, 1),    # Hari Lahir Pancasila
    (8, 17),   # Hari Kemerdekaan RI
    (12, 25),  # Hari Raya Natal
]

# =========================================================
# 2. HARI LIBUR BERBASIS KALENDER HIJRIYAH / KEAGAMAAN
#    & TANGGAL LAIN YANG BERUBAH SETIAP TAHUN
# =========================================================
# Bersumber dari penetapan SKB 3 Menteri / Kemenag RI tiap tahun.
# Format: "YYYY-MM-DD"
HARI_LIBUR_HIJRIYAH = {
    2021: [
        "2021-02-12",  # Tahun Baru Imlek
        "2021-03-11",  # Isra Mikraj
        "2021-03-14",  # Hari Suci Nyepi
        "2021-04-02",  # Wafat Isa Almasih
        "2021-05-13",  # Hari Raya Waisak
        "2021-05-14",  # Hari Raya Idul Fitri
        "2021-05-26",  # Kenaikan Isa Almasih
        "2021-06-01",  # Hari Lahir Pancasila
        "2021-07-20",  # Hari Raya Idul Adha
        "2021-08-11",  # Tahun Baru Islam
        "2021-10-20",  # Maulid Nabi Muhammad
    ],
    2022: [
        "2022-02-01",  # Tahun Baru Imlek
        "2022-02-28",  # Isra Mikraj
        "2022-03-03",  # Hari Suci Nyepi
        "2022-04-15",  # Wafat Isa Almasih
        "2022-05-02",  # Hari Raya Idul Fitri (1)
        "2022-05-03",  # Hari Raya Idul Fitri (2)
        "2022-05-16",  # Hari Raya Waisak
        "2022-05-26",  # Kenaikan Isa Almasih
        "2022-07-09",  # Hari Raya Idul Adha
        "2022-07-30",  # Tahun Baru Islam
        "2022-10-08",  # Maulid Nabi Muhammad
    ],
    2023: [
        "2023-01-22",  # Tahun Baru Imlek
        "2023-02-18",  # Isra Mikraj
        "2023-03-22",  # Hari Suci Nyepi
        "2023-04-07",  # Wafat Isa Almasih
        "2023-04-21",  # cuti bersama Idul Fitri
        "2023-04-22",  # Hari Raya Idul Fitri (1)
        "2023-04-23",  # Hari Raya Idul Fitri (2)
        "2023-05-18",  # Kenaikan Isa Almasih
        "2023-06-04",  # Hari Raya Waisak
        "2023-06-29",  # Hari Raya Idul Adha
        "2023-07-19",  # Tahun Baru Islam
        "2023-09-28",  # Maulid Nabi Muhammad
    ],
    2024: [
        "2024-02-08",  # Isra Mikraj
        "2024-02-10",  # Tahun Baru Imlek
        "2024-03-11",  # Hari Suci Nyepi
        "2024-03-29",  # Wafat Isa Almasih
        "2024-04-10",  # Hari Raya Idul Fitri (1)
        "2024-04-11",  # Hari Raya Idul Fitri (2)
        "2024-05-09",  # Kenaikan Isa Almasih
        "2024-05-23",  # Hari Raya Waisak
        "2024-06-17",  # Hari Raya Idul Adha
        "2024-07-07",  # Tahun Baru Islam
        "2024-09-16",  # Maulid Nabi Muhammad
    ],
    2025: [
        "2025-01-27",  # Isra Mikraj
        "2025-01-29",  # Tahun Baru Imlek
        "2025-03-27",  # Hari Suci Nyepi (Catatan: lihat data, 27 & 28 keduanya 1)
        "2025-03-28",  # cuti bersama Nyepi
        "2025-03-31",  # Hari Raya Idul Fitri (1)
        "2025-04-01",  # Hari Raya Idul Fitri (2)
        "2025-05-12",  # Hari Raya Waisak
        "2025-05-29",  # Kenaikan Isa Almasih
        "2025-06-06",  # Hari Raya Idul Adha
        "2025-06-27",  # Tahun Baru Islam
        "2025-09-05",  # Maulid Nabi Muhammad
    ],
    2026: [
        "2026-01-16",  # Isra Mikraj
        "2026-02-16",  # Tahun Baru Imlek (cuti bersama)
        "2026-02-17",  # Tahun Baru Imlek
        "2026-03-18",  # cuti bersama Nyepi
        "2026-03-19",  # Hari Suci Nyepi
        "2026-03-20",  # Hari Raya Idul Fitri (cuti)
        "2026-03-21",  # Hari Raya Idul Fitri (1)
        "2026-03-22",  # Hari Raya Idul Fitri (2)
        "2026-03-23",  # cuti bersama Idul Fitri
        "2026-03-24",  # cuti bersama Idul Fitri
        "2026-04-03",  # Wafat Isa Almasih
        "2026-04-05",  # Kenaikan Isa Almasih
        "2026-05-14",  # Hari Raya Waisak (cuti)
        "2026-05-15",  # Hari Raya Waisak
        "2026-05-27",  # Hari Raya Idul Adha (terkonfirmasi SKB 3 Menteri)
        "2026-05-28",  # cuti bersama Idul Adha (terkonfirmasi)
        "2026-05-31",  # Hari Raya Waisak (sesuai data asli)
        "2026-06-16",  # Tahun Baru Islam (terkonfirmasi SKB 3 Menteri)
        "2026-08-25",  # Maulid Nabi Muhammad SAW (terkonfirmasi SKB 3 Menteri)
        "2026-12-24",  # cuti bersama Natal
    ],
}

# =========================================================
# 3. PERIODE "EFEK LEBARAN" (WINDOW DI SEKITAR IDUL FITRI)
# =========================================================
# Berdasarkan analisis dataset asli: window ini selalu 11 hari,
# dimulai sekitar 8-9 hari SEBELUM Idul Fitri hari pertama, dan
# berakhir 1-2 hari SETELAH Idul Fitri hari kedua (efek mudik &
# kenaikan harga ayam/pakan menjelang & sesudah Lebaran).
#
# Tanggal Idul Fitri hari pertama per tahun (acuan utama window):
IDUL_FITRI_HARI_1 = {
    2021: "2021-05-13",
    2022: "2022-05-02",
    2023: "2023-04-22",
    2024: "2024-04-10",
    2025: "2025-03-31",
    2026: "2026-03-21",
    # 2027 dan seterusnya: TAMBAHKAN di sini setelah tanggal resmi
    # Idul Fitri tahun tersebut ditetapkan oleh Kemenag RI.
}

# Offset window efek lebaran (hari sebelum & sesudah Idul Fitri hari-1)
# Disesuaikan agar match dengan window 11 hari pada data asli.
EFEK_LEBARAN_HARI_SEBELUM = 7
EFEK_LEBARAN_HARI_SESUDAH = 3


def _semua_hari_libur_tetap(tahun: int):
    """Generate tanggal hari libur fixed-date untuk satu tahun tertentu."""
    return [pd.Timestamp(year=tahun, month=m, day=d) for (m, d) in HARI_LIBUR_TETAP]


def is_hari_raya(tanggal: pd.Timestamp) -> int:
    """
    Mengembalikan 1 jika tanggal tersebut adalah hari libur nasional
    (gabungan hari libur tetap + hari libur hijriyah/keagamaan yang
    sudah terdaftar di HARI_LIBUR_HIJRIYAH), selain itu 0.

    Jika tahun tidak terdaftar di HARI_LIBUR_HIJRIYAH, fungsi ini
    hanya akan mendeteksi hari libur dengan tanggal tetap
    (Tahun Baru, Hari Buruh, dst) -- hari libur hijriyah akan
    terlewat dan perlu ditambahkan manual.
    """
    tanggal = pd.Timestamp(tanggal).normalize()
    tahun = tanggal.year

    libur_tetap = _semua_hari_libur_tetap(tahun)
    if tanggal in libur_tetap:
        return 1

    libur_hijriyah = HARI_LIBUR_HIJRIYAH.get(tahun, [])
    libur_hijriyah_ts = [pd.Timestamp(t) for t in libur_hijriyah]
    if tanggal in libur_hijriyah_ts:
        return 1

    return 0


def efek_lebaran(tanggal: pd.Timestamp) -> int:
    """
    Mengembalikan 1 jika tanggal berada di dalam window "efek Lebaran"
    (periode menjelang & sesudah Idul Fitri yang biasanya berkorelasi
    dengan kenaikan harga ayam & pakan akibat permintaan mudik/lebaran).

    Jika tahun tidak terdaftar di IDUL_FITRI_HARI_1, mengembalikan 0
    (tidak bisa menentukan tanpa data referensi Idul Fitri tahun tsb).
    """
    tanggal = pd.Timestamp(tanggal).normalize()
    tahun = tanggal.year

    if tahun not in IDUL_FITRI_HARI_1:
        return 0

    idul_fitri = pd.Timestamp(IDUL_FITRI_HARI_1[tahun])
    mulai = idul_fitri - pd.Timedelta(days=EFEK_LEBARAN_HARI_SEBELUM)
    selesai = idul_fitri + pd.Timedelta(days=EFEK_LEBARAN_HARI_SESUDAH)

    return int(mulai <= tanggal <= selesai)


def tambahkan_flag_kalender(df: pd.DataFrame, kolom_tanggal: str = "Tanggal") -> pd.DataFrame:
    """
    Helper untuk menambahkan kolom Is_Hari_Raya & Efek_Lebaran pada
    sebuah DataFrame berdasarkan kolom tanggal yang ada, MENIMPA nilai
    yang sudah ada (digunakan khusus untuk baris-baris hasil forecast
    yang belum punya flag kalender yang benar).
    """
    df = df.copy()
    df["Is_Hari_Raya"] = df[kolom_tanggal].apply(is_hari_raya)
    df["Efek_Lebaran"] = df[kolom_tanggal].apply(efek_lebaran)
    return df


if __name__ == "__main__":
    # Self-test ringan: bandingkan dengan beberapa tanggal yang
    # diketahui dari dataset asli.
    contoh = [
        ("2021-01-01", 1, 0),
        ("2024-04-10", 1, 1),
        ("2025-06-06", 1, 0),
        ("2026-03-21", 1, 1),
        ("2026-06-20", 0, 0),
    ]
    for tgl, exp_raya, exp_lebaran in contoh:
        r = is_hari_raya(tgl)
        l = efek_lebaran(tgl)
        status = "OK" if (r == exp_raya) else "CEK ULANG"
        print(f"{tgl} -> Is_Hari_Raya={r} (exp {exp_raya}) | Efek_Lebaran={l} (exp {exp_lebaran}) [{status}]")
