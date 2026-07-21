"""
buat_grafik_window_metrics.py
=========================================================
Menghasilkan grafik_window_vs_metrics.png -- hubungan window size
(7/14/21/30 hari) terhadap MAPE dan RMSE hasil grid search di
optimize.py, sesuai models/optimize_results.json terverifikasi
(dataset 2.024 baris, s.d. 17 Juli 2026).

Catatan: rentang sumbu-Y sengaja dibuat proporsional terhadap sebaran
data asli (bukan dipaksa sangat sempit) supaya tidak melebih-lebihkan
perbedaan antar-window yang sebenarnya kecil (~Rp1-5, dalam batas
noise training) -- lihat pembahasan sebelumnya soal grafik app yang
skalanya terlalu sempit dan jadi menyesatkan secara visual.

CARA MENJALANKAN:
    python buat_grafik_window_metrics.py
Menghasilkan grafik_window_vs_metrics.png di folder yang sama.
"""

import matplotlib
matplotlib.use("Agg")  # tidak butuh display, langsung simpan ke file
import matplotlib.pyplot as plt

# =========================================================
# DATA (dari models/optimize_results.json -- window_grid_search)
# =========================================================
WINDOWS = [7, 14, 21, 30]
MAPE = [0.4251, 0.4515, 0.4426, 0.4169]   # persen
RMSE = [450.73, 449.25, 452.25, 454.19]   # Rupiah
WINDOW_TERPILIH = 14

COLOR_MAPE = "#1F4E79"
COLOR_RMSE = "#C00000"
COLOR_TERPILIH = "#D4A017"


def buat_grafik_window_vs_metrics():
    """Grafik dual-axis: MAPE (%) & RMSE (Rp) vs window size, window
    terpilih ditandai garis vertikal."""
    fig, ax1 = plt.subplots(figsize=(8, 5.5), dpi=200)

    ax1.plot(WINDOWS, MAPE, marker="o", color=COLOR_MAPE, linewidth=2, label="MAPE (%)")
    ax1.set_xlabel("Window Size (Hari)")
    ax1.set_ylabel("MAPE (%)", color=COLOR_MAPE, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=COLOR_MAPE)
    for w, v in zip(WINDOWS, MAPE):
        ax1.annotate(f"{v:.2f}%", (w, v), textcoords="offset points", xytext=(0, 10),
                     ha="center", color=COLOR_MAPE, fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(WINDOWS, RMSE, marker="s", linestyle="--", color=COLOR_RMSE, linewidth=2, label="RMSE (Rp)")
    ax2.set_ylabel("RMSE (Rp)", color=COLOR_RMSE, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=COLOR_RMSE)
    for w, v in zip(WINDOWS, RMSE):
        ax2.annotate(f"Rp{v:.0f}", (w, v), textcoords="offset points", xytext=(0, -15),
                     ha="center", color=COLOR_RMSE, fontsize=9)

    # Skala proporsional terhadap sebaran data asli -- bukan dipaksa sempit
    ax1.set_ylim(0.40, 0.47)
    ax2.set_ylim(445, 460)

    ax1.axvline(WINDOW_TERPILIH, color=COLOR_TERPILIH, linestyle=":", linewidth=2, alpha=0.7)
    ax1.annotate("Window terpilih (RMSE terkecil)", xy=(WINDOW_TERPILIH, 0.465), ha="center",
                 fontsize=9, color="#8a6d0a", fontweight="bold")

    ax1.set_xticks(WINDOWS)
    ax1.set_title(
        "Hubungan Window Size terhadap MAPE dan RMSE\n"
        "(dataset 2.024 baris, s.d. 17 Juli 2026 -- via optimize.py)",
        fontsize=12,
    )
    ax1.grid(axis="both", linestyle="--", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
               bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig.tight_layout()
    fig.savefig("grafik_window_vs_metrics.png", dpi=200)
    plt.close(fig)
    print("Tersimpan: grafik_window_vs_metrics.png")


if __name__ == "__main__":
    buat_grafik_window_vs_metrics()