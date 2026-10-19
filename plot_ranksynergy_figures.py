from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ============================================================
# 0. Global configuration
# ============================================================
OUT_DIR = Path("paper_figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    # 用 matplotlib 自带字体，避免 Ubuntu 上 Arial fallback 导致错位
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "axes.unicode_minus": False,
})

# ============================================================
# 1. Color palette
# ============================================================
COLORS = {
    "BOIL": "#8C8C8C",
    "HyperSynergy": "#4C78A8",
    "MetaSynergy": "#F58518",
    "RankSynergy": "#54A24B",
    "Anchor": "#7F7F7F",
    "Ablation1": "#9C755F",
    "Ablation2": "#E45756",
    "Ablation3": "#72B7B2",
    "LowSim": "#B279A2",
    "Dark": "#333333",
}

# ============================================================
# 2. Experimental data
# ============================================================
shots = ["5-shot", "10-shot", "30-shot"]

main_data = {
    "BOIL": {
        "MSE": [0.110, 0.108, 0.104],
        "SCC": [0.443, 0.448, 0.496],
        "R2":  [0.204, 0.215, 0.249],
    },
    "HyperSynergy": {
        "MSE": [0.115, 0.111, 0.105],
        "SCC": [0.508, 0.530, 0.545],
        "R2":  [0.181, 0.210, 0.250],
    },
    "MetaSynergy": {
        "MSE": [0.108, 0.107, 0.104],
        "SCC": [0.513, 0.530, 0.552],
        "R2":  [0.225, 0.235, 0.262],
    },
    "RankSynergy": {
        "MSE": [0.107642, 0.105645, 0.099093],
        "SCC": [0.513416, 0.533855, 0.554646],
        "R2":  [0.226918, 0.249463, 0.300219],
    },
}

main_std = {
    "BOIL": {
        "MSE": [0.002, 0.002, 0.002],
        "SCC": [0.003, 0.002, 0.003],
        "R2":  [0.010, 0.003, 0.002],
    },
    "HyperSynergy": {
        "MSE": [0.002, 0.003, 0.002],
        "SCC": [0.004, 0.004, 0.004],
        "R2":  [0.004, 0.008, 0.007],
    },
    "MetaSynergy": {
        "MSE": [0.001, 0.001, 0.002],
        "SCC": [0.011, 0.010, 0.004],
        "R2":  [0.005, 0.011, 0.005],
    },
    "RankSynergy": {
        "MSE": [0.001653, 0.000853, 0.000761],
        "SCC": [0.001815, 0.001614, 0.001888],
        "R2":  [0.002774, 0.002512, 0.003773],
    },
}

ablation_names = [
    "Anchor only",
    "w/o Sparse support\ndensification",
    "w/o MSE-guarded\nSCC selection",
    "w/o Value-preserving\nrank transfer",
    "Full RankSynergy",
]

ablation_data = {
    "MSE":  [0.111285, 0.111062, 0.109686, 0.109586, 0.107642],
    "RMSE": [0.333584, 0.333250, 0.331179, 0.331027, 0.328078],
    "SCC":  [0.499057, 0.507358, 0.513858, 0.518978, 0.513416],
    "PCC":  [0.501481, 0.498796, 0.503466, 0.503747, 0.509425],
    "R2":   [0.200755, 0.202350, 0.212238, 0.212957, 0.226918],
}

lowsim_rank = {
    "MSE":  [0.014722, 0.014626, 0.013520],
    "RMSE": [0.121336, 0.120939, 0.116277],
    "SCC":  [0.735411, 0.737232, 0.750153],
    "PCC":  [0.895556, 0.896877, 0.903805],
    "R2":   [0.799156, 0.800550, 0.815620],
    "AUC":  [0.983223, 0.985709, 0.982638],
    "AUPR": [0.277434, 0.286699, 0.298877],
    "AUPR@Top10%": [0.925537, 0.926243, 0.931989],
}

lowsim_anchor = {
    "MSE":  [0.015041, 0.014061, 0.013189],
    "SCC":  [0.710611, 0.725414, 0.740223],
    "R2":   [0.794815, 0.808255, 0.820146],
    "AUPR": [0.257311, 0.284612, 0.312055],
}

# ============================================================
# 3. Utilities
# ============================================================
def save_all(fig, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            OUT_DIR / f"{name}.{ext}",
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white"
        )
    print(f"[saved] {name}.png / .pdf / .svg")


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", linewidth=0.55, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.5, pad=3)


def panel_label(ax, label):
    ax.text(
        -0.10, 1.03, label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

# ============================================================
# 4. Figure 1: Main comparison
# ============================================================
def plot_fig1_main_comparison():
    methods = ["BOIL", "HyperSynergy", "MetaSynergy", "RankSynergy"]
    metrics = ["MSE", "SCC", "R2"]
    ylabels = {"MSE": "MSE ↓", "SCC": "SCC ↑", "R2": r"$R^2$ ↑"}

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    x = np.arange(len(shots))
    width = 0.18

    for ax, metric, letter in zip(axes, metrics, ["A", "B", "C"]):
        for i, method in enumerate(methods):
            vals = main_data[method][metric]
            stds = main_std[method][metric]
            ax.bar(
                x + (i - 1.5) * width,
                vals,
                width=width,
                yerr=stds,
                capsize=2.2,
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.45,
                alpha=0.96,
                label=method if metric == "MSE" else None,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(shots)
        ax.set_ylabel(ylabels[metric], labelpad=6)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        style_axis(ax, "y")
        panel_label(ax, letter)

        if metric == "MSE":
            ax.set_ylim(0.096, 0.118)
        elif metric == "SCC":
            ax.set_ylim(0.42, 0.57)
        else:
            ax.set_ylim(0.16, 0.32)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        columnspacing=1.3,
        handlelength=1.6,
    )

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.78, wspace=0.38)
    save_all(fig, "fig1_main_comparison")
    plt.close(fig)

# ============================================================
# 5. Figure 2: Ablation
# ============================================================
def plot_fig2_ablation():
    metrics = ["MSE", "SCC", "R2"]
    xlabels = {"MSE": "MSE ↓", "SCC": "SCC ↑", "R2": r"$R^2$ ↑"}

    colors = [
        COLORS["Anchor"],
        COLORS["Ablation1"],
        COLORS["Ablation2"],
        COLORS["Ablation3"],
        COLORS["RankSynergy"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
    y = np.arange(len(ablation_names))

    for ax, metric, letter in zip(axes, metrics, ["A", "B", "C"]):
        vals = ablation_data[metric]
        bars = ax.barh(
            y,
            vals,
            color=colors,
            edgecolor="black",
            linewidth=0.45,
            alpha=0.96,
        )

        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(ablation_names, fontsize=9)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel(xlabels[metric], labelpad=5)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        style_axis(ax, "x")
        panel_label(ax, letter)

        if metric == "MSE":
            ax.set_xlim(0.1065, 0.1123)
        elif metric == "SCC":
            ax.set_xlim(0.495, 0.522)
        else:
            ax.set_xlim(0.198, 0.230)

        x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
        for b in bars:
            w = b.get_width()
            ax.text(
                w + x_span * 0.010,
                b.get_y() + b.get_height() / 2,
                f"{w:.3f}",
                va="center",
                ha="left",
                fontsize=8,
            )

    fig.subplots_adjust(left=0.33, right=0.99, bottom=0.20, top=0.90, wspace=0.30)
    save_all(fig, "fig2_ablation")
    plt.close(fig)

# ============================================================
# 6. Figure 3: Low-sim robustness trend
# ============================================================
def plot_fig3_lowsim_trend():
    x = np.array([5, 10, 30])
    xticklabels = ["5-shot", "10-shot", "30-shot"]
    metrics = ["MSE", "SCC", "AUPR", "AUPR@Top10%"]
    ylabels = {
        "MSE": "MSE ↓",
        "SCC": "SCC ↑",
        "AUPR": "AUPR ↑",
        "AUPR@Top10%": "AUPR@Top10% ↑",
    }
    ylims = {
        "MSE": (0.0130, 0.0152),
        "SCC": (0.728, 0.755),
        "AUPR": (0.265, 0.305),
        "AUPR@Top10%": (0.920, 0.936),
    }

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.2))

    for ax, metric, letter in zip(axes, metrics, ["A", "B", "C", "D"]):
        vals = lowsim_rank[metric]
        ax.plot(
            x,
            vals,
            marker="o",
            markersize=5.2,
            linewidth=1.9,
            color=COLORS["LowSim"],
            markerfacecolor=COLORS["LowSim"],
            markeredgecolor="black",
            markeredgewidth=0.55,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=15)
        ax.set_ylabel(ylabels[metric], labelpad=6)
        ax.set_ylim(*ylims[metric])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        style_axis(ax, "y")
        panel_label(ax, letter)

        y0, y1 = ylims[metric]
        for xi, yi in zip(x, vals):
            ax.text(
                xi,
                yi + (y1 - y0) * 0.035,
                f"{yi:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.24, top=0.90, wspace=0.42)
    save_all(fig, "fig3_lowsim_trend")
    plt.close(fig)

# ============================================================
# 7. Figure 4: Low-sim paired comparison
# ============================================================
def plot_fig4_lowsim_paired():
    metrics = ["MSE", "SCC", "R2", "AUPR"]
    ylabels = {"MSE": "MSE ↓", "SCC": "SCC ↑", "R2": r"$R^2$ ↑", "AUPR": "AUPR ↑"}
    ylims = {
        "MSE": (0.0128, 0.0155),
        "SCC": (0.700, 0.755),
        "R2": (0.790, 0.825),
        "AUPR": (0.240, 0.325),
    }

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.35))
    x = np.arange(len(shots))
    width = 0.34

    for ax, metric, letter in zip(axes, metrics, ["A", "B", "C", "D"]):
        ax.bar(
            x - width / 2,
            lowsim_anchor[metric],
            width,
            color=COLORS["Anchor"],
            edgecolor="black",
            linewidth=0.45,
            label="Anchor" if metric == "MSE" else None,
        )
        ax.bar(
            x + width / 2,
            lowsim_rank[metric],
            width,
            color=COLORS["RankSynergy"],
            edgecolor="black",
            linewidth=0.45,
            label="RankSynergy" if metric == "MSE" else None,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(shots, rotation=15)
        ax.set_ylabel(ylabels[metric], labelpad=6)
        ax.set_ylim(*ylims[metric])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        style_axis(ax, "y")
        panel_label(ax, letter)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        columnspacing=1.5,
    )

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.24, top=0.78, wspace=0.42)
    save_all(fig, "fig4_lowsim_paired")
    plt.close(fig)

# ============================================================
# 8. Figure 5: Delta SCC emphasis
# ============================================================
def plot_fig5_delta_scc():
    delta_scc = np.array(lowsim_rank["SCC"]) - np.array(lowsim_anchor["SCC"])
    delta_mse = np.array(lowsim_rank["MSE"]) - np.array(lowsim_anchor["MSE"])
    delta_r2 = np.array(lowsim_rank["R2"]) - np.array(lowsim_anchor["R2"])
    delta_aupr = np.array(lowsim_rank["AUPR"]) - np.array(lowsim_anchor["AUPR"])

    x = np.arange(len(shots))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))

    # A
    bars = axes[0].bar(
        x,
        delta_scc,
        width=0.50,
        color=COLORS["RankSynergy"],
        edgecolor="black",
        linewidth=0.45,
    )
    axes[0].axhline(0, color=COLORS["Dark"], linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(shots)
    axes[0].set_ylabel(r"$\Delta$SCC", labelpad=6)
    axes[0].set_ylim(0.0, 0.028)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    style_axis(axes[0], "y")
    panel_label(axes[0], "A")

    for b, v in zip(bars, delta_scc):
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            v + 0.0006,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # B
    axes[1].bar(
        x - 1.5 * width, delta_mse, width,
        color="#A0CBE8", edgecolor="black", linewidth=0.45, label=r"$\Delta$MSE"
    )
    axes[1].bar(
        x - 0.5 * width, delta_scc, width,
        color=COLORS["RankSynergy"], edgecolor="black", linewidth=0.45, label=r"$\Delta$SCC"
    )
    axes[1].bar(
        x + 0.5 * width, delta_r2, width,
        color="#FF9DA6", edgecolor="black", linewidth=0.45, label=r"$\Delta R^2$"
    )
    axes[1].bar(
        x + 1.5 * width, delta_aupr, width,
        color=COLORS["LowSim"], edgecolor="black", linewidth=0.45, label=r"$\Delta$AUPR"
    )
    axes[1].axhline(0, color=COLORS["Dark"], linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(shots)
    axes[1].set_ylabel("Metric change", labelpad=6)
    style_axis(axes[1], "y")
    panel_label(axes[1], "B")
    axes[1].legend(
        frameon=False, ncol=2, loc="upper right",
        handlelength=1.2, columnspacing=0.9
    )

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.90, wspace=0.30)
    save_all(fig, "fig5_lowsim_delta")
    plt.close(fig)

# ============================================================
# 9. Run all
# ============================================================
if __name__ == "__main__":
    plot_fig1_main_comparison()
    plot_fig2_ablation()
    plot_fig3_lowsim_trend()
    plot_fig4_lowsim_paired()
    plot_fig5_delta_scc()
    print("\nAll figures saved to:", OUT_DIR.resolve())