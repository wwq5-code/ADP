import os
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "output.txt"
FIG_DIR = BASE_DIR / "figures"
TABLE_DIR = BASE_DIR / "tables"

os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None


SECTION_RE = re.compile(r"^===== (?P<section>.+?) =====$")
METRIC_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_]+)="
    r"(?P<mean>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
    r"(?:±(?P<std>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?))?"
)


PALETTE = {
    "deleteleak": "#797BB7",
    "dvia": "#B595BF",
    "likelihood": "#E1C855",
    "mia_after": "#86BBD8",
    "mia_before": "#E07A5F",
    "logits_cos": "#B595BF",
    "posterior_cos": "#797BB7",
    "logits_dist": "#E1C855",
    "posterior_dist": "#86BBD8",
}


def parse_output(results_file=RESULTS_FILE):
    """Parse sections with lines like key=mean±std into one DataFrame."""
    rows = []
    current_section = None
    with results_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            header_match = SECTION_RE.match(line)
            if header_match:
                current_section = header_match.group("section")
                continue

            if not line.startswith("unlearning_ratio="):
                continue

            row = {"section": current_section}
            for metric_match in METRIC_RE.finditer(line):
                key = metric_match.group("key")
                row[key] = float(metric_match.group("mean"))
                if metric_match.group("std") is not None:
                    row[f"{key}_std"] = float(metric_match.group("std"))
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No parsed rows found in {results_file}")
    return pd.DataFrame(rows)


def get_section(df, section):
    section_data = df[df["section"] == section].copy()
    if section_data.empty:
        raise RuntimeError(f"Missing section: {section}")
    section_data = section_data.sort_values("unlearning_ratio").reset_index(drop=True)
    section_data["unlearning_percent"] = section_data["unlearning_ratio"] * 100.0
    return section_data


def get_first_existing_section(df, sections):
    for section in sections:
        section_data = df[df["section"] == section].copy()
        if not section_data.empty:
            section_data = section_data.sort_values("unlearning_ratio").reset_index(drop=True)
            section_data["unlearning_percent"] = section_data["unlearning_ratio"] * 100.0
            return section_data
    raise RuntimeError(f"Missing all candidate sections: {sections}")


def configure_plot_style():
    if sns is not None:
        sns.set_theme(style="ticks", context="paper")
    else:
        plt.style.use("default")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "axes.labelsize": 32,
        "axes.titlesize": 22,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 17,
        "axes.linewidth": 1.8,
        "axes.edgecolor": "#222222",
        "grid.color": "#B8B8B8",
        "grid.alpha": 0.75,
        "grid.linestyle": "--",
        "grid.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def polish_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, which="major", axis="both", color="#B8B8B8", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.8)


def style_legend(legend, alpha=0.72):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(alpha)
    frame.set_edgecolor("#666666")
    frame.set_linewidth(0.8)


def save_tables(df):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "parsed_cifar10_deleteleak_results.csv", index=False)

    summary_rows = []
    for section, section_data in df.groupby("section"):
        row = {"section": section}
        for col in section_data.select_dtypes(include=[np.number]).columns:
            if not col.endswith("_std"):
                row[f"avg_{col}"] = section_data[col].mean()
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "cifar10_deleteleak_section_summary.csv", index=False)


def save_figure(fig, name, rect=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(FIG_DIR / f"{name}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_with_band(ax, x, mean, std, label, color, marker, linestyle="-", linewidth=4.2):
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.zeros_like(mean) if std is None else np.asarray(std, dtype=float)

    ax.plot(
        x,
        mean,
        linestyle=linestyle,
        color=color,
        marker=marker,
        fillstyle="full",
        label=label,
        linewidth=linewidth,
        markersize=12,
        markeredgewidth=2.2,
    )
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.18,
        linewidth=0,
    )


def plot_grouped_bars(ax, groups, series, ylabel, xlabel="Deleted Data (%)", width=0.24):
    x = np.arange(len(groups), dtype=float)
    offsets = (np.arange(len(series), dtype=float) - (len(series) - 1) / 2.0) * width
    y_values = []

    for offset, item in zip(offsets, series):
        mean = np.asarray(item["mean"], dtype=float)
        std = item.get("std")
        std = np.zeros_like(mean) if std is None else np.asarray(std, dtype=float)
        y_values.extend(mean + std)
        y_values.extend(mean - std)
        ax.bar(
            x + offset,
            mean,
            width=width,
            yerr=std,
            label=item["label"],
            color=item["color"],
            edgecolor="#222222",
            linewidth=0.9,
            alpha=0.92,
            capsize=3,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )

    y_values = np.asarray(y_values, dtype=float)
    y_values = y_values[np.isfinite(y_values)]
    y_max = float(np.nanmax(y_values)) if y_values.size else 1.0
    ax.set_ylim(0, np.ceil(y_max * 1.10 + 1.0))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{group:g}" for group in groups])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def metric_std(df, metric):
    std_col = f"{metric}_std"
    if std_col not in df:
        return None
    return df[std_col].to_numpy()


def plot_attack_auc(deleteleak, dvia, mia, output_prefix="CIFAR10_DeleteLeak"):
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    x = deleteleak["unlearning_percent"]

    plot_with_band(
        ax, x, deleteleak["posterior_clr_neighbor_attack_auc"],
        metric_std(deleteleak, "posterior_clr_neighbor_attack_auc"),
        "NeighborLeak", PALETTE["deleteleak"], "o", "-"
    )
    plot_with_band(
        ax, dvia["unlearning_percent"], dvia["dual_view_mia_auc"], metric_std(dvia, "dual_view_mia_auc"),
        "DVIA", PALETTE["dvia"], "d", "-."
    )
    plot_with_band(
        ax, dvia["unlearning_percent"], dvia["dual_view_likelihood_auc"],
        metric_std(dvia, "dual_view_likelihood_auc"),
        "Likelihood", PALETTE["likelihood"], "^", "--"
    )
    plot_with_band(
        ax, mia["unlearning_percent"], mia["mia_after_mia_auc"], metric_std(mia, "mia_after_mia_auc"),
        "MIA after unlearning", PALETTE["mia_after"], "s", ":"
    )
    plot_with_band(
        ax, mia["unlearning_percent"], mia["mia_before_mia_auc"], metric_std(mia, "mia_before_mia_auc"),
        "MIA before unlearning", PALETTE["mia_before"], "v", "--"
    )

    ax.axhline(50, color="gray", linestyle=":", linewidth=2.0)
    ax.set_xlabel("Deleted Data (%)")
    ax.set_ylabel("Attack AUC (%)")
    ax.set_xticks(np.arange(1, 6))
    ax.set_yticks(np.arange(50, 96, 10))
    ax.set_ylim(48, 95)
    style_legend(ax.legend(loc="best", fancybox=True, shadow=False, ncol=1, handletextpad=0.7))
    polish_axes(ax)
    save_figure(fig, f"{output_prefix}_attack_auc")


def plot_tpr_at_low_fpr(deleteleak, dvia, mia, output_prefix="CIFAR10_DeleteLeak"):
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    x = deleteleak["unlearning_percent"]

    plot_with_band(
        ax, x, deleteleak["posterior_clr_neighbor_attack_tpr_at_1fpr"],
        metric_std(deleteleak, "posterior_clr_neighbor_attack_tpr_at_1fpr"),
        "NeighborLeak", PALETTE["deleteleak"], "o", "-"
    )
    plot_with_band(
        ax, dvia["unlearning_percent"], dvia["dual_view_mia_tpr_at_1fpr"],
        metric_std(dvia, "dual_view_mia_tpr_at_1fpr"),
        "DVIA", PALETTE["dvia"], "d", "-."
    )
    plot_with_band(
        ax, dvia["unlearning_percent"], dvia["dual_view_likelihood_tpr_at_1fpr"],
        metric_std(dvia, "dual_view_likelihood_tpr_at_1fpr"),
        "Likelihood", PALETTE["likelihood"], "^", "--"
    )
    plot_with_band(
        ax, mia["unlearning_percent"], mia["mia_after_mia_tpr_at_1fpr"],
        metric_std(mia, "mia_after_mia_tpr_at_1fpr"),
        "MIA after unlearning", PALETTE["mia_after"], "s", ":"
    )
    plot_with_band(
        ax, mia["unlearning_percent"], mia["mia_before_mia_tpr_at_1fpr"],
        metric_std(mia, "mia_before_mia_tpr_at_1fpr"),
        "MIA before unlearning", PALETTE["mia_before"], "v", "--"
    )

    ax.set_xlabel("Deleted Data (%)")
    ax.set_ylabel("TPR@1%FPR (%)")
    ax.set_xticks(np.arange(1, 6))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylim(0, 100)
    style_legend(ax.legend(loc="best", fancybox=True, shadow=False, ncol=1, handletextpad=0.7))
    polish_axes(ax)
    save_figure(fig, f"{output_prefix}_tpr_at_1fpr")


def plot_attack_acc(deleteleak, dvia, mia, output_prefix="CIFAR10_DeleteLeak"):
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    x = deleteleak["unlearning_percent"]

    plot_with_band(
        ax, x, deleteleak["posterior_clr_neighbor_attack_acc"],
        metric_std(deleteleak, "posterior_clr_neighbor_attack_acc"),
        "NeighborLeak", PALETTE["deleteleak"], "o", "-"
    )
    plot_with_band(
        ax, dvia["unlearning_percent"], dvia["dual_view_mia_acc"], metric_std(dvia, "dual_view_mia_acc"),
        "DVIA", PALETTE["dvia"], "d", "-."
    )
    plot_with_band(
        ax, dvia["unlearning_percent"], dvia["dual_view_likelihood_acc"],
        metric_std(dvia, "dual_view_likelihood_acc"),
        "Likelihood", PALETTE["likelihood"], "^", "--"
    )
    plot_with_band(
        ax, mia["unlearning_percent"], mia["mia_after_mia_acc"], metric_std(mia, "mia_after_mia_acc"),
        "MIA after unlearning", PALETTE["mia_after"], "s", ":"
    )
    plot_with_band(
        ax, mia["unlearning_percent"], mia["mia_before_mia_acc"], metric_std(mia, "mia_before_mia_acc"),
        "MIA before unlearning", PALETTE["mia_before"], "v", "--"
    )

    ax.axhline(50, color="gray", linestyle=":", linewidth=2.0)
    ax.set_xlabel("Deleted Data (%)")
    ax.set_ylabel("Attack Accuracy (%)")
    ax.set_xticks(np.arange(1, 6))
    ax.set_yticks(np.arange(50, 91, 10))
    ax.set_ylim(48, 90)
    style_legend(ax.legend(loc="best", fancybox=True, shadow=False, ncol=1, handletextpad=0.7))
    polish_axes(ax)
    save_figure(fig, f"{output_prefix}_attack_acc")


def plot_relation_shift_gap(deleteleak, output_prefix="CIFAR10_DeleteLeak"):
    gap_df = deleteleak[["unlearning_percent"]].copy()
    gap_specs = [
        (
            "dfd_logits_geom_cosine_shift",
            "Logit Cosine",
            PALETTE["logits_cos"],
            "d",
            "-.",
        ),
        (
            "dfd_posterior_clr_geom_cosine_shift",
            "Posterior-CLR Cosine",
            PALETTE["posterior_cos"],
            "o",
            "-",
        ),
        (
            "dfd_logits_geom_distance_shift",
            "Logit Distance",
            PALETTE["logits_dist"],
            "^",
            "--",
        ),
        (
            "dfd_posterior_clr_geom_distance_shift",
            "Posterior-CLR Distance",
            PALETTE["posterior_dist"],
            "s",
            ":",
        ),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    for base, label, color, marker, linestyle in gap_specs:
        forget = deleteleak[f"{base}_forget_mean"]
        control = deleteleak[f"{base}_control_mean"]
        gap = (forget - control) * 100.0
        gap_std = np.sqrt(
            deleteleak[f"{base}_forget_mean_std"].to_numpy() ** 2
            + deleteleak[f"{base}_control_mean_std"].to_numpy() ** 2
        ) * 100.0

        gap_df[f"{base}_gap"] = gap / 100.0
        gap_df[f"{base}_gap_std"] = gap_std / 100.0
        plot_with_band(
            ax,
            deleteleak["unlearning_percent"],
            gap,
            gap_std,
            label,
            color,
            marker,
            linestyle,
        )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    gap_df.to_csv(TABLE_DIR / f"{output_prefix.lower()}_relation_shift_gaps.csv", index=False)

    ax.axhline(0, color="gray", linestyle=":", linewidth=2.0)
    ax.set_xlabel("Deleted Data (%)")
    ax.set_ylabel("Forget-Control Gap (%)")
    ax.set_xticks(np.arange(1, 6))
    style_legend(ax.legend(loc="best", fancybox=True, shadow=False, ncol=1, handletextpad=0.7))
    polish_axes(ax)
    save_figure(fig, f"{output_prefix}_relation_shift_gap")


def plot_relation_effect_size(deleteleak, output_prefix="CIFAR10_DeleteLeak"):
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    specs = [
        (
            "dfd_logits_geom_cosine_shift_cliffs_delta",
            "Logit Cosine",
            PALETTE["logits_cos"],
            "d",
            "-.",
        ),
        (
            "dfd_posterior_clr_geom_cosine_shift_cliffs_delta",
            "Posterior-CLR Cosine",
            PALETTE["posterior_cos"],
            "o",
            "-",
        ),
        (
            "dfd_logits_geom_distance_shift_cliffs_delta",
            "Logit Distance",
            PALETTE["logits_dist"],
            "^",
            "--",
        ),
        (
            "dfd_posterior_clr_geom_distance_shift_cliffs_delta",
            "Posterior-CLR Distance",
            PALETTE["posterior_dist"],
            "s",
            ":",
        ),
    ]
    y_values = []
    for metric, label, color, marker, linestyle in specs:
        mean = deleteleak[metric].to_numpy()
        std = metric_std(deleteleak, metric)
        if std is None:
            std = np.zeros_like(mean)
        y_values.extend(mean - std)
        y_values.extend(mean + std)
        plot_with_band(
            ax,
            deleteleak["unlearning_percent"],
            mean,
            std,
            label,
            color,
            marker,
            linestyle,
        )

    ax.axhline(0, color="gray", linestyle=":", linewidth=2.0)
    ax.set_xlabel("Deleted Data (%)")
    ax.set_ylabel("Cliff's Delta")
    ax.set_xticks(np.arange(1, 6))
    y_values = np.asarray(y_values, dtype=float)
    y_min = min(0.0, float(np.nanmin(y_values)))
    y_max = max(0.0, float(np.nanmax(y_values)))
    margin = max(0.03, 0.08 * (y_max - y_min))
    y_min = np.floor((y_min - margin) / 0.1) * 0.1
    y_max = np.ceil((y_max + margin) / 0.1) * 0.1
    ax.set_yticks(np.arange(y_min, y_max + 0.001, 0.1))
    ax.set_ylim(y_min, y_max)
    style_legend(ax.legend(loc="best", fancybox=True, shadow=False, ncol=1, handletextpad=0.7))
    polish_axes(ax)
    save_figure(fig, f"{output_prefix}_relation_effect_size")


def plot_utility_attacking_time(deleteleak, dvia, mia, output_prefix="CIFAR10_DeleteLeak"):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    plot_with_band(
        axes[0], deleteleak["unlearning_percent"], deleteleak["acc_after"] * 100,
        metric_std(deleteleak, "acc_after") * 100, "NeighborLeak", PALETTE["deleteleak"], "o", "-"
    )
    plot_with_band(
        axes[0], dvia["unlearning_percent"], dvia["acc_after"] * 100,
        metric_std(dvia, "acc_after") * 100, "DVIA", PALETTE["dvia"], "d", "-."
    )
    plot_with_band(
        axes[0], mia["unlearning_percent"], mia["acc_after"] * 100,
        metric_std(mia, "acc_after") * 100, "MIA", PALETTE["mia_after"], "s", ":"
    )
    axes[0].set_xlabel("Deleted Data (%)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(np.arange(1, 6))
    acc_values = np.concatenate([
        (deleteleak["acc_after"].to_numpy() + metric_std(deleteleak, "acc_after")) * 100,
        (deleteleak["acc_after"].to_numpy() - metric_std(deleteleak, "acc_after")) * 100,
        (dvia["acc_after"].to_numpy() + metric_std(dvia, "acc_after")) * 100,
        (dvia["acc_after"].to_numpy() - metric_std(dvia, "acc_after")) * 100,
        (mia["acc_after"].to_numpy() + metric_std(mia, "acc_after")) * 100,
        (mia["acc_after"].to_numpy() - metric_std(mia, "acc_after")) * 100,
    ])
    acc_min = np.floor(acc_values.min()) - 1
    acc_max = np.ceil(acc_values.max()) + 1
    axes[0].set_yticks(np.arange(acc_min, acc_max + 0.1, 1))
    axes[0].set_ylim(acc_min, acc_max)

    plot_grouped_bars(
        axes[1],
        deleteleak["unlearning_percent"],
        [
            {
                "mean": deleteleak["dfd_attacking_running_time"],
                "std": metric_std(deleteleak, "dfd_attacking_running_time"),
                "label": "NeighborLeak",
                "color": PALETTE["deleteleak"],
            },
            {
                "mean": dvia["dual_view_attacking_running_time"],
                "std": metric_std(dvia, "dual_view_attacking_running_time"),
                "label": "DVIA",
                "color": PALETTE["dvia"],
            },
            {
                "mean": mia["mia_attacking_running_time"],
                "std": metric_std(mia, "mia_attacking_running_time"),
                "label": "MIA",
                "color": PALETTE["mia_after"],
            },
        ],
        "Attacking Time (s)",
    )
    polish_axes(axes[0])
    polish_axes(axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig_legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        fancybox=True,
        shadow=False,
        fontsize=17,
    )
    style_legend(fig_legend)
    axes[0].legend().remove()
    axes[1].legend().remove()
    save_figure(fig, f"{output_prefix}_utility_attacking_time", rect=(0, 0, 1, 0.93))


def plot_attacking_time(deleteleak, dvia, mia, output_prefix="CIFAR10_DeleteLeak"):
    fig, ax = plt.subplots(figsize=(8.0, 5.6))

    dfd_time_col = "dfd_attacking_running_time"
    dfd_time_label = "NeighborLeak"
    if dfd_time_col not in deleteleak:
        dfd_time_col = "attacking_running_time"
        dfd_time_label = "NeighborLeak+MIA"

    plot_grouped_bars(
        ax,
        deleteleak["unlearning_percent"],
        [
            {
                "mean": deleteleak[dfd_time_col],
                "std": metric_std(deleteleak, dfd_time_col),
                "label": dfd_time_label,
                "color": PALETTE["deleteleak"],
            },
            {
                "mean": dvia["dual_view_attacking_running_time"],
                "std": metric_std(dvia, "dual_view_attacking_running_time"),
                "label": "DVIA",
                "color": PALETTE["dvia"],
            },
            {
                "mean": mia["mia_attacking_running_time"],
                "std": metric_std(mia, "mia_attacking_running_time"),
                "label": "MIA",
                "color": PALETTE["mia_after"],
            },
        ],
        "Attacking Time (s)",
    )
    style_legend(ax.legend(loc="best", fancybox=True, shadow=False, ncol=1, handletextpad=0.7))
    polish_axes(ax)
    save_figure(fig, f"{output_prefix}_attacking_time")


def print_summary(method_name, deleteleak, dvia, mia):
    print(f"CIFAR-10 {method_name} NeighborLeak analysis")
    print(
        "NeighborLeak mean AUC (%): "
        f"{deleteleak['posterior_clr_neighbor_attack_auc'].mean():.4f}"
    )
    print(f"DVIA mean AUC (%): {dvia['dual_view_mia_auc'].mean():.4f}")
    print(f"MIA after unlearning mean AUC (%): {mia['mia_after_mia_auc'].mean():.4f}")
    print(
        "NeighborLeak mean TPR@1%FPR (%): "
        f"{deleteleak['posterior_clr_neighbor_attack_tpr_at_1fpr'].mean():.4f}"
    )
    print(f"DVIA mean TPR@1%FPR (%): {dvia['dual_view_mia_tpr_at_1fpr'].mean():.4f}")
    print(f"MIA after unlearning mean TPR@1%FPR (%): {mia['mia_after_mia_tpr_at_1fpr'].mean():.4f}")
    print(f"MIA before unlearning mean TPR@1%FPR (%): {mia['mia_before_mia_tpr_at_1fpr'].mean():.4f}")
    print(
        "MIA after-before unlearning TPR@1%FPR gap (%): "
        f"{(mia['mia_after_mia_tpr_at_1fpr'] - mia['mia_before_mia_tpr_at_1fpr']).mean():.4f}"
    )
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Tables saved to: {TABLE_DIR}")


def main():
    configure_plot_style()
    df = parse_output(RESULTS_FILE)
    save_tables(df)

    method_specs = [
        ("RFU", ["CIFAR10_RFU_DeleteLeak"]),
        ("Retrain", ["CIFAR10_Retrain_DeleteLeak"]),
        ("SalUn", ["CIFAR10_SalUn_DeleteLeak"]),
    ]

    for method_name, output_prefixes in method_specs:
        deleteleak = get_first_existing_section(
            df,
            [
                f"{method_name} DeleteLeak and mia summary",
                f"{method_name} DeleteLeak summary",
            ],
        )
        dvia = get_section(df, f"{method_name} DVIA summary")

        if {"mia_before_mia_auc", "mia_after_mia_auc"}.issubset(deleteleak.columns):
            mia = deleteleak
        else:
            mia = get_section(df, f"{method_name} MIA summary")

        # Convert attack scores to percentage scale for presentation figures.
        for data, metrics in [
            (deleteleak, [
                "posterior_clr_neighbor_attack_auc",
                "posterior_clr_neighbor_attack_acc",
                "posterior_clr_neighbor_attack_tpr_at_1fpr",
            ]),
            (dvia, [
                "dual_view_mia_auc",
                "dual_view_likelihood_auc",
                "dual_view_mia_acc",
                "dual_view_likelihood_acc",
                "dual_view_mia_tpr_at_1fpr",
                "dual_view_likelihood_tpr_at_1fpr",
            ]),
            (mia, ["mia_before_mia_acc", "mia_before_mia_auc", "mia_after_mia_acc", "mia_after_mia_auc"]),
            (mia, ["mia_before_mia_tpr_at_1fpr", "mia_after_mia_tpr_at_1fpr"]),
        ]:
            for metric in metrics:
                if metric not in data:
                    continue
                data[metric] = data[metric] * 100.0
                std_col = f"{metric}_std"
                if std_col in data:
                    data[std_col] = data[std_col] * 100.0

        for output_prefix in output_prefixes:
            plot_attack_auc(deleteleak, dvia, mia, output_prefix)
            plot_tpr_at_low_fpr(deleteleak, dvia, mia, output_prefix)
            plot_attack_acc(deleteleak, dvia, mia, output_prefix)
            plot_relation_shift_gap(deleteleak, output_prefix)
            plot_relation_effect_size(deleteleak, output_prefix)
            plot_utility_attacking_time(deleteleak, dvia, mia, output_prefix)
            plot_attacking_time(deleteleak, dvia, mia, output_prefix)
        print_summary(method_name, deleteleak, dvia, mia)


if __name__ == "__main__":
    main()
