import os
import re
import warnings
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
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


# The shadow-reference decoder receives no target-model membership labels.  It
# is trained using memberships controlled by the attacker in auxiliary
# reference-shadow models.
EXPERIMENTS = [
    {
        "dataset": "Fashion-MNIST",
        "supervised": (
            BASE_DIR
            / "On_FMNIST/results/"
            "Baseline_CNN_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt"
        ),
        "shadow_reference": (
            BASE_DIR
            / "On_FMNIST/results/"
            "Shadow_Reference_CNN_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt"
        ),
    },
    {
        "dataset": "CIFAR-10",
        "supervised": (
            BASE_DIR
            / "On_CIFAR10/results/"
            "Baseline_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt"
        ),
        "shadow_reference": (
            BASE_DIR
            / "On_CIFAR10/results/"
            "Shadow_Reference_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt"
        ),
    },
]

AUC_FIELDS = (
    "static_auc",
    "passive_deletion_auc",
    "sequential_adp_auc",
    "active_adp_auc",
)

METHOD_STYLES = {
    "static_auc": {
        "label": "Static",
        "color": "#707070",
        "marker": "v",
        "linestyle": ":",
    },
    "passive_deletion_auc": {
        "label": "Passive-1",
        "color": "#E1C855",
        "marker": "s",
        "linestyle": "--",
    },
    "sequential_adp_auc": {
        "label": "ADP-Random-$k$",
        "color": "#86BBD8",
        "marker": "d",
        "linestyle": "-.",
    },
    "active_adp_auc": {
        "label": "ADP-Planned-$k$",
        "color": "#797BB7",
        "marker": "o",
        "linestyle": "-",
    },
}

PROTOCOL_STYLES = {
    "supervised": {"label": "Supervised", "linestyle": "-", "alpha": 1.0},
    "shadow_reference": {
        "label": "Shadow-reference",
        "linestyle": "--",
        "alpha": 0.92,
    },
}


def configure_plot_style():
    if sns is not None:
        sns.set_theme(style="ticks", context="paper")
    else:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.labelsize": 24,
            "axes.titlesize": 21,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 15,
            "axes.linewidth": 1.8,
            "axes.edgecolor": "#222222",
            "grid.color": "#B8B8B8",
            "grid.alpha": 0.75,
            "grid.linestyle": "--",
            "grid.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polish_axes(
    ax,
    grid_linewidth=1.0,
    grid_alpha=0.75,
    spine_linewidth=1.8,
):
    ax.set_facecolor("white")
    ax.grid(
        True,
        which="major",
        axis="both",
        color="#B8B8B8",
        linestyle="--",
        linewidth=grid_linewidth,
        alpha=grid_alpha,
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(spine_linewidth)


def style_legend(legend, alpha=0.76):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(alpha)
    frame.set_edgecolor("#666666")
    frame.set_linewidth(0.8)


def save_figure(fig, name, rect=None, bbox_inches="tight"):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(FIG_DIR / f"{name}.pdf", dpi=300, bbox_inches=bbox_inches)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches=bbox_inches)
    plt.close(fig)


def plot_line(
    ax,
    x,
    y,
    label,
    color,
    marker,
    linestyle,
    alpha=1.0,
    linewidth=3.6,
    markersize=10,
    markeredgewidth=1.8,
):
    ax.plot(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        alpha=alpha,
    )


def load_summary(summary_file):
    """Extract completed scalar AUROC results without parsing large diagnostics."""
    if not summary_file.exists():
        warnings.warn(f"Missing summary file: {summary_file}")
        return pd.DataFrame(columns=("deletion_budget", *AUC_FIELDS))

    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    patterns = {
        "deletion_budget": re.compile(
            rf"'num_unlearning_operations':\s*({number})"
        ),
        **{
            field: re.compile(rf"'{field}':\s*({number})")
            for field in AUC_FIELDS
        },
    }

    rows = []
    with summary_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line.startswith("result:"):
                continue

            row = {}
            for field, pattern in patterns.items():
                match = pattern.search(line)
                if match is None:
                    raise RuntimeError(f"Missing {field} in result row from {summary_file}")
                row[field] = float(match.group(1))
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=("deletion_budget", *AUC_FIELDS))

    data = pd.DataFrame(rows).sort_values("deletion_budget").reset_index(drop=True)
    data["deletion_budget"] = data["deletion_budget"].astype(int)
    for field in AUC_FIELDS:
        data[field] = data[field] * 100.0
    return data


def build_comparison(dataset, supervised, shadow_reference):
    supervised = supervised.rename(
        columns={field: f"supervised_{field}" for field in AUC_FIELDS}
    )
    shadow_reference = shadow_reference.rename(
        columns={field: f"shadow_reference_{field}" for field in AUC_FIELDS}
    )
    comparison = supervised.merge(
        shadow_reference,
        on="deletion_budget",
        how="inner",
        validate="one_to_one",
    )
    comparison.insert(0, "dataset", dataset)

    for protocol in PROTOCOL_STYLES:
        static = comparison[f"{protocol}_static_auc"]
        for method in ("passive_deletion_auc", "sequential_adp_auc", "active_adp_auc"):
            comparison[f"{protocol}_{method}_gain"] = (
                comparison[f"{protocol}_{method}"] - static
            )

    for method in AUC_FIELDS:
        comparison[f"{method}_supervision_gap"] = (
            comparison[f"supervised_{method}"]
            - comparison[f"shadow_reference_{method}"]
        )
    return comparison


def load_comparisons():
    comparisons = []
    for experiment in EXPERIMENTS:
        supervised = load_summary(experiment["supervised"])
        shadow_reference = load_summary(experiment["shadow_reference"])
        if supervised.empty or shadow_reference.empty:
            missing_protocols = []
            if supervised.empty:
                missing_protocols.append("supervised")
            if shadow_reference.empty:
                missing_protocols.append("shadow-reference")
            warnings.warn(
                f"Skipping {experiment['dataset']}: no completed "
                f"{', '.join(missing_protocols)} result rows."
            )
            continue

        comparison = build_comparison(
            experiment["dataset"],
            supervised,
            shadow_reference,
        )
        if comparison.empty:
            warnings.warn(
                f"Skipping {experiment['dataset']}: supervised and shadow-reference "
                "results have no common deletion budgets."
            )
            continue
        comparisons.append((experiment["dataset"], comparison))

    if not comparisons:
        raise RuntimeError(
            "No paired supervised/shadow-reference results were found."
        )
    return comparisons


def save_analysis_tables(comparisons):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_points = pd.concat(
        [comparison for _, comparison in comparisons],
        ignore_index=True,
    )
    all_points.to_csv(
        TABLE_DIR / "ADP_supervised_unsupervised_by_budget.csv",
        index=False,
    )

    summary_rows = []
    for dataset, data in comparisons:
        for protocol in PROTOCOL_STYLES:
            for method in ("sequential_adp_auc", "active_adp_auc"):
                gain = data[f"{protocol}_{method}_gain"]
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol,
                        "sequence": (
                            "ADP-Random"
                            if method.startswith("sequential")
                            else "ADP-Planned"
                        ),
                        "mean_auc": data[f"{protocol}_{method}"].mean(),
                        "mean_gain_over_static": gain.mean(),
                        "max_gain_over_static": gain.max(),
                        "final_budget_gain_over_static": gain.iloc[-1],
                    }
                )

    pd.DataFrame(summary_rows).to_csv(
        TABLE_DIR / "ADP_supervised_unsupervised_summary.csv",
        index=False,
    )


def add_shared_legend(fig, ax, ncol, anchor_y=1.02):
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=ncol,
        fancybox=True,
        shadow=False,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    style_legend(legend)


def plot_absolute_auc(comparisons):
    """Place each dataset on one row for direct protocol comparison."""
    num_datasets = len(comparisons)
    fig, axes = plt.subplots(
        num_datasets,
        2,
        figsize=(11.8, 8.4),
        sharex="row",
        sharey=True,
        squeeze=False,
    )

    all_auc_values = []
    for row, (dataset, data) in enumerate(comparisons):
        budgets = data["deletion_budget"]
        for column, protocol in enumerate(("supervised", "shadow_reference")):
            ax = axes[row, column]
            for metric, style in METHOD_STYLES.items():
                values = data[f"{protocol}_{metric}"]
                all_auc_values.extend(values.to_numpy())
                plot_line(
                    ax,
                    budgets,
                    values,
                    style["label"],
                    style["color"],
                    style["marker"],
                    style["linestyle"],
                    linewidth=3.5,
                    markersize=9.0,
                    markeredgewidth=1.6,
                )

            ax.axhline(50, color="#888888", linestyle=":", linewidth=1.6, zorder=0)
            ax.set_title(
                f"{dataset} — {PROTOCOL_STYLES[protocol]['label']}",
                fontsize=19,
            )
            ax.set_xticks(budgets)
            if row == num_datasets - 1:
                ax.set_xlabel("Deletion Budget $k$", fontsize=19)
            if column == 0:
                ax.set_ylabel("Attack AUROC (%)", fontsize=19)
            ax.tick_params(axis="both", labelsize=14)
            polish_axes(
                ax,
                grid_linewidth=0.9,
                grid_alpha=0.65,
                spine_linewidth=1.5,
            )

    lower = min(50.0, np.floor((min(all_auc_values) - 2.0) / 5.0) * 5.0)
    upper = np.ceil((max(all_auc_values) + 2.0) / 5.0) * 5.0
    axes[0, 0].set_ylim(lower, upper)
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=5))
    add_shared_legend(fig, axes[0, 0], ncol=4, anchor_y=0.955)
    save_figure(
        fig,
        "ADP_supervised_unsupervised_absolute_auc",
        rect=(0, 0, 1, 0.90),
        bbox_inches=None,
    )


def plot_incremental_gain(comparisons):
    """Compare deletion-channel gains after removing each protocol's Static AUC."""
    num_datasets = len(comparisons)
    fig, axes = plt.subplots(
        1,
        num_datasets,
        figsize=(6.3 * num_datasets, 5.5),
        sharey=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    all_gains = []

    for ax, (dataset, data) in zip(axes, comparisons):
        budgets = data["deletion_budget"]
        for protocol in ("supervised", "shadow_reference"):
            protocol_style = PROTOCOL_STYLES[protocol]
            for metric in ("sequential_adp_auc", "active_adp_auc"):
                method_style = METHOD_STYLES[metric]
                gain = data[f"{protocol}_{metric}_gain"]
                all_gains.extend(gain.to_numpy())
                plot_line(
                    ax,
                    budgets,
                    gain,
                    f"{protocol_style['label']} {method_style['label']}",
                    method_style["color"],
                    method_style["marker"],
                    protocol_style["linestyle"],
                    alpha=protocol_style["alpha"],
                )

        ax.axhline(0, color="#888888", linestyle=":", linewidth=1.6, zorder=0)
        ax.set_title(dataset)
        ax.set_xlabel("Deletion Budget $k$")
        ax.set_xticks(budgets)
        polish_axes(ax)

    axes[0].set_ylabel("AUROC Gain over Matched Static (points)")
    lower = min(0.0, np.floor(min(all_gains) - 1.0))
    upper = np.ceil((max(all_gains) + 1.0) / 2.0) * 2.0
    axes[0].set_ylim(lower, upper)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=5))
    add_shared_legend(fig, axes[0], ncol=4)
    save_figure(
        fig,
        "ADP_supervised_unsupervised_gain_over_static",
        rect=(0, 0, 1, 0.91),
    )


def plot_supervision_gap(comparisons):
    """Quantify the AUROC lost when target-model membership labels are removed."""
    num_datasets = len(comparisons)
    fig, axes = plt.subplots(
        1,
        num_datasets,
        figsize=(6.3 * num_datasets, 5.5),
        sharey=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    all_gaps = []

    for ax, (dataset, data) in zip(axes, comparisons):
        budgets = data["deletion_budget"]
        for metric in ("sequential_adp_auc", "active_adp_auc"):
            style = METHOD_STYLES[metric]
            gap = data[f"{metric}_supervision_gap"]
            all_gaps.extend(gap.to_numpy())
            plot_line(
                ax,
                budgets,
                gap,
                style["label"],
                style["color"],
                style["marker"],
                style["linestyle"],
            )

        ax.axhline(0, color="#888888", linestyle=":", linewidth=1.6, zorder=0)
        ax.set_title(dataset)
        ax.set_xlabel("Deletion Budget $k$")
        ax.set_xticks(budgets)
        polish_axes(ax)

    axes[0].set_ylabel("Supervised $-$ Shadow-reference AUROC (points)")
    lower = min(0.0, np.floor(min(all_gaps) - 1.0))
    upper = np.ceil(max(all_gaps) + 1.0)
    axes[0].set_ylim(lower, upper)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=5))
    add_shared_legend(fig, axes[0], ncol=2)
    save_figure(
        fig,
        "ADP_supervised_unsupervised_gap",
        rect=(0, 0, 1, 0.91),
    )


def print_analysis(comparisons):
    for dataset, data in comparisons:
        print(f"{dataset} supervised vs. shadow-reference ADP")
        for protocol in ("supervised", "shadow_reference"):
            random_gain = data[f"{protocol}_sequential_adp_auc_gain"].mean()
            active_gain = data[f"{protocol}_active_adp_auc_gain"].mean()
            print(
                f"  {PROTOCOL_STYLES[protocol]['label']}: "
                f"mean ADP-Random-k gain={random_gain:.2f} points, "
                f"mean ADP-Planned-k gain={active_gain:.2f} points"
            )
        print(
            "  Mean supervised-minus-shadow-reference gap: "
            f"ADP-Random-k={data['sequential_adp_auc_supervision_gap'].mean():.2f} points, "
            f"ADP-Planned-k={data['active_adp_auc_supervision_gap'].mean():.2f} points"
        )


def main():
    configure_plot_style()
    comparisons = load_comparisons()
    save_analysis_tables(comparisons)
    plot_absolute_auc(comparisons)
    plot_incremental_gain(comparisons)
    plot_supervision_gap(comparisons)
    print_analysis(comparisons)


if __name__ == "__main__":
    main()
