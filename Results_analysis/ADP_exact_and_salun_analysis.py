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


# Compare exact retraining with SalUn while keeping the attack construction,
# dataset, model, and deletion budget matched.
EXPERIMENTS = [
    {
        "dataset": "Fashion-MNIST",
        "retrain": (
            BASE_DIR
            / "On_FMNIST/results/"
            "Baseline_CNN_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt"
        ),
        "salun": (
            BASE_DIR
            / "On_FMNIST/results/"
            "Baseline_CNN_SalUn_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt"
        ),
    },
    {
        "dataset": "CIFAR-10",
        "retrain": (
            BASE_DIR
            / "On_CIFAR10/results/"
            "Baseline_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt"
        ),
        "salun": (
            BASE_DIR
            / "On_CIFAR10/results/"
            "Baseline_SalUn_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt"
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
    "retrain": {"label": "Retrain", "linestyle": "-", "alpha": 1.0},
    "salun": {
        "label": "SalUn",
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


def polish_axes(ax):
    ax.set_facecolor("white")
    ax.grid(
        True,
        which="major",
        axis="both",
        color="#B8B8B8",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.8)


def style_legend(legend, alpha=0.76):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(alpha)
    frame.set_edgecolor("#666666")
    frame.set_linewidth(0.8)


def save_figure(fig, name, rect=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(FIG_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
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


def build_comparison(dataset, retrain, salun):
    retrain = retrain.rename(
        columns={field: f"retrain_{field}" for field in AUC_FIELDS}
    )
    salun = salun.rename(
        columns={field: f"salun_{field}" for field in AUC_FIELDS}
    )
    comparison = retrain.merge(
        salun,
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
        comparison[f"{method}_salun_minus_retrain"] = (
            comparison[f"salun_{method}"]
            - comparison[f"retrain_{method}"]
        )
    return comparison


def load_comparisons():
    comparisons = []
    for experiment in EXPERIMENTS:
        retrain = load_summary(experiment["retrain"])
        salun = load_summary(experiment["salun"])
        if retrain.empty or salun.empty:
            missing_protocols = []
            if retrain.empty:
                missing_protocols.append("retrain")
            if salun.empty:
                missing_protocols.append("SalUn")
            warnings.warn(
                f"Skipping {experiment['dataset']}: no completed "
                f"{', '.join(missing_protocols)} result rows."
            )
            continue

        comparison = build_comparison(
            experiment["dataset"],
            retrain,
            salun,
        )
        if comparison.empty:
            warnings.warn(
                f"Skipping {experiment['dataset']}: retrain and SalUn "
                "results have no common deletion budgets."
            )
            continue
        comparisons.append((experiment["dataset"], comparison))

    if not comparisons:
        raise RuntimeError(
            "No paired retrain/SalUn results were found."
        )
    return comparisons


def save_analysis_tables(comparisons):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_points = pd.concat(
        [comparison for _, comparison in comparisons],
        ignore_index=True,
    )
    all_points.to_csv(
        TABLE_DIR / "ADP_retrain_salun_by_budget.csv",
        index=False,
    )

    summary_rows = []
    for dataset, data in comparisons:
        for method in ("sequential_adp_auc", "active_adp_auc"):
            difference = data[f"{method}_salun_minus_retrain"]
            summary_rows.append(
                {
                    "dataset": dataset,
                    "sequence": (
                        "ADP-Random"
                        if method.startswith("sequential")
                        else "ADP-Planned"
                    ),
                    "mean_retrain_auc": data[f"retrain_{method}"].mean(),
                    "mean_salun_auc": data[f"salun_{method}"].mean(),
                    "mean_salun_minus_retrain": difference.mean(),
                    "final_budget_salun_minus_retrain": difference.iloc[-1],
                    "mean_retrain_gain_over_static": data[
                        f"retrain_{method}_gain"
                    ].mean(),
                    "mean_salun_gain_over_static": data[
                        f"salun_{method}_gain"
                    ].mean(),
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        TABLE_DIR / "ADP_retrain_salun_summary.csv",
        index=False,
    )


def add_shared_legend(fig, ax, ncol):
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
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
        figsize=(12.6, 4.6 * num_datasets),
        sharex="row",
        sharey=True,
        squeeze=False,
    )

    all_auc_values = []
    for row, (dataset, data) in enumerate(comparisons):
        budgets = data["deletion_budget"]
        for column, protocol in enumerate(("retrain", "salun")):
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
                    linewidth=4.0,
                    markersize=11,
                    markeredgewidth=2.0,
                )

            ax.axhline(50, color="#888888", linestyle=":", linewidth=1.6, zorder=0)
            ax.set_title(f"{dataset} — {PROTOCOL_STYLES[protocol]['label']}")
            ax.set_xticks(budgets)
            if row == num_datasets - 1:
                ax.set_xlabel("Deletion Budget $k$")
            if column == 0:
                ax.set_ylabel("Attack AUROC (%)")
            polish_axes(ax)

    lower = min(50.0, np.floor((min(all_auc_values) - 2.0) / 5.0) * 5.0)
    upper = np.ceil((max(all_auc_values) + 2.0) / 5.0) * 5.0
    axes[0, 0].set_ylim(lower, upper)
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=5))
    add_shared_legend(fig, axes[0, 0], ncol=4)
    save_figure(
        fig,
        "ADP_retrain_salun_absolute_auc",
        rect=(0, 0, 1, 0.94),
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
        for protocol in ("retrain", "salun"):
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
        "ADP_retrain_salun_gain_over_static",
        rect=(0, 0, 1, 0.91),
    )


def plot_salun_minus_retrain(comparisons):
    """Plot the matched AUROC difference between SalUn and exact retraining."""
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
            gap = data[f"{metric}_salun_minus_retrain"]
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

    axes[0].set_ylabel("SalUn $-$ Retrain AUROC (points)")
    limit = np.ceil(max(abs(min(all_gaps)), abs(max(all_gaps))) + 1.0)
    lower = -limit
    upper = limit
    axes[0].set_ylim(lower, upper)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=5))
    add_shared_legend(fig, axes[0], ncol=2)
    save_figure(
        fig,
        "ADP_salun_minus_retrain_auc",
        rect=(0, 0, 1, 0.91),
    )


def print_analysis(comparisons):
    for dataset, data in comparisons:
        print(f"{dataset} retrain vs. SalUn ADP")
        for protocol in ("retrain", "salun"):
            random_gain = data[f"{protocol}_sequential_adp_auc_gain"].mean()
            active_gain = data[f"{protocol}_active_adp_auc_gain"].mean()
            print(
                f"  {PROTOCOL_STYLES[protocol]['label']}: "
                f"mean ADP-Random-k gain={random_gain:.2f} points, "
                f"mean ADP-Planned-k gain={active_gain:.2f} points"
            )
        print(
            "  Mean SalUn-minus-Retrain difference: "
            f"ADP-Random-k={data['sequential_adp_auc_salun_minus_retrain'].mean():.2f} points, "
            f"ADP-Planned-k={data['active_adp_auc_salun_minus_retrain'].mean():.2f} points"
        )


def main():
    configure_plot_style()
    comparisons = load_comparisons()
    save_analysis_tables(comparisons)
    plot_absolute_auc(comparisons)
    plot_incremental_gain(comparisons)
    plot_salun_minus_retrain(comparisons)
    print_analysis(comparisons)


if __name__ == "__main__":
    main()
