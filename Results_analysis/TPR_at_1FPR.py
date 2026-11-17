import ast
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


TARGET_FPR = 0.01
DELETION_BUDGETS = (3, 5, 7, 9, 11, 13, 15)

EXPERIMENTS = [
    {
        "dataset": "Fashion-MNIST",
        "retrain": BASE_DIR / "On_FMNIST/results/"
        "Baseline_CNN_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt",
        "salun": BASE_DIR / "On_FMNIST/results/"
        "Baseline_CNN_SalUn_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt",
    },
    {
        "dataset": "CIFAR-10",
        "retrain": BASE_DIR / "On_CIFAR10/results/"
        "Baseline_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt",
        "salun": BASE_DIR / "On_CIFAR10/results/"
        "Baseline_SalUn_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt",
    },
    {
        "dataset": "SVHN",
        "retrain": BASE_DIR / "On_SVHN/results/"
        "Baseline_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt",
        "salun": BASE_DIR / "On_SVHN/results/"
        "Baseline_SalUn_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt",
    },
]

PROTOCOLS = (
    ("retrain", "Retrain"),
    ("salun", "SalUn"),
)

METHODS = {
    "static": {
        "label": "Static",
        "color": "#707070",
        "marker": "v",
        "linestyle": ":",
        "linewidth": 2.0,
        "alpha": 0.78,
    },
    "passive": {
        "label": "Passive-1",
        "color": "#D0B638",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.0,
        "alpha": 0.82,
    },
    "sequential": {
        "label": "ADP-Random-$k$",
        "color": "#5A9FC6",
        "marker": "D",
        "linestyle": "-.",
        "linewidth": 3.0,
        "alpha": 1.0,
    },
    "active": {
        "label": "ADP-Planned-$k$",
        "color": "#6D6AAF",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 3.0,
        "alpha": 1.0,
    },
}

ROC_FIELDS = {
    method: (f"{method}_roc_fpr", f"{method}_roc_tpr")
    for method in METHODS
}


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.labelsize": 19,
            "axes.titlesize": 19,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 15,
            "axes.linewidth": 1.5,
            "axes.edgecolor": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_run_index(summary_file):
    """Extract deletion budgets and per-budget result filenames."""
    if not summary_file.exists():
        warnings.warn(f"Missing summary file: {summary_file}")
        return []

    budget_pattern = re.compile(r"'num_unlearning_operations':\s*(\d+)")
    file_pattern = re.compile(r"'result_file':\s*'([^']+)'")
    runs = []
    with summary_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("result:"):
                continue
            budget_match = budget_pattern.search(line)
            file_match = file_pattern.search(line)
            if budget_match is None or file_match is None:
                raise RuntimeError(f"Malformed result row in {summary_file}")
            runs.append(
                {
                    "deletion_budget": int(budget_match.group(1)),
                    "result_file": summary_file.parent / file_match.group(1),
                }
            )

    budgets = [run["deletion_budget"] for run in runs]
    if len(budgets) != len(set(budgets)):
        raise RuntimeError(f"Duplicate deletion budgets in {summary_file}")
    return sorted(runs, key=lambda run: run["deletion_budget"])


def load_roc_arrays(result_file):
    """Load only the ROC arrays required for the low-FPR comparison."""
    if not result_file.exists():
        warnings.warn(f"Missing result file: {result_file}")
        return None

    required_fields = {
        field
        for field_pair in ROC_FIELDS.values()
        for field in field_pair
    }
    arrays = {}
    with result_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            key, separator, value = raw_line.partition(":")
            key = key.strip()
            if separator and key in required_fields:
                arrays[key] = np.asarray(
                    ast.literal_eval(value.strip()),
                    dtype=float,
                )

    missing = sorted(required_fields.difference(arrays))
    if missing:
        raise RuntimeError(f"Missing ROC arrays {missing} in {result_file}")
    return arrays


def tpr_at_fpr(fpr, tpr, target_fpr=TARGET_FPR):
    """Best empirical TPR attainable without exceeding target_fpr."""
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.shape != tpr.shape or fpr.ndim != 1:
        raise ValueError("FPR and TPR must be one-dimensional arrays of equal length.")
    if fpr.size == 0 or not np.all(np.isfinite(fpr)) or not np.all(np.isfinite(tpr)):
        raise ValueError("ROC arrays must be non-empty and finite.")

    valid = fpr <= target_fpr + 1e-12
    if not np.any(valid):
        return 0.0, np.nan
    return float(np.max(tpr[valid])), float(np.max(fpr[valid]))


def build_results_table():
    rows = []
    for experiment in EXPERIMENTS:
        for protocol, protocol_label in PROTOCOLS:
            for run in load_run_index(experiment[protocol]):
                arrays = load_roc_arrays(run["result_file"])
                if arrays is None:
                    continue

                for method, (fpr_field, tpr_field) in ROC_FIELDS.items():
                    tpr_value, empirical_fpr = tpr_at_fpr(
                        arrays[fpr_field],
                        arrays[tpr_field],
                    )
                    rows.append(
                        {
                            "dataset": experiment["dataset"],
                            "protocol": protocol,
                            "protocol_label": protocol_label,
                            "deletion_budget": run["deletion_budget"],
                            "method": method,
                            "method_label": METHODS[method]["label"],
                            "target_fpr": TARGET_FPR,
                            "empirical_fpr": empirical_fpr,
                            "tpr_at_1pct_fpr": tpr_value,
                            "tpr_at_1pct_fpr_percent": 100.0 * tpr_value,
                            "result_file": run["result_file"].name,
                        }
                    )

    if not rows:
        raise RuntimeError("No ROC results were found.")
    return pd.DataFrame(rows)


def polish_axes(ax):
    ax.grid(
        True,
        which="major",
        axis="both",
        color="#B8B8B8",
        linestyle="--",
        linewidth=0.9,
        alpha=0.65,
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.5)


def plot_small_multiples(results):
    fig, axes = plt.subplots(
        len(PROTOCOLS),
        len(EXPERIMENTS),
        figsize=(14.0, 8.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    max_tpr = float(results["tpr_at_1pct_fpr_percent"].max())
    y_upper = max(5.0, 5.0 * np.ceil((max_tpr + 1.0) / 5.0))

    for row, (protocol, protocol_label) in enumerate(PROTOCOLS):
        for column, experiment in enumerate(EXPERIMENTS):
            ax = axes[row, column]
            panel = results[
                (results["protocol"] == protocol)
                & (results["dataset"] == experiment["dataset"])
            ]

            for method, style in METHODS.items():
                series = panel[panel["method"] == method].sort_values(
                    "deletion_budget"
                )
                if series.empty:
                    continue
                ax.plot(
                    series["deletion_budget"],
                    series["tpr_at_1pct_fpr_percent"],
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=7.5,
                    markeredgewidth=1.3,
                    alpha=style["alpha"],
                )

            ax.set_title(f"{experiment['dataset']} — {protocol_label}")
            ax.set_xticks(DELETION_BUDGETS)
            ax.set_xlim(min(DELETION_BUDGETS) - 0.5, max(DELETION_BUDGETS) + 0.5)
            ax.set_ylim(0.0, y_upper)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
            if row == len(PROTOCOLS) - 1:
                ax.set_xlabel("Deletion budget $k$")
            if column == 0:
                ax.set_ylabel("TPR at 1% FPR (%)")
            polish_axes(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(METHODS),
        fancybox=True,
        columnspacing=1.6,
        handletextpad=0.6,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#666666")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.82)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(
            FIG_DIR / f"ADP_TPR_at_1pct_FPR_small_multiples.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def save_table(results):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        TABLE_DIR / "ADP_TPR_at_1pct_FPR.csv",
        index=False,
    )


def main():
    configure_plot_style()
    results = build_results_table()
    save_table(results)
    plot_small_multiples(results)


if __name__ == "__main__":
    main()
