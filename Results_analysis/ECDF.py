"""Plot attack-score ECDFs from the saved ADP experiment results.

The script generates separate figures for deletion budgets k=3, k=9, and
k=13.
Each panel compares ADP-Random and ADP-Planned, with separate ECDFs for members
and non-members.  The saved ``measurement_membership_ground_truth`` array is
used to split the scores, so the script does not rely on an assumed sample
ordering.
"""

import ast
import csv
import os
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
TABLE_DIR = BASE_DIR / "tables"

os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np


# Generate one figure and one summary table for each budget.
DELETION_BUDGETS = (3, 9, 13)

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
    {
        "dataset": "SVHN",
        "retrain": (
            BASE_DIR
            / "On_SVHN/results/"
            "Baseline_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt"
        ),
        "salun": (
            BASE_DIR
            / "On_SVHN/results/"
            "Baseline_SalUn_Probability_Active_ADP_Different_Unlearning_Sizes_Summary.txt"
        ),
    },
    {
        "dataset": "TinyImageNet",
        "retrain": (
            BASE_DIR
            / "On_TinyImageNet/results/"
            "TinyImageNet_ResNet18_imagenet1k_pretrained_Probability_Active_ADP_"
            "Different_Unlearning_Sizes_Summary.txt"
        ),
        "salun": (
            BASE_DIR
            / "On_TinyImageNet/results/"
            "TinyImageNet_ResNet18_imagenet1k_pretrained_SalUn_Probability_Active_ADP_"
            "Different_Unlearning_Sizes_Summary.txt"
        ),
    },
]

PROTOCOLS = (
    ("retrain", "Retrain"),
    ("salun", "SalUn"),
)

METHODS = {
    "sequential": {
        "label": "ADP-Random",
        "score_field": "sequential_attack_scores",
        "auc_field": "sequential_adp_auc",
        "color": "#4C78A8",
    },
    "active": {
        "label": "ADP-Planned",
        "score_field": "active_attack_scores",
        "auc_field": "active_adp_auc",
        "color": "#D65F5F",
    },
}

MEMBERSHIP_GROUPS = (
    (1, "Member", "-"),
    (0, "Non-member", "--"),
)


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.labelsize": 16,
            "axes.titlesize": 17,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "axes.linewidth": 1.2,
            "axes.edgecolor": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def result_file_for_budget(summary_file, deletion_budget):
    """Resolve a per-budget result file from an experiment summary."""
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_file}")

    budget_pattern = re.compile(r"'num_unlearning_operations':\s*(\d+)")
    file_pattern = re.compile(r"'result_file':\s*'([^']+)'")
    matches = []

    with summary_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("result:"):
                continue
            budget_match = budget_pattern.search(line)
            file_match = file_pattern.search(line)
            if budget_match is None or file_match is None:
                raise RuntimeError(f"Malformed result row in {summary_file}")
            if int(budget_match.group(1)) == deletion_budget:
                matches.append(summary_file.parent / file_match.group(1))

    if not matches:
        raise RuntimeError(
            f"No k={deletion_budget} result is listed in {summary_file}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple k={deletion_budget} results are listed in {summary_file}"
        )
    if not matches[0].exists():
        raise FileNotFoundError(f"Missing result file: {matches[0]}")
    return matches[0]


def load_attack_data(result_file):
    """Load membership labels, attack scores, and reported AUCs."""
    required_fields = {"measurement_membership_ground_truth"}
    for method in METHODS.values():
        required_fields.add(method["score_field"])
        required_fields.add(method["auc_field"])

    values = {}
    with result_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            key, separator, value = raw_line.partition(":")
            key = key.strip()
            if separator and key in required_fields:
                values[key] = ast.literal_eval(value.strip())

    missing = sorted(required_fields.difference(values))
    if missing:
        raise RuntimeError(f"Missing fields {missing} in {result_file}")

    labels = np.asarray(
        values["measurement_membership_ground_truth"], dtype=int
    )
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"Invalid membership labels in {result_file}")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError(f"Membership labels must be binary in {result_file}")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError(f"Both membership classes are required in {result_file}")

    attacks = {}
    for method_key, method in METHODS.items():
        scores = np.asarray(values[method["score_field"]], dtype=float)
        if scores.ndim != 1 or scores.shape != labels.shape:
            raise ValueError(
                f"{method['score_field']} does not match the membership labels "
                f"in {result_file}"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                f"{method['score_field']} contains non-finite values in {result_file}"
            )
        attacks[method_key] = {
            "scores": scores,
            "auc": float(values[method["auc_field"]]),
        }

    return labels, attacks


def ecdf(values):
    """Return coordinates for a right-continuous empirical CDF."""
    sorted_values = np.sort(np.asarray(values, dtype=float))
    probabilities = np.arange(1, sorted_values.size + 1) / sorted_values.size
    return sorted_values, probabilities


def summary_row(
    dataset,
    protocol,
    deletion_budget,
    method,
    membership,
    scores,
    auc,
    result_file,
):
    return {
        "dataset": dataset,
        "protocol": protocol,
        "deletion_budget": deletion_budget,
        "method": method,
        "membership": membership,
        "n": scores.size,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "q10": float(np.quantile(scores, 0.10)),
        "median": float(np.median(scores)),
        "q90": float(np.quantile(scores, 0.90)),
        "auc": auc,
        "result_file": str(result_file.relative_to(BASE_DIR)),
    }


def draw_panel(
    ax,
    dataset,
    protocol_key,
    summary_file,
    deletion_budget,
    summary_rows,
):
    result_file = result_file_for_budget(summary_file, deletion_budget)
    labels, attacks = load_attack_data(result_file)

    for method_key, method in METHODS.items():
        attack = attacks[method_key]
        for membership_value, membership_label, linestyle in MEMBERSHIP_GROUPS:
            class_scores = attack["scores"][labels == membership_value]
            x_values, y_values = ecdf(class_scores)
            ax.step(
                x_values,
                y_values,
                where="post",
                color=method["color"],
                linestyle=linestyle,
                linewidth=2.15,
                alpha=0.96,
            )
            summary_rows.append(
                summary_row(
                    dataset,
                    protocol_key,
                    deletion_budget,
                    method["label"],
                    membership_label,
                    class_scores,
                    attack["auc"],
                    result_file,
                )
            )

    random_auc = attacks["sequential"]["auc"]
    active_auc = attacks["active"]["auc"]
    ax.text(
        0.97,
        0.05,
        f"AUC R/P: {random_auc:.3f}/{active_auc:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid(True, color="#D9D9D9", linewidth=0.75, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_summary_table(rows, deletion_budget):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output_file = (
        TABLE_DIR / f"ADP_attack_score_ECDF_summary_k{deletion_budget}.csv"
    )
    fieldnames = [
        "dataset",
        "protocol",
        "deletion_budget",
        "method",
        "membership",
        "n",
        "mean",
        "std",
        "q10",
        "median",
        "q90",
        "auc",
        "result_file",
    ]
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_file


def plot_ecdf(deletion_budget):
    configure_plot_style()
    summary_rows = []
    figure, axes = plt.subplots(
        len(PROTOCOLS),
        len(EXPERIMENTS),
        figsize=(17.2, 7.4),
        sharex=True,
        sharey=True,
    )

    for row_index, (protocol_key, protocol_label) in enumerate(PROTOCOLS):
        for column_index, experiment in enumerate(EXPERIMENTS):
            ax = axes[row_index, column_index]
            draw_panel(
                ax,
                experiment["dataset"],
                protocol_key,
                experiment[protocol_key],
                deletion_budget,
                summary_rows,
            )
            if row_index == 0:
                ax.set_title(experiment["dataset"], pad=10)
            if column_index == 0:
                ax.set_ylabel("ECDF")

        figure.text(
            0.018,
            0.675 if row_index == 0 else 0.285,
            f"{protocol_label}\n($k={deletion_budget}$)",
            ha="center",
            va="center",
            rotation=90,
            fontsize=16,
            fontweight="bold",
        )

    legend_handles = []
    for method in METHODS.values():
        for _, membership_label, linestyle in MEMBERSHIP_GROUPS:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=method["color"],
                    linestyle=linestyle,
                    linewidth=2.5,
                    label=f"{method['label']} {membership_label}",
                )
            )

    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.6,
    )
    figure.supxlabel(
        "Membership score (higher indicates more member-like)",
        fontsize=16,
        y=0.045,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.88,
        bottom=0.13,
        wspace=0.13,
        hspace=0.18,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = FIG_DIR / f"ADP_attack_score_ECDF_k{deletion_budget}"
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    table_file = save_summary_table(summary_rows, deletion_budget)
    print(f"Saved: {output_stem.with_suffix('.pdf')}")
    print(f"Saved: {output_stem.with_suffix('.png')}")
    print(f"Saved: {table_file}")


if __name__ == "__main__":
    for budget in DELETION_BUDGETS:
        plot_ecdf(budget)
