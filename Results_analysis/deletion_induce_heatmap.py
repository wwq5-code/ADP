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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


# Each panel compares a matched unlearning implementation against the Static
# attack from the same summary file.  TinyImageNet SalUn currently has no k=15
# result; the corresponding cells are left missing rather than interpolated.
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

METHODS = (
    ("passive_deletion_auc", "Passive-1"),
    ("sequential_adp_auc", "ADP-Random-$k$"),
    ("active_adp_auc", "ADP-Planned-$k$"),
)

DELETION_BUDGETS = (3, 5, 7, 9, 11, 13, 15)


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_summary(summary_file):
    """Read the scalar AUC fields needed for deletion-induced leakage."""
    columns = (
        "deletion_budget",
        "static_auc",
        *(field for field, _ in METHODS),
    )
    if not summary_file.exists():
        warnings.warn(f"Missing summary file: {summary_file}")
        return pd.DataFrame(columns=columns)

    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    patterns = {
        "deletion_budget": re.compile(
            rf"'num_unlearning_operations':\s*({number})"
        ),
        "static_auc": re.compile(rf"'static_auc':\s*({number})"),
        **{
            field: re.compile(rf"'{field}':\s*({number})")
            for field, _ in METHODS
        },
    }

    rows = []
    with summary_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("result:"):
                continue

            row = {}
            for field, pattern in patterns.items():
                match = pattern.search(line)
                if match is None:
                    raise RuntimeError(
                        f"Missing {field} in a result row from {summary_file}"
                    )
                row[field] = float(match.group(1))
            rows.append(row)

    if not rows:
        warnings.warn(f"No completed result rows in: {summary_file}")
        return pd.DataFrame(columns=columns)

    data = pd.DataFrame(rows).sort_values("deletion_budget").reset_index(drop=True)
    data["deletion_budget"] = data["deletion_budget"].astype(int)
    if data["deletion_budget"].duplicated().any():
        raise RuntimeError(f"Duplicate deletion budgets in {summary_file}")
    return data


def build_long_table():
    """Return matched AUROC gains over Static, measured in percentage points."""
    rows = []
    for experiment in EXPERIMENTS:
        for protocol, protocol_label in PROTOCOLS:
            data = load_summary(experiment[protocol])
            for _, result in data.iterrows():
                for method, method_label in METHODS:
                    rows.append(
                        {
                            "dataset": experiment["dataset"],
                            "protocol": protocol,
                            "protocol_label": protocol_label,
                            "method": method,
                            "method_label": method_label,
                            "deletion_budget": int(result["deletion_budget"]),
                            "method_auc": 100.0 * result[method],
                            "static_auc": 100.0 * result["static_auc"],
                            "gain_over_static": 100.0
                            * (result[method] - result["static_auc"]),
                        }
                    )

    if not rows:
        raise RuntimeError("No deletion-induced leakage results were found.")
    return pd.DataFrame(rows)


def matrix_for_protocol(results, protocol):
    row_index = pd.MultiIndex.from_product(
        [
            [experiment["dataset"] for experiment in EXPERIMENTS],
            [method_label for _, method_label in METHODS],
        ],
        names=("dataset", "method_label"),
    )
    matrix = (
        results[results["protocol"] == protocol]
        .pivot(
            index=["dataset", "method_label"],
            columns="deletion_budget",
            values="gain_over_static",
        )
        .reindex(index=row_index, columns=DELETION_BUDGETS)
    )
    return matrix


def annotate_heatmap(ax, values, color_limit):
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                ax.text(
                    column,
                    row,
                    "—",
                    ha="center",
                    va="center",
                    color="#777777",
                    fontsize=12,
                )
                continue

            text_color = "white" if abs(value) >= 0.58 * color_limit else "#202020"
            ax.text(
                column,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )


def format_heatmap_axes(ax, matrix, title, show_y_labels):
    ax.set_title(title, pad=12, fontweight="medium")
    ax.set_xlabel("Deletion budget $k$", labelpad=10)
    ax.set_xticks(np.arange(len(DELETION_BUDGETS)))
    ax.set_xticklabels(DELETION_BUDGETS)

    ax.set_yticks(np.arange(len(matrix.index)))
    if show_y_labels:
        ax.set_yticklabels(
            [f"{dataset} — {method}" for dataset, method in matrix.index]
        )
    else:
        ax.tick_params(axis="y", labelleft=False)

    ax.set_xticks(np.arange(-0.5, len(DELETION_BUDGETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_table(results):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        TABLE_DIR / "ADP_deletion_induced_leakage.csv",
        index=False,
    )


def plot_heatmap(results):
    matrices = [matrix_for_protocol(results, protocol) for protocol, _ in PROTOCOLS]
    finite_values = np.concatenate(
        [matrix.to_numpy().ravel() for matrix in matrices]
    )
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise RuntimeError("No finite deletion-induced leakage values were found.")

    color_limit = max(1.0, float(np.ceil(np.max(np.abs(finite_values)))))
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E6E6E6")

    fig, axes = plt.subplots(
        1,
        len(PROTOCOLS),
        figsize=(18.0, 9.2),
        sharey=True,
        constrained_layout=True,
    )

    image = None
    for panel, (ax, matrix, (_, protocol_label)) in enumerate(
        zip(axes, matrices, PROTOCOLS)
    ):
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(
            np.ma.masked_invalid(values),
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="none",
        )
        annotate_heatmap(ax, values, color_limit)
        format_heatmap_axes(
            ax,
            matrix,
            protocol_label,
            show_y_labels=(panel == 0),
        )

    fig.suptitle(
        "Deletion-Induced Leakage Relative to the Static Attack",
        fontsize=24,
        fontweight="medium",
    )
    colorbar = fig.colorbar(
        image,
        ax=axes,
        location="right",
        shrink=0.88,
        pad=0.025,
    )
    colorbar.set_label("AUROC gain over Static (percentage points)", fontsize=17)
    colorbar.ax.tick_params(labelsize=14)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(
            FIG_DIR / f"ADP_deletion_induced_leakage_heatmap.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    configure_plot_style()
    results = build_long_table()
    save_table(results)
    plot_heatmap(results)


if __name__ == "__main__":
    main()
