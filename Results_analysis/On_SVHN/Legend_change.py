"""Regenerate SVHN ROC figures with paper-consistent legend names.

The script reads saved ROC coordinates and AUROC values from ``results/*.txt``;
it does not rerun training, unlearning, or attack inference.  By default it
redraws all seven budgets for Retrain, SalUn, and shadow-reference results.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALL_BUDGETS = (3, 5, 7, 9, 11, 13, 15)

PLOT_CONFIGS = {
    "retrain": {
        "result_prefix": "Baseline_Probability_Active_ADP_Unlearning_Size_",
        "figure_prefix": (
            "ROC_Baseline_Probability_Active_ADP_Unlearning_Size_"
        ),
        "title": "SVHN membership attacks",
        "dvia_label": "DVIA UCD likelihood ratio",
        "budgets": ALL_BUDGETS,
    },
    "salun": {
        "result_prefix": (
            "Baseline_SalUn_Probability_Active_ADP_Unlearning_Size_"
        ),
        "figure_prefix": (
            "ROC_Baseline_SalUn_Probability_Active_ADP_Unlearning_Size_"
        ),
        "title": "SVHN SalUn membership attacks",
        "dvia_label": "DVIA SalUn UCD likelihood ratio",
        "budgets": ALL_BUDGETS,
    },
    "shadow-reference": {
        "result_prefix": (
            "Shadow_Reference_Probability_Active_ADP_Unlearning_Size_"
        ),
        "figure_prefix": (
            "ROC_Shadow_Reference_Probability_Active_ADP_Unlearning_Size_"
        ),
        "title": "SVHN shadow/reference membership attacks",
        "dvia_label": "Shadow-calibrated DVIA",
        "budgets": ALL_BUDGETS,
    },
}

CURVES = (
    ("static", "steelblue", "-", "Static probability MIA"),
    ("passive", "darkorange", "-", "Passive-1 MIA"),
    ("sequential", "forestgreen", "-", "ADP-Random MIA"),
    ("active", "crimson", "-", "ADP-Planned MIA"),
    ("dvia", "saddlebrown", "-.", "DVIA UCD likelihood ratio"),
)

FIELD_NAMES = {
    "static": ("static_roc_fpr", "static_roc_tpr", "static_auc"),
    "passive": (
        "passive_roc_fpr",
        "passive_roc_tpr",
        "passive_deletion_auc",
    ),
    "sequential": (
        "sequential_roc_fpr",
        "sequential_roc_tpr",
        "sequential_adp_auc",
    ),
    "active": ("active_roc_fpr", "active_roc_tpr", "active_adp_auc"),
    "dvia": ("dvia_roc_fpr", "dvia_roc_tpr", "dvia_auc"),
}

REQUIRED_KEYS = {"num_unlearning_operations"}
for _fpr_key, _tpr_key, _auc_key in FIELD_NAMES.values():
    REQUIRED_KEYS.update((_fpr_key, _tpr_key, _auc_key))


def read_result_file(path: Path) -> dict[str, object]:
    """Read only the scalar and ROC fields required for one figure."""
    values: dict[str, object] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            key, separator, raw_value = line.partition(":")
            if not separator or key not in REQUIRED_KEYS:
                continue
            try:
                values[key] = ast.literal_eval(raw_value.strip())
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    f"Cannot parse {key!r} at {path}:{line_number}"
                ) from error

    missing = sorted(REQUIRED_KEYS.difference(values))
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")

    for curve_name, (fpr_key, tpr_key, _) in FIELD_NAMES.items():
        if len(values[fpr_key]) != len(values[tpr_key]):
            raise ValueError(
                f"{path}: {curve_name} FPR/TPR arrays have different lengths"
            )
    return values


def draw_roc_figure(
    values: dict[str, object],
    output_path: Path,
    title: str,
    dvia_label: str,
) -> None:
    """Reproduce the original plot styling with updated legend labels."""
    budget = int(values["num_unlearning_operations"])
    figure, axes = plt.subplots(figsize=(6.4, 4.8))

    for curve_name, color, linestyle, label in CURVES:
        fpr_key, tpr_key, auc_key = FIELD_NAMES[curve_name]
        displayed_label = dvia_label if curve_name == "dvia" else label
        axes.plot(
            values[fpr_key],
            values[tpr_key],
            color=color,
            linewidth=2,
            linestyle=linestyle,
            label=f"{displayed_label} (area = {float(values[auc_key]):.2f})",
        )

    axes.plot([0, 1], [0, 1], color="navy", linewidth=2, linestyle="--")
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)
    axes.set_xlabel("False Positive Rate")
    axes.set_ylabel("True Positive Rate")
    axes.set_title(f"{title} (unlearning size={budget})")
    axes.legend(loc="lower right")
    figure.savefig(output_path)
    plt.close(figure)


def regenerate_figures(
    results_dir: Path,
    methods: Iterable[str],
    requested_budgets: Sequence[int] | None,
) -> None:
    """Validate all requested inputs before replacing any ROC figure."""
    loaded = []
    for method in methods:
        config = PLOT_CONFIGS[method]
        budgets = requested_budgets or config["budgets"]
        for budget in budgets:
            input_path = results_dir / f"{config['result_prefix']}{budget}.txt"
            if not input_path.is_file():
                raise FileNotFoundError(f"Missing result file: {input_path}")
            values = read_result_file(input_path)
            recorded_budget = int(values["num_unlearning_operations"])
            if recorded_budget != budget:
                raise ValueError(
                    f"{input_path} records budget {recorded_budget}, "
                    f"expected {budget}"
                )
            loaded.append((budget, values, config))

    results_dir.mkdir(parents=True, exist_ok=True)
    for budget, values, config in loaded:
        output_path = results_dir / f"{config['figure_prefix']}{budget}.png"
        draw_roc_figure(
            values,
            output_path,
            title=str(config["title"]),
            dvia_label=str(config["dvia_label"]),
        )
        print(f"Regenerated {output_path}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Redraw SVHN ROC plots with paper-consistent legends."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results",
        help="Directory containing result TXT files and output PNG files.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="Optional deletion budgets; defaults to every available budget.",
    )
    parser.add_argument(
        "--methods",
        choices=tuple(PLOT_CONFIGS),
        nargs="+",
        default=tuple(PLOT_CONFIGS),
        help="Plot families (default: retrain salun shadow-reference).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regenerate_figures(
        args.results_dir.resolve(), args.methods, args.budgets
    )


if __name__ == "__main__":
    main()
