"""Controlled evidence for membership-conditioned retained-set re-equilibration.

This diagnostic implements the matched worlds used in the ADP paper.  For a
target ``x*`` and a class-matched replacement ``x_tilde``, it constructs

    D_0^1 = R union A union {x*}
    D_0^0 = R union A union {x_tilde}

where ``A`` is an attacker-owned probe pool.  The same probe ``a`` is removed
from both worlds, and every before/after model is trained from scratch.  All
four trainings for a target/seed pair use the same initialization and sampler
seed.  The observable deletion response is

    delta^b = p_after^b(x*) - p_before^b(x*),

and the empirical membership-conditioned response is ``delta^1 - delta^0``.

The script reports:

* the paired difference in the correct-label log-odds response (an empirical
  scalar version of Gamma in the paper), with a target-pair bootstrap CI;
* full-posterior response separation;
* group-held-out AUROC for the initial posterior, the deletion response, the
  response residualized on the initial posterior, and their concatenation;
* JSON/CSV records and a PNG diagnostic figure.

This is a controlled mechanism diagnostic, not the main target-service attack
evaluation and not a direct validation of the paper's local influence
approximation.  Its intended interpretation is evidence consistent with
membership-conditioned re-equilibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

try:
    from .baseline.model import CNN
except ImportError:
    from baseline.model import CNN


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "FASHION_MNIST_DATA_ROOT",
        "/home/wwq/Data/data/fashion_mnist",
    )
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "re_equilibration_evidence"


@dataclass(frozen=True)
class PairDefinition:
    """One matched H1/H0 construction and its shared deletion request."""

    pair_id: int
    target_train_index: int
    replacement_test_index: int
    deleted_probe_train_index: int
    target_label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure membership-conditioned posterior responses to the same "
            "single-record deletion in matched Fashion-MNIST worlds."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download Fashion-MNIST if it is absent.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device such as cuda:0, mps, or cpu; 'auto' chooses an accelerator.",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=5000,
        help="Number of common retained records in R (default: 5000).",
    )
    parser.add_argument(
        "--probe-pool-size",
        type=int,
        default=30,
        help="Number of attacker-owned records in A (default: 30).",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=12,
        help="Number of class-matched target/replacement pairs (default: 12).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Matched training repetitions per target pair (default: 3).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for every exact-retraining model (default: 50).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help=(
            "Shuffle training data with a matched generator. By default the "
            "loader is deterministic and unshuffled, matching main_baseline.py."
        ),
    )
    parser.add_argument("--feature-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--training-seed", type=int, default=3000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--utility-samples",
        type=int,
        default=1000,
        help="Test records used for utility checks; use 0 to disable.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke-test settings: 256 base records, 2 pairs, 1 seed, and "
            "1 epoch. These results are not suitable for the paper."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "base_size",
        "probe_pool_size",
        "num_pairs",
        "num_seeds",
        "epochs",
        "batch_size",
        "cv_folds",
        "bootstrap_samples",
    )
    for name in positive_integer_fields:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.ridge <= 0:
        raise ValueError("--ridge must be positive.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1).")
    if args.utility_samples < 0:
        raise ValueError("--utility-samples cannot be negative.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_determinism(seed: int) -> None:
    """Reset every RNG used by model initialization and optimization."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_or_make_permutation(
    path: Path, length: int, seed: int
) -> torch.Tensor:
    """Reuse the main experiment split when available, without creating it."""

    if path.exists():
        permutation = torch.load(path, map_location="cpu")
        permutation = torch.as_tensor(permutation, dtype=torch.long).flatten()
        if len(permutation) != length:
            raise ValueError(
                f"{path} has {len(permutation)} entries; expected {length}."
            )
        if len(torch.unique(permutation)) != length:
            raise ValueError(f"{path} is not a permutation.")
        if int(permutation.min()) != 0 or int(permutation.max()) != length - 1:
            raise ValueError(f"{path} contains indices outside [0, {length - 1}].")
        return permutation
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator)


def select_experiment_records(
    train_targets: torch.Tensor,
    test_targets: torch.Tensor,
    train_permutation: torch.Tensor,
    test_permutation: torch.Tensor,
    base_size: int,
    probe_pool_size: int,
    num_pairs: int,
) -> tuple[list[int], list[int], list[PairDefinition]]:
    """Create equal-size, class-matched worlds with disjoint roles."""

    ordered_train = [int(index) for index in train_permutation]
    ordered_test = [int(index) for index in test_permutation]
    required_train = probe_pool_size + num_pairs + base_size
    if required_train > len(ordered_train):
        raise ValueError(
            "Requested probe, target, and base records exceed the training set: "
            f"{required_train} > {len(ordered_train)}."
        )

    probe_indices = ordered_train[:probe_pool_size]
    target_indices = ordered_train[probe_pool_size : probe_pool_size + num_pairs]
    reserved = set(probe_indices) | set(target_indices)
    base_indices = [
        index for index in ordered_train if index not in reserved
    ][:base_size]

    replacements_by_class: dict[int, list[int]] = {label: [] for label in range(10)}
    for index in ordered_test:
        replacements_by_class[int(test_targets[index])].append(index)

    used_per_class = {label: 0 for label in range(10)}
    pairs: list[PairDefinition] = []
    for pair_id, target_index in enumerate(target_indices):
        label = int(train_targets[target_index])
        replacement_position = used_per_class[label]
        candidates = replacements_by_class[label]
        if replacement_position >= len(candidates):
            raise ValueError(f"No unused class-{label} test replacement remains.")
        replacement_index = candidates[replacement_position]
        used_per_class[label] += 1
        pairs.append(
            PairDefinition(
                pair_id=pair_id,
                target_train_index=target_index,
                replacement_test_index=replacement_index,
                deleted_probe_train_index=probe_indices[pair_id % probe_pool_size],
                target_label=label,
            )
        )
    return base_indices, probe_indices, pairs


def materialize(dataset, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in indices:
        image, label = dataset[int(index)]
        images.append(image)
        labels.append(int(label))
    if not images:
        return torch.empty((0, 1, 28, 28)), torch.empty((0,), dtype=torch.long)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def make_model(args: argparse.Namespace) -> CNN:
    return CNN(
        num_classes=10,
        feature_size=args.feature_size,
        dropout=args.dropout,
    )


@torch.no_grad()
def posterior(model: nn.Module, image: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(image.unsqueeze(0).to(device))
    return torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float64)


@torch.no_grad()
def accuracy(
    model: nn.Module,
    images: torch.Tensor | None,
    labels: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> float | None:
    if images is None or labels is None or len(images) == 0:
        return None
    model.eval()
    correct = 0
    loader = DataLoader(
        TensorDataset(images, labels), batch_size=batch_size, shuffle=False
    )
    for batch_images, batch_labels in loader:
        predictions = model(batch_images.to(device)).argmax(dim=1).cpu()
        correct += int((predictions == batch_labels).sum())
    return correct / len(labels)


def train_and_measure(
    images: torch.Tensor,
    labels: torch.Tensor,
    target_image: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    utility_images: torch.Tensor | None,
    utility_labels: torch.Tensor | None,
) -> tuple[np.ndarray, float, float | None]:
    """Train one exact-retraining model from scratch and query x*."""

    set_determinism(seed)
    model = make_model(args).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    final_loss = float("nan")
    for _ in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_labels)
            total_count += len(batch_labels)
        final_loss = total_loss / total_count
    target_posterior = posterior(model, target_image, device)
    test_accuracy = accuracy(
        model,
        utility_images,
        utility_labels,
        args.batch_size,
        device,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return target_posterior, final_loss, test_accuracy


def stable_log_odds(probability: float, epsilon: float = 1e-6) -> float:
    clipped = float(np.clip(probability, epsilon, 1.0 - epsilon))
    return float(np.log(clipped) - np.log1p(-clipped))


def run_pair_seed(
    pair: PairDefinition,
    repetition: int,
    seed: int,
    base_images: torch.Tensor,
    base_labels: torch.Tensor,
    probe_images: torch.Tensor,
    probe_labels: torch.Tensor,
    probe_index_to_position: dict[int, int],
    train_dataset,
    test_dataset,
    args: argparse.Namespace,
    device: torch.device,
    utility_images: torch.Tensor | None,
    utility_labels: torch.Tensor | None,
) -> dict:
    target_image, target_label = train_dataset[pair.target_train_index]
    replacement_image, replacement_label = test_dataset[pair.replacement_test_index]
    if int(target_label) != pair.target_label or int(replacement_label) != pair.target_label:
        raise RuntimeError("The matched-world class invariant was violated.")

    member_before_images = torch.cat(
        [base_images, probe_images, target_image.unsqueeze(0)], dim=0
    )
    member_before_labels = torch.cat(
        [base_labels, probe_labels, torch.tensor([target_label])], dim=0
    )
    nonmember_before_images = torch.cat(
        [base_images, probe_images, replacement_image.unsqueeze(0)], dim=0
    )
    nonmember_before_labels = torch.cat(
        [base_labels, probe_labels, torch.tensor([replacement_label])], dim=0
    )

    deleted_position = probe_index_to_position[pair.deleted_probe_train_index]
    keep_probe = torch.ones(len(probe_images), dtype=torch.bool)
    keep_probe[deleted_position] = False
    retained_probe_images = probe_images[keep_probe]
    retained_probe_labels = probe_labels[keep_probe]
    member_after_images = torch.cat(
        [base_images, retained_probe_images, target_image.unsqueeze(0)], dim=0
    )
    member_after_labels = torch.cat(
        [base_labels, retained_probe_labels, torch.tensor([target_label])], dim=0
    )
    nonmember_after_images = torch.cat(
        [base_images, retained_probe_images, replacement_image.unsqueeze(0)], dim=0
    )
    nonmember_after_labels = torch.cat(
        [base_labels, retained_probe_labels, torch.tensor([replacement_label])], dim=0
    )

    # Equal-size matched worlds are essential: H1/H0 differ only in the record
    # occupying the final world-record slot. All four fits use the same seed.
    if len(member_before_images) != len(nonmember_before_images):
        raise RuntimeError("Pre-deletion matched worlds have unequal size.")
    if len(member_after_images) != len(nonmember_after_images):
        raise RuntimeError("Post-deletion matched worlds have unequal size.")

    outputs: dict[str, tuple[np.ndarray, float, float | None]] = {}
    training_sets = {
        "member_before": (member_before_images, member_before_labels),
        "member_after": (member_after_images, member_after_labels),
        "nonmember_before": (nonmember_before_images, nonmember_before_labels),
        "nonmember_after": (nonmember_after_images, nonmember_after_labels),
    }
    for name, (images, labels) in training_sets.items():
        outputs[name] = train_and_measure(
            images,
            labels,
            target_image,
            seed,
            args,
            device,
            utility_images,
            utility_labels,
        )

    p0_member, loss0_member, accuracy0_member = outputs["member_before"]
    p1_member, loss1_member, accuracy1_member = outputs["member_after"]
    p0_nonmember, loss0_nonmember, accuracy0_nonmember = outputs[
        "nonmember_before"
    ]
    p1_nonmember, loss1_nonmember, accuracy1_nonmember = outputs[
        "nonmember_after"
    ]
    delta_member = p1_member - p0_member
    delta_nonmember = p1_nonmember - p0_nonmember
    label = pair.target_label
    u0_member = stable_log_odds(p0_member[label])
    u1_member = stable_log_odds(p1_member[label])
    u0_nonmember = stable_log_odds(p0_nonmember[label])
    u1_nonmember = stable_log_odds(p1_nonmember[label])

    return {
        **asdict(pair),
        "repetition": repetition,
        "training_seed": seed,
        "p0_member": p0_member.tolist(),
        "p1_member": p1_member.tolist(),
        "p0_nonmember": p0_nonmember.tolist(),
        "p1_nonmember": p1_nonmember.tolist(),
        "delta_member": delta_member.tolist(),
        "delta_nonmember": delta_nonmember.tolist(),
        "posterior_gamma": (delta_member - delta_nonmember).tolist(),
        "u0_member": u0_member,
        "u1_member": u1_member,
        "u0_nonmember": u0_nonmember,
        "u1_nonmember": u1_nonmember,
        "delta_log_odds_member": u1_member - u0_member,
        "delta_log_odds_nonmember": u1_nonmember - u0_nonmember,
        "gamma_log_odds": (u1_member - u0_member) - (u1_nonmember - u0_nonmember),
        "delta_true_probability_member": float(delta_member[label]),
        "delta_true_probability_nonmember": float(delta_nonmember[label]),
        "gamma_true_probability": float(
            delta_member[label] - delta_nonmember[label]
        ),
        "final_loss_member_before": loss0_member,
        "final_loss_member_after": loss1_member,
        "final_loss_nonmember_before": loss0_nonmember,
        "final_loss_nonmember_after": loss1_nonmember,
        "test_accuracy_member_before": accuracy0_member,
        "test_accuracy_member_after": accuracy1_member,
        "test_accuracy_nonmember_before": accuracy0_nonmember,
        "test_accuracy_nonmember_after": accuracy1_nonmember,
    }


def _stack_decoder_data(
    observations: Sequence[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p0: list[list[float]] = []
    delta: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    seeds: list[int] = []
    for observation in observations:
        for world, label in (("member", 1), ("nonmember", 0)):
            p0.append(observation[f"p0_{world}"])
            delta.append(observation[f"delta_{world}"])
            labels.append(label)
            groups.append(int(observation["pair_id"]))
            seeds.append(int(observation["training_seed"]))
    return (
        np.asarray(p0, dtype=np.float64),
        np.asarray(delta, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
        np.asarray(seeds, dtype=np.int64),
    )


def grouped_oof_auc(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    requested_folds: int,
    seed: int,
    conditioners: np.ndarray | None = None,
    ridge: float = 1e-4,
) -> float | None:
    """Cross-validate by target pair so no seed replicate leaks across folds."""

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return None
    folds = min(requested_folds, len(unique_groups))
    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)
    group_folds = np.array_split(shuffled_groups, folds)
    scores = np.full(len(labels), np.nan, dtype=np.float64)

    for test_groups in group_folds:
        test_mask = np.isin(groups, test_groups)
        train_mask = ~test_mask
        train_features = features[train_mask]
        test_features = features[test_mask]
        if conditioners is not None:
            response_model = Ridge(alpha=ridge)
            response_model.fit(conditioners[train_mask], train_features)
            train_features = train_features - response_model.predict(
                conditioners[train_mask]
            )
            test_features = test_features - response_model.predict(
                conditioners[test_mask]
            )
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                random_state=seed,
            ),
        )
        classifier.fit(train_features, labels[train_mask])
        scores[test_mask] = classifier.predict_proba(test_features)[:, 1]
    if np.isnan(scores).any():
        raise RuntimeError("Grouped cross-validation did not score every record.")
    return float(roc_auc_score(labels, scores))


def bootstrap_pair_mean_ci(
    values: np.ndarray, samples: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("Bootstrap values cannot be empty.")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def sign_flip_p_value(values: np.ndarray, seed: int, samples: int = 20000) -> float:
    """Two-sided paired randomization test over target-pair mean effects."""

    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    if len(values) <= 18:
        assignments = np.arange(2 ** len(values), dtype=np.uint64)[:, None]
        bits = (assignments >> np.arange(len(values), dtype=np.uint64)) & 1
        signs = bits.astype(np.float64) * 2.0 - 1.0
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(samples, len(values)))
    null_statistics = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null_statistics >= observed) + 1) / (len(null_statistics) + 1))


def regularized_mahalanobis(
    member: np.ndarray, nonmember: np.ndarray, ridge: float
) -> float:
    difference = member.mean(axis=0) - nonmember.mean(axis=0)
    centered_member = member - member.mean(axis=0)
    centered_nonmember = nonmember - nonmember.mean(axis=0)
    denominator = max(len(member) + len(nonmember) - 2, 1)
    covariance = (
        centered_member.T @ centered_member
        + centered_nonmember.T @ centered_nonmember
    ) / denominator
    scale = max(float(np.trace(covariance) / covariance.shape[0]), 1e-12)
    covariance = covariance + ridge * scale * np.eye(covariance.shape[0])
    squared = float(difference @ np.linalg.pinv(covariance) @ difference)
    return float(np.sqrt(max(squared, 0.0)))


def aggregate_pair_values(observations: Sequence[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pair_ids = np.asarray(sorted({int(item["pair_id"]) for item in observations}))
    means = np.asarray(
        [
            np.mean(
                [
                    float(item[key])
                    for item in observations
                    if int(item["pair_id"]) == pair_id
                ]
            )
            for pair_id in pair_ids
        ],
        dtype=np.float64,
    )
    return pair_ids, means


def summarize(
    observations: Sequence[dict], args: argparse.Namespace
) -> dict:
    p0, delta, membership, groups, _ = _stack_decoder_data(observations)
    joint = np.concatenate([p0, delta], axis=1)
    aucs = {
        "initial_posterior": grouped_oof_auc(
            p0, membership, groups, args.cv_folds, args.split_seed
        ),
        "deletion_response": grouped_oof_auc(
            delta, membership, groups, args.cv_folds, args.split_seed
        ),
        "baseline_conditioned_response": grouped_oof_auc(
            delta,
            membership,
            groups,
            args.cv_folds,
            args.split_seed,
            conditioners=p0,
            ridge=args.ridge,
        ),
        "initial_plus_response": grouped_oof_auc(
            joint, membership, groups, args.cv_folds, args.split_seed
        ),
    }
    pair_ids, pair_gamma = aggregate_pair_values(observations, "gamma_log_odds")
    gamma_mean, gamma_lower, gamma_upper = bootstrap_pair_mean_ci(
        pair_gamma, args.bootstrap_samples, args.split_seed
    )
    member_delta = np.asarray(
        [item["delta_member"] for item in observations], dtype=np.float64
    )
    nonmember_delta = np.asarray(
        [item["delta_nonmember"] for item in observations], dtype=np.float64
    )
    posterior_gamma = member_delta - nonmember_delta
    return {
        "interpretation": (
            "Evidence consistent with membership-conditioned retained-set "
            "re-equilibration; not direct proof of the local influence approximation."
        ),
        "num_target_pairs": len(pair_ids),
        "num_seed_repetitions": args.num_seeds,
        "num_paired_observations": len(observations),
        "gamma_log_odds": {
            "pair_averaged_mean": gamma_mean,
            "pair_bootstrap_95_ci": [gamma_lower, gamma_upper],
            "paired_sign_flip_p_value": sign_flip_p_value(
                pair_gamma, args.split_seed
            ),
            "pair_values": pair_gamma.tolist(),
        },
        "full_posterior_response": {
            "mean_gamma_vector": posterior_gamma.mean(axis=0).tolist(),
            "mean_gamma_l2": float(np.linalg.norm(posterior_gamma.mean(axis=0))),
            "regularized_mahalanobis": regularized_mahalanobis(
                member_delta, nonmember_delta, args.ridge
            ),
        },
        "group_held_out_auroc": aucs,
        "auroc_note": (
            "Each fold holds out complete target pairs, including all seed "
            "replicates. The conditioned response first regresses delta on p0 "
            "without membership labels inside each training fold."
        ),
        "inference_note": (
            "The bootstrap interval and sign-flip test operate on target-pair "
            "averages. Pairs share the common retained set and probe pool, so "
            "these quantities are controlled diagnostic summaries rather than "
            "population-level significance claims. A near-zero signed scalar "
            "Gamma can also coexist with multivariate separation when response "
            "directions vary across targets or probes."
        ),
    }


def atomic_json_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def write_csv(observations: Sequence[dict], path: Path) -> None:
    scalar_fields = [
        "pair_id",
        "repetition",
        "training_seed",
        "target_train_index",
        "replacement_test_index",
        "deleted_probe_train_index",
        "target_label",
        "u0_member",
        "u1_member",
        "u0_nonmember",
        "u1_nonmember",
        "delta_log_odds_member",
        "delta_log_odds_nonmember",
        "gamma_log_odds",
        "delta_true_probability_member",
        "delta_true_probability_nonmember",
        "gamma_true_probability",
        "test_accuracy_member_before",
        "test_accuracy_member_after",
        "test_accuracy_nonmember_before",
        "test_accuracy_nonmember_after",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for observation in observations:
            writer.writerow({field: observation.get(field) for field in scalar_fields})


def make_figure(observations: Sequence[dict], summary: dict, path: Path) -> None:
    pair_ids = sorted({int(item["pair_id"]) for item in observations})
    member_by_pair = np.asarray(
        [
            np.mean(
                [
                    item["delta_log_odds_member"]
                    for item in observations
                    if int(item["pair_id"]) == pair_id
                ]
            )
            for pair_id in pair_ids
        ]
    )
    nonmember_by_pair = np.asarray(
        [
            np.mean(
                [
                    item["delta_log_odds_nonmember"]
                    for item in observations
                    if int(item["pair_id"]) == pair_id
                ]
            )
            for pair_id in pair_ids
        ]
    )
    gamma_by_pair = member_by_pair - nonmember_by_pair

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    axis = axes[0]
    axis.scatter(nonmember_by_pair, member_by_pair, color="#2864dc", s=35)
    lower = min(nonmember_by_pair.min(), member_by_pair.min())
    upper = max(nonmember_by_pair.max(), member_by_pair.max())
    padding = max((upper - lower) * 0.08, 1e-3)
    axis.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        linestyle="--",
        color="0.45",
        linewidth=1,
    )
    axis.set_xlabel("Non-member response")
    axis.set_ylabel("Member response")
    axis.set_title("Correct-label log-odds change")

    axis = axes[1]
    colors = np.where(gamma_by_pair >= 0, "#2c9c5b", "#d04a4a")
    axis.bar(np.arange(len(pair_ids)), gamma_by_pair, color=colors)
    axis.axhline(0, color="0.35", linewidth=1)
    axis.set_xlabel("Matched target pair")
    axis.set_ylabel(r"$\Delta u^1-\Delta u^0$")
    ci = summary["gamma_log_odds"]["pair_bootstrap_95_ci"]
    mean = summary["gamma_log_odds"]["pair_averaged_mean"]
    axis.set_title(f"Empirical $\\Gamma$: {mean:.3g}\n95% CI [{ci[0]:.3g}, {ci[1]:.3g}]")

    axis = axes[2]
    auc_items = [
        ("Initial", summary["group_held_out_auroc"]["initial_posterior"]),
        ("Response", summary["group_held_out_auroc"]["deletion_response"]),
        (
            "Response\n| initial",
            summary["group_held_out_auroc"]["baseline_conditioned_response"],
        ),
        ("Joint", summary["group_held_out_auroc"]["initial_plus_response"]),
    ]
    valid_auc_items = [(name, value) for name, value in auc_items if value is not None]
    axis.bar(
        [name for name, _ in valid_auc_items],
        [value for _, value in valid_auc_items],
        color=["#9aa4b2", "#4b8cf5", "#7556d8", "#ef8c34"][: len(valid_auc_items)],
    )
    axis.axhline(0.5, color="0.35", linestyle="--", linewidth=1)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Group-held-out AUROC")
    axis.set_title("Membership evidence")

    figure.suptitle("Fashion-MNIST retained-set re-equilibration diagnostic")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def format_auc(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    if args.quick:
        args.base_size = 256
        args.num_pairs = 2
        args.num_seeds = 1
        args.epochs = 1
        args.utility_samples = min(args.utility_samples, 128)
        args.bootstrap_samples = min(args.bootstrap_samples, 1000)
    validate_args(args)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}", flush=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    train_dataset = datasets.FashionMNIST(
        root=args.data_root,
        train=True,
        download=args.download,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=args.data_root,
        train=False,
        download=args.download,
        transform=transform,
    )
    train_permutation = load_or_make_permutation(
        SCRIPT_DIR / "original_indices_fmnist.pt",
        len(train_dataset),
        args.split_seed,
    )
    test_permutation = load_or_make_permutation(
        SCRIPT_DIR / "original_indices_test_fmnist.pt",
        len(test_dataset),
        args.split_seed + 1,
    )
    base_indices, probe_indices, pairs = select_experiment_records(
        torch.as_tensor(train_dataset.targets),
        torch.as_tensor(test_dataset.targets),
        train_permutation,
        test_permutation,
        args.base_size,
        args.probe_pool_size,
        args.num_pairs,
    )
    base_images, base_labels = materialize(train_dataset, base_indices)
    probe_images, probe_labels = materialize(train_dataset, probe_indices)
    probe_index_to_position = {
        index: position for position, index in enumerate(probe_indices)
    }
    if args.utility_samples:
        replacement_indices = {
            pair.replacement_test_index for pair in pairs
        }
        utility_indices = [
            int(index)
            for index in test_permutation
            if int(index) not in replacement_indices
        ][: args.utility_samples]
        utility_images, utility_labels = materialize(test_dataset, utility_indices)
    else:
        utility_images = utility_labels = None

    protocol = {
        "config": {
            **vars(args),
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir),
            "device": str(device),
        },
        "base_train_indices": base_indices,
        "probe_train_indices": probe_indices,
        "pairs": [asdict(pair) for pair in pairs],
        "matched_randomness": (
            "For each target/repetition, H1/H0 and before/after models use the "
            "same initialization and DataLoader generator seed. H1/H0 have "
            "identical sizes and record positions; after models omit the same probe."
        ),
    }
    atomic_json_dump(protocol, args.output_dir / "protocol.json")

    observations: list[dict] = []
    total = len(pairs) * args.num_seeds
    completed = 0
    for pair in pairs:
        for repetition in range(args.num_seeds):
            seed = args.training_seed + repetition
            completed += 1
            print(
                f"[{completed}/{total}] pair={pair.pair_id} "
                f"label={pair.target_label} seed={seed}",
                flush=True,
            )
            observation = run_pair_seed(
                pair,
                repetition,
                seed,
                base_images,
                base_labels,
                probe_images,
                probe_labels,
                probe_index_to_position,
                train_dataset,
                test_dataset,
                args,
                device,
                utility_images,
                utility_labels,
            )
            observations.append(observation)
            atomic_json_dump(
                observations, args.output_dir / "observations.partial.json"
            )

    summary = summarize(observations, args)
    atomic_json_dump(observations, args.output_dir / "observations.json")
    atomic_json_dump(summary, args.output_dir / "summary.json")
    write_csv(observations, args.output_dir / "observations.csv")
    make_figure(observations, summary, args.output_dir / "evidence.png")
    partial_path = args.output_dir / "observations.partial.json"
    if partial_path.exists():
        partial_path.unlink()

    gamma = summary["gamma_log_odds"]
    aucs = summary["group_held_out_auroc"]
    print("\n=== Re-equilibration evidence summary ===")
    print(
        "Paired log-odds Gamma: "
        f"{gamma['pair_averaged_mean']:.6f} "
        f"(95% pair-bootstrap CI "
        f"[{gamma['pair_bootstrap_95_ci'][0]:.6f}, "
        f"{gamma['pair_bootstrap_95_ci'][1]:.6f}])"
    )
    print(f"Paired sign-flip p-value: {gamma['paired_sign_flip_p_value']:.6g}")
    print(f"Initial-posterior AUROC: {format_auc(aucs['initial_posterior'])}")
    print(f"Deletion-response AUROC: {format_auc(aucs['deletion_response'])}")
    print(
        "Baseline-conditioned response AUROC: "
        f"{format_auc(aucs['baseline_conditioned_response'])}"
    )
    print(f"Initial + response AUROC: {format_auc(aucs['initial_plus_response'])}")
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
