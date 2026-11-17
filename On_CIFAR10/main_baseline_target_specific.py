"""CIFAR-10 target-specific ADP diagnostic (exact retraining, k=3).

This is an oracle/diagnostic ablation rather than the main ADP experiment. It
compares ADP-Random, ADP-Global-Borda, and ADP-Target-Specific on the same 100
targets. No planner consumes target-service outputs or membership labels. The
target-specific condition is diagnostic because it uses each evaluation target
and class label to plan a separate, precommitted deletion trajectory.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

if __package__:
    from .ADP_active_lira.ADP_active_lira import select_conditional_probe_sequences
    from .baseline.baseline_prob import ActiveADP, SequentialADP, train_model_with_loader
    from .baseline.model import CNN
else:
    from ADP_active_lira.ADP_active_lira import select_conditional_probe_sequences
    from baseline.baseline_prob import ActiveADP, SequentialADP, train_model_with_loader
    from baseline.model import CNN


# Fixed diagnostic configuration.
channel = 3
num_classes = 10
target_training_epochs = 50
target_training_lr = 1e-3
batch_size = 256
num_workers = 4

num_unlearning_operations = 3
raw_owned_pool_size = 30
planning_candidate_pool_size = 10
num_evaluation_members = 50
num_evaluation_nonmembers = 50
num_evaluation_targets = num_evaluation_members + num_evaluation_nonmembers

num_global_design_targets = 10
design_shadow_size = 1000
num_design_shadow_pairs = 2
selection_epochs = 5
selection_lr = 1e-3
selection_ridge = 1e-4
attack_epochs = 30
attack_lr = 1e-3
attack_hidden_size = 128

split_seed = 0
prefilter_seed = 101
random_schedule_seed = 102
global_planning_seed = 200
target_specific_planning_seed = 300
attack_seed = 1000
exact_retraining_seed = 3000
exact_retraining_cache_version = 2
bootstrap_repetitions = 2000
bootstrap_confidence = 0.95
bootstrap_seed = 4000

gpu = 0
device = torch.device(
    f"cuda:{gpu}" if torch.cuda.is_available() and gpu != -1 else "cpu"
)
script_dir = Path(__file__).resolve().parent
results_dir = script_dir / "results"
data_root = os.environ.get("CIFAR10_DATA_ROOT", "/home/wwq/Data/data/cifar")
target_model_path = script_dir / "target_model.pth"
train_indices_path = script_dir / "original_indices"
test_indices_path = script_dir / "original_indices_test"
global_plan_cache_path = script_dir / (
    "target_specific_diagnostic_cifar10_global_borda_raw30_prefilter10_k3.json"
)
target_specific_plan_cache_path = script_dir / (
    "target_specific_diagnostic_cifar10_per_target_100_targets_"
    "raw30_prefilter10_k3.json"
)
result_json_path = (
    results_dir / "Target_Specific_Diagnostic_CIFAR10_CNN_k3_100_targets.json"
)
result_text_path = (
    results_dir / "Target_Specific_Diagnostic_CIFAR10_CNN_k3_100_targets.txt"
)
roc_path = (
    results_dir / "ROC_Target_Specific_Diagnostic_CIFAR10_CNN_k3_100_targets.png"
)


def build_model():
    return CNN(channel, num_classes)


def set_experiment_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, indices=None, shuffle=False):
    selected = dataset if indices is None else Subset(dataset, indices)
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def materialize(loader):
    images, labels = [], []
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels)
    return torch.cat(images), torch.cat(labels)


def load_or_create_permutation(path, length, seed):
    if path.exists():
        permutation = torch.load(path, map_location="cpu")
        if len(permutation) != length:
            raise ValueError(f"Cached permutation {path} has the wrong length.")
        return permutation
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(length, generator=generator)
    torch.save(permutation, path)
    return permutation


def label_balanced_prefilter(labels, output_size, seed):
    """Cheap target-independent 30-to-10 filter using owned-record labels."""
    if output_size > len(labels):
        raise ValueError("The prefilter output cannot exceed the raw pool size.")
    traversal = np.random.default_rng(seed).permutation(len(labels)).tolist()
    selected, seen_classes = [], set()
    for position in traversal:
        label = int(labels[position])
        if label not in seen_classes:
            selected.append(int(position))
            seen_classes.add(label)
            if len(selected) == output_size:
                return selected
    selected_set = set(selected)
    for position in traversal:
        position = int(position)
        if position not in selected_set:
            selected.append(position)
            selected_set.add(position)
            if len(selected) == output_size:
                break
    return selected


def read_matching_cache(path, metadata):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as cache_file:
            cache = json.load(cache_file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Ignoring unreadable planning cache {path}: {error}")
        return None
    if cache.get("metadata") != metadata:
        print(f"Ignoring planning cache with mismatched metadata: {path}")
        return None
    return cache


def write_json_atomic(path, payload):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
    os.replace(temporary_path, path)


def borda_aggregate(schedules, candidate_count, schedule_length):
    scores = np.zeros(candidate_count, dtype=np.float64)
    for schedule in schedules:
        for rank, position in enumerate(schedule[:schedule_length]):
            scores[int(position)] += schedule_length - rank
    selected = sorted(
        range(candidate_count), key=lambda position: (-scores[position], position)
    )[:schedule_length]
    return selected, scores


def schedule_positions_to_training_indices(schedule, candidate_train_indices):
    return [int(candidate_train_indices[int(position)]) for position in schedule]


def target_release_checkpoint_path(deleted_prefix):
    prefix = tuple(int(index) for index in deleted_prefix)
    round_index = len(prefix)
    round_seed = exact_retraining_seed + round_index
    prefix_tag = "_".join(str(index) for index in prefix)
    return script_dir / (
        "target_model_cifar10_cnn_exact_paired_"
        f"v{exact_retraining_cache_version}_round_{round_index:02d}_"
        f"seed_{round_seed}_delete_{prefix_tag}.pth"
    )


def load_or_train_target_release(train_dataset, deleted_prefix):
    """Load one exact-retraining release without retaining it globally."""
    prefix = tuple(int(index) for index in deleted_prefix)
    if not prefix:
        raise ValueError("The initial release is loaded separately.")
    checkpoint_path = target_release_checkpoint_path(prefix)
    if checkpoint_path.exists():
        model = build_model().to(device)
        print(f"Loading target release after prefix {list(prefix)}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model

    deleted = set(prefix)
    retained_indices = [
        index for index in range(len(train_dataset)) if index not in deleted
    ]
    retained_loader = make_loader(train_dataset, retained_indices, shuffle=False)
    round_seed = exact_retraining_seed + len(prefix)
    set_experiment_seed(round_seed)
    model = build_model().to(device)
    print(
        f"Retraining target release after prefix {list(prefix)} for "
        f"{target_training_epochs} epochs (seed={round_seed})",
        flush=True,
    )
    train_model_with_loader(
        model,
        retained_loader,
        training_epochs=target_training_epochs,
        lr=target_training_lr,
        device=device,
    )
    torch.save(model.state_dict(), checkpoint_path)
    return model


def load_target_trajectory(train_dataset, initial_model, schedule_train_indices):
    trajectory, prefix = [initial_model], []
    for training_index in schedule_train_indices:
        prefix.append(int(training_index))
        trajectory.append(load_or_train_target_release(train_dataset, prefix))
    return trajectory


def release_trajectory(trajectory):
    # trajectory[0] is the shared initial target model.
    for model in trajectory[1:]:
        model.to("cpu")
    del trajectory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_unique_nonempty_prefixes(schedules):
    prefixes = set()
    for schedule in schedules:
        schedule = tuple(int(value) for value in schedule)
        for prefix_length in range(1, len(schedule) + 1):
            prefixes.add(schedule[:prefix_length])
    return prefixes


def paired_stratified_bootstrap(labels, scores_by_method, repetitions, confidence, seed):
    """Paired AUC bootstrap with identical stratified resamples per method."""
    labels = np.asarray(labels, dtype=np.int64)
    scores_by_method = {
        name: np.asarray(scores, dtype=np.float64)
        for name, scores in scores_by_method.items()
    }
    members = np.flatnonzero(labels == 1)
    nonmembers = np.flatnonzero(labels == 0)
    if not len(members) or not len(nonmembers):
        raise ValueError("Paired bootstrap requires both membership classes.")
    rng = np.random.default_rng(seed)
    bootstrap_aucs = {
        name: np.empty(repetitions, dtype=np.float64) for name in scores_by_method
    }
    for repetition in range(repetitions):
        sampled = np.concatenate([
            rng.choice(members, len(members), replace=True),
            rng.choice(nonmembers, len(nonmembers), replace=True),
        ])
        sampled_labels = labels[sampled]
        for name, scores in scores_by_method.items():
            bootstrap_aucs[name][repetition] = roc_auc_score(
                sampled_labels, scores[sampled]
            )

    alpha = (1.0 - confidence) / 2.0

    def summarize(values, point):
        lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
        return {
            "point": float(point),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
        }

    point_aucs = {
        name: float(roc_auc_score(labels, scores))
        for name, scores in scores_by_method.items()
    }
    method_intervals = {
        name: summarize(bootstrap_aucs[name], point_aucs[name])
        for name in scores_by_method
    }
    comparisons = {}
    for left, right in [
        ("ADP-Global-Borda", "ADP-Random"),
        ("ADP-Target-Specific", "ADP-Random"),
        ("ADP-Target-Specific", "ADP-Global-Borda"),
    ]:
        differences = bootstrap_aucs[left] - bootstrap_aucs[right]
        comparisons[f"{left}_minus_{right}"] = summarize(
            differences, point_aucs[left] - point_aucs[right]
        )
    return {
        "repetitions": int(repetitions),
        "confidence": float(confidence),
        "seed": int(seed),
        "method_auc": method_intervals,
        "paired_auc_differences": comparisons,
    }


def plan_global_borda(
    design_target_images,
    design_target_labels,
    design_shadow_images,
    design_shadow_labels,
    candidate_data,
    candidate_labels,
    metadata,
):
    cached = read_matching_cache(global_plan_cache_path, metadata)
    if cached is None:
        set_experiment_seed(global_planning_seed)
        schedules, diagnostics = select_conditional_probe_sequences(
            design_target_images,
            design_target_labels,
            design_shadow_images,
            design_shadow_labels,
            candidate_data,
            candidate_labels,
            num_unlearning_operations=num_unlearning_operations,
            num_design_shadow_models=num_design_shadow_pairs,
            epochs=selection_epochs,
            lr=selection_lr,
            channel=channel,
            num_classes=num_classes,
            selection_ridge=selection_ridge,
            device=device,
            unlearning_method="retrain",
        )
        cached = {
            "metadata": metadata,
            "design_schedules": schedules,
            "diagnostics": diagnostics,
        }
        write_json_atomic(global_plan_cache_path, cached)
    schedules = cached["design_schedules"]
    selected, scores = borda_aggregate(
        schedules, planning_candidate_pool_size, num_unlearning_operations
    )
    return selected, scores, schedules, cached["diagnostics"]


def plan_target_specific_incrementally(
    target_images,
    target_class_labels,
    design_shadow_images,
    design_shadow_labels,
    candidate_data,
    candidate_labels,
    metadata,
):
    """Plan and persist one evaluation target at a time."""
    cached = read_matching_cache(target_specific_plan_cache_path, metadata)
    if cached is None:
        cached = {"metadata": metadata, "schedules": [], "diagnostics": []}
    schedules = cached.get("schedules", [])
    diagnostics = cached.get("diagnostics", [])
    if len(schedules) != len(diagnostics) or len(schedules) > len(target_images):
        raise ValueError("Target-specific planning cache is inconsistent.")

    for target_position in range(len(schedules), len(target_images)):
        print(
            f"\n===== Target-specific planning {target_position + 1}/"
            f"{len(target_images)} =====",
            flush=True,
        )
        set_experiment_seed(target_specific_planning_seed + target_position)
        one_schedule, one_diagnostics = select_conditional_probe_sequences(
            target_images[target_position:target_position + 1],
            target_class_labels[target_position:target_position + 1],
            design_shadow_images,
            design_shadow_labels,
            candidate_data,
            candidate_labels,
            num_unlearning_operations=num_unlearning_operations,
            num_design_shadow_models=num_design_shadow_pairs,
            epochs=selection_epochs,
            lr=selection_lr,
            channel=channel,
            num_classes=num_classes,
            selection_ridge=selection_ridge,
            device=device,
            unlearning_method="retrain",
        )
        schedules.append(one_schedule[0])
        diagnostics.append(one_diagnostics[0])
        cached["schedules"] = schedules
        cached["diagnostics"] = diagnostics
        write_json_atomic(target_specific_plan_cache_path, cached)
    return schedules, diagnostics


def score_common_schedule(
    schedule_positions,
    decoder,
    decoder_seed,
    candidate_train_indices,
    train_dataset,
    initial_model,
    attack_member_loader,
    attack_nonmember_loader,
    target_images,
    target_labels,
):
    schedule_train_indices = schedule_positions_to_training_indices(
        schedule_positions, candidate_train_indices
    )
    trajectory = load_target_trajectory(
        train_dataset, initial_model, schedule_train_indices
    )
    set_experiment_seed(decoder_seed)
    scores = decoder(
        trajectory,
        attack_member_loader,
        attack_nonmember_loader,
        target_images,
        target_labels,
        attack_hidden_size,
        attack_epochs,
        attack_lr,
        device,
    )
    release_trajectory(trajectory)
    return np.asarray(scores, dtype=np.float64), schedule_train_indices


def score_target_specific_schedules(
    schedules,
    candidate_train_indices,
    train_dataset,
    initial_model,
    attack_member_loader,
    attack_nonmember_loader,
    target_images,
    target_labels,
):
    """Fit one decoder per unique schedule and score its assigned targets."""
    schedule_groups = {}
    for target_position, schedule in enumerate(schedules):
        key = tuple(int(position) for position in schedule)
        schedule_groups.setdefault(key, []).append(target_position)

    all_scores = np.empty(len(schedules), dtype=np.float64)
    ordered_groups = sorted(schedule_groups.items())
    for group_number, (schedule, target_positions) in enumerate(
        ordered_groups, start=1
    ):
        print(
            f"\n===== Target-specific execution group {group_number}/"
            f"{len(ordered_groups)}: {len(target_positions)} target(s), "
            f"schedule={list(schedule)} =====",
            flush=True,
        )
        group_index = torch.as_tensor(target_positions, dtype=torch.long)
        group_scores, _ = score_common_schedule(
            schedule,
            ActiveADP,
            # Reuse the same decoder seed for every schedule so differences are
            # not driven by schedule-dependent MLP initialization or shuffling.
            attack_seed,
            candidate_train_indices,
            train_dataset,
            initial_model,
            attack_member_loader,
            attack_nonmember_loader,
            target_images[group_index],
            target_labels[group_index],
        )
        all_scores[group_index.numpy()] = group_scores.reshape(-1)
    return all_scores, schedule_groups


def main():
    if num_unlearning_operations != 3 or num_evaluation_targets != 100:
        raise ValueError("This diagnostic is fixed to k=3 and 100 targets.")
    if planning_candidate_pool_size < num_unlearning_operations:
        raise ValueError("The prefiltered pool must cover the deletion budget.")
    results_dir.mkdir(parents=True, exist_ok=True)
    set_experiment_seed(split_seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda image: image.view(3, 32, 32)),
    ])
    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    train_loader = make_loader(train_dataset)
    test_loader = make_loader(test_dataset)
    train_images, train_labels = materialize(train_loader)
    test_images, test_labels = materialize(test_loader)
    original_train_indices = load_or_create_permutation(
        train_indices_path, len(train_dataset), split_seed
    )
    original_test_indices = load_or_create_permutation(
        test_indices_path, len(test_dataset), split_seed
    )

    raw_probe_train_indices = [
        int(index) for index in original_train_indices[:raw_owned_pool_size]
    ]
    raw_probe_labels = train_labels[raw_probe_train_indices]
    prefilter_positions = label_balanced_prefilter(
        raw_probe_labels, planning_candidate_pool_size, prefilter_seed
    )
    candidate_train_indices = [
        raw_probe_train_indices[position] for position in prefilter_positions
    ]
    candidate_data = train_images[candidate_train_indices]
    candidate_labels = train_labels[candidate_train_indices]
    random_schedule_positions = [
        int(value)
        for value in np.random.default_rng(random_schedule_seed).permutation(
            planning_candidate_pool_size
        )[:num_unlearning_operations]
    ]

    evaluation_member_indices = original_train_indices[
        raw_owned_pool_size:raw_owned_pool_size + num_evaluation_members
    ]
    attack_member_indices = original_train_indices[
        raw_owned_pool_size + num_evaluation_members:
    ]
    evaluation_nonmember_indices = original_test_indices[:num_evaluation_nonmembers]
    design_target_start = num_evaluation_nonmembers
    design_target_end = design_target_start + num_global_design_targets
    design_shadow_end = design_target_end + design_shadow_size
    if design_shadow_end >= len(test_dataset):
        raise ValueError("The requested test split leaves no attack references.")
    design_target_indices = original_test_indices[design_target_start:design_target_end]
    design_shadow_indices = original_test_indices[design_target_end:design_shadow_end]
    attack_nonmember_indices = original_test_indices[design_shadow_end:]

    attack_reference_count = min(len(attack_member_indices), len(attack_nonmember_indices))
    attack_member_indices = attack_member_indices[:attack_reference_count]
    attack_nonmember_indices = attack_nonmember_indices[:attack_reference_count]
    attack_member_loader = make_loader(
        train_dataset, attack_member_indices.tolist(), shuffle=False
    )
    attack_nonmember_loader = make_loader(
        test_dataset, attack_nonmember_indices.tolist(), shuffle=False
    )

    target_images = torch.cat([
        train_images[evaluation_member_indices],
        test_images[evaluation_nonmember_indices],
    ])
    target_class_labels = torch.cat([
        train_labels[evaluation_member_indices],
        test_labels[evaluation_nonmember_indices],
    ])
    target_membership_labels = np.asarray(
        [1] * num_evaluation_members + [0] * num_evaluation_nonmembers,
        dtype=np.int64,
    )
    design_target_images = test_images[design_target_indices]
    design_target_labels = test_labels[design_target_indices]
    design_shadow_images = test_images[design_shadow_indices]
    design_shadow_labels = test_labels[design_shadow_indices]
    planned_target_classes = (
        set(target_class_labels.tolist()) | set(design_target_labels.tolist())
    )
    missing_classes = sorted(
        planned_target_classes - set(design_shadow_labels.tolist())
    )
    if missing_classes:
        raise ValueError(f"Design shadows lack controls for labels {missing_classes}.")

    common_metadata = {
        "experiment_scope": "target_specific_oracle_diagnostic",
        "dataset": "CIFAR10",
        "unlearning_method": "exact_retraining",
        "k": num_unlearning_operations,
        "raw_owned_pool_size": raw_owned_pool_size,
        "planning_candidate_pool_size": planning_candidate_pool_size,
        "raw_probe_train_indices": raw_probe_train_indices,
        "prefilter_positions_in_raw_pool": prefilter_positions,
        "candidate_train_indices": candidate_train_indices,
        "candidate_class_labels": candidate_labels.tolist(),
        "design_shadow_indices": design_shadow_indices.tolist(),
        "num_design_shadow_pairs": num_design_shadow_pairs,
        "selection_epochs": selection_epochs,
        "selection_lr": selection_lr,
        "selection_ridge": selection_ridge,
        "model": "CIFAR10_ShadowModel",
    }
    global_metadata = {
        **common_metadata,
        "planner": "global_truncated_borda",
        "planning_seed": global_planning_seed,
        "design_target_indices": design_target_indices.tolist(),
    }
    (
        global_schedule_positions,
        global_borda_scores,
        global_design_schedules,
        global_planning_diagnostics,
    ) = plan_global_borda(
        design_target_images,
        design_target_labels,
        design_shadow_images,
        design_shadow_labels,
        candidate_data,
        candidate_labels,
        global_metadata,
    )

    target_specific_metadata = {
        **common_metadata,
        "planner": "per_evaluation_target_oracle_diagnostic",
        "planning_seed": target_specific_planning_seed,
        "evaluation_member_indices": evaluation_member_indices.tolist(),
        "evaluation_nonmember_indices": evaluation_nonmember_indices.tolist(),
        "number_of_targets": num_evaluation_targets,
        "planner_inputs": (
            "target record and class label only; no target membership label "
            "and no target-service response"
        ),
    }
    target_specific_schedules, target_specific_planning_diagnostics = (
        plan_target_specific_incrementally(
            target_images,
            target_class_labels,
            design_shadow_images,
            design_shadow_labels,
            candidate_data,
            candidate_labels,
            target_specific_metadata,
        )
    )

    per_target_shadow_trainings = 2 * num_design_shadow_pairs * (
        1 + sum(
            planning_candidate_pool_size - round_index
            for round_index in range(num_unlearning_operations)
        )
    )
    print("\n===== Diagnostic workload =====")
    print(f"Evaluation targets: {num_evaluation_targets}")
    print(
        "Estimated target-specific planning shadow trainings: "
        f"{num_evaluation_targets * per_target_shadow_trainings} "
        f"({per_target_shadow_trainings} per target)"
    )
    print(
        "Estimated auxiliary Global-Borda planning shadow trainings: "
        f"{num_global_design_targets * per_target_shadow_trainings}"
    )
    print(
        "Target-specific exact-retraining upper bound before prefix reuse: "
        f"{num_evaluation_targets * num_unlearning_operations}"
    )

    # Planning above is entirely offline. Reset before model construction so
    # an absent initial checkpoint is reproducible despite planning randomness.
    set_experiment_seed(split_seed)
    initial_model = build_model().to(device)
    if target_model_path.exists():
        print("Loading trained initial target model")
        initial_model.load_state_dict(torch.load(target_model_path, map_location=device))
    else:
        print(f"Training initial target for {target_training_epochs} epochs")
        train_model_with_loader(
            initial_model,
            train_loader,
            training_epochs=target_training_epochs,
            lr=target_training_lr,
            device=device,
        )
        torch.save(initial_model.state_dict(), target_model_path)

    print("\n===== ADP-Random execution =====")
    random_scores, random_schedule_train_indices = score_common_schedule(
        random_schedule_positions,
        SequentialADP,
        attack_seed,
        candidate_train_indices,
        train_dataset,
        initial_model,
        attack_member_loader,
        attack_nonmember_loader,
        target_images,
        target_class_labels,
    )
    print("\n===== ADP-Global-Borda execution =====")
    global_scores, global_schedule_train_indices = score_common_schedule(
        global_schedule_positions,
        ActiveADP,
        attack_seed,
        candidate_train_indices,
        train_dataset,
        initial_model,
        attack_member_loader,
        attack_nonmember_loader,
        target_images,
        target_class_labels,
    )
    target_specific_scores, schedule_groups = score_target_specific_schedules(
        target_specific_schedules,
        candidate_train_indices,
        train_dataset,
        initial_model,
        attack_member_loader,
        attack_nonmember_loader,
        target_images,
        target_class_labels,
    )

    scores_by_method = {
        "ADP-Random": random_scores,
        "ADP-Global-Borda": global_scores,
        "ADP-Target-Specific": target_specific_scores,
    }
    bootstrap = paired_stratified_bootstrap(
        target_membership_labels,
        scores_by_method,
        bootstrap_repetitions,
        bootstrap_confidence,
        bootstrap_seed,
    )

    roc_coordinates = {}
    colors = {
        "ADP-Random": "forestgreen",
        "ADP-Global-Borda": "crimson",
        "ADP-Target-Specific": "darkviolet",
    }
    for method, scores in scores_by_method.items():
        fpr, tpr, _ = roc_curve(target_membership_labels, scores)
        method_auc = bootstrap["method_auc"][method]["point"]
        roc_coordinates[method] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        plt.plot(
            fpr,
            tpr,
            color=colors[method],
            linewidth=2,
            label=f"{method} (area = {method_auc:.2f})",
        )
    plt.plot([0, 1], [0, 1], color="navy", linewidth=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CIFAR-10 target-specific diagnostic (Retrain, k=3)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    target_specific_train_schedules = [
        schedule_positions_to_training_indices(schedule, candidate_train_indices)
        for schedule in target_specific_schedules
    ]
    target_specific_prefixes = collect_unique_nonempty_prefixes(
        target_specific_train_schedules
    )
    all_execution_prefixes = collect_unique_nonempty_prefixes([
        random_schedule_train_indices,
        global_schedule_train_indices,
        *target_specific_train_schedules,
    ])
    result = {
        "experiment_scope": "target_specific_oracle_diagnostic_ablation",
        "main_threat_model_result": False,
        "interpretation": (
            "Target-specific planning uses each evaluated target and class label, "
            "but never its membership label or target-service response. Each "
            "schedule is fixed before its corresponding trajectory is queried."
        ),
        "dataset": "CIFAR10",
        "unlearning_method": "exact_retraining",
        "k": num_unlearning_operations,
        "number_of_evaluation_targets": num_evaluation_targets,
        "number_of_evaluation_members": num_evaluation_members,
        "number_of_evaluation_nonmembers": num_evaluation_nonmembers,
        "raw_owned_pool_size": raw_owned_pool_size,
        "planning_candidate_pool_size": planning_candidate_pool_size,
        "prefilter_policy": "seeded_label_balanced_owned-record-only",
        "raw_probe_train_indices": raw_probe_train_indices,
        "prefilter_positions_in_raw_pool": prefilter_positions,
        "candidate_train_indices": candidate_train_indices,
        "candidate_class_labels": candidate_labels.tolist(),
        "evaluation_member_indices": evaluation_member_indices.tolist(),
        "evaluation_nonmember_indices": evaluation_nonmember_indices.tolist(),
        "attack_reference_records_per_membership_class": int(attack_reference_count),
        "random_schedule_positions": random_schedule_positions,
        "random_schedule_train_indices": random_schedule_train_indices,
        "global_design_target_indices": design_target_indices.tolist(),
        "global_design_schedules": global_design_schedules,
        "global_borda_scores": global_borda_scores.tolist(),
        "global_schedule_positions": global_schedule_positions,
        "global_schedule_train_indices": global_schedule_train_indices,
        "target_specific_schedules_positions": target_specific_schedules,
        "target_specific_schedules_train_indices": target_specific_train_schedules,
        "number_of_unique_target_specific_schedules": len(schedule_groups),
        "number_of_unique_target_specific_release_prefixes": len(target_specific_prefixes),
        "number_of_unique_release_prefixes_across_all_methods": len(all_execution_prefixes),
        "estimated_shadow_trainings_per_planned_target": per_target_shadow_trainings,
        "estimated_target_specific_planning_shadow_trainings": (
            num_evaluation_targets * per_target_shadow_trainings
        ),
        "estimated_global_planning_shadow_trainings": (
            num_global_design_targets * per_target_shadow_trainings
        ),
        "target_specific_retraining_upper_bound_without_prefix_reuse": (
            num_evaluation_targets * num_unlearning_operations
        ),
        "selection_configuration": {
            "num_design_shadow_pairs": num_design_shadow_pairs,
            "design_shadow_size": design_shadow_size,
            "epochs": selection_epochs,
            "learning_rate": selection_lr,
            "ridge": selection_ridge,
        },
        "decoder_configuration": {
            "hidden_units": attack_hidden_size,
            "epochs": attack_epochs,
            "learning_rate": attack_lr,
            "feature": "ordered posterior levels and consecutive differences",
            "calibration": "disjoint known target-model memberships",
        },
        "bootstrap": bootstrap,
        "membership_ground_truth": target_membership_labels.tolist(),
        "scores": {method: scores.tolist() for method, scores in scores_by_method.items()},
        "roc_coordinates": roc_coordinates,
        "planning_cache_files": {
            "global_borda": str(global_plan_cache_path),
            "target_specific": str(target_specific_plan_cache_path),
        },
        "global_planning_diagnostics": global_planning_diagnostics,
        "target_specific_planning_diagnostics": target_specific_planning_diagnostics,
        "seeds": {
            "split": split_seed,
            "prefilter": prefilter_seed,
            "random_schedule": random_schedule_seed,
            "global_planning": global_planning_seed,
            "target_specific_planning": target_specific_planning_seed,
            "attack": attack_seed,
            "exact_retraining": exact_retraining_seed,
            "bootstrap": bootstrap_seed,
        },
        "device": str(device),
        "roc_figure": str(roc_path),
    }
    write_json_atomic(result_json_path, result)

    print("\n===== Paired-bootstrap results =====")
    lines = [
        "CIFAR-10 target-specific oracle/diagnostic ablation",
        "Unlearning: exact retraining; k=3; targets=50 members + 50 non-members",
        (
            "Planning candidate pool: 30 owned records -> 10 records via "
            "target-independent label-balanced prefilter"
        ),
    ]
    for method, interval in bootstrap["method_auc"].items():
        line = (
            f"{method} AUROC: {interval['point']:.6f} "
            f"({bootstrap_confidence:.0%} paired-bootstrap CI "
            f"[{interval['ci_lower']:.6f}, {interval['ci_upper']:.6f}])"
        )
        lines.append(line)
        print(line)
    for comparison, interval in bootstrap["paired_auc_differences"].items():
        line = (
            f"{comparison}: {interval['point']:+.6f} "
            f"({bootstrap_confidence:.0%} paired-bootstrap CI "
            f"[{interval['ci_lower']:+.6f}, {interval['ci_upper']:+.6f}])"
        )
        lines.append(line)
        print(line)
    lines.extend([
        f"Unique target-specific schedules: {len(schedule_groups)}",
        (
            "Unique target-specific exact-retraining prefixes after reuse: "
            f"{len(target_specific_prefixes)}"
        ),
        "Interpret as a target-specific oracle/diagnostic ablation, not the main ADP result.",
        f"Full machine-readable results: {result_json_path}",
    ])
    with result_text_path.open("w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")
    print(f"\nSaved ROC figure to: {roc_path}")
    print(f"Saved summary to: {result_text_path}")
    print(f"Saved full results to: {result_json_path}")


if __name__ == "__main__":
    main()
