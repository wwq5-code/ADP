import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, auc, roc_curve
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision import transforms

if __package__:
    from .baseline.active_selection import select_global_probe_sequence
    from .baseline.baseline_prob import (
        calculate_accuracy,
        train_model_with_loader,
    )
    from .baseline.dataset import TinyImageNetDataset
    from .baseline.model import ResNet18
else:
    from baseline.active_selection import select_global_probe_sequence
    from baseline.baseline_prob import (
        calculate_accuracy,
        train_model_with_loader,
    )
    from baseline.dataset import TinyImageNetDataset
    from baseline.model import ResNet18


# -------------------------------------------------------------------------
dataset_name = 'TinyImageNet'
input_shape = (3, 128, 128)
image_size = 128
num_classes = 200
model_name = 'resnet18'
use_pretrained = os.environ.get(
    'TINYIMAGENET_PRETRAINED', '1'
).strip().lower() not in {'0', 'false', 'no'}
initialization_name = (
    'imagenet1k_pretrained' if use_pretrained else 'scratch'
)
epochs = 40
lr = 1e-4
batch_size = 128
measurement_number = 1000  # members and non-members, respectively
unlearning_sequence_sizes = (3, 5, 7, 9, 11, 13, 15)
max_num_unlearning_operations = max(unlearning_sequence_sizes)

active_probe_pool_size = 30
num_active_design_targets = 10
active_design_shadow_size = 1000
num_design_shadow_models = 2
active_selection_epochs = 5
active_selection_lr = 1e-4
active_selection_ridge = 1e-4
active_selection_batch_size = 64
active_selection_seed = 0

attack_epochs = 30
attack_lr = 1e-3
attack_hidden_size = 128
attack_random_seed = 1000
trajectory_top_nonmember_classes = 10
trajectory_log_odds_epsilon = 1e-6
dvia_ucd_epsilon = 1e-6
dvia_gaussian_variance_floor = 1e-8
dvia_likelihood_ratio_threshold = 1.0
# TinyImageNet validation contains 50 records per class. These values leave
# room for balanced measurement/design splits while providing 2,000 reference
# candidates and 5,000 base records for attacker-constructed reference models.
decoder_reference_records_per_class = 10
decoder_shadow_base_records_per_class = 25
num_decoder_shadow_models = 4
decoder_shadow_epochs = 10
decoder_shadow_lr = 1e-4
decoder_shadow_seed = 2000
decoder_cache_version = 1
gpu = 1
device = torch.device(
    f'cuda:{gpu}' if torch.cuda.is_available() and gpu != -1 else 'cpu'
)
num_workers = int(os.environ.get('TINYIMAGENET_NUM_WORKERS', '8'))
pin_memory = device.type == 'cuda'

script_dir = Path(__file__).resolve().parent
results_dir = script_dir / 'results'
data_root = os.environ.get(
    'TINYIMAGENET_DATA_ROOT',
    '/home/wwq/Data/data/tiny-imagenet-200/tiny-imagenet-200',
)
target_model_path = script_dir / (
    f'target_model_tinyimagenet_resnet18_{initialization_name}.pth'
)
train_indices_path = script_dir / 'original_indices_tinyimagenet.pt'
test_indices_path = script_dir / 'original_indices_test_tinyimagenet.pt'
sequence_size_env = 'TINYIMAGENET_UNLEARNING_SEQUENCE_SIZE'
# -------------------------------------------------------------------------


def set_experiment_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model():
    return ResNet18(
        num_classes=num_classes, pretrained=use_pretrained
    )


def get_tensor_probability_vectors(model, data, inference_batch_size=64):
    """Extract target-model posteriors without membership labels."""
    loader = DataLoader(
        TensorDataset(data),
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    probabilities = []
    model.eval()
    with torch.no_grad():
        for (images,) in loader:
            logits = model(images.to(device))
            probabilities.append(F.softmax(logits, dim=1).cpu())
    if not probabilities:
        raise ValueError('Probability query data must not be empty.')
    return torch.cat(probabilities, dim=0)


def get_dataset_probability_vectors(
    model,
    dataset,
    inference_batch_size=64,
):
    """Query a disk-backed auxiliary dataset in deterministic order."""
    loader = DataLoader(
        dataset,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    probabilities = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device))
            probabilities.append(F.softmax(logits, dim=1).cpu())
    if not probabilities:
        raise ValueError('Reference probability query must not be empty.')
    return torch.cat(probabilities, dim=0)


def fixed_top_class_log_odds_features(
    probability_trajectory,
    class_labels,
    feature_kind,
):
    """Encode a trajectory on classes fixed by each record's M_0 output.

    The selected coordinates are the correct class followed by the ten
    highest-probability non-correct classes under M_0.  Their class identities
    and M_0 ranking are reused at every later release, so consecutive
    differences always compare the same classes.
    """
    if not probability_trajectory:
        raise ValueError('A probability trajectory must not be empty.')
    if feature_kind == 'passive' and len(probability_trajectory) < 2:
        raise ValueError('Passive deletion needs M_0 and M_1.')
    if feature_kind == 'sequential' and len(probability_trajectory) < 3:
        raise ValueError(
            'Sequential ADP needs M_0 and at least two deletion releases.'
        )
    if feature_kind not in {'static', 'passive', 'sequential'}:
        raise ValueError(f'Unknown trajectory feature kind: {feature_kind}')

    initial_probabilities = probability_trajectory[0]
    labels = torch.as_tensor(class_labels, dtype=torch.long).cpu()
    if initial_probabilities.ndim != 2:
        raise ValueError('Probability releases must be rank-two tensors.')
    if len(labels) != len(initial_probabilities):
        raise ValueError('Each probability row needs one class label.')
    if initial_probabilities.shape[1] <= trajectory_top_nonmember_classes:
        raise ValueError(
            'The class count must exceed the requested non-correct top-k.'
        )
    if torch.any(labels < 0) or torch.any(
        labels >= initial_probabilities.shape[1]
    ):
        raise ValueError('A class label is outside the posterior columns.')
    expected_shape = tuple(initial_probabilities.shape)
    if any(tuple(release.shape) != expected_shape for release in probability_trajectory):
        raise ValueError('All probability releases must have the same shape.')

    noncorrect_initial = initial_probabilities.clone()
    noncorrect_initial.scatter_(1, labels[:, None], float('-inf'))
    top_noncorrect_classes = torch.topk(
        noncorrect_initial,
        k=trajectory_top_nonmember_classes,
        dim=1,
        largest=True,
        sorted=True,
    ).indices
    fixed_class_indices = torch.cat(
        [labels[:, None], top_noncorrect_classes], dim=1
    )

    log_odds_trajectory = []
    for release in probability_trajectory:
        selected_probabilities = release.gather(1, fixed_class_indices).clamp(
            min=trajectory_log_odds_epsilon,
            max=1.0 - trajectory_log_odds_epsilon,
        )
        log_odds_trajectory.append(
            torch.log(selected_probabilities)
            - torch.log1p(-selected_probabilities)
        )

    if feature_kind == 'static':
        return log_odds_trajectory[0]
    if feature_kind == 'passive':
        before, after = log_odds_trajectory[:2]
        return torch.cat([before, after, after - before], dim=1)

    deltas = [
        log_odds_trajectory[index] - log_odds_trajectory[index - 1]
        for index in range(1, len(log_odds_trajectory))
    ]
    return torch.cat([*log_odds_trajectory, *deltas], dim=1)


def dvia_ucd_from_probabilities(
    before_probabilities,
    after_probabilities,
    class_labels,
):
    """Transform correct-class confidence changes from cached posteriors.

    Class labels identify the queried probability column; they are not
    membership labels. The same transform is used for shadows and targets.
    """
    before = np.asarray(before_probabilities, dtype=np.float64)
    after = np.asarray(after_probabilities, dtype=np.float64)
    labels = np.asarray(class_labels, dtype=np.int64)
    if before.ndim != 2 or after.shape != before.shape:
        raise ValueError(
            'DVIA needs matching before/after probability matrices.'
        )
    if labels.ndim != 1 or len(labels) != len(before) or len(labels) == 0:
        raise ValueError(
            'DVIA needs one class label per non-empty probability row.'
        )
    if np.any(labels < 0) or np.any(labels >= before.shape[1]):
        raise ValueError(
            'DVIA class label is outside the probability columns.'
        )
    rows = np.arange(len(labels))
    ucd = np.clip(
        after[rows, labels] - before[rows, labels],
        -1.0 + dvia_ucd_epsilon,
        1.0 - dvia_ucd_epsilon,
    )
    return np.log1p(ucd) - np.log1p(-ucd)


def gaussian_log_likelihood(values, mean, variance):
    values = np.asarray(values, dtype=np.float64)
    variance = max(float(variance), dvia_gaussian_variance_floor)
    return -0.5 * (
        np.log(2.0 * np.pi * variance)
        + np.square(values - float(mean)) / variance
    )


def fit_reference_dvia(member_ucd, nonmember_ucd):
    """Calibrate DVIA solely on attacker-controlled shadow IN/OUT features.

    Neither target-model outputs nor target membership labels enter the fit.
    """
    member_ucd = np.asarray(member_ucd, dtype=np.float64)
    nonmember_ucd = np.asarray(nonmember_ucd, dtype=np.float64)
    for values in (member_ucd, nonmember_ucd):
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError(
                'DVIA calibration needs non-empty finite UCD vectors.'
            )
    return {
        'member_transformed_ucd_mean': float(np.mean(member_ucd)),
        'member_transformed_ucd_variance': max(
            float(np.var(member_ucd)), dvia_gaussian_variance_floor
        ),
        'nonmember_transformed_ucd_mean': float(np.mean(nonmember_ucd)),
        'nonmember_transformed_ucd_variance': max(
            float(np.var(nonmember_ucd)), dvia_gaussian_variance_floor
        ),
    }


def score_reference_dvia(target_ucd, gaussian_fit):
    """Score targets using fixed shadow-calibrated Gaussian parameters."""
    return gaussian_log_likelihood(
        target_ucd,
        gaussian_fit['member_transformed_ucd_mean'],
        gaussian_fit['member_transformed_ucd_variance'],
    ) - gaussian_log_likelihood(
        target_ucd,
        gaussian_fit['nonmember_transformed_ucd_mean'],
        gaussian_fit['nonmember_transformed_ucd_variance'],
    )


def fit_and_score_reference_attack(
    member_features,
    nonmember_features,
    target_features,
    random_seed,
):
    """Fit an MLP only on attacker-controlled reference memberships."""
    if len(member_features) != len(nonmember_features):
        raise ValueError('Reference member/non-member features must balance.')
    if member_features.shape[1] != target_features.shape[1]:
        raise ValueError('Reference and target feature dimensions must match.')

    set_experiment_seed(random_seed)
    features = torch.cat([member_features, nonmember_features], dim=0)
    labels = torch.cat([
        torch.ones(len(member_features), 1),
        torch.zeros(len(nonmember_features), 1),
    ], dim=0)
    loader_generator = torch.Generator().manual_seed(random_seed)
    attack_loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=128,
        shuffle=True,
        generator=loader_generator,
    )
    classifier = nn.Sequential(
        nn.Linear(features.shape[1], attack_hidden_size),
        nn.ReLU(),
        nn.Linear(attack_hidden_size, 1),
        nn.Sigmoid(),
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=attack_lr)

    for epoch_index in range(attack_epochs):
        classifier.train()
        running_loss = 0.0
        for batch_features, batch_labels in attack_loader:
            optimizer.zero_grad()
            predictions = classifier(batch_features.to(device))
            loss = criterion(predictions, batch_labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (
            epoch_index == 0
            or (epoch_index + 1) % 10 == 0
            or epoch_index == attack_epochs - 1
        ):
            print(
                f'[Reference-trained probability MIA] epoch '
                f'{epoch_index + 1}/{attack_epochs} | '
                f'loss={running_loss / len(attack_loader):.4f}',
                flush=True,
            )

    classifier.eval()
    with torch.no_grad():
        scores = classifier(target_features.to(device)).squeeze(1)
    return scores.cpu().numpy()


def take_balanced_indices(
    candidate_indices,
    labels,
    records_per_class,
    split_name,
):
    """Take a deterministic class-balanced subset and return the remainder."""
    candidate_indices = [int(index) for index in candidate_indices]
    selected = []
    selected_set = set()
    for class_index in range(num_classes):
        class_indices = [
            index for index in candidate_indices
            if int(labels[index]) == class_index
        ]
        if len(class_indices) < records_per_class:
            raise ValueError(
                f'{split_name} needs {records_per_class} class-{class_index} '
                f'records, but only {len(class_indices)} remain.'
            )
        chosen = class_indices[:records_per_class]
        selected.extend(chosen)
        selected_set.update(chosen)
    remaining = [
        index for index in candidate_indices if index not in selected_set
    ]
    return torch.tensor(selected, dtype=torch.long), remaining


def build_complementary_reference_masks(reference_labels):
    """Make each reference candidate IN in half and OUT in half of shadows."""
    if num_decoder_shadow_models < 2 or num_decoder_shadow_models % 2 != 0:
        raise ValueError(
            'num_decoder_shadow_models must be a positive even number.'
        )
    masks = []
    generator = torch.Generator().manual_seed(decoder_shadow_seed)
    for _ in range(num_decoder_shadow_models // 2):
        member_mask = torch.zeros(len(reference_labels), dtype=torch.bool)
        for class_index in range(num_classes):
            class_positions = torch.nonzero(
                reference_labels == class_index,
                as_tuple=False,
            ).squeeze(1)
            if len(class_positions) == 0 or len(class_positions) % 2 != 0:
                raise ValueError(
                    'Each reference class needs a positive even number of '
                    'records for complementary IN/OUT shadows.'
                )
            permutation = torch.randperm(
                len(class_positions), generator=generator
            )
            member_mask[
                class_positions[permutation[:len(class_positions) // 2]]
            ] = True
        masks.extend([member_mask, ~member_mask])
    return masks


def stable_reference_state_seed(shadow_index, deleted_positions):
    """Use one deterministic seed per shadow world and retained set."""
    state_seed = int(decoder_shadow_seed + 1009 * shadow_index)
    for probe_position in sorted(int(value) for value in deleted_positions):
        state_seed = (
            state_seed * 1000003 + probe_position + 9176
        ) % (2 ** 31 - 1)
    return state_seed


# Run each budget in an isolated process so GPU memory is released between
# experiments. Exact-retraining checkpoints and the maximum active schedule
# are shared across child processes.
requested_sequence_size = os.environ.get(sequence_size_env)
summary_path = results_dir / (
    f'TinyImageNet_ResNet18_{initialization_name}_Shadow_Reference_'
    'Probability_Active_ADP_'
    'Different_Unlearning_Sizes_Summary.txt'
)
if __name__ == '__main__' and requested_sequence_size is None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8') as summary_file:
        summary_file.write(
            f'unlearning_sequence_sizes: {list(unlearning_sequence_sizes)}\n'
        )
        summary_file.write(
            'attacks: static, passive_deletion, sequential_adp, '
            'active_adp, dvia\n'
        )
        summary_file.write(f'active_probe_pool_size: {active_probe_pool_size}\n')
        summary_file.write(
            'attack_decoder_label_source: '
            'attacker_controlled_shadow_reference_memberships_only\n'
        )
        summary_file.write(
            'dvia_calibration_label_source: '
            'attacker_controlled_shadow_reference_memberships_only\n'
        )
        summary_file.write(
            f'num_decoder_shadow_models: {num_decoder_shadow_models}\n'
        )

    for sequence_size in unlearning_sequence_sizes:
        child_environment = os.environ.copy()
        child_environment[sequence_size_env] = str(sequence_size)
        print(
            f'\n===== Running unlearning sequence size {sequence_size} =====',
            flush=True,
        )
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            check=True,
            env=child_environment,
        )
    print(f'All sequence-size results saved under: {results_dir}')
    raise SystemExit(0)

try:
    num_unlearning_operations = int(
        requested_sequence_size
        if requested_sequence_size is not None
        else unlearning_sequence_sizes[0]
    )
except ValueError as error:
    raise ValueError(
        f'{sequence_size_env} must be an integer, got '
        f'{requested_sequence_size!r}.'
    ) from error
if num_unlearning_operations not in unlearning_sequence_sizes:
    raise ValueError(
        f'Unsupported unlearning sequence size {num_unlearning_operations}; '
        f'choose one of {list(unlearning_sequence_sizes)}.'
    )

results_dir.mkdir(parents=True, exist_ok=True)


def load_or_create_permutation(path, dataset_size):
    if path.exists():
        permutation = torch.load(path, map_location='cpu')
        valid = (
            permutation.ndim == 1
            and permutation.numel() == dataset_size
            and int(permutation.min()) == 0
            and int(permutation.max()) == dataset_size - 1
            and torch.unique(permutation).numel() == dataset_size
        )
        if valid:
            return permutation.long()
        print(f'Regenerating stale index permutation: {path}')
    permutation = torch.randperm(dataset_size)
    torch.save(permutation, path)
    return permutation


def materialize_subset(dataset, indices, loader_batch_size=batch_size):
    normalized_indices = [int(index) for index in indices]
    if not normalized_indices:
        raise ValueError('Cannot materialize an empty dataset subset.')
    loader = DataLoader(
        Subset(dataset, normalized_indices),
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    images = []
    labels = []
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels)
    return torch.cat(images, dim=0), torch.cat(labels, dim=0)


# VIB.py uses local tiny-imagenet-200, 128 x 128 images, and ImageNet
# normalization.  Training augmentation and deterministic evaluation views
# share the same stable, sorted file-to-index mapping.
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    ),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])
evaluation_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

train_dataset = TinyImageNetDataset(
    data_root, transform=train_transform, train=True
)
train_evaluation_dataset = TinyImageNetDataset(
    data_root, transform=evaluation_transform, train=True
)
test_dataset = TinyImageNetDataset(
    data_root, transform=evaluation_transform, train=False
)
reference_auxiliary_train_dataset = TinyImageNetDataset(
    data_root, transform=train_transform, train=False
)
if len(train_dataset) != len(train_evaluation_dataset):
    raise ValueError('Training and evaluation views have different lengths.')
if (
    reference_auxiliary_train_dataset.data != test_dataset.data
    or reference_auxiliary_train_dataset.labels != test_dataset.labels
):
    raise ValueError(
        'Reference training/evaluation views have different validation indices.'
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
train_evaluation_loader = DataLoader(
    train_evaluation_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
print(
    f'TinyImageNet root: {train_dataset.root_dir} | '
    f'train={len(train_dataset)} | val={len(test_dataset)}'
)


target_model = build_model().to(device)
if not target_model_path.exists():
    print(
        f'Training TinyImageNet ResNet-18 ({initialization_name}) '
        f'for {epochs} epochs'
    )
    train_model_with_loader(
        target_model,
        train_loader,
        training_epochs=epochs,
        lr=lr,
        device=device,
    )
    torch.save(target_model.state_dict(), target_model_path)
else:
    print(f'Loading TinyImageNet ResNet-18: {target_model_path}')
    target_model.load_state_dict(
        torch.load(target_model_path, map_location=device)
    )

train_accuracy = calculate_accuracy(
    target_model, train_evaluation_loader, device
)
test_accuracy = calculate_accuracy(target_model, test_loader, device)
print(f'Training accuracy: {train_accuracy:.2f}%')
print(f'Validation accuracy: {test_accuracy:.2f}%')
print('----------------------------------')
print('Probability baseline attack knowledge:')
print(
    'Attack classifier: memberships controlled only in attacker-constructed '
    'shadow/reference models'
)
print('Target-model measurement membership: evaluation metrics only')
print('DVIA: shadow IN/OUT Gaussian calibration; known class labels for UCD')
print('Active selector: disjoint auxiliary design targets and design shadows')
print('----------------------------------')


if not unlearning_sequence_sizes:
    raise ValueError('At least one unlearning sequence size is required.')
if tuple(sorted(set(unlearning_sequence_sizes))) != unlearning_sequence_sizes:
    raise ValueError('unlearning_sequence_sizes must be unique and increasing.')
if min(unlearning_sequence_sizes) < 2:
    raise ValueError('Sequential ADP requires at least two deletions.')
if active_probe_pool_size < max_num_unlearning_operations:
    raise ValueError('active_probe_pool_size must cover the maximum budget.')
if num_active_design_targets < 1:
    raise ValueError('Active ADP needs at least one design target.')
if active_design_shadow_size < 2:
    raise ValueError('Active ADP needs at least two design-shadow records.')

original_indices = load_or_create_permutation(
    train_indices_path, len(train_dataset)
)
probe_pool_indices = [
    int(index) for index in original_indices[:active_probe_pool_size]
]
fixed_deletion_positions = list(range(num_unlearning_operations))
fixed_deletion_indices = [
    probe_pool_indices[position] for position in fixed_deletion_positions
]
if measurement_number % num_classes != 0:
    raise ValueError('measurement_number must be divisible by num_classes.')
measurement_train_indices, _ = take_balanced_indices(
    original_indices[active_probe_pool_size:].tolist(),
    train_evaluation_dataset.labels,
    measurement_number // num_classes,
    'Member measurement split',
)
probe_pool_data, probe_pool_labels = materialize_subset(
    train_evaluation_dataset, probe_pool_indices
)
measurement_train_images, measurement_train_labels = materialize_subset(
    train_evaluation_dataset, measurement_train_indices.tolist()
)


# Exact retraining is cached by ordered deletion prefix.  Each new release is
# initialized as a fresh ResNet-18 and trained on the retained augmented set.
target_model_cache = {(): target_model}
release_test_accuracy_cache = {(): test_accuracy}


def load_or_train_target_release(deleted_prefix):
    prefix_key = tuple(int(index) for index in deleted_prefix)
    if prefix_key in target_model_cache:
        return target_model_cache[prefix_key]

    deleted_set = set(prefix_key)
    retained_indices = [
        index for index in range(len(train_dataset))
        if index not in deleted_set
    ]
    retained_loader = DataLoader(
        Subset(train_dataset, retained_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    release_model = build_model().to(device)
    prefix_tag = '_'.join(str(index) for index in prefix_key)
    release_path = script_dir / (
        f'target_model_tinyimagenet_resnet18_{initialization_name}_'
        f'retrain_prefix_{prefix_tag}.pth'
    )
    if not release_path.exists():
        print(
            f'Retraining ResNet-18 after deletion prefix '
            f'{list(prefix_key)} for {epochs} epochs'
        )
        train_model_with_loader(
            release_model,
            retained_loader,
            training_epochs=epochs,
            lr=lr,
            device=device,
        )
        torch.save(release_model.state_dict(), release_path)
    else:
        print(f'Loading target release after prefix {list(prefix_key)}')
        release_model.load_state_dict(
            torch.load(release_path, map_location=device)
        )

    release_accuracy = calculate_accuracy(release_model, test_loader, device)
    print(
        f'Validation accuracy after prefix {list(prefix_key)}: '
        f'{release_accuracy:.2f}%'
    )
    target_model_cache[prefix_key] = release_model
    release_test_accuracy_cache[prefix_key] = release_accuracy
    return release_model


target_models = [target_model]
fixed_prefix = []
for deletion_index in fixed_deletion_indices:
    fixed_prefix.append(deletion_index)
    target_models.append(load_or_train_target_release(fixed_prefix))
fixed_release_test_accuracies = [
    release_test_accuracy_cache[tuple(fixed_deletion_indices[:prefix_length])]
    for prefix_length in range(num_unlearning_operations + 1)
]
print(f'Attacker-owned probe pool: {probe_pool_indices}')
print(f'Fixed Sequential deletion sequence: {fixed_deletion_indices}')


# Partition validation records by class without materializing the complete
# image set. This prevents class-composition leakage and guarantees enough
# records for mutually disjoint measurement, design, and reference pools.
original_indices_test = load_or_create_permutation(
    test_indices_path, len(test_dataset)
)
required_test_records = (
    measurement_number
    + active_design_shadow_size
    + num_active_design_targets
    + num_classes * decoder_reference_records_per_class
    + num_classes * decoder_shadow_base_records_per_class
)
if required_test_records > len(test_dataset):
    raise ValueError(
        'TinyImageNet validation split is too small for measurement, active '
        'design, and shadow/reference decoder pools.'
    )
if active_design_shadow_size % num_classes != 0:
    raise ValueError(
        'active_design_shadow_size must be divisible by num_classes.'
    )

validation_pool = original_indices_test.tolist()
measurement_test_indices, validation_pool = take_balanced_indices(
    validation_pool,
    test_dataset.labels,
    measurement_number // num_classes,
    'Non-member measurement split',
)
active_design_shadow_indices, validation_pool = take_balanced_indices(
    validation_pool,
    test_dataset.labels,
    active_design_shadow_size // num_classes,
    'Active design shadow split',
)
shadow_label_counts = Counter(
    test_dataset.labels[int(index)] for index in active_design_shadow_indices
)
selected_label_counts = Counter()
active_design_target_indices = []
post_design_target_pool = []
for index in validation_pool:
    label = test_dataset.labels[int(index)]
    if (
        len(active_design_target_indices) < num_active_design_targets
        and selected_label_counts[label] < shadow_label_counts[label]
    ):
        active_design_target_indices.append(int(index))
        selected_label_counts[label] += 1
    else:
        post_design_target_pool.append(int(index))
if len(active_design_target_indices) != num_active_design_targets:
    raise ValueError('Could not construct class-matched Active design targets.')

decoder_reference_indices, post_design_target_pool = take_balanced_indices(
    post_design_target_pool,
    test_dataset.labels,
    decoder_reference_records_per_class,
    'Decoder reference candidate split',
)
decoder_shadow_base_indices, post_design_target_pool = take_balanced_indices(
    post_design_target_pool,
    test_dataset.labels,
    decoder_shadow_base_records_per_class,
    'Decoder shadow base split',
)
decoder_reference_labels = torch.tensor(
    [test_dataset.labels[int(index)] for index in decoder_reference_indices],
    dtype=torch.long,
)
decoder_membership_masks = build_complementary_reference_masks(
    decoder_reference_labels
)
decoder_reference_dataset = Subset(
    test_dataset, decoder_reference_indices.tolist()
)
decoder_shadow_base_dataset = Subset(
    reference_auxiliary_train_dataset,
    decoder_shadow_base_indices.tolist(),
)

validation_split_sets = {
    'measurement_nonmembers': set(measurement_test_indices.tolist()),
    'active_design_targets': set(active_design_target_indices),
    'active_design_shadows': set(active_design_shadow_indices.tolist()),
    'decoder_reference_candidates': set(decoder_reference_indices.tolist()),
    'decoder_shadow_base': set(decoder_shadow_base_indices.tolist()),
}
validation_split_names = list(validation_split_sets)
for left_index, left_name in enumerate(validation_split_names):
    for right_name in validation_split_names[left_index + 1:]:
        overlap = (
            validation_split_sets[left_name]
            & validation_split_sets[right_name]
        )
        if overlap:
            raise RuntimeError(
                f'Auxiliary/evaluation split overlap between {left_name} '
                f'and {right_name}: {sorted(overlap)[:10]}'
            )
if set(probe_pool_indices) & set(measurement_train_indices.tolist()):
    raise RuntimeError('Attacker-owned probes overlap member evaluation targets.')

measurement_test_images, measurement_test_labels = materialize_subset(
    test_dataset, measurement_test_indices.tolist()
)
active_design_target_images, active_design_target_labels = materialize_subset(
    test_dataset, active_design_target_indices
)
active_design_shadow_images, active_design_shadow_labels = materialize_subset(
    test_dataset, active_design_shadow_indices.tolist()
)


# The selector is greedy and therefore has a prefix property. Compute/cache the
# maximum global schedule once, then use the requested prefix for every budget.
# It never queries target-model releases or the measurement set.
active_selection_cache_path = script_dir / (
    f'active_global_greedy_mean_selection_tinyimagenet_resnet18_'
    f'{initialization_name}_'
    f'pool_{active_probe_pool_size}_'
    f'max_unlearning_{max_num_unlearning_operations}.json'
)
active_selection_metadata = {
    'probe_pool_indices': probe_pool_indices,
    'active_design_target_indices': [
        int(index) for index in active_design_target_indices
    ],
    'active_design_shadow_indices': active_design_shadow_indices.tolist(),
    'max_num_unlearning_operations': max_num_unlearning_operations,
    'num_design_shadow_models': num_design_shadow_models,
    'active_selection_epochs': active_selection_epochs,
    'active_selection_lr': active_selection_lr,
    'active_selection_ridge': active_selection_ridge,
    'active_selection_batch_size': active_selection_batch_size,
    'active_selection_seed': active_selection_seed,
    'model': model_name,
    'pretrained': use_pretrained,
    'initialization': initialization_name,
    'num_classes': num_classes,
    'image_size': image_size,
    'selection_policy': (
        'shared_prefix_global_greedy_mean_conditional_score_v1'
    ),
}
cached_selection = None
if active_selection_cache_path.exists():
    try:
        with active_selection_cache_path.open(
            'r', encoding='utf-8'
        ) as cache_file:
            candidate_cache = json.load(cache_file)
        if candidate_cache.get('metadata') == active_selection_metadata:
            cached_selection = candidate_cache
            print(
                f'Loading cached maximum active schedule from '
                f'{active_selection_cache_path}'
            )
        else:
            print('Ignoring active-selection cache with mismatched metadata.')
    except (OSError, json.JSONDecodeError) as error:
        print(f'Ignoring unreadable active-selection cache: {error}')

if cached_selection is None:
    set_experiment_seed(active_selection_seed)
    (
        max_active_deletion_positions,
        max_active_selection_diagnostics,
    ) = select_global_probe_sequence(
        active_design_target_images,
        active_design_target_labels,
        active_design_shadow_images,
        active_design_shadow_labels,
        probe_pool_data,
        probe_pool_labels,
        num_unlearning_operations=max_num_unlearning_operations,
        num_design_shadow_models=num_design_shadow_models,
        epochs=active_selection_epochs,
        lr=active_selection_lr,
        num_classes=num_classes,
        pretrained=use_pretrained,
        selection_ridge=active_selection_ridge,
        device=device,
        batch_size=active_selection_batch_size,
    )
    cached_selection = {
        'metadata': active_selection_metadata,
        'active_deletion_positions': max_active_deletion_positions,
        'active_selection_diagnostics': max_active_selection_diagnostics,
    }
    with active_selection_cache_path.open(
        'w', encoding='utf-8'
    ) as cache_file:
        json.dump(cached_selection, cache_file)
else:
    max_active_deletion_positions = cached_selection[
        'active_deletion_positions'
    ]
    max_active_selection_diagnostics = cached_selection[
        'active_selection_diagnostics'
    ]

active_deletion_positions = max_active_deletion_positions[
    :num_unlearning_operations
]
active_selection_diagnostics = max_active_selection_diagnostics[
    :num_unlearning_operations
]
active_deletion_indices = [
    probe_pool_indices[position] for position in active_deletion_positions
]
active_target_models = [target_model]
active_prefix = []
for deletion_index in active_deletion_indices:
    active_prefix.append(deletion_index)
    active_target_models.append(load_or_train_target_release(active_prefix))
active_release_test_accuracies = [
    release_test_accuracy_cache[tuple(active_deletion_indices[:prefix_length])]
    for prefix_length in range(num_unlearning_operations + 1)
]
selected_mean_scores = [
    round_info['mean_conditional_score']
    for round_info in active_selection_diagnostics
]
print(
    'Global Active ADP selected mean conditional scores: '
    f'{selected_mean_scores}'
)
print(f'Global Active ADP deletion sequence: {active_deletion_indices}')
del active_design_target_images
del active_design_target_labels
del active_design_shadow_images
del active_design_shadow_labels


measurement_images = torch.cat(
    [measurement_train_images, measurement_test_images], dim=0
)
# DVIA uses image class labels, independently of membership ground truth.
measurement_labels = torch.cat(
    [measurement_train_labels, measurement_test_labels], dim=0
)
measurement_ref = np.asarray(
    [1] * len(measurement_train_images)
    + [0] * len(measurement_test_images)
)
del measurement_train_images
del measurement_train_labels
del measurement_test_images
del measurement_test_labels
print(f'Measurement sample size: {len(measurement_images)}')

# ----------------Shadow/reference-trained attack decoders------------------
reference_cache_metadata = {
    'decoder_cache_version': decoder_cache_version,
    'dataset': dataset_name,
    'model': model_name,
    'pretrained': use_pretrained,
    'initialization': initialization_name,
    'image_size': image_size,
    'num_classes': num_classes,
    'batch_size': batch_size,
    'reference_training_transform': 'target_training_augmentation',
    'reference_query_transform': 'deterministic_evaluation',
    'probe_pool_indices': probe_pool_indices,
    'decoder_reference_indices': decoder_reference_indices.tolist(),
    'decoder_shadow_base_indices': decoder_shadow_base_indices.tolist(),
    'decoder_membership_masks': [
        mask.to(dtype=torch.int8).tolist()
        for mask in decoder_membership_masks
    ],
    'num_decoder_shadow_models': num_decoder_shadow_models,
    'decoder_shadow_epochs': decoder_shadow_epochs,
    'decoder_shadow_lr': decoder_shadow_lr,
    'decoder_shadow_seed': decoder_shadow_seed,
    'unlearning_method': 'exact_retraining',
}
reference_cache_fingerprint = hashlib.sha256(
    json.dumps(reference_cache_metadata, sort_keys=True).encode('utf-8')
).hexdigest()[:16]
reference_state_cache_dir = (
    results_dir
    / 'tinyimagenet_reference_state_cache'
    / reference_cache_fingerprint
)
reference_state_cache_dir.mkdir(parents=True, exist_ok=True)
reference_probability_cache = {}


def load_or_train_reference_state(shadow_index, deleted_positions):
    deleted_positions = tuple(sorted(int(value) for value in deleted_positions))
    state_key = (int(shadow_index), deleted_positions)
    if state_key in reference_probability_cache:
        return reference_probability_cache[state_key]

    deleted_tag = (
        'none' if not deleted_positions
        else '_'.join(str(value) for value in deleted_positions)
    )
    cache_path = reference_state_cache_dir / (
        f'shadow_{shadow_index:02d}_deleted_{deleted_tag}.pt'
    )
    if cache_path.exists():
        cached_probabilities = torch.load(cache_path, map_location='cpu')
        if (
            isinstance(cached_probabilities, torch.Tensor)
            and tuple(cached_probabilities.shape)
            == (len(decoder_reference_indices), num_classes)
        ):
            print(f'Loading cached reference state: {cache_path.name}')
            reference_probability_cache[state_key] = cached_probabilities
            return cached_probabilities
        print(f'Ignoring malformed reference-state cache: {cache_path}')

    membership_mask = decoder_membership_masks[shadow_index]
    reference_member_indices = decoder_reference_indices[
        membership_mask
    ].tolist()
    deleted_position_set = set(deleted_positions)
    remaining_probe_positions = [
        position for position in range(active_probe_pool_size)
        if position not in deleted_position_set
    ]
    remaining_probe_indices = [
        probe_pool_indices[position]
        for position in remaining_probe_positions
    ]
    probe_dataset = Subset(
        train_dataset,
        remaining_probe_indices,
    )
    reference_train_dataset = ConcatDataset([
        decoder_shadow_base_dataset,
        Subset(reference_auxiliary_train_dataset, reference_member_indices),
        probe_dataset,
    ])
    state_seed = stable_reference_state_seed(
        shadow_index, deleted_positions
    )
    set_experiment_seed(state_seed)
    loader_generator = torch.Generator().manual_seed(state_seed)
    reference_loader = DataLoader(
        reference_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(
        f'Training reference shadow {shadow_index + 1}/'
        f'{num_decoder_shadow_models}, deleted probe positions '
        f'{list(deleted_positions)}',
        flush=True,
    )
    reference_model = build_model().to(device)
    train_model_with_loader(
        reference_model,
        reference_loader,
        training_epochs=decoder_shadow_epochs,
        lr=decoder_shadow_lr,
        device=device,
    )
    probabilities = get_dataset_probability_vectors(
        reference_model, decoder_reference_dataset
    )
    torch.save(probabilities, cache_path)
    reference_probability_cache[state_key] = probabilities
    del reference_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probabilities


def reference_probability_trajectory(shadow_index, deletion_schedule):
    trajectory = [load_or_train_reference_state(shadow_index, ())]
    deleted_prefix = []
    for probe_position in deletion_schedule:
        deleted_prefix.append(int(probe_position))
        trajectory.append(
            load_or_train_reference_state(shadow_index, deleted_prefix)
        )
    return trajectory


reference_member_parts = {
    'static': [], 'passive': [], 'sequential': [], 'active': [],
}
reference_nonmember_parts = {
    'static': [], 'passive': [], 'sequential': [], 'active': [],
}
reference_dvia_member_parts = []
reference_dvia_nonmember_parts = []
for shadow_index, membership_mask in enumerate(decoder_membership_masks):
    fixed_reference_trajectory = reference_probability_trajectory(
        shadow_index, fixed_deletion_positions
    )
    active_reference_trajectory = reference_probability_trajectory(
        shadow_index, active_deletion_positions
    )
    # Match the original/final fixed-release views used on the target model.
    # Membership here is relative only to the constructed reference shadow.
    reference_dvia_ucd = dvia_ucd_from_probabilities(
        fixed_reference_trajectory[0],
        fixed_reference_trajectory[-1],
        decoder_reference_labels,
    )
    reference_dvia_member_mask = np.asarray(membership_mask, dtype=bool)
    reference_dvia_member_parts.append(
        reference_dvia_ucd[reference_dvia_member_mask]
    )
    reference_dvia_nonmember_parts.append(
        reference_dvia_ucd[~reference_dvia_member_mask]
    )
    method_features = {
        'static': fixed_top_class_log_odds_features(
            fixed_reference_trajectory,
            decoder_reference_labels,
            'static',
        ),
        'passive': fixed_top_class_log_odds_features(
            fixed_reference_trajectory,
            decoder_reference_labels,
            'passive',
        ),
        'sequential': fixed_top_class_log_odds_features(
            fixed_reference_trajectory,
            decoder_reference_labels,
            'sequential',
        ),
        'active': fixed_top_class_log_odds_features(
            active_reference_trajectory,
            decoder_reference_labels,
            'sequential',
        ),
    }
    for method_name, features in method_features.items():
        reference_member_parts[method_name].append(
            features[membership_mask]
        )
        reference_nonmember_parts[method_name].append(
            features[~membership_mask]
        )

reference_member_features = {
    method_name: torch.cat(feature_parts, dim=0)
    for method_name, feature_parts in reference_member_parts.items()
}
reference_nonmember_features = {
    method_name: torch.cat(feature_parts, dim=0)
    for method_name, feature_parts in reference_nonmember_parts.items()
}
reference_dvia_member_ucd = np.concatenate(reference_dvia_member_parts)
reference_dvia_nonmember_ucd = np.concatenate(
    reference_dvia_nonmember_parts
)
# Freeze calibration before querying the target measurement records.
dvia_gaussian_fit = fit_reference_dvia(
    reference_dvia_member_ucd, reference_dvia_nonmember_ucd
)

# Target-model membership ground truth is absent from this query, feature, and
# classifier-fitting path. It first enters the computation in roc_curve below.
fixed_target_probability_trajectory = [
    get_tensor_probability_vectors(model, measurement_images)
    for model in target_models
]
active_target_probability_trajectory = [
    get_tensor_probability_vectors(model, measurement_images)
    for model in active_target_models
]
dvia_transformed_ucd = dvia_ucd_from_probabilities(
    fixed_target_probability_trajectory[0],
    fixed_target_probability_trajectory[-1],
    measurement_labels,
)
target_method_features = {
    'static': fixed_top_class_log_odds_features(
        fixed_target_probability_trajectory,
        measurement_labels,
        'static',
    ),
    'passive': fixed_top_class_log_odds_features(
        fixed_target_probability_trajectory,
        measurement_labels,
        'passive',
    ),
    'sequential': fixed_top_class_log_odds_features(
        fixed_target_probability_trajectory,
        measurement_labels,
        'sequential',
    ),
    'active': fixed_top_class_log_odds_features(
        active_target_probability_trajectory,
        measurement_labels,
        'sequential',
    ),
}

static_scores = fit_and_score_reference_attack(
    reference_member_features['static'],
    reference_nonmember_features['static'],
    target_method_features['static'],
    attack_random_seed,
)
passive_scores = fit_and_score_reference_attack(
    reference_member_features['passive'],
    reference_nonmember_features['passive'],
    target_method_features['passive'],
    attack_random_seed + 1,
)
sequential_scores = fit_and_score_reference_attack(
    reference_member_features['sequential'],
    reference_nonmember_features['sequential'],
    target_method_features['sequential'],
    attack_random_seed + 2,
)
active_scores = fit_and_score_reference_attack(
    reference_member_features['active'],
    reference_nonmember_features['active'],
    target_method_features['active'],
    attack_random_seed + 3,
)
dvia_scores = score_reference_dvia(dvia_transformed_ucd, dvia_gaussian_fit)

static_fpr, static_tpr, _ = roc_curve(measurement_ref, static_scores)
passive_fpr, passive_tpr, _ = roc_curve(measurement_ref, passive_scores)
sequential_fpr, sequential_tpr, _ = roc_curve(
    measurement_ref, sequential_scores
)
active_fpr, active_tpr, _ = roc_curve(measurement_ref, active_scores)
dvia_fpr, dvia_tpr, _ = roc_curve(measurement_ref, dvia_scores)
static_auc = auc(static_fpr, static_tpr)
passive_auc = auc(passive_fpr, passive_tpr)
sequential_auc = auc(sequential_fpr, sequential_tpr)
active_auc = auc(active_fpr, active_tpr)
dvia_auc = auc(dvia_fpr, dvia_tpr)
dvia_predictions = (
    dvia_scores > np.log(dvia_likelihood_ratio_threshold)
).astype(np.int64)
dvia_accuracy = accuracy_score(measurement_ref, dvia_predictions)
passive_deletion_gain = passive_auc - static_auc
sequential_gain_over_static = sequential_auc - static_auc
sequential_gain_over_passive = sequential_auc - passive_auc
active_gain_over_static = active_auc - static_auc
active_gain_over_passive = active_auc - passive_auc
active_gain_over_sequential = active_auc - sequential_auc
dvia_gain_over_static = dvia_auc - static_auc
active_gain_over_dvia = active_auc - dvia_auc

print('--------------')
print(f'Reference-trained Static probability MIA AUC: {static_auc:.4f}')
print(f'Reference-trained Passive-deletion probability MIA AUC: {passive_auc:.4f}')
print(f'Reference-trained Sequential ADP probability MIA AUC: {sequential_auc:.4f}')
print(f'Reference-trained Global Active ADP probability MIA AUC: {active_auc:.4f}')
print(
    f'Shadow-calibrated DVIA UCD likelihood-ratio AUC: {dvia_auc:.4f} '
    f'(accuracy at likelihood-ratio threshold 1: {dvia_accuracy:.4f})'
)
print(f'Passive gain over Static: {passive_deletion_gain:+.4f}')
print(f'Sequential gain over Static: {sequential_gain_over_static:+.4f}')
print(f'Sequential gain over Passive: {sequential_gain_over_passive:+.4f}')
print(f'Active gain over Static: {active_gain_over_static:+.4f}')
print(f'Active gain over Passive: {active_gain_over_passive:+.4f}')
print(f'Active gain over fixed Sequential: {active_gain_over_sequential:+.4f}')
print(
    f'DVIA gain over Static: {dvia_gain_over_static:+.4f} '
    f'({100.0 * dvia_gain_over_static:+.2f} percentage points)'
)
print(
    f'Active ADP gain over DVIA: {active_gain_over_dvia:+.4f} '
    f'({100.0 * active_gain_over_dvia:+.2f} percentage points)'
)
print('--------------')

plt.plot(
    static_fpr, static_tpr, color='steelblue', lw=2,
    label=f'Static probability MIA (area = {static_auc:.2f})',
)
plt.plot(
    passive_fpr, passive_tpr, color='darkorange', lw=2,
    label=f'Passive-deletion MIA (area = {passive_auc:.2f})',
)
plt.plot(
    sequential_fpr, sequential_tpr, color='forestgreen', lw=2,
    label=f'Sequential ADP (area = {sequential_auc:.2f})',
)
plt.plot(
    active_fpr, active_tpr, color='crimson', lw=2,
    label=f'Global Active ADP (area = {active_auc:.2f})',
)
plt.plot(
    dvia_fpr,
    dvia_tpr,
    color='saddlebrown',
    lw=2,
    linestyle='-.',
    label=f'Shadow-calibrated DVIA (area = {dvia_auc:.2f})',
)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(
    'TinyImageNet ResNet-18 Reference-Trained Probability MIA '
    f'(unlearning size={num_unlearning_operations})'
)
plt.legend(loc='lower right')
plot_path = results_dir / (
    f'ROC_TinyImageNet_ResNet18_{initialization_name}_'
    f'Shadow_Reference_Probability_Active_ADP_'
    f'Unlearning_Size_{num_unlearning_operations}.png'
)
plt.savefig(plot_path)
plt.close()


filename = results_dir / (
    f'TinyImageNet_ResNet18_{initialization_name}_'
    f'Shadow_Reference_Probability_Active_ADP_'
    f'Unlearning_Size_{num_unlearning_operations}.txt'
)
hyperparams = {
    'dataset': dataset_name,
    'data_root': str(train_dataset.root_dir),
    'input_shape': input_shape,
    'image_size': image_size,
    'num_classes': num_classes,
    'model': model_name,
    'pretrained': use_pretrained,
    'initialization': initialization_name,
    'epochs': epochs,
    'lr': lr,
    'batch_size': batch_size,
    'measurement_number': measurement_number,
    'tested_unlearning_sequence_sizes': list(unlearning_sequence_sizes),
    'num_unlearning_operations': num_unlearning_operations,
    'active_probe_pool_size': active_probe_pool_size,
    'num_active_design_targets': num_active_design_targets,
    'active_design_shadow_size': active_design_shadow_size,
    'num_design_shadow_models': num_design_shadow_models,
    'active_selection_epochs': active_selection_epochs,
    'active_selection_lr': active_selection_lr,
    'active_selection_ridge': active_selection_ridge,
    'active_selection_batch_size': active_selection_batch_size,
    'active_selection_seed': active_selection_seed,
    'active_selection_max_cached_budget': max_num_unlearning_operations,
    'active_selection_cache_path': str(active_selection_cache_path),
    'attack_epochs': attack_epochs,
    'attack_lr': attack_lr,
    'attack_hidden_size': attack_hidden_size,
    'attack_random_seed': attack_random_seed,
    'trajectory_top_nonmember_classes': trajectory_top_nonmember_classes,
    'trajectory_feature_classes_per_record': (
        trajectory_top_nonmember_classes + 1
    ),
    'trajectory_log_odds_epsilon': trajectory_log_odds_epsilon,
    'dvia_views': 'original_and_final_fixed_exact_unlearning_release',
    'dvia_attack_feature': (
        'phi(delta)=log((1+delta)/(1-delta)); '
        'delta=p_unlearned(y|x)-p_original(y|x)'
    ),
    'dvia_distribution_model': (
        'member_and_nonmember_univariate_gaussians'
    ),
    'dvia_distribution_calibration': (
        'pooled_shadow_reference_in_out_ucd_from_fixed_deletion_trajectories'
    ),
    'dvia_calibration_label_source': (
        'attacker_controlled_shadow_reference_memberships_only'
    ),
    'dvia_target_membership_used_for_calibration': False,
    'dvia_class_label_usage': (
        'known_class_labels_for_correct_label_confidence'
    ),
    'dvia_score': (
        'log_member_likelihood_minus_log_nonmember_likelihood'
    ),
    'dvia_ucd_epsilon': dvia_ucd_epsilon,
    'dvia_gaussian_variance_floor': dvia_gaussian_variance_floor,
    'dvia_likelihood_ratio_threshold': dvia_likelihood_ratio_threshold,
    'dvia_calibration_member_records': len(reference_dvia_member_ucd),
    'dvia_calibration_nonmember_records': len(
        reference_dvia_nonmember_ucd
    ),
    'attack_decoder_label_source': (
        'attacker_controlled_shadow_reference_memberships_only'
    ),
    'target_model_membership_ground_truth_usage': (
        'measurement_ground_truth_is_roc_auc_accuracy_evaluation_only;_'
        'attacker_knows_only_its_owned_probe_membership'
    ),
    'decoder_reference_records_per_class': (
        decoder_reference_records_per_class
    ),
    'decoder_shadow_base_records_per_class': (
        decoder_shadow_base_records_per_class
    ),
    'num_decoder_shadow_models': num_decoder_shadow_models,
    'decoder_shadow_epochs': decoder_shadow_epochs,
    'decoder_shadow_lr': decoder_shadow_lr,
    'decoder_shadow_seed': decoder_shadow_seed,
    'decoder_cache_version': decoder_cache_version,
    'reference_cache_fingerprint': reference_cache_fingerprint,
    'reference_state_cache_dir': str(reference_state_cache_dir),
    'device': str(device),
    'unlearning_method': 'exact_retraining',
    'fixed_sequence_policy': 'fixed_precommitted_random_order',
    'active_sequence_policy': (
        'shared_prefix_global_greedy_mean_conditional_score'
    ),
    'active_selection_aggregation': 'mean_across_design_targets',
    'probe_pool_indices': probe_pool_indices,
    'fixed_deletion_positions': fixed_deletion_positions,
    'fixed_deletion_indices': fixed_deletion_indices,
    'active_selection_diagnostics': active_selection_diagnostics,
    'active_deletion_positions': active_deletion_positions,
    'active_deletion_indices': active_deletion_indices,
    'trajectory_class_selection': (
        'correct_class_then_M0_top10_noncorrect_classes_fixed_across_releases'
    ),
    'static_attack_feature': 'fixed_top11_M0_class_log_odds',
    'passive_attack_feature': (
        'fixed_top11_M0_class_log_odds_before_after_and_delta'
    ),
    'sequential_attack_feature': (
        'fixed_top11_M0_class_log_odds_levels_and_consecutive_deltas'
    ),
    'active_attack_feature': (
        'fixed_top11_M0_class_log_odds_levels_and_consecutive_deltas'
    ),
    'active_design_target_records': len(active_design_target_indices),
    'active_design_shadow_records': len(active_design_shadow_indices),
    'measurement_train_indices': measurement_train_indices.tolist(),
    'measurement_test_indices': measurement_test_indices.tolist(),
    'decoder_reference_indices': decoder_reference_indices.tolist(),
    'decoder_shadow_base_indices': decoder_shadow_base_indices.tolist(),
    'decoder_membership_masks': [
        mask.to(dtype=torch.int8).tolist()
        for mask in decoder_membership_masks
    ],
    'attack_member_records': len(reference_member_features['static']),
    'attack_nonmember_records': len(reference_nonmember_features['static']),
    'target_train_accuracy': train_accuracy,
    'target_validation_accuracy': test_accuracy,
    'fixed_release_validation_accuracies': fixed_release_test_accuracies,
    'active_release_validation_accuracies': active_release_test_accuracies,
    'static_auc': static_auc,
    'passive_deletion_auc': passive_auc,
    'passive_deletion_gain_auc': passive_deletion_gain,
    'sequential_adp_auc': sequential_auc,
    'sequential_gain_over_static_auc': sequential_gain_over_static,
    'sequential_gain_over_passive_auc': sequential_gain_over_passive,
    'active_adp_auc': active_auc,
    'active_gain_over_static_auc': active_gain_over_static,
    'active_gain_over_passive_auc': active_gain_over_passive,
    'active_gain_over_sequential_auc': active_gain_over_sequential,
    'dvia_auc': dvia_auc,
    'dvia_accuracy_at_likelihood_ratio_threshold': dvia_accuracy,
    'dvia_gain_over_static_auc': dvia_gain_over_static,
    'active_gain_over_dvia_auc': active_gain_over_dvia,
    'dvia_gaussian_fit': dvia_gaussian_fit,
    'dvia_reference_member_transformed_ucd': (
        reference_dvia_member_ucd.tolist()
    ),
    'dvia_reference_nonmember_transformed_ucd': (
        reference_dvia_nonmember_ucd.tolist()
    ),
    # Raw attack outputs and exact ROC coordinates used by the PNG. This
    # experimenter-only ground truth first enters at roc_curve and is saved in
    # the same sample order solely for metric reproducibility.
    'measurement_membership_ground_truth': measurement_ref.tolist(),
    'static_attack_scores': np.asarray(static_scores).tolist(),
    'static_roc_fpr': static_fpr.tolist(),
    'static_roc_tpr': static_tpr.tolist(),
    'passive_attack_scores': np.asarray(passive_scores).tolist(),
    'passive_roc_fpr': passive_fpr.tolist(),
    'passive_roc_tpr': passive_tpr.tolist(),
    'sequential_attack_scores': np.asarray(sequential_scores).tolist(),
    'sequential_roc_fpr': sequential_fpr.tolist(),
    'sequential_roc_tpr': sequential_tpr.tolist(),
    'active_attack_scores': np.asarray(active_scores).tolist(),
    'active_roc_fpr': active_fpr.tolist(),
    'active_roc_tpr': active_tpr.tolist(),
    'dvia_transformed_ucd': dvia_transformed_ucd.tolist(),
    'dvia_attack_scores': dvia_scores.tolist(),
    'dvia_roc_fpr': dvia_fpr.tolist(),
    'dvia_roc_tpr': dvia_tpr.tolist(),
}
with filename.open('w', encoding='utf-8') as handle:
    for name, value in hyperparams.items():
        handle.write(f'{name}: {value}\n')

summary_result = {
    'num_unlearning_operations': num_unlearning_operations,
    'result_file': filename.name,
    'plot_file': plot_path.name,
    'static_auc': static_auc,
    'passive_deletion_auc': passive_auc,
    'sequential_adp_auc': sequential_auc,
    'active_adp_auc': active_auc,
    'dvia_auc': dvia_auc,
    'dvia_accuracy_at_likelihood_ratio_threshold': dvia_accuracy,
    'dvia_gain_over_static_auc': dvia_gain_over_static,
    'active_gain_over_dvia_auc': active_gain_over_dvia,
    'fixed_deletion_indices': fixed_deletion_indices,
    'active_deletion_indices': active_deletion_indices,
}
with summary_path.open('a', encoding='utf-8') as summary_file:
    summary_file.write(f'result: {summary_result}\n')
