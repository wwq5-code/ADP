"""SVHN practical shadow-reference membership attacks under exact retraining.

No target-model membership labels are used to fit the attack MLPs or DVIA.
Only attacker-controlled shadow memberships provide calibration labels.
SVHN_DATA_ROOT selects the official train/test .mat directory; extra is unused.
Target releases, data splits, and the global active schedule share the SVHN
baseline caches, while decoder reference states have their own cache.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

if __package__:
    from .baseline.baseline_prob import (
        calculate_accuracy,
        train_model_with_loader,
    )
    from .baseline.model import CNN
    from .baseline.active_selection import (
        select_global_conditional_probe_sequence,
    )
else:
    from baseline.baseline_prob import (
        calculate_accuracy,
        train_model_with_loader,
    )
    from baseline.model import CNN
    from baseline.active_selection import (
        select_global_conditional_probe_sequence,
    )


#-----------------------------------------------------------------------------------
dataset_name = 'SVHN'
input_shape = (3, 32, 32)
channel = 3
num_classes=10
hidden_size = 512
output_size = 10
epochs = 50
lr = 1e-3
measurement_number=1000  # measured members and non-members, respectively
unlearning_sequence_sizes=(3, 5, 7, 9, 11, 13, 15)
max_num_unlearning_operations=max(unlearning_sequence_sizes)
# Fixed Sequential uses the first k records.  Active ADP selects k records
# from the same 30-record attacker-owned pool using auxiliary design shadows.
active_probe_pool_size=30
num_active_design_targets=10
active_design_shadow_size=1000
num_design_shadow_models=2
active_selection_epochs=5
active_selection_lr=1e-3
active_selection_ridge=1e-4
active_selection_seed=0
attack_epochs=30
attack_lr=1e-3
attack_hidden_size=128
attack_random_seed=1000
dvia_ucd_epsilon=1e-6
dvia_gaussian_variance_floor=1e-8
dvia_likelihood_ratio_threshold=1.0
# The attack decoder is supervised only by memberships controlled by the
# attacker in auxiliary reference models. Attacker-owned deletion probes may
# appear in those models, but no other target-model membership label is used.
decoder_reference_records_per_class=50
decoder_shadow_base_records_per_class=600
num_decoder_shadow_models=4
decoder_shadow_epochs=20
decoder_shadow_lr=1e-3
decoder_shadow_seed=2000
decoder_cache_version=1
exact_retraining_seed=3000
exact_retraining_cache_version=2
gpu = 0
device = torch.device(
    'cuda:{}'.format(gpu)
    if torch.cuda.is_available() and gpu != -1
    else 'cpu'
)


def set_experiment_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model():
    return CNN(channel, num_classes)


def load_or_create_permutation(path, dataset_size):
    """Keep record identities stable across budgets and reject stale splits."""
    if path.exists():
        indices = torch.load(path, map_location='cpu')
        if (
            not isinstance(indices, torch.Tensor)
            or indices.dtype != torch.long
            or indices.ndim != 1
            or len(indices) != dataset_size
            or not torch.equal(indices.sort().values, torch.arange(dataset_size))
        ):
            raise ValueError(f'Invalid SVHN permutation cache: {path}')
        return indices
    indices = torch.randperm(dataset_size)
    torch.save(indices, path)
    return indices


def materialize_subset(dataset, indices, loader_batch_size=256):
    """Materialize only attack targets/design records, not the whole dataset."""
    indices = [int(index) for index in indices]
    if not indices:
        raise ValueError('Cannot materialize an empty SVHN subset.')
    loader = DataLoader(
        Subset(dataset, indices), batch_size=loader_batch_size, shuffle=False
    )
    images, labels = [], []
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels)
    return torch.cat(images), torch.cat(labels)


def get_tensor_probability_vectors(model, data, inference_batch_size=256):
    """Extract black-box posterior vectors without membership labels."""
    loader = DataLoader(
        TensorDataset(data),
        batch_size=inference_batch_size,
        shuffle=False,
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


def passive_probability_features(probability_trajectory):
    if len(probability_trajectory) < 2:
        raise ValueError('Passive deletion needs M_0 and M_1.')
    before, after = probability_trajectory[0], probability_trajectory[1]
    return torch.cat([before, after, after - before], dim=1)


def sequential_probability_features(probability_trajectory):
    if len(probability_trajectory) < 3:
        raise ValueError(
            'Sequential ADP needs M_0 and at least two deletion releases.'
        )
    deltas = [
        probability_trajectory[index]
        - probability_trajectory[index - 1]
        for index in range(1, len(probability_trajectory))
    ]
    return torch.cat([*probability_trajectory, *deltas], dim=1)


def dvia_ucd_from_probabilities(before_probabilities, after_probabilities, class_labels):
    """Transform correct-class confidence changes from cached CPU posteriors.

    Class labels identify the queried probability column; they are not
    membership labels. The same transform is used for shadows and targets.
    """
    before = np.asarray(before_probabilities, dtype=np.float64)
    after = np.asarray(after_probabilities, dtype=np.float64)
    labels = np.asarray(class_labels, dtype=np.int64)
    if before.ndim != 2 or after.shape != before.shape:
        raise ValueError('DVIA needs matching before/after probability matrices.')
    if labels.ndim != 1 or len(labels) != len(before) or len(labels) == 0:
        raise ValueError('DVIA needs one class label per non-empty probability row.')
    if np.any(labels < 0) or np.any(labels >= before.shape[1]):
        raise ValueError('DVIA class label is outside the probability columns.')
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

    This follows this script's no-target-membership-supervision setting.
    Neither target-model outputs nor target membership labels enter the fit.
    """
    member_ucd = np.asarray(member_ucd, dtype=np.float64)
    nonmember_ucd = np.asarray(nonmember_ucd, dtype=np.float64)
    for values in (member_ucd, nonmember_ucd):
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError('DVIA calibration needs non-empty finite UCD vectors.')
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
    """Apply fixed shadow-calibrated Gaussians without target membership labels."""
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


def take_balanced_auxiliary_indices(
    candidate_indices,
    labels,
    records_per_class,
    split_name,
):
    """Take a class-balanced, deterministic subset from an auxiliary pool."""
    candidate_indices = [int(index) for index in candidate_indices]
    selected = []
    selected_set = set()
    for class_index in range(num_classes):
        class_indices = [
            index
            for index in candidate_indices
            if int(labels[index].item()) == class_index
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
    """Make every candidate IN in half and OUT in half of the shadows."""
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
            in_positions = class_positions[
                permutation[:len(class_positions) // 2]
            ]
            member_mask[in_positions] = True
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

script_dir = Path(__file__).resolve().parent
results_dir = script_dir / 'results'
data_root = os.environ.get('SVHN_DATA_ROOT', '/home/wwq/Data/data/svhn')
target_model_path = script_dir / 'target_model_svhn_cnn.pth'
train_indices_path = script_dir / 'original_indices_svhn_train.pt'
test_indices_path = script_dir / 'original_indices_svhn_test.pt'
sequence_size_env = 'SVHN_UNSUPERVISED_UNLEARNING_SEQUENCE_SIZE'
#-----------------------------------------------------------------------------------


# Run every requested sequence size in its own process. This releases GPU memory
# between experiments, while the exact-retraining checkpoint cache still lets
# longer fixed sequences reuse model releases created by shorter sequences.
requested_sequence_size = os.environ.get(sequence_size_env)
summary_path = (
    results_dir /
    'Shadow_Reference_Probability_Active_ADP_'
    'Different_Unlearning_Sizes_Summary.txt'
)
if __name__ == '__main__' and requested_sequence_size is None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8') as summary_file:
        summary_file.write(f'dataset: {dataset_name}\n')
        summary_file.write(f'exact_retraining_seed: {exact_retraining_seed}\n')
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
        child_environment['PYTHONUNBUFFERED'] = '1'
        print(
            f'\n===== Running unlearning sequence size {sequence_size} =====',
            flush=True,
        )
        subprocess.run(
            [sys.executable, '-u', str(Path(__file__).resolve())],
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


# SVHN cropped digits are already RGB, 32 x 32. Use the same deterministic
# [-1, 1] pixel scaling as the reference baseline, with no digit-flipping
# augmentation. Torchvision maps SVHN's original label 10 (digit zero) to 0.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.SVHN(
    root=data_root, split='train', download=True, transform=transform
)
test_dataset = datasets.SVHN(
    root=data_root, split='test', download=True, transform=transform
)
for split_dataset in (train_dataset, test_dataset):
    labels = np.asarray(split_dataset.labels)
    if labels.ndim != 1 or not np.all((labels >= 0) & (labels < num_classes)):
        raise ValueError('Expected SVHN labels in [0, 9] after torchvision loading.')
print(
    f'SVHN root: {data_root} | train={len(train_dataset)} | '
    f'test={len(test_dataset)} | device={device} | '
    f'unlearning budget={num_unlearning_operations}',
    flush=True,
)


batch_size = 128 * 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





target_model=build_model().to(device)
if not target_model_path.exists():
    print(f'Training target CNN on SVHN for {epochs} epochs', flush=True)
    train_model_with_loader(target_model, train_loader, epochs, lr,device)
    torch.save(target_model.state_dict(), target_model_path)

else:
    print("Loading trained Target Model")
    target_model.load_state_dict(
        torch.load(target_model_path, map_location=device)
    )
    target_model.to(device)

# Calculate training and test accuracy
train_accuracy = calculate_accuracy(target_model, train_loader, device)
test_accuracy = calculate_accuracy(target_model, test_loader, device)
print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")


print("----------------------------------")
print("Probability baseline attack knowledge:")
print(
    "Attack classifier: memberships controlled only in attacker-constructed "
    "shadow/reference models"
)
print("Target-model measurement membership: evaluation metrics only")
print("DVIA: shadow IN/OUT Gaussian calibration; known class labels for UCD")
print("Active selector: disjoint auxiliary design targets and design shadows")
print("----------------------------------")


original_indices = load_or_create_permutation(
    train_indices_path, len(train_dataset)
)

# The probe pool is disjoint from measurements and probability-classifier
# training records.  Fixed Sequential uses the first k pool records, while
# Active ADP later selects k records from the complete pool.
if not unlearning_sequence_sizes:
    raise ValueError('At least one unlearning sequence size is required.')
if tuple(sorted(set(unlearning_sequence_sizes))) != unlearning_sequence_sizes:
    raise ValueError('Unlearning sequence sizes must be unique and increasing.')
if min(unlearning_sequence_sizes) < 2:
    raise ValueError('Sequential ADP requires at least two deletion operations.')
if active_probe_pool_size < max_num_unlearning_operations:
    raise ValueError('active_probe_pool_size must cover the deletion budget.')
if num_active_design_targets < 1:
    raise ValueError('Active ADP needs at least one auxiliary design target.')
if active_design_shadow_size < 2:
    raise ValueError('Active ADP needs at least two design-shadow records.')
if active_probe_pool_size + measurement_number >= len(train_dataset):
    raise ValueError('Probe/measurement splits leave no retained evaluation members.')
probe_pool_indices = [
    int(index) for index in original_indices[:active_probe_pool_size]
]
fixed_deletion_positions = list(range(num_unlearning_operations))
fixed_deletion_indices = [
    probe_pool_indices[position] for position in fixed_deletion_positions
]
measurement_train_indices = original_indices[
    active_probe_pool_size:active_probe_pool_size+measurement_number
]
measurement_train_images, measurement_train_labels = materialize_subset(
    train_dataset, measurement_train_indices
)
probe_pool_data, probe_pool_labels = materialize_subset(
    train_dataset, probe_pool_indices
)


# Exact machine unlearning: cache every ordered prefix.  Both the fixed and
# active paths are initialized and retrained from scratch on D_0 with their
# cumulative deletion prefix removed.
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
        shuffle=False,
    )
    round_index = len(prefix_key)
    round_seed = exact_retraining_seed + round_index
    prefix_tag = '_'.join(str(index) for index in prefix_key)
    # SVHN checkpoints are separate from any copied CIFAR-10 artifacts.
    target_model_after_path = script_dir / (
        f'target_model_svhn_cnn_exact_paired_'
        f'v{exact_retraining_cache_version}_round_{round_index:02d}_'
        f'seed_{round_seed}_delete_{prefix_tag}.pth'
    )

    if not target_model_after_path.exists():
        set_experiment_seed(round_seed)
        target_model_after = build_model().to(device)
        print(
            f'Retraining target release after deletion prefix '
            f'{list(prefix_key)} for {epochs} epochs with '
            f'round seed {round_seed}'
        )
        train_model_with_loader(
            target_model_after,
            retained_loader,
            training_epochs=epochs,
            lr=lr,
            device=device,
        )
        torch.save(target_model_after.state_dict(), target_model_after_path)
    else:
        target_model_after = build_model().to(device)
        print(
            f'Loading target release after deletion prefix '
            f'{list(prefix_key)}'
        )
        target_model_after.load_state_dict(
            torch.load(target_model_after_path, map_location=device)
        )
        target_model_after.to(device)

    release_accuracy = calculate_accuracy(
        target_model_after, test_loader, device
    )
    print(
        f'Test accuracy after prefix {list(prefix_key)}: '
        f'{release_accuracy:.2f}%'
    )
    target_model_cache[prefix_key] = target_model_after
    release_test_accuracy_cache[prefix_key] = release_accuracy
    return target_model_after


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


original_indices_test = load_or_create_permutation(
    test_indices_path, len(test_dataset)
)
# SVHN exposes digit labels through .labels; loading labels needs no image copy.
test_labels = torch.tensor(test_dataset.labels, dtype=torch.long)

required_test_records = (
    measurement_number
    + num_active_design_targets
    + active_design_shadow_size
    + num_classes * decoder_reference_records_per_class
    + num_classes * decoder_shadow_base_records_per_class
)
if required_test_records > len(original_indices_test):
    raise ValueError(
        'SVHN auxiliary split is too small for measurement, active '
        'design, and shadow/reference decoder pools.'
    )

measurement_test_indices = original_indices_test[:measurement_number]
design_target_start = measurement_number
design_target_end = design_target_start + num_active_design_targets
design_shadow_end = design_target_end + active_design_shadow_size
active_design_target_indices = original_indices_test[
    design_target_start:design_target_end
]
active_design_shadow_indices = original_indices_test[
    design_target_end:design_shadow_end
]
measurement_test_images, measurement_test_labels = materialize_subset(
    test_dataset, measurement_test_indices
)
active_design_target_images, active_design_target_labels = materialize_subset(
    test_dataset, active_design_target_indices
)
active_design_shadow_images, active_design_shadow_labels = materialize_subset(
    test_dataset, active_design_shadow_indices
)
missing_design_classes = sorted(
    set(active_design_target_labels.tolist())
    - set(active_design_shadow_labels.tolist())
)
if missing_design_classes:
    raise ValueError(
        'Design-shadow split has no class-matched replacements for labels: '
        f'{missing_design_classes}'
    )

# Decoder fitting uses only an auxiliary pool disjoint from evaluation and
# active probe design. Target-model membership of these records is neither
# queried nor used; only their attacker-controlled shadow memberships label
# the decoder features.
decoder_auxiliary_pool = original_indices_test[design_shadow_end:].tolist()
decoder_reference_indices, decoder_auxiliary_pool = (
    take_balanced_auxiliary_indices(
        decoder_auxiliary_pool,
        test_labels,
        decoder_reference_records_per_class,
        'Decoder reference candidate split',
    )
)
decoder_shadow_base_indices, decoder_auxiliary_pool = (
    take_balanced_auxiliary_indices(
        decoder_auxiliary_pool,
        test_labels,
        decoder_shadow_base_records_per_class,
        'Decoder shadow base split',
    )
)
decoder_reference_images, decoder_reference_labels = materialize_subset(
    test_dataset, decoder_reference_indices
)
decoder_shadow_base_images, decoder_shadow_base_labels = materialize_subset(
    test_dataset, decoder_shadow_base_indices
)
decoder_membership_masks = build_complementary_reference_masks(
    decoder_reference_labels
)
test_split_sets = {
    'measurement_nonmembers': set(measurement_test_indices.tolist()),
    'active_design_targets': set(active_design_target_indices.tolist()),
    'active_design_shadows': set(active_design_shadow_indices.tolist()),
    'decoder_reference_candidates': set(decoder_reference_indices.tolist()),
    'decoder_shadow_base': set(decoder_shadow_base_indices.tolist()),
}
test_split_names = list(test_split_sets)
for left_index, left_name in enumerate(test_split_names):
    for right_name in test_split_names[left_index + 1:]:
        overlap = test_split_sets[left_name] & test_split_sets[right_name]
        if overlap:
            raise RuntimeError(
                f'Auxiliary/evaluation split overlap between {left_name} '
                f'and {right_name}: {sorted(overlap)[:10]}'
            )
if set(probe_pool_indices) & set(measurement_train_indices.tolist()):
    raise RuntimeError('Attacker-owned probes overlap member evaluation targets.')


# --------Active probe selection (auxiliary design data only)---------------
# The original target model and measurement records are not queried here.
# Global greedy selection has a prefix property. Compute the longest shared
# schedule once, cache it with the exact split/config metadata, and use the
# requested prefix in each sequence-size experiment.
active_selection_cache_path = script_dir / (
    f'active_global_greedy_selection_svhn_pool_{active_probe_pool_size}_'
    f'max_unlearning_{max_num_unlearning_operations}.json'
)
active_selection_metadata = {
    'dataset': dataset_name,
    'data_root': str(Path(data_root).expanduser().resolve()),
    'normalization_mean': [0.5, 0.5, 0.5],
    'normalization_std': [0.5, 0.5, 0.5],
    'probe_pool_indices': probe_pool_indices,
    'active_design_target_indices': active_design_target_indices.tolist(),
    'active_design_shadow_indices': active_design_shadow_indices.tolist(),
    'max_num_unlearning_operations': max_num_unlearning_operations,
    'num_design_shadow_models': num_design_shadow_models,
    'active_selection_epochs': active_selection_epochs,
    'active_selection_lr': active_selection_lr,
    'active_selection_ridge': active_selection_ridge,
    'active_selection_seed': active_selection_seed,
    'num_classes': num_classes,
    'unlearning_method': 'retrain',
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
                f'Loading cached maximum global active schedule from '
                f'{active_selection_cache_path}'
            )
        else:
            print('Ignoring active-selection cache with mismatched metadata.')
    except (OSError, json.JSONDecodeError) as error:
        print(f'Ignoring unreadable active-selection cache: {error}')

if cached_selection is None:
    print(
        f'Computing global active schedule up to '
        f'k={max_num_unlearning_operations}; '
        f'this run evaluates k={num_unlearning_operations}.',
        flush=True,
    )
    set_experiment_seed(active_selection_seed)
    (
        max_active_deletion_positions,
        max_active_selection_diagnostics,
    ) = select_global_conditional_probe_sequence(
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
        channel=channel,
        num_classes=num_classes,
        selection_ridge=active_selection_ridge,
        device=device,
        unlearning_method='retrain',
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
fixed_utility_trajectory = [
    {
        'num_deletions': prefix_length,
        'deleted_indices': fixed_deletion_indices[:prefix_length],
        'test_accuracy': fixed_release_test_accuracies[prefix_length],
    }
    for prefix_length in range(num_unlearning_operations + 1)
]
active_utility_trajectory = [
    {
        'num_deletions': prefix_length,
        'deleted_indices': active_deletion_indices[:prefix_length],
        'test_accuracy': active_release_test_accuracies[prefix_length],
    }
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
print(f'Original model test accuracy: {test_accuracy:.2f}%')
print(f'Fixed unlearning utility trajectory: {fixed_utility_trajectory}')
print(f'Active unlearning utility trajectory: {active_utility_trajectory}')

measurement_images=torch.cat([measurement_train_images,measurement_test_images])
# DVIA uses image class labels, independently of the membership ground truth.
measurement_labels=torch.cat([measurement_train_labels,measurement_test_labels])
measurement_ref=np.array(
    [1]*len(measurement_train_images)+[0]*len(measurement_test_images)
)

print("Measurement Sample Size:",len(measurement_images))

# ----------------Shadow/reference-trained attack decoders------------------
# Exact retraining depends only on the retained set. Cache each reference
# shadow state by shadow ID and deleted probe set so fixed/active trajectories
# and different sequence-size child processes can safely reuse it.
reference_cache_metadata = {
    'decoder_cache_version': decoder_cache_version,
    'dataset': dataset_name,
    'data_root': str(Path(data_root).expanduser().resolve()),
    'model': 'SVHNCNN',
    'channel': channel,
    'num_classes': num_classes,
    'batch_size': batch_size,
    'normalization': 'mean=(0.5,0.5,0.5);std=(0.5,0.5,0.5)',
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
    / 'svhn_reference_state_cache'
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
            == (len(decoder_reference_images), num_classes)
        ):
            print(f'Loading cached reference state: {cache_path.name}')
            reference_probability_cache[state_key] = cached_probabilities
            return cached_probabilities
        print(f'Ignoring malformed reference-state cache: {cache_path}')

    membership_mask = decoder_membership_masks[shadow_index]
    deleted_position_set = set(deleted_positions)
    remaining_probe_positions = [
        position for position in range(active_probe_pool_size)
        if position not in deleted_position_set
    ]
    reference_train_images = torch.cat([
        decoder_shadow_base_images,
        decoder_reference_images[membership_mask],
        probe_pool_data[remaining_probe_positions],
    ], dim=0)
    reference_train_labels = torch.cat([
        decoder_shadow_base_labels,
        decoder_reference_labels[membership_mask],
        probe_pool_labels[remaining_probe_positions],
    ], dim=0)
    state_seed = stable_reference_state_seed(
        shadow_index, deleted_positions
    )
    set_experiment_seed(state_seed)
    loader_generator = torch.Generator().manual_seed(state_seed)
    reference_loader = DataLoader(
        TensorDataset(reference_train_images, reference_train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
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
    probabilities = get_tensor_probability_vectors(
        reference_model, decoder_reference_images
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
        'static': fixed_reference_trajectory[0],
        'passive': passive_probability_features(fixed_reference_trajectory),
        'sequential': sequential_probability_features(
            fixed_reference_trajectory
        ),
        'active': sequential_probability_features(
            active_reference_trajectory
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
reference_dvia_nonmember_ucd = np.concatenate(reference_dvia_nonmember_parts)
# Freeze calibration before querying the target measurement records.
dvia_gaussian_fit = fit_reference_dvia(
    reference_dvia_member_ucd, reference_dvia_nonmember_ucd
)

# The experimenter-only measurement membership vector is absent from the
# query, feature construction, and classifier-fitting path. It first enters
# the computation in roc_curve below.
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
    'static': fixed_target_probability_trajectory[0],
    'passive': passive_probability_features(
        fixed_target_probability_trajectory
    ),
    'sequential': sequential_probability_features(
        fixed_target_probability_trajectory
    ),
    'active': sequential_probability_features(
        active_target_probability_trajectory
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
print("--------------")
print(f'Reference-trained Static probability MIA AUC: {static_auc:.4f}')
print(f'Reference-trained Passive-deletion probability MIA AUC: {passive_auc:.4f}')
print(f'Reference-trained Sequential ADP probability MIA AUC: {sequential_auc:.4f}')
print(f'Reference-trained Global Active ADP probability MIA AUC: {active_auc:.4f}')
print(
    f'Shadow-calibrated DVIA UCD likelihood-ratio AUC: {dvia_auc:.4f} '
    f'(accuracy at likelihood-ratio threshold 1: {dvia_accuracy:.4f})'
)
print(
    f'Passive deletion gain (Delta AUC): {passive_deletion_gain:+.4f} '
    f'({100.0 * passive_deletion_gain:+.2f} percentage points)'
)
print(
    f'Sequential gain over Static (Delta AUC): '
    f'{sequential_gain_over_static:+.4f} '
    f'({100.0 * sequential_gain_over_static:+.2f} percentage points)'
)
print(
    f'Sequential gain over Passive (Delta AUC): '
    f'{sequential_gain_over_passive:+.4f} '
    f'({100.0 * sequential_gain_over_passive:+.2f} percentage points)'
)
print(
    f'Active gain over Static (Delta AUC): {active_gain_over_static:+.4f} '
    f'({100.0 * active_gain_over_static:+.2f} percentage points)'
)
print(
    f'Active gain over Passive (Delta AUC): {active_gain_over_passive:+.4f} '
    f'({100.0 * active_gain_over_passive:+.2f} percentage points)'
)
print(
    f'Active gain over fixed Sequential (Delta AUC): '
    f'{active_gain_over_sequential:+.4f} '
    f'({100.0 * active_gain_over_sequential:+.2f} percentage points)'
)
print(
    f'DVIA gain over Static (Delta AUC): {dvia_gain_over_static:+.4f} '
    f'({100.0 * dvia_gain_over_static:+.2f} percentage points)'
)
print(
    f'Active ADP gain over DVIA (Delta AUC): {active_gain_over_dvia:+.4f} '
    f'({100.0 * active_gain_over_dvia:+.2f} percentage points)'
)
print("-------------")
plt.plot(
    static_fpr,
    static_tpr,
    color='steelblue',
    lw=2,
    label='Static probability MIA (area = %0.2f)' % static_auc,
)
plt.plot(
    passive_fpr,
    passive_tpr,
    color='darkorange',
    lw=2,
    label='Passive-deletion probability MIA (area = %0.2f)' % passive_auc,
)
plt.plot(
    sequential_fpr,
    sequential_tpr,
    color='forestgreen',
    lw=2,
    label='Sequential ADP probability MIA (area = %0.2f)' % sequential_auc,
)
plt.plot(
    active_fpr,
    active_tpr,
    color='crimson',
    lw=2,
    label='Global Active ADP probability MIA (area = %0.2f)' % active_auc,
)
plt.plot(
    dvia_fpr,
    dvia_tpr,
    color='saddlebrown',
    lw=2,
    linestyle='-.',
    label='Shadow-calibrated DVIA (area = %0.2f)' % dvia_auc,
)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(
    'SVHN shadow/reference membership attacks '
    f'(unlearning size={num_unlearning_operations})'
)
plt.legend(loc="lower right")
plt.savefig(
    results_dir /
    f'ROC_Shadow_Reference_Probability_Active_ADP_Unlearning_Size_'
    f'{num_unlearning_operations}.png'
)
plt.close()


filename = (
    results_dir /
    f'Shadow_Reference_Probability_Active_ADP_Unlearning_Size_'
    f'{num_unlearning_operations}.txt'
)

hyperparams = {
    "dataset": dataset_name,
    "dataset_train_split": "train",
    "dataset_test_split": "test",
    "uses_extra_split": False,
    "label_mapping": "torchvision.SVHN maps original label 10 to digit 0",
    "normalization_mean": (0.5, 0.5, 0.5),
    "normalization_std": (0.5, 0.5, 0.5),
    "augmentation": "none",
    "exact_retraining_seed": exact_retraining_seed,
    "exact_retraining_cache_version": exact_retraining_cache_version,
    "input_shape": input_shape,
    "channel": channel,
    "num_classes": num_classes,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "epochs": epochs,
    "lr": lr,
    "measurement_number": measurement_number,
    "tested_unlearning_sequence_sizes": list(unlearning_sequence_sizes),
    "num_unlearning_operations": num_unlearning_operations,
    "active_probe_pool_size": active_probe_pool_size,
    "num_active_design_targets": num_active_design_targets,
    "active_design_shadow_size": active_design_shadow_size,
    "num_design_shadow_models": num_design_shadow_models,
    "active_selection_epochs": active_selection_epochs,
    "active_selection_lr": active_selection_lr,
    "active_selection_ridge": active_selection_ridge,
    "active_selection_seed": active_selection_seed,
    "active_selection_max_cached_budget": max_num_unlearning_operations,
    "active_selection_cache_path": str(active_selection_cache_path),
    "attack_epochs": attack_epochs,
    "attack_lr": attack_lr,
    "attack_hidden_size": attack_hidden_size,
    "attack_random_seed": attack_random_seed,
    "dvia_views": "original_and_final_fixed_exact_unlearning_release",
    "dvia_attack_feature": (
        "phi(delta)=log((1+delta)/(1-delta)); "
        "delta=p_unlearned(y|x)-p_original(y|x)"
    ),
    "dvia_distribution_model": "member_and_nonmember_univariate_gaussians",
    "dvia_distribution_calibration": (
        "pooled_shadow_reference_in_out_ucd_from_fixed_deletion_trajectories"
    ),
    "dvia_calibration_label_source": (
        "attacker_controlled_shadow_reference_memberships_only"
    ),
    "dvia_target_membership_used_for_calibration": False,
    "dvia_class_label_usage": "known_class_labels_for_correct_label_confidence",
    "dvia_score": "log_member_likelihood_minus_log_nonmember_likelihood",
    "dvia_ucd_epsilon": dvia_ucd_epsilon,
    "dvia_gaussian_variance_floor": dvia_gaussian_variance_floor,
    "dvia_likelihood_ratio_threshold": dvia_likelihood_ratio_threshold,
    "dvia_calibration_member_records": len(reference_dvia_member_ucd),
    "dvia_calibration_nonmember_records": len(reference_dvia_nonmember_ucd),
    "attack_decoder_label_source": (
        "attacker_controlled_shadow_reference_memberships_only"
    ),
    "target_model_membership_ground_truth_usage": (
        "measurement_ground_truth_is_roc_auc_accuracy_evaluation_only;_"
        "attacker_knows_only_its_owned_probe_membership"
    ),
    "decoder_reference_records_per_class": (
        decoder_reference_records_per_class
    ),
    "decoder_shadow_base_records_per_class": (
        decoder_shadow_base_records_per_class
    ),
    "num_decoder_shadow_models": num_decoder_shadow_models,
    "decoder_shadow_epochs": decoder_shadow_epochs,
    "decoder_shadow_lr": decoder_shadow_lr,
    "decoder_shadow_seed": decoder_shadow_seed,
    "decoder_cache_version": decoder_cache_version,
    "reference_cache_fingerprint": reference_cache_fingerprint,
    "reference_state_cache_dir": str(reference_state_cache_dir),
    "device": str(device),
    "unlearning_method": "exact_retraining",
    "fixed_sequence_policy": "fixed_precommitted_random_order",
    "active_sequence_policy": (
        "shared_prefix_global_greedy_mean_conditional_score"
    ),
    "active_selection_aggregation": "mean_across_design_targets",
    "probe_pool_indices": probe_pool_indices,
    "fixed_deletion_positions": fixed_deletion_positions,
    "fixed_deletion_indices": fixed_deletion_indices,
    "active_selection_diagnostics": active_selection_diagnostics,
    "active_deletion_positions": active_deletion_positions,
    "active_deletion_indices": active_deletion_indices,
    "static_attack_feature": "full_probability_vector",
    "passive_attack_feature": "before_after_and_probability_delta",
    "sequential_attack_feature": "all_posteriors_and_consecutive_deltas",
    "active_attack_feature": "all_posteriors_and_consecutive_deltas",
    "active_design_target_records": len(active_design_target_indices),
    "active_design_shadow_records": len(active_design_shadow_indices),
    "measurement_train_indices": measurement_train_indices.tolist(),
    "measurement_test_indices": measurement_test_indices.tolist(),
    "decoder_reference_indices": decoder_reference_indices.tolist(),
    "decoder_shadow_base_indices": decoder_shadow_base_indices.tolist(),
    "decoder_membership_masks": [
        mask.to(dtype=torch.int8).tolist()
        for mask in decoder_membership_masks
    ],
    "attack_member_records": len(reference_member_features['static']),
    "attack_nonmember_records": len(reference_nonmember_features['static']),
    "data_root": data_root,
    # Utility before unlearning and after every cumulative deletion prefix.
    # The legacy fields below are retained for backward compatibility.
    "original_model_test_accuracy": test_accuracy,
    "fixed_unlearning_model_test_accuracies": (
        fixed_release_test_accuracies[1:]
    ),
    "active_unlearning_model_test_accuracies": (
        active_release_test_accuracies[1:]
    ),
    "fixed_utility_trajectory": fixed_utility_trajectory,
    "active_utility_trajectory": active_utility_trajectory,
    "target_train_accuracy_before": train_accuracy,
    "target_test_accuracy_before": test_accuracy,
    "fixed_release_test_accuracies": fixed_release_test_accuracies,
    "active_release_test_accuracies": active_release_test_accuracies,
    "static_auc": static_auc,
    "passive_deletion_auc": passive_auc,
    "passive_deletion_gain_auc": passive_deletion_gain,
    "sequential_adp_auc": sequential_auc,
    "sequential_gain_over_static_auc": sequential_gain_over_static,
    "sequential_gain_over_passive_auc": sequential_gain_over_passive,
    "active_adp_auc": active_auc,
    "active_gain_over_static_auc": active_gain_over_static,
    "active_gain_over_passive_auc": active_gain_over_passive,
    "active_gain_over_sequential_auc": active_gain_over_sequential,
    "dvia_auc": dvia_auc,
    "dvia_accuracy_at_likelihood_ratio_threshold": dvia_accuracy,
    "dvia_gain_over_static_auc": dvia_gain_over_static,
    "active_gain_over_dvia_auc": active_gain_over_dvia,
    "dvia_gaussian_fit": dvia_gaussian_fit,
    "dvia_reference_member_transformed_ucd": reference_dvia_member_ucd.tolist(),
    "dvia_reference_nonmember_transformed_ucd": (
        reference_dvia_nonmember_ucd.tolist()
    ),
    # Raw attack outputs and the exact ROC coordinates used by the PNG. This
    # experimenter-only ground truth first enters at roc_curve and is saved in
    # the same sample order solely for metric reproducibility.
    "measurement_membership_ground_truth": measurement_ref.tolist(),
    "static_attack_scores": np.asarray(static_scores).tolist(),
    "static_roc_fpr": static_fpr.tolist(),
    "static_roc_tpr": static_tpr.tolist(),
    "passive_attack_scores": np.asarray(passive_scores).tolist(),
    "passive_roc_fpr": passive_fpr.tolist(),
    "passive_roc_tpr": passive_tpr.tolist(),
    "sequential_attack_scores": np.asarray(sequential_scores).tolist(),
    "sequential_roc_fpr": sequential_fpr.tolist(),
    "sequential_roc_tpr": sequential_tpr.tolist(),
    "active_attack_scores": np.asarray(active_scores).tolist(),
    "active_roc_fpr": active_fpr.tolist(),
    "active_roc_tpr": active_tpr.tolist(),
    "dvia_transformed_ucd": dvia_transformed_ucd.tolist(),
    "dvia_attack_scores": dvia_scores.tolist(),
    "dvia_roc_fpr": dvia_fpr.tolist(),
    "dvia_roc_tpr": dvia_tpr.tolist(),
}


results_dir.mkdir(parents=True, exist_ok=True)


with open(filename, 'w', encoding='utf-8') as f:
    for param_name, param_value in hyperparams.items():
        f.write(f"{param_name}: {param_value}\n")

summary_result = {
    "num_unlearning_operations": num_unlearning_operations,
    "result_file": filename.name,
    "static_auc": static_auc,
    "passive_deletion_auc": passive_auc,
    "sequential_adp_auc": sequential_auc,
    "active_adp_auc": active_auc,
    "dvia_auc": dvia_auc,
    "dvia_accuracy_at_likelihood_ratio_threshold": dvia_accuracy,
    "dvia_gain_over_static_auc": dvia_gain_over_static,
    "active_gain_over_dvia_auc": active_gain_over_dvia,
    "fixed_deletion_indices": fixed_deletion_indices,
    "active_deletion_indices": active_deletion_indices,
    "original_model_test_accuracy": test_accuracy,
    "fixed_unlearning_model_test_accuracies": (
        fixed_release_test_accuracies[1:]
    ),
    "active_unlearning_model_test_accuracies": (
        active_release_test_accuracies[1:]
    ),
    "fixed_utility_trajectory": fixed_utility_trajectory,
    "active_utility_trajectory": active_utility_trajectory,
}
with summary_path.open('a', encoding='utf-8') as summary_file:
    summary_file.write(f'result: {summary_result}\n')
