"""Supervised SVHN membership attacks across sequential SalUn budgets.

SVHN_DATA_ROOT selects the official train/test .mat directory. Initial target
weights and record permutations are shared with the SVHN retraining baseline;
SalUn releases and the global active schedule use method-specific caches.
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, accuracy_score
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

if __package__:
    from .baseline.baseline_prob import (
        ActiveADP,
        MIA,
        PassiveMIA,
        SequentialADP,
        calculate_accuracy,
        train_model_with_loader,
    )
    from .baseline.model import CNN
    from .baseline.active_selection import (
        select_global_conditional_probe_sequence,
    )
    from .salun_utils import (
        generate_salun_weight_mask,
        salun_unlearn,
        set_random_seed,
    )
else:
    from baseline.baseline_prob import (
        ActiveADP,
        MIA,
        PassiveMIA,
        SequentialADP,
        calculate_accuracy,
        train_model_with_loader,
    )
    from baseline.model import CNN
    from baseline.active_selection import (
        select_global_conditional_probe_sequence,
    )
    from salun_utils import (
        generate_salun_weight_mask,
        salun_unlearn,
        set_random_seed,
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
salun_mask_ratio=0.5
salun_lr=lr
salun_unlearning_epochs=40
salun_seed=0
attack_epochs=30
attack_lr=1e-3
attack_hidden_size=128
attack_random_seed=1000
dvia_ucd_epsilon=1e-6
dvia_gaussian_variance_floor=1e-8
dvia_likelihood_ratio_threshold=1.0
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


@torch.no_grad()
def collect_dvia_transformed_ucd(original_model, unlearned_model, data_loader):
    """Collect DVIA's transformed correct-label confidence difference."""
    original_model.eval()
    unlearned_model.eval()
    transformed_ucd = []
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        original_probabilities = F.softmax(original_model(images), dim=1)
        unlearned_probabilities = F.softmax(unlearned_model(images), dim=1)
        label_column = labels.view(-1, 1)
        original_confidence = original_probabilities.gather(
            1, label_column
        ).squeeze(1)
        unlearned_confidence = unlearned_probabilities.gather(
            1, label_column
        ).squeeze(1)
        ucd = (unlearned_confidence - original_confidence).clamp(
            min=-1.0 + dvia_ucd_epsilon,
            max=1.0 - dvia_ucd_epsilon,
        )
        transformed_ucd.append(
            (torch.log1p(ucd) - torch.log1p(-ucd)).cpu()
        )

    if not transformed_ucd:
        raise ValueError('DVIA received an empty data loader.')
    return torch.cat(transformed_ucd).numpy().astype(np.float64, copy=False)


def gaussian_log_likelihood(values, mean, variance):
    values = np.asarray(values, dtype=np.float64)
    variance = max(float(variance), dvia_gaussian_variance_floor)
    return -0.5 * (
        np.log(2.0 * np.pi * variance)
        + np.square(values - float(mean)) / variance
    )


def DVIA(
    original_model,
    unlearned_model,
    calibration_member_loader,
    calibration_nonmember_loader,
    measurement_loader,
):
    """Dual-view UCD attack with Gaussian likelihood-ratio inference.

    To match the supervised baseline comparison in this script, the two
    group-level Gaussians are fitted on the same disjoint known-membership
    calibration records used by the other attacks.  Evaluation records remain
    held out, and the target is never part of the deletion pool.
    """
    member_ucd = collect_dvia_transformed_ucd(
        original_model, unlearned_model, calibration_member_loader
    )
    nonmember_ucd = collect_dvia_transformed_ucd(
        original_model, unlearned_model, calibration_nonmember_loader
    )
    member_mean = float(np.mean(member_ucd))
    nonmember_mean = float(np.mean(nonmember_ucd))
    member_variance = max(
        float(np.var(member_ucd)), dvia_gaussian_variance_floor
    )
    nonmember_variance = max(
        float(np.var(nonmember_ucd)), dvia_gaussian_variance_floor
    )

    measurement_ucd = collect_dvia_transformed_ucd(
        original_model, unlearned_model, measurement_loader
    )
    log_likelihood_ratio = gaussian_log_likelihood(
        measurement_ucd, member_mean, member_variance
    ) - gaussian_log_likelihood(
        measurement_ucd, nonmember_mean, nonmember_variance
    )
    fit = {
        'member_transformed_ucd_mean': member_mean,
        'member_transformed_ucd_variance': member_variance,
        'nonmember_transformed_ucd_mean': nonmember_mean,
        'nonmember_transformed_ucd_variance': nonmember_variance,
    }
    return log_likelihood_ratio, measurement_ucd, fit


script_dir = Path(__file__).resolve().parent
results_dir = script_dir / 'results'
data_root = os.environ.get('SVHN_DATA_ROOT', '/home/wwq/Data/data/svhn')
target_model_path = script_dir / 'target_model_svhn_cnn.pth'
train_indices_path = script_dir / 'original_indices_svhn_train.pt'
test_indices_path = script_dir / 'original_indices_svhn_test.pt'
sequence_size_env = 'SVHN_SALUN_UNLEARNING_SEQUENCE_SIZE'
#-----------------------------------------------------------------------------------


# Run every requested sequence size in its own process. This releases GPU memory
# between experiments, while path-dependent SalUn checkpoints let longer fixed
# sequences reuse model releases created by shorter sequences.
requested_sequence_size = os.environ.get(sequence_size_env)
summary_path = (
    results_dir /
    'Baseline_SalUn_Probability_Active_ADP_'
    'Different_Unlearning_Sizes_Summary.txt'
)
if __name__ == '__main__' and requested_sequence_size is None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8') as summary_file:
        summary_file.write(f'dataset: {dataset_name}\n')
        summary_file.write(
            f'unlearning_sequence_sizes: {list(unlearning_sequence_sizes)}\n'
        )
        summary_file.write(
            'attacks: static, passive_deletion, sequential_adp, '
            'active_adp, dvia\n'
        )
        summary_file.write(f'active_probe_pool_size: {active_probe_pool_size}\n')
        summary_file.write(f'salun_mask_ratio: {salun_mask_ratio}\n')
        summary_file.write(f'salun_lr: {salun_lr}\n')
        summary_file.write(
            f'salun_unlearning_epochs: {salun_unlearning_epochs}\n'
        )
        summary_file.write(f'salun_seed: {salun_seed}\n')

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

if not 0.0 <= salun_mask_ratio <= 1.0:
    raise ValueError('salun_mask_ratio must lie in [0, 1].')
if salun_lr <= 0.0:
    raise ValueError('salun_lr must be positive.')
if salun_unlearning_epochs < 1:
    raise ValueError('salun_unlearning_epochs must be at least 1.')

results_dir.mkdir(parents=True, exist_ok=True)


# Match the SVHN retraining baseline: RGB 32 x 32, deterministic [-1, 1]
# scaling, no horizontal flips, and no extra split. Torchvision maps digit
# zero from the original SVHN label 10 to the PyTorch class index 0.
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
    f'SalUn budget={num_unlearning_operations}',
    flush=True,
)


batch_size = 128 * 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
salun_args = SimpleNamespace(
    device=device,
    num_classes=num_classes,
    salun_mask_ratio=salun_mask_ratio,
    salun_lr=salun_lr,
    unlearning_epochs=salun_unlearning_epochs,
    batch_size=batch_size,
)





set_random_seed(salun_seed)
target_model=CNN(channel, num_classes).to(device)
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
print("Attack classifier: disjoint target-train/test records with known membership")
print("DVIA: the same disjoint records calibrate its UCD Gaussian distributions")
print("Active selector: disjoint auxiliary design targets and design shadows")
print("----------------------------------")


if not train_indices_path.exists():
    set_random_seed(salun_seed + 101)
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
    raise ValueError('Probe/measurement splits leave no attack-training members.')
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
attack_train_indices = original_indices[
    active_probe_pool_size+measurement_number:
]
measurement_train_images, measurement_train_labels = materialize_subset(
    train_dataset, measurement_train_indices
)
probe_pool_data, probe_pool_labels = materialize_subset(
    train_dataset, probe_pool_indices
)


# Every released model applies one SalUn update to its immediate predecessor.
# Cache by the ordered deletion prefix because SalUn is path-dependent.
target_model_cache = {(): target_model}
release_test_accuracy_cache = {(): test_accuracy}
target_release_diagnostics = {}


def salun_seed_for_prefix(prefix_key):
    seed = int(salun_seed) % (2 ** 31 - 1)
    for deletion_index in prefix_key:
        seed = (
            seed * 1000003 + int(deletion_index) + 9176
        ) % (2 ** 31 - 1)
    return seed


def salun_checkpoint_tag():
    mask_tag = str(salun_mask_ratio).replace('.', 'p')
    lr_tag = str(salun_lr).replace('.', 'p')
    return (
        f'mask_{mask_tag}_epochs_{salun_unlearning_epochs}_lr_{lr_tag}_'
        f'seed_{salun_seed}'
    )


def load_or_apply_salun_target_release(deleted_prefix):
    prefix_key = tuple(int(index) for index in deleted_prefix)
    if prefix_key in target_model_cache:
        return target_model_cache[prefix_key]
    if not prefix_key:
        return target_model
    if len(set(prefix_key)) != len(prefix_key):
        raise ValueError('A SalUn deletion trajectory cannot repeat a record.')
    if min(prefix_key) < 0 or max(prefix_key) >= len(train_dataset):
        raise ValueError('SalUn deletion index is outside the training dataset.')

    prefix_tag = '_'.join(str(index) for index in prefix_key)
    target_model_after_path = script_dir / (
        f'target_model_svhn_cnn_probability_baseline_salun_'
        f'{salun_checkpoint_tag()}_delete_sequence_{prefix_tag}.pth'
    )
    target_model_after = CNN(channel, num_classes).to(device)
    if target_model_after_path.exists():
        print(f'Loading SalUn target release: {target_model_after_path}')
        target_model_after.load_state_dict(
            torch.load(target_model_after_path, map_location=device)
        )
        target_model_after.eval()
        target_release_diagnostics[prefix_key] = {
            'deleted_index': int(prefix_key[-1]),
            'checkpoint_path': str(target_model_after_path),
            'loaded_from_checkpoint': True,
        }
    else:
        parent_model = load_or_apply_salun_target_release(prefix_key[:-1])
        target_model_after = copy.deepcopy(parent_model).to(device)
        deleted_index = int(prefix_key[-1])
        deleted_set = set(prefix_key)
        retained_indices = [
            index for index in range(len(train_dataset))
            if index not in deleted_set
        ]
        forget_loader = DataLoader(
            Subset(train_dataset, [deleted_index]),
            batch_size=1,
            shuffle=False,
        )
        retained_loader = DataLoader(
            Subset(train_dataset, retained_indices),
            batch_size=batch_size,
            shuffle=True,
        )
        request_seed = salun_seed_for_prefix(prefix_key)
        set_random_seed(request_seed)
        print(
            f'Applying SalUn target release after deletion prefix '
            f'{list(prefix_key)} for {salun_unlearning_epochs} epochs',
            flush=True,
        )
        loss_fn = nn.CrossEntropyLoss()
        mask = generate_salun_weight_mask(
            model=target_model_after,
            forget_loader=forget_loader,
            loss_fn=loss_fn,
            args=salun_args,
            verbose=True,
        )
        selected_parameters = sum(
            int(mask_tensor.sum().item()) for mask_tensor in mask.values()
        )
        total_parameters = sum(
            mask_tensor.numel() for mask_tensor in mask.values()
        )
        salun_unlearn(
            forget_loader=forget_loader,
            retain_loader=retained_loader,
            model=target_model_after,
            loss_fn=loss_fn,
            args=salun_args,
            mask=mask,
            verbose=True,
        )
        target_model_after.eval()
        torch.save(target_model_after.state_dict(), target_model_after_path)
        target_release_diagnostics[prefix_key] = {
            'deleted_index': deleted_index,
            'request_seed': request_seed,
            'checkpoint_path': str(target_model_after_path),
            'selected_mask_parameters': selected_parameters,
            'total_mask_parameters': total_parameters,
            'effective_mask_ratio': (
                selected_parameters / max(total_parameters, 1)
            ),
            'loaded_from_checkpoint': False,
        }

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
    target_models.append(load_or_apply_salun_target_release(fixed_prefix))

fixed_release_test_accuracies = [
    release_test_accuracy_cache[tuple(fixed_deletion_indices[:prefix_length])]
    for prefix_length in range(num_unlearning_operations + 1)
]
print(f'Attacker-owned probe pool: {probe_pool_indices}')
print(f'Fixed Sequential deletion sequence: {fixed_deletion_indices}')


if not test_indices_path.exists():
    set_random_seed(salun_seed + 102)
original_indices_test = load_or_create_permutation(
    test_indices_path, len(test_dataset)
)

required_test_records = (
    measurement_number
    + num_active_design_targets
    + active_design_shadow_size
)
if required_test_records >= len(original_indices_test):
    raise ValueError(
        'measurement/design splits leave no non-members for the probability '
        'attack classifier.'
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
attack_test_indices = original_indices_test[design_shadow_end:]
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


# --------Active probe selection (auxiliary design data only)---------------
# The original target model and measurement records are not queried here.
# Global greedy selection has a prefix property. Compute the longest shared
# schedule once, cache it with the exact split/config metadata, and use the
# requested prefix in each sequence-size experiment.
active_selection_cache_path = script_dir / (
    f'active_global_greedy_selection_svhn_salun_'
    f'{salun_checkpoint_tag()}_'
    f'pool_{active_probe_pool_size}_'
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
    'unlearning_method': 'salun',
    'salun_mask_ratio': salun_mask_ratio,
    'salun_lr': salun_lr,
    'salun_unlearning_epochs': salun_unlearning_epochs,
    'salun_batch_size': batch_size,
    'salun_seed': salun_seed,
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
        f'Computing SalUn global active schedule up to '
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
        unlearning_method='salun',
        salun_mask_ratio=salun_mask_ratio,
        salun_lr=salun_lr,
        salun_unlearning_epochs=salun_unlearning_epochs,
        salun_batch_size=batch_size,
        salun_seed=salun_seed,
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
    active_target_models.append(
        load_or_apply_salun_target_release(active_prefix)
    )
active_release_test_accuracies = [
    release_test_accuracy_cache[tuple(active_deletion_indices[:prefix_length])]
    for prefix_length in range(num_unlearning_operations + 1)
]
fixed_release_diagnostics = [
    {
        'num_deletions': prefix_length,
        'deleted_indices': fixed_deletion_indices[:prefix_length],
        'test_accuracy': fixed_release_test_accuracies[prefix_length],
        'salun_diagnostics': (
            None if prefix_length == 0 else target_release_diagnostics.get(
                tuple(fixed_deletion_indices[:prefix_length])
            )
        ),
    }
    for prefix_length in range(num_unlearning_operations + 1)
]
active_release_diagnostics = [
    {
        'num_deletions': prefix_length,
        'deleted_indices': active_deletion_indices[:prefix_length],
        'test_accuracy': active_release_test_accuracies[prefix_length],
        'salun_diagnostics': (
            None if prefix_length == 0 else target_release_diagnostics.get(
                tuple(active_deletion_indices[:prefix_length])
            )
        ),
    }
    for prefix_length in range(num_unlearning_operations + 1)
]
fixed_utility_trajectory = fixed_release_diagnostics
active_utility_trajectory = active_release_diagnostics
selected_mean_scores = [
    round_info['mean_conditional_score']
    for round_info in active_selection_diagnostics
]
print(
    'Global Active ADP selected mean conditional scores: '
    f'{selected_mean_scores}'
)
print(f'Global Active ADP deletion sequence: {active_deletion_indices}')
print(f'Fixed SalUn utility trajectory: {fixed_release_diagnostics}')
print(f'Active SalUn utility trajectory: {active_release_diagnostics}')

# Exclude evaluation targets from the attack classifier's member/non-member
# training data, exclude all active-selection data, and balance both classes.
attack_sample_count = min(len(attack_train_indices), len(attack_test_indices))
attack_train_indices = attack_train_indices[:attack_sample_count]
attack_test_indices = attack_test_indices[:attack_sample_count]
attack_member_loader = DataLoader(
    Subset(train_dataset, attack_train_indices.tolist()),
    batch_size=batch_size,
    shuffle=False,
)
attack_nonmember_loader = DataLoader(
    Subset(test_dataset, attack_test_indices.tolist()),
    batch_size=batch_size,
    shuffle=False,
)


measurement_images=torch.cat([measurement_train_images,measurement_test_images])
measurement_ref=np.array(
    [1]*len(measurement_train_images)+[0]*len(measurement_test_images)
)
measurement_labels=torch.cat([measurement_train_labels,measurement_test_labels])
measurement_loader = DataLoader(
    TensorDataset(measurement_images, measurement_labels),
    batch_size=batch_size,
    shuffle=False,
)

print("Measurement Sample Size:",len(measurement_images))


set_experiment_seed(attack_random_seed)
static_scores = MIA(
    target_model,
    attack_member_loader,
    attack_nonmember_loader,
    measurement_images,
    measurement_labels,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device,
)
set_experiment_seed(attack_random_seed + 1)
passive_scores = PassiveMIA(
    target_models[0],
    target_models[1],
    attack_member_loader,
    attack_nonmember_loader,
    measurement_images,
    measurement_labels,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device,
)
set_experiment_seed(attack_random_seed + 2)
sequential_scores = SequentialADP(
    target_models,
    attack_member_loader,
    attack_nonmember_loader,
    measurement_images,
    measurement_labels,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device,
)
set_experiment_seed(attack_random_seed + 3)
active_scores = ActiveADP(
    active_target_models,
    attack_member_loader,
    attack_nonmember_loader,
    measurement_images,
    measurement_labels,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device,
)
dvia_scores, dvia_transformed_ucd, dvia_gaussian_fit = DVIA(
    target_models[0],
    target_models[-1],
    attack_member_loader,
    attack_nonmember_loader,
    measurement_loader,
)

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
print(f'Static probability MIA AUC: {static_auc:.4f}')
print(f'Passive-deletion SalUn probability MIA AUC: {passive_auc:.4f}')
print(f'Sequential ADP SalUn probability MIA AUC: {sequential_auc:.4f}')
print(f'Global Active ADP SalUn probability MIA AUC: {active_auc:.4f}')
print(
    f'DVIA SalUn UCD likelihood-ratio AUC: {dvia_auc:.4f} '
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
    label='Passive-deletion SalUn probability MIA (area = %0.2f)' % passive_auc,
)
plt.plot(
    sequential_fpr,
    sequential_tpr,
    color='forestgreen',
    lw=2,
    label='Sequential ADP SalUn probability MIA (area = %0.2f)' % sequential_auc,
)
plt.plot(
    active_fpr,
    active_tpr,
    color='crimson',
    lw=2,
    label='Global Active ADP SalUn probability MIA (area = %0.2f)' % active_auc,
)
plt.plot(
    dvia_fpr,
    dvia_tpr,
    color='saddlebrown',
    lw=2,
    linestyle='-.',
    label='DVIA SalUn UCD likelihood ratio (area = %0.2f)' % dvia_auc,
)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(
    'SVHN SalUn membership attacks '
    f'(unlearning size={num_unlearning_operations})'
)
plt.legend(loc="lower right")
plot_path = results_dir / (
    f'ROC_Baseline_SalUn_Probability_Active_ADP_Unlearning_Size_'
    f'{num_unlearning_operations}.png'
)
plt.savefig(plot_path)
plt.close()


filename = (
    results_dir /
    f'Baseline_SalUn_Probability_Active_ADP_Unlearning_Size_'
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
    "input_shape": input_shape,
    "channel": channel,
    "num_classes": num_classes,
    "model": "CNN",
    "hidden_size": hidden_size,
    "output_size": output_size,
    "epochs": epochs,
    "lr": lr,
    "batch_size": batch_size,
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
    "salun_mask_ratio": salun_mask_ratio,
    "salun_lr": salun_lr,
    "salun_unlearning_epochs": salun_unlearning_epochs,
    "salun_batch_size": batch_size,
    "salun_seed": salun_seed,
    "salun_checkpoint_tag": salun_checkpoint_tag(),
    "attack_epochs": attack_epochs,
    "attack_lr": attack_lr,
    "attack_hidden_size": attack_hidden_size,
    "attack_random_seed": attack_random_seed,
    "dvia_views": "original_and_final_fixed_salun_unlearning_release",
    "dvia_attack_feature": (
        "phi(delta)=log((1+delta)/(1-delta)); "
        "delta=p_unlearned(y|x)-p_original(y|x)"
    ),
    "dvia_distribution_model": (
        "member_and_nonmember_univariate_gaussians"
    ),
    "dvia_distribution_calibration": (
        "same_disjoint_known_membership_records_as_other_attacks"
    ),
    "dvia_score": "log_member_likelihood_minus_log_nonmember_likelihood",
    "dvia_ucd_epsilon": dvia_ucd_epsilon,
    "dvia_gaussian_variance_floor": dvia_gaussian_variance_floor,
    "dvia_likelihood_ratio_threshold": dvia_likelihood_ratio_threshold,
    "device": str(device),
    "unlearning_method": "salun_sequential_update_from_parent_release",
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
    "attack_member_records": len(attack_member_loader.dataset),
    "attack_nonmember_records": len(attack_nonmember_loader.dataset),
    "data_root": data_root,
    "target_train_accuracy": train_accuracy,
    "original_model_test_accuracy": test_accuracy,
    "fixed_unlearning_model_test_accuracies": (
        fixed_release_test_accuracies[1:]
    ),
    "active_unlearning_model_test_accuracies": (
        active_release_test_accuracies[1:]
    ),
    "fixed_utility_trajectory": fixed_utility_trajectory,
    "active_utility_trajectory": active_utility_trajectory,
    "target_test_accuracy_before": test_accuracy,
    "fixed_release_test_accuracies": fixed_release_test_accuracies,
    "active_release_test_accuracies": active_release_test_accuracies,
    "target_release_diagnostics": {
        ','.join(str(index) for index in prefix): diagnostic
        for prefix, diagnostic in target_release_diagnostics.items()
    },
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
    # Raw attack outputs and the exact ROC coordinates used by the PNG.
    # The ground truth is saved in the same sample order as every score array.
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
    "dvia_transformed_ucd": np.asarray(dvia_transformed_ucd).tolist(),
    "dvia_attack_scores": np.asarray(dvia_scores).tolist(),
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
    "plot_file": plot_path.name,
    "unlearning_method": "salun_sequential_update_from_parent_release",
    "salun_mask_ratio": salun_mask_ratio,
    "salun_lr": salun_lr,
    "salun_unlearning_epochs": salun_unlearning_epochs,
    "salun_seed": salun_seed,
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
