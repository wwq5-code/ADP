import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, accuracy_score
import math
import matplotlib.pyplot as plt 
import json
import os
import subprocess
import sys
from pathlib import Path
from torch.utils.data import DataLoader, Subset
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
    from .ADP_active_lira.ADP_active_lira import (
        select_conditional_probe_sequences,
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
    from ADP_active_lira.ADP_active_lira import (
        select_conditional_probe_sequences,
    )


#-----------------------------------------------------------------------------------
input_shape = (1, 28, 28)
input_size = 28 * 28
num_classes=10
feature_size = 256
model_dropout = 0.1
output_size = 10
epochs = 50
lr = 1e-3
measurement_number=1000  # measured members and non-members, respectively
unlearning_sequence_sizes=(3, 5, 7, 9, 11, 13, 15)
max_num_unlearning_operations=max(unlearning_sequence_sizes)
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
gpu = 0
device = torch.device(
    'cuda:{}'.format(gpu)
    if torch.cuda.is_available() and gpu != -1
    else 'cpu'
)


def build_model():
    return CNN(
        num_classes=num_classes,
        feature_size=feature_size,
        dropout=model_dropout,
    )


def set_experiment_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


script_dir = Path(__file__).resolve().parent
results_dir = script_dir / 'results'
data_root = os.environ.get(
    'FASHION_MNIST_DATA_ROOT', '/home/wwq/Data/data/fashion_mnist'
)
target_model_path = script_dir / 'target_model_fmnist_cnn.pth'
train_indices_path = script_dir / 'original_indices_fmnist.pt'
test_indices_path = script_dir / 'original_indices_test_fmnist.pt'
sequence_size_env = 'FMNIST_UNLEARNING_SEQUENCE_SIZE'
#-----------------------------------------------------------------------------------


# Run each sequence-size experiment in an isolated process so GPU memory is
# released between budgets. Exact-retraining checkpoints are shared by prefix.
requested_sequence_size = os.environ.get(sequence_size_env)
summary_path = (
    results_dir /
    'Baseline_CNN_Probability_Active_ADP_'
    'Different_Unlearning_Sizes_Summary.txt'
)
if __name__ == '__main__' and requested_sequence_size is None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8') as summary_file:
        summary_file.write(
            f'unlearning_sequence_sizes: {list(unlearning_sequence_sizes)}\n'
        )
        summary_file.write(f'active_probe_pool_size: {active_probe_pool_size}\n')

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


# Load Fashion-MNIST dataset. These statistics are computed over its training set.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

train_dataset = datasets.FashionMNIST(
    root=data_root, train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root=data_root, train=False, download=True, transform=transform
)


batch_size = 128 * 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





target_model=build_model().to(device)
if not target_model_path.exists():
    print(f"Training Fashion-MNIST CNN on Epochs: {epochs}")
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
print("Active selector: disjoint auxiliary design targets and design shadows")
print("----------------------------------")


train_images=[]
train_labels=[]
for images, labels in train_loader:
    train_images.append(images)
    train_labels.append(labels)
train_images=torch.cat(train_images)
train_labels=torch.cat(train_labels)

if not train_indices_path.exists():
    original_indices=torch.randperm(len(train_images))
    torch.save(original_indices, train_indices_path)
else:
    original_indices=torch.load(train_indices_path, map_location='cpu')

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
measurement_train_images = train_images[measurement_train_indices]
measurement_train_labels = train_labels[measurement_train_indices]
probe_pool_data = train_images[probe_pool_indices]
probe_pool_labels = train_labels[probe_pool_indices]


# Exact machine unlearning: cache models and accuracies by ordered deletion
# prefix.  Every non-empty prefix is initialized and retrained from scratch.
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
    target_model_after = build_model().to(device)
    if len(prefix_key) == 1:
        target_model_after_path = (
            script_dir /
            f'target_model_fmnist_cnn_retrain_delete_{prefix_key[0]}.pth'
        )
    else:
        prefix_tag = '_'.join(str(index) for index in prefix_key)
        target_model_after_path = (
            script_dir /
            f'target_model_fmnist_cnn_retrain_delete_sequence_{prefix_tag}.pth'
        )

    if not target_model_after_path.exists():
        print(
            f'Retraining target release after deletion prefix '
            f'{list(prefix_key)} for {epochs} epochs'
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
        print(f'Loading target release after prefix {list(prefix_key)}')
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


test_images=[]
test_labels=[]
for images, labels in test_loader:
    test_images.append(images)
    test_labels.append(labels)
test_images=torch.cat(test_images)
test_labels=torch.cat(test_labels)

if not test_indices_path.exists():
    original_indices_test=torch.randperm(len(test_images))
    torch.save(original_indices_test, test_indices_path)
else:
    original_indices_test=torch.load(test_indices_path, map_location='cpu')

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
measurement_test_images = test_images[measurement_test_indices]
measurement_test_labels = test_labels[measurement_test_indices]
active_design_target_images = test_images[active_design_target_indices]
active_design_target_labels = test_labels[active_design_target_indices]
active_design_shadow_images = test_images[active_design_shadow_indices]
active_design_shadow_labels = test_labels[active_design_shadow_indices]
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
# Greedy conditional selection has a prefix property. Compute the longest
# per-target schedules once, validate/cache them with the exact experiment
# metadata, and use the requested prefix for each sequence-size experiment.
active_selection_cache_path = script_dir / (
    f'active_design_selection_fmnist_cnn_pool_{active_probe_pool_size}_'
    f'max_unlearning_{max_num_unlearning_operations}.json'
)
active_selection_metadata = {
    'probe_pool_indices': probe_pool_indices,
    'active_design_target_indices': active_design_target_indices.tolist(),
    'active_design_shadow_indices': active_design_shadow_indices.tolist(),
    'max_num_unlearning_operations': max_num_unlearning_operations,
    'num_design_shadow_models': num_design_shadow_models,
    'active_selection_epochs': active_selection_epochs,
    'active_selection_lr': active_selection_lr,
    'active_selection_ridge': active_selection_ridge,
    'active_selection_seed': active_selection_seed,
    'model': 'SmallCNN',
    'feature_size': feature_size,
    'model_dropout': model_dropout,
    'num_classes': num_classes,
    'unlearning_method': 'retrain',
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
                f'Loading cached maximum active schedules from '
                f'{active_selection_cache_path}'
            )
        else:
            print('Ignoring active-selection cache with mismatched metadata.')
    except (OSError, json.JSONDecodeError) as error:
        print(f'Ignoring unreadable active-selection cache: {error}')

if cached_selection is None:
    set_experiment_seed(active_selection_seed)
    max_design_schedules, max_active_selection_diagnostics = (
        select_conditional_probe_sequences(
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
            input_size=input_size,
            hidden_size=feature_size,
            num_classes=num_classes,
            selection_ridge=active_selection_ridge,
            device=device,
            unlearning_method='retrain',
        )
    )
    cached_selection = {
        'metadata': active_selection_metadata,
        'design_schedules': max_design_schedules,
        'active_selection_diagnostics': max_active_selection_diagnostics,
    }
    with active_selection_cache_path.open(
        'w', encoding='utf-8'
    ) as cache_file:
        json.dump(cached_selection, cache_file)
else:
    max_design_schedules = cached_selection['design_schedules']
    max_active_selection_diagnostics = cached_selection[
        'active_selection_diagnostics'
    ]

design_schedules = [
    schedule[:num_unlearning_operations]
    for schedule in max_design_schedules
]
active_selection_diagnostics = [
    diagnostics[:num_unlearning_operations]
    for diagnostics in max_active_selection_diagnostics
]

active_rank_scores = np.zeros(active_probe_pool_size, dtype=np.float64)
for schedule in design_schedules:
    for rank, probe_position in enumerate(schedule):
        active_rank_scores[probe_position] += num_unlearning_operations - rank
active_deletion_positions = sorted(
    range(active_probe_pool_size),
    key=lambda position: (-active_rank_scores[position], position),
)[:num_unlearning_operations]
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
print(f'Active design schedules (probe positions): {design_schedules}')
print(f'Active Borda scores: {active_rank_scores.tolist()}')
print(f'Global Active ADP deletion sequence: {active_deletion_indices}')

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

static_fpr, static_tpr, _ = roc_curve(measurement_ref, static_scores)
passive_fpr, passive_tpr, _ = roc_curve(measurement_ref, passive_scores)
sequential_fpr, sequential_tpr, _ = roc_curve(
    measurement_ref, sequential_scores
)
active_fpr, active_tpr, _ = roc_curve(measurement_ref, active_scores)
static_auc = auc(static_fpr, static_tpr)
passive_auc = auc(passive_fpr, passive_tpr)
sequential_auc = auc(sequential_fpr, sequential_tpr)
active_auc = auc(active_fpr, active_tpr)
passive_deletion_gain = passive_auc - static_auc
sequential_gain_over_static = sequential_auc - static_auc
sequential_gain_over_passive = sequential_auc - passive_auc
active_gain_over_static = active_auc - static_auc
active_gain_over_passive = active_auc - passive_auc
active_gain_over_sequential = active_auc - sequential_auc
print("--------------")
print(f'Static probability MIA AUC: {static_auc:.4f}')
print(f'Passive-deletion probability MIA AUC: {passive_auc:.4f}')
print(f'Sequential ADP probability MIA AUC: {sequential_auc:.4f}')
print(f'Global Active ADP probability MIA AUC: {active_auc:.4f}')
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
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(
    'Fashion-MNIST CNN Static, Passive, Sequential, and Global Active MIA '
    f'(unlearning size={num_unlearning_operations})'
)
plt.legend(loc="lower right")
plt.savefig(
    results_dir /
    f'ROC_Baseline_CNN_Probability_Active_ADP_Unlearning_Size_'
    f'{num_unlearning_operations}.png'
)
plt.close()


filename = (
    results_dir /
    f'Baseline_CNN_Probability_Active_ADP_Unlearning_Size_'
    f'{num_unlearning_operations}.txt'
)

hyperparams = {
    "dataset": "FashionMNIST",
    "input_shape": input_shape,
    "input_size": input_size,
    "num_classes": num_classes,
    "model": "SmallCNN",
    "feature_size": feature_size,
    "model_dropout": model_dropout,
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
    "device": str(device),
    "unlearning_method": "exact_retraining",
    "fixed_sequence_policy": "fixed_precommitted_random_order",
    "active_sequence_policy": (
        "design_shadow_conditional_selection_with_global_borda_aggregation"
    ),
    "probe_pool_indices": probe_pool_indices,
    "fixed_deletion_positions": fixed_deletion_positions,
    "fixed_deletion_indices": fixed_deletion_indices,
    "active_design_schedules": design_schedules,
    "active_selection_diagnostics": active_selection_diagnostics,
    "active_rank_scores": active_rank_scores.tolist(),
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
    "fixed_deletion_indices": fixed_deletion_indices,
    "active_deletion_indices": active_deletion_indices,
}
with summary_path.open('a', encoding='utf-8') as summary_file:
    summary_file.write(f'result: {summary_result}\n')
