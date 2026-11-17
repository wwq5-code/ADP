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
import os
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
    from .baseline.model import MLP
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
    from baseline.model import MLP
    from ADP_active_lira.ADP_active_lira import (
        select_conditional_probe_sequences,
    )


#-----------------------------------------------------------------------------------
input_shape = (1, 28, 28)
input_size = 28 * 28
num_classes=10
hidden_size = 256
output_size = 10
epochs = 50
lr = 1e-3
measurement_number=1000  # measured members and non-members, respectively
num_unlearning_operations=3
active_probe_pool_size=5
num_active_design_targets=10
active_design_shadow_size=1000
num_design_shadow_models=2
active_selection_epochs=5
active_selection_lr=1e-3
active_selection_ridge=1e-4
attack_epochs=30
attack_lr=1e-3
attack_hidden_size=128
gpu = 0
device = torch.device(
    'cuda:{}'.format(gpu)
    if torch.cuda.is_available() and gpu != -1
    else 'cpu'
)

script_dir = Path(__file__).resolve().parent
results_dir = script_dir / 'results'
data_root = os.environ.get('MNIST_DATA_ROOT', '/home/wwq/Data/data/mnist')
target_model_path = script_dir / 'target_model_mnist_mlp.pth'
train_indices_path = script_dir / 'original_indices_mnist.pt'
test_indices_path = script_dir / 'original_indices_test_mnist.pt'
#-----------------------------------------------------------------------------------

results_dir.mkdir(parents=True, exist_ok=True)


# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(
    root=data_root, train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root=data_root, train=False, download=True, transform=transform
)


batch_size = 128 * 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





target_model=MLP(input_size, hidden_size, num_classes).to(device)
if not target_model_path.exists():
    print(f"Training Target Model on MNIST on Epochs: {epochs}")
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

if num_unlearning_operations < 2:
    raise ValueError('Sequential ADP requires at least two deletion operations.')
if active_probe_pool_size < num_unlearning_operations:
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
    target_model_after = MLP(input_size, hidden_size, num_classes).to(device)
    if len(prefix_key) == 1:
        target_model_after_path = (
            script_dir /
            f'target_model_mnist_mlp_retrain_delete_{prefix_key[0]}.pth'
        )
    else:
        prefix_tag = '_'.join(str(index) for index in prefix_key)
        target_model_after_path = (
            script_dir /
            f'target_model_mnist_mlp_retrain_delete_sequence_{prefix_tag}.pth'
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
# Per-design-target schedules are selected without querying the original
# target model or any measurement membership, then aggregated into one global
# schedule so the target-release cost does not scale with measurement_number.
design_schedules, active_selection_diagnostics = (
    select_conditional_probe_sequences(
        active_design_target_images,
        active_design_target_labels,
        active_design_shadow_images,
        active_design_shadow_labels,
        probe_pool_data,
        probe_pool_labels,
        num_unlearning_operations=num_unlearning_operations,
        num_design_shadow_models=num_design_shadow_models,
        epochs=active_selection_epochs,
        lr=active_selection_lr,
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        selection_ridge=active_selection_ridge,
        device=device,
        unlearning_method='retrain',
    )
)

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
plt.title('MNIST Static, Passive, Sequential, and Global Active MIA')
plt.legend(loc="lower right")
plt.savefig(results_dir / 'ROC_Baseline_Probability_Active_ADP.png')
plt.close()


filename = results_dir / 'Baseline_Probability_Active_ADP.txt'

hyperparams = {
    "dataset": "MNIST",
    "input_shape": input_shape,
    "input_size": input_size,
    "num_classes": num_classes,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "epochs": epochs,
    "lr": lr,
    "measurement_number": measurement_number,
    "num_unlearning_operations": num_unlearning_operations,
    "active_probe_pool_size": active_probe_pool_size,
    "num_active_design_targets": num_active_design_targets,
    "active_design_shadow_size": active_design_shadow_size,
    "num_design_shadow_models": num_design_shadow_models,
    "active_selection_epochs": active_selection_epochs,
    "active_selection_lr": active_selection_lr,
    "active_selection_ridge": active_selection_ridge,
    "attack_epochs": attack_epochs,
    "attack_lr": attack_lr,
    "attack_hidden_size": attack_hidden_size,
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
}


results_dir.mkdir(parents=True, exist_ok=True)


with open(filename, 'w') as f:
    for param_name, param_value in hyperparams.items():
        f.write(f"{param_name}: {param_value}\n")
