import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 
import json
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
try:
    from .ADP_active_lira.ADP_active_lira import (
        calculate_accuracy,
        estimate_sequential_distributions,
        run_over_active_MIA,
        run_over_passive_MIA,
        run_over_sequential_MIA,
        run_over_static_MIA,
        select_conditional_probe_sequences,
        train_model_with_loader,
    )
    from .ADP_active_lira.model import CNN
except ImportError:
    from ADP_active_lira.ADP_active_lira import (
        calculate_accuracy,
        estimate_sequential_distributions,
        run_over_active_MIA,
        run_over_passive_MIA,
        run_over_sequential_MIA,
        run_over_static_MIA,
        select_conditional_probe_sequences,
        train_model_with_loader,
    )
    from ADP_active_lira.model import CNN


#-----------------------------------------------------------------------------------
input_shape = (1, 28, 28)
input_size = 28 * 28
num_classes=10
hidden_size = 256
output_size = 10
epochs = 50
lr = 1e-3
perc=0.0   # amount of actual training data available to the attacker
perc_test=0.20    # amount of testing data available to the attacker ( similar distribution to training data)
meausurement_number=10   # number of target samples to be measured from each training and non-training data
num_shadow_models=20
num_unlearning_operations=3
active_probe_pool_size=5
num_design_shadow_models=5
lr_shadow_model=1e-3
epochs_shadow_model=20
covariance_ridge=1e-6
selection_ridge=1e-4
gpu = 1
device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
#-----------------------------------------------------------------------------------


if not os.path.exists('results'):
    os.makedirs('results')

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

data_root = os.environ.get('MNIST_DATA_ROOT', '/home/wwq/Data/data/mnist')
train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)


batch_size = 128 * 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





target_model = CNN(input_size, hidden_size, num_classes).to(device)
target_model_path = 'target_model_mnist_cnn.pth'
if not os.path.exists(target_model_path):
    print(f"Training Target Model on MNIST on Epochs: {epochs}")
    train_model_with_loader(
        target_model,
        train_loader,
        training_epochs=epochs,
        lr=lr,
        device=device,
        verbose=True,
        progress_label='MNIST target CNN',
    )
    torch.save(target_model.state_dict(), target_model_path)

else:
    print("Loading trained Target Model")
    target_model.load_state_dict(torch.load(target_model_path, map_location=device))
    target_model.to(device)

# Calculate training and test accuracy
train_accuracy = calculate_accuracy(target_model, train_loader, device)
test_accuracy = calculate_accuracy(target_model, test_loader, device)
print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")





num_samples_train = int(perc*len(train_loader.dataset))
num_samples_test=int(perc_test*len(test_loader.dataset))
print("----------------------------------")
print(f"Attackers knowledge:")
print(f"Training Dataset Info: {num_samples_train}/{len(train_loader.dataset)} = {num_samples_train/len(train_loader.dataset)*100}%")
print(f"Testing Dataset Info: {num_samples_test}/{len(test_loader.dataset)} = {num_samples_test/len(test_loader.dataset)*100}%")
print("----------------------------------")


train_images=[]
train_labels=[]
for images, labels in train_loader:
    train_images.append(images)
    train_labels.append(labels)
train_images=torch.cat(train_images)
train_labels=torch.cat(train_labels)

train_indices_path = 'original_indices_mnist.pt'
if not os.path.exists(train_indices_path):
    original_indices=torch.randperm(len(train_images))
    torch.save(original_indices, train_indices_path)
else:
    original_indices=torch.load(train_indices_path)

# The first records form the attacker-owned probe pool. The fixed Sequential
# baseline uses its first k records; Active ADP selects k records from the same
# pool for each target using design shadows only.
if active_probe_pool_size < num_unlearning_operations:
    raise ValueError('active_probe_pool_size must cover the deletion budget.')
probe_pool_indices = [
    int(index) for index in original_indices[:active_probe_pool_size]
]
fixed_deletion_positions = list(range(num_unlearning_operations))
fixed_deletion_indices = [
    probe_pool_indices[position] for position in fixed_deletion_positions
]
indices = original_indices[
    active_probe_pool_size:active_probe_pool_size + num_samples_train
]
anti_indices = original_indices[
    active_probe_pool_size + num_samples_train:
    active_probe_pool_size + num_samples_train + meausurement_number
]
attacker_train_images = train_images[indices]
attacker_train_labels = train_labels[indices]

measurement_train_images = train_images[anti_indices]
measurement_train_labels = train_labels[anti_indices]
probe_pool_data = train_images[probe_pool_indices]
probe_pool_labels = train_labels[probe_pool_indices]


# Every released model is exact retraining from scratch. Cache by the ordered
# deletion prefix so schedules shared by multiple targets reuse the same model.
target_model_cache = {(): target_model}


def load_or_train_target_release(deleted_prefix):
    prefix_key = tuple(int(index) for index in deleted_prefix)
    if prefix_key in target_model_cache:
        return target_model_cache[prefix_key]

    deleted_set = set(prefix_key)
    retained_indices = [
        index for index in range(len(train_dataset)) if index not in deleted_set
    ]
    retained_loader = DataLoader(
        Subset(train_dataset, retained_indices),
        batch_size=batch_size,
        shuffle=False,
    )
    target_model_after = CNN(input_size, hidden_size, num_classes).to(device)
    prefix_tag = '_'.join(str(index) for index in prefix_key)
    target_model_after_path = (
        f'target_model_mnist_cnn_delete_sequence_{prefix_tag}.pth'
    )
    if not os.path.exists(target_model_after_path):
        print(
            f'Retraining target release after deletion prefix '
            f'{list(prefix_key)} on Epochs: {epochs}'
        )
        train_model_with_loader(
            target_model_after,
            retained_loader,
            training_epochs=epochs,
            lr=lr,
            device=device,
            verbose=True,
            progress_label=f'MNIST target CNN prefix {list(prefix_key)}',
        )
        torch.save(target_model_after.state_dict(), target_model_after_path)
    else:
        print(f'Loading target release: {target_model_after_path}')
        target_model_after.load_state_dict(
            torch.load(target_model_after_path, map_location=device)
        )
        target_model_after.to(device)

    target_model_cache[prefix_key] = target_model_after
    return target_model_after


target_models = [target_model]
fixed_prefix = []
for deletion_index in fixed_deletion_indices:
    fixed_prefix.append(deletion_index)
    target_models.append(load_or_train_target_release(fixed_prefix))

print(f'Attacker-owned probe pool: {probe_pool_indices}')
print(f'Fixed Sequential deletion sequence: {fixed_deletion_indices}')


test_images=[]
test_labels=[]
for images, labels in test_loader:
    test_images.append(images)
    test_labels.append(labels)
test_images=torch.cat(test_images)
test_labels=torch.cat(test_labels)

test_indices_path = 'original_indices_test_mnist.pt'
if not os.path.exists(test_indices_path):
    original_indices_test=torch.randperm(len(test_images))
    torch.save(original_indices_test, test_indices_path)
else:
    original_indices_test=torch.load(test_indices_path)


indices = original_indices_test[:num_samples_test]
anti_indices = original_indices_test[num_samples_test:num_samples_test+meausurement_number]


attacker_test_images = test_images[indices]
attacker_test_labels = test_labels[indices]

measurement_test_images = test_images[anti_indices]
measurement_test_labels = test_labels[anti_indices]

shadow_images=torch.cat([attacker_train_images,attacker_test_images])
shadow_labels=torch.cat([attacker_train_labels,attacker_test_labels])
shadow_split = len(shadow_images) // 2
if shadow_split == 0 or shadow_split == len(shadow_images):
    raise ValueError('Need non-empty, disjoint design and calibration shadow pools.')
design_shadow_images = shadow_images[:shadow_split]
design_shadow_labels = shadow_labels[:shadow_split]
calibration_shadow_images = shadow_images[shadow_split:]
calibration_shadow_labels = shadow_labels[shadow_split:]


measurement_images=torch.cat([measurement_train_images,measurement_test_images])
measurement_ref=np.array([1]*len(measurement_train_images)+[0]*len(measurement_test_images))
measurement_labels=torch.cat([measurement_train_labels,measurement_test_labels])

print("Measurement Sample Size:",len(measurement_images))



# --------Conditional probe selection (design shadows only)-----------------
active_schedules, active_selection_diagnostics = (
    select_conditional_probe_sequences(
        measurement_images,
        measurement_labels,
        design_shadow_images,
        design_shadow_labels,
        probe_pool_data,
        probe_pool_labels,
        num_unlearning_operations=num_unlearning_operations,
        num_design_shadow_models=num_design_shadow_models,
        epochs=epochs_shadow_model,
        lr=lr_shadow_model,
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        selection_ridge=selection_ridge,
        device=device,
    )
)
active_deletion_indices = [
    [probe_pool_indices[position] for position in schedule]
    for schedule in active_schedules
]
print('Active ADP target-specific deletion sequences:')
for target_index, sequence in enumerate(active_deletion_indices):
    print(f'  target {target_index}: {sequence}')


# The schedule is now fixed. Only this execution stage obtains target-model
# releases; no realized target response feeds back into probe selection.
active_target_model_trajectories = []
for sequence in active_deletion_indices:
    models = [target_model]
    active_prefix = []
    for deletion_index in sequence:
        active_prefix.append(deletion_index)
        models.append(load_or_train_target_release(active_prefix))
    active_target_model_trajectories.append(models)


# --------Static, Passive, fixed Sequential, and Active trajectory LiRA------
fixed_distribution_statistics = estimate_sequential_distributions(
    measurement_images,
    measurement_labels,
    calibration_shadow_images,
    calibration_shadow_labels,
    probe_pool_data,
    probe_pool_labels,
    deletion_schedules=fixed_deletion_positions,
    num_shadow_models=num_shadow_models,
    epochs=epochs_shadow_model,
    lr=lr_shadow_model,
    input_size=input_size,
    hidden_size=hidden_size,
    num_classes=num_classes,
    covariance_ridge=covariance_ridge,
    progress_label='Fixed Sequential calibration',
    device=device,
)
active_distribution_statistics = estimate_sequential_distributions(
    measurement_images,
    measurement_labels,
    calibration_shadow_images,
    calibration_shadow_labels,
    probe_pool_data,
    probe_pool_labels,
    deletion_schedules=active_schedules,
    num_shadow_models=num_shadow_models,
    epochs=epochs_shadow_model,
    lr=lr_shadow_model,
    input_size=input_size,
    hidden_size=hidden_size,
    num_classes=num_classes,
    covariance_ridge=covariance_ridge,
    progress_label='Active ADP calibration',
    device=device,
)

static_scores = run_over_static_MIA(
    target_models[0],
    measurement_images,
    measurement_labels,
    fixed_distribution_statistics['static'],
    device=device,
)
passive_scores = run_over_passive_MIA(
    target_models[:2],
    measurement_images,
    measurement_labels,
    fixed_distribution_statistics['passive'],
    device=device,
)
sequential_scores = run_over_sequential_MIA(
    target_models,
    measurement_images,
    measurement_labels,
    fixed_distribution_statistics['sequential'],
    device=device,
)
active_scores = run_over_active_MIA(
    active_target_model_trajectories,
    measurement_images,
    measurement_labels,
    active_distribution_statistics['sequential'],
    device=device,
)

#-----------------------------------------------------------------------------------

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
print(f'Static LiRA AUC: {static_auc:.4f}')
print(f'Passive Deletion LiRA AUC: {passive_auc:.4f}')
print(f'Sequential ADP LiRA AUC: {sequential_auc:.4f}')
print(f'Active ADP LiRA AUC: {active_auc:.4f}')
print(
    f'Passive Deletion gain (Delta AUC): {passive_deletion_gain:+.4f} '
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
    f'Active gain over Sequential (Delta AUC): '
    f'{active_gain_over_sequential:+.4f} '
    f'({100.0 * active_gain_over_sequential:+.2f} percentage points)'
)
print("-------------")
plt.figure()
plt.plot(static_fpr,
         static_tpr,
         color='steelblue',
         lw=2,
         label='Static LiRA (area = %0.2f)' % static_auc)
plt.plot(passive_fpr,
         passive_tpr,
         color='darkorange',
         lw=2,
         label='Passive Deletion LiRA (area = %0.2f)' % passive_auc)
plt.plot(sequential_fpr,
         sequential_tpr,
         color='forestgreen',
         lw=2,
         label='Sequential ADP LiRA (area = %0.2f)' % sequential_auc)
plt.plot(active_fpr,
         active_tpr,
         color='crimson',
         lw=2,
         label='Active ADP LiRA (area = %0.2f)' % active_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Static, Passive, Sequential, and Active ADP LiRA')
plt.legend(loc="lower right")
plt.savefig('results/ROC_ADP_Active_LiRA.png')
plt.close()




results = {
    'configuration': {
        'input_shape': input_shape,
        'input_size': input_size,
        'num_classes': num_classes,
        'model': 'SmallCNN',
        'hidden_size': hidden_size,
        'output_size': output_size,
        'epochs': epochs,
        'lr': lr,
        'perc': perc,
        'perc_test': perc_test,
        'measurement_number_per_class': meausurement_number,
        'measurement_sample_size': len(measurement_images),
        'design_shadow_pool_size': len(design_shadow_images),
        'calibration_shadow_pool_size': len(calibration_shadow_images),
        'num_shadow_models': num_shadow_models,
        'num_design_shadow_models': num_design_shadow_models,
        'num_unlearning_operations': num_unlearning_operations,
        'active_probe_pool_size': active_probe_pool_size,
        'lr_shadow_model': lr_shadow_model,
        'epochs_shadow_model': epochs_shadow_model,
        'covariance_ridge': covariance_ridge,
        'selection_ridge': selection_ridge,
        'device': str(device),
        'sequential_policy': 'fixed_precommitted_random_order',
        'active_policy': 'target_specific_conditional_gaussian_logdet',
        'decoder': 'joint_gaussian_trajectory_lira',
        'conditional_adp': False,
        'conditional_probe_selection': True,
        'active_adp': True,
    },
    'probe_pool_indices': probe_pool_indices,
    'fixed_sequential_deletion_indices': fixed_deletion_indices,
    'active_deletion_indices_by_target': active_deletion_indices,
    'active_selection_diagnostics': active_selection_diagnostics,
    'metrics': {
        'static_lira': {
            'auroc': float(static_auc),
        },
        'passive_deletion_lira': {
            'auroc': float(passive_auc),
            'gain_over_static': float(passive_deletion_gain),
        },
        'sequential_adp_lira': {
            'auroc': float(sequential_auc),
            'gain_over_static': float(sequential_gain_over_static),
            'gain_over_passive': float(sequential_gain_over_passive),
        },
        'active_adp_lira': {
            'auroc': float(active_auc),
            'gain_over_static': float(active_gain_over_static),
            'gain_over_passive': float(active_gain_over_passive),
            'gain_over_sequential': float(active_gain_over_sequential),
        },
    },
}

json_filename = 'results/ADP_Active_LiRA.json'
text_filename = 'results/ADP_Active_LiRA.txt'
with open(json_filename, 'w') as file:
    json.dump(results, file, indent=2)
with open(text_filename, 'w') as file:
    file.write(json.dumps(results, indent=2))
