import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, accuracy_score
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
import math
import os
import matplotlib.pyplot as plt 
from pathlib import Path

if __package__:
    from .rmia.rmia import RMIA, calculate_accuracy, train_model_with_loader
    from .rmia.model import CNN
else:
    from rmia.rmia import RMIA, calculate_accuracy, train_model_with_loader
    from rmia.model import CNN


#-----------------------------------------------------------------------------------
input_shape = (3, 32, 32)
channel = 3
num_classes=10
hidden_size = 512
output_size = 10
epochs = 50
lr = 1e-3
perc=0.0   # amount of actual training data available to the attacker
perc_test=0.20    # amount of testing data available to the attacker ( similar distribution to training data)
meausurement_number=10 # number of target samples to be measured from each training and non-training data
num_shadow_models=50
lr_shadow_model=1e-3
epochs_shadow_model=20
random_sample_number=100
gamma=0.5
gpu = 0
device = torch.device(
    'cuda:{}'.format(gpu)
    if torch.cuda.is_available() and gpu != -1
    else 'cpu'
)

script_dir = Path(__file__).resolve().parent
results_dir = script_dir / 'results'
data_root = os.environ.get('CIFAR10_DATA_ROOT', '/home/wwq/Data/data/cifar')
target_model_path = script_dir / 'target_model.pth'
train_indices_path = script_dir / 'original_indices'
test_indices_path = script_dir / 'original_indices_test'

#-----------------------------------------------------------------------------------
results_dir.mkdir(parents=True, exist_ok=True)

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Lambda(lambda x: x.view(3, 32, 32))
])

train_dataset = datasets.CIFAR10(
    root=data_root, train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root=data_root, train=False, download=True, transform=transform
)


batch_size = 128 * 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




target_model=CNN(channel, num_classes).to(device)
if not target_model_path.exists():
    print(f"Training Target Model on CIFAR-10 on Epochs: {epochs}")
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

if not train_indices_path.exists():
    original_indices=torch.randperm(len(train_images))
    torch.save(original_indices, train_indices_path)
else:
    original_indices=torch.load(train_indices_path, map_location='cpu')

indices = original_indices[:num_samples_train]
anti_indices = original_indices[num_samples_train:num_samples_train+meausurement_number]
attacker_train_images = train_images[indices]
attacker_train_labels = train_labels[indices]

measurement_train_images = train_images[anti_indices]
measurement_train_labels = train_labels[anti_indices]


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


indices = original_indices_test[:num_samples_test]
anti_indices = original_indices_test[num_samples_test:num_samples_test+meausurement_number]


attacker_test_images = test_images[indices]
attacker_test_labels = test_labels[indices]

measurement_test_images = test_images[anti_indices]
measurement_test_labels = test_labels[anti_indices]

shadow_images=torch.cat([attacker_train_images,attacker_test_images])
shadow_labels=torch.cat([attacker_train_labels,attacker_test_labels])


measurement_images=torch.cat([measurement_train_images,measurement_test_images])
measurement_ref=np.array(
    [1]*len(measurement_train_images)+[0]*len(measurement_test_images)
)
measurement_labels=torch.cat([measurement_train_labels,measurement_test_labels])

print("Measurement Sample Size:",len(measurement_images))


scores=RMIA(measurement_images,
            measurement_labels,
            target_model,
            shadow_images,
            shadow_labels,
            num_shadow_models,
            epochs_shadow_model,
            lr_shadow_model,
            random_sample_number,
            gamma, device)

fpr, tpr, _ = roc_curve(measurement_ref, scores)
rmia_auc = auc(fpr, tpr)
print("--------------")
print(f'AUC: {rmia_auc}  |')
print("-------------")
plt.plot(
    fpr,
    tpr,
    color='darkorange',
    lw=2,
    label='ROC curve (area = %0.2f)' % rmia_auc,
)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(results_dir / 'ROC_RMIA.png')
plt.close()



filename = results_dir / 'RMIA.txt'

hyperparams = {
    "input_shape": input_shape,
    "channel": channel,
    "num_classes": num_classes,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "epochs": epochs,
    "lr": lr,
    "perc": perc,
    "perc_test": perc_test,
    "measurement_number": meausurement_number,
    "num_shadow_models": num_shadow_models,
    "lr_shadow_model": lr_shadow_model,
    "epochs_shadow_model": epochs_shadow_model,
    "random_sample_number": random_sample_number,
    "gamma": gamma,
    "device": str(device),
    "data_root": data_root,
    "auc": rmia_auc,
}


results_dir.mkdir(parents=True, exist_ok=True)

with open(filename, 'w') as f:
    for param_name, param_value in hyperparams.items():
        f.write(f"{param_name}: {param_value}\n")
