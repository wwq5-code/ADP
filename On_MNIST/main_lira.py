import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
try:
    from .lira.lira import (
        calculate_accuracy,
        estimate_loss_distributions,
        run_over_MIA,
        train_model_with_loader,
    )
    from .lira.model import MLP
except ImportError:  # Support: python On_MNIST/main_lira.py
    from lira.lira import (
        calculate_accuracy,
        estimate_loss_distributions,
        run_over_MIA,
        train_model_with_loader,
    )
    from lira.model import MLP


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
num_shadow_models=50
lr_shadow_model=1e-3
epochs_shadow_model=20
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





target_model = MLP(input_size, hidden_size, num_classes).to(device)
target_model_path = 'target_model_mnist_mlp.pth'
if not os.path.exists(target_model_path):
    print(f"Training Target Model on MNIST on Epochs: {epochs}")
    train_model_with_loader(
        target_model,
        train_loader,
        training_epochs=epochs,
        lr=lr,
        device=device,
        verbose=True,
        progress_label='MNIST target MLP',
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


measurement_images=torch.cat([measurement_train_images,measurement_test_images])
measurement_ref=np.array([0]*len(measurement_train_images)+[1]*len(measurement_test_images))
measurement_labels=torch.cat([measurement_train_labels,measurement_test_labels])

print("Measurement Sample Size:",len(measurement_images))



# --------LIRA-----------------------------------------------------------------
# shadow zone
in_mean, in_std, out_mean, out_std = estimate_loss_distributions(measurement_images, 
                                                                 measurement_labels, 
                                                                 shadow_images, 
                                                                 shadow_labels,
                                                                 num_shadow_models=num_shadow_models, 
                                                                 epochs=epochs_shadow_model, 
                                                                 lr=lr_shadow_model,
                                                                 input_size=input_size,
                                                                 hidden_size=hidden_size,
                                                                 num_classes=num_classes,
                                                                 device=device)
# attack zone
scores = run_over_MIA(target_model, 
                          measurement_images, 
                          measurement_labels, 
                          in_mean, 
                          in_std, 
                          out_mean, 
                          out_std,
                          device=device)

#-----------------------------------------------------------------------------------

tpr, fpr, roc = roc_curve(measurement_ref, scores)
print("--------------")
print(f'AUC: {auc(fpr, tpr)}  |')
print("-------------")
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(f'results/ROC_LIRA Attack.png')




filename = "results/LIRA.txt"

hyperparams = {
    "input_shape": input_shape,
    "input_size": input_size,
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
    "device": str(device)
}


os.makedirs("results", exist_ok=True)

with open(filename, 'w') as f:
    for param_name, param_value in hyperparams.items():
        f.write(f"{param_name}: {param_value}\n")
