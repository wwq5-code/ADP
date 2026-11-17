import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import norm
try:
    from .shadow import ShadowModel
except ImportError:  # Support importing lira when main_lira.py is run directly.
    from lira.shadow import ShadowModel






def calculate_accuracy(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total



# Function to train a model
def train_model_with_raw_tensors(
    model,
    train_data,
    train_labels,
    epochs=100,
    lr=0.01,
    bs=128 * 2,
    device='cuda',
    verbose=False,
    progress_label='Shadow model',
):
    dataset= torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            img, label = batch
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == label).sum().item()
            total += batch_size
        if verbose:
            print(
                f'[{progress_label}] epoch {epoch + 1}/{epochs} | '
                f'loss={total_loss / total:.4f} | '
                f'accuracy={100.0 * correct / total:.2f}%',
                flush=True,
            )
    return model


def train_model_with_loader(
    model,
    train_loader,
    training_epochs=100,
    lr=0.01,
    device='cuda',
    verbose=True,
    progress_label='Target model',
):
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(training_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            data, target = batch[0].to(device), batch[1].to(device)
            output = model(data) 
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            correct += (output.argmax(dim=1) == target).sum().item()
            total += batch_size
        if verbose:
            print(
                f'[{progress_label}] epoch {epoch + 1}/{training_epochs} | '
                f'loss={total_loss / total:.4f} | '
                f'accuracy={100.0 * correct / total:.2f}%',
                flush=True,
            )

    return model


def stable_phi(output, target, num_classes=10, device=None):
    if device is None:
        device = output.device
    batch_size = output.size(0)
    
    # Compute CrossEntropyLoss for the entire batch
    loss = nn.CrossEntropyLoss(reduction='none')(output, target)
    loss = torch.clamp(loss, min=1e-15, max=10.0)
    
    # Create anti-targets for each sample in the batch
    anti_targets = []
    for i in range(batch_size):
        sample_anti_targets = [j for j in range(num_classes) if j != target[i].item()]
        anti_targets.append(torch.tensor(sample_anti_targets).to(device))
    
    # Compute exp terms
    exp_term = torch.exp(-loss)
    
    # Compute anti_exp_term for each sample
    anti_exp_term = []
    for i in range(batch_size):
        sample_anti_exp = torch.exp(-nn.CrossEntropyLoss(reduction='none')(
            output[i].unsqueeze(0).repeat(num_classes-1, 1),
            anti_targets[i]
        ))
        anti_exp_term.append(sample_anti_exp)
    
    anti_exp_term = torch.stack(anti_exp_term)
    
    # Compute phi for each sample
    positive_term = torch.log(exp_term)
    negative_term = torch.log(torch.sum(anti_exp_term, dim=1))
    phi = positive_term - negative_term
    
    return phi


# Function to estimate loss distributions using shadow models
def estimate_loss_distributions(
    target_data,
    target_label,
    shadow_imgs,
    shadow_labs,
    num_shadow_models=2,
    epochs=100,
    lr=0.01,
    input_size=28 * 28,
    hidden_size=256,
    num_classes=10,
    device=None,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_losses = [[] for _ in range(target_data.size(0))]
    out_losses = [[] for _ in range(target_data.size(0))]
    total_shadow_models = target_data.size(0) * num_shadow_models

    for omeaga in range(target_data.size(0)):
        for ind in range(num_shadow_models):
            completed = omeaga * num_shadow_models + ind
            print(
                f'[LiRA shadow] target {omeaga + 1}/{target_data.size(0)} | '
                f'shadow {ind + 1}/{num_shadow_models} | '
                f'overall {completed + 1}/{total_shadow_models} '
                f'({100.0 * (completed + 1) / total_shadow_models:.1f}%)',
                flush=True,
            )

            # split the shadow_imgs and shadow_labs into two parts (random)
            indices = torch.randperm(shadow_imgs.size(0))
            shadow_imgs1 = shadow_imgs[indices[:len(indices)//2]]
            shadow_labs1 = shadow_labs[indices[:len(indices)//2]]
            shadow_imgs2 = shadow_imgs[indices[len(indices)//2:]]
            shadow_labs2 = shadow_labs[indices[len(indices)//2:]]

            # Include the target example in the dataset
            model_in = ShadowModel(input_size, hidden_size, num_classes).to(device)
            train_data_in = torch.cat((shadow_imgs1, target_data[omeaga:omeaga+1]), 0)
            train_labels_in = torch.cat((shadow_labs1, target_label[omeaga:omeaga+1]), 0)
            model_in = train_model_with_raw_tensors(
                model_in, train_data_in, train_labels_in, epochs, lr,
                device=device,
            )
            model_in.eval()
            with torch.no_grad():
                output_in = model_in(target_data[omeaga:omeaga+1].to(device))
                loss_in = stable_phi(
                    output_in,
                    target_label[omeaga:omeaga+1].to(device),
                    num_classes=num_classes,
                    device=device,
                ).cpu().numpy()
                in_losses[omeaga].append(loss_in)

            # Exclude the target example from the dataset
            model_out = ShadowModel(input_size, hidden_size, num_classes).to(device)
            train_data_out=shadow_imgs2
            train_labels_out = shadow_labs2
            model_out = train_model_with_raw_tensors(
                model_out, train_data_out, train_labels_out, epochs, lr,
                device=device,
            )
            model_out.eval()
            with torch.no_grad():
                output_out = model_out(target_data[omeaga:omeaga+1].to(device))
                loss_out = stable_phi(
                    output_out,
                    target_label[omeaga:omeaga+1].to(device),
                    num_classes=num_classes,
                    device=device,
                ).cpu().numpy()
                out_losses[omeaga].append(loss_out)


    in_mean_list = [np.mean(losses) for losses in in_losses]
    out_mean_list = [np.mean(losses) for losses in out_losses]
    in_std_list = [np.std(losses) for losses in in_losses]
    out_std_list = [np.std(losses) for losses in out_losses]

    return in_mean_list, in_std_list, out_mean_list, out_std_list


# Function to perform membership inference
def membership_inference(model, target_data, target_label, in_mean, in_std, out_mean, out_std, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        output = model(target_data.to(device))
        loss = stable_phi(
            output,
            target_label.to(device),
            num_classes=output.size(1),
            device=device,
        ).item()

    # Calculate likelihoods assuming Gaussian distributions
    #------Version-1-------------------------------------------------------------------------------------
    # p_in = (1 / (in_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((loss - in_mean) / in_std) ** 2)
    # p_out = (1 / (out_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((loss - out_mean) / out_std) ** 2)

    # return p_in/(p_out+1e-30)
    #----------------------------------------------------------------------------------------------------

    #-----Version-2--------------------------------------------------------------------------------------
    p_in = norm.pdf(loss, loc=in_mean, scale=in_std)
    p_out = norm.pdf(loss, loc=out_mean, scale=out_std)

    return p_in/(p_out+1e-30)
    #----------------------------------------------------------------------------------------------------

    #------Version-3-------------------------------------------------------------------------------------
    # p_in = -norm.logpdf(loss,in_mean, in_std + 1e-30)
    # p_out = -norm.logpdf(loss, out_mean, out_std + 1e-30)
    # return p_in - p_out
    #----------------------------------------------------------------------------------------------------


def run_over_MIA(model,target_data_col,target_label_col,in_mean_col,in_std_col,out_mean_col,out_std_col,device=None):
    result_col=[]
    for i in range(len(target_data_col)):
        result = membership_inference(
            model,
            target_data_col[i:i+1],
            target_label_col[i:i+1],
            in_mean_col[i],
            in_std_col[i],
            out_mean_col[i],
            out_std_col[i],
            device=device,
        )
        result_col.append(result)
    return np.array(result_col)
