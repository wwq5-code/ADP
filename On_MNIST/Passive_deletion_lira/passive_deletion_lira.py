import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import norm
try:
    from .shadow import ShadowModel
except ImportError:
    from Passive_deletion_lira.shadow import ShadowModel






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


def _train_shadow_and_measure(train_data,
                              train_labels,
                              target_data,
                              target_label,
                              input_size,
                              hidden_size,
                              num_classes,
                              epochs,
                              lr,
                              device):
    model = ShadowModel(input_size, hidden_size, num_classes).to(device)
    model = train_model_with_raw_tensors(model,
                                         train_data,
                                         train_labels,
                                         epochs,
                                         lr,
                                         device=device)
    model.eval()
    with torch.no_grad():
        output = model(target_data.to(device))
        utility = stable_phi(output,
                             target_label.to(device),
                             num_classes=num_classes,
                             device=device).item()
    return utility


# Same LiRA distribution fitting, but fit deletion-induced deltas instead of
# fitting the utility of a single model.
def estimate_deletion_distributions(target_data,
                                    target_label,
                                    shadow_imgs,
                                    shadow_labs,
                                    deletion_data,
                                    deletion_label,
                                    num_shadow_models=2,
                                    epochs=100,
                                    lr=0.01,
                                    input_size=28 * 28,
                                    hidden_size=256,
                                    num_classes=10,
                                    device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_deltas = [[] for _ in range(target_data.size(0))]
    out_deltas = [[] for _ in range(target_data.size(0))]
    static_in_utilities = [[] for _ in range(target_data.size(0))]
    static_out_utilities = [[] for _ in range(target_data.size(0))]
    total_shadow_models = target_data.size(0) * num_shadow_models

    for omeaga in range(target_data.size(0)):
        for ind in range(num_shadow_models):
            completed = omeaga * num_shadow_models + ind
            print(
                f'[Passive Deletion LiRA] target {omeaga + 1}/{target_data.size(0)} | '
                f'shadow {ind + 1}/{num_shadow_models} | '
                f'overall {completed + 1}/{total_shadow_models} '
                f'({100.0 * (completed + 1) / total_shadow_models:.1f}%)',
                flush=True,
            )

            # Paired H1/H0 worlds use the same base records.
            indices = torch.randperm(shadow_imgs.size(0))
            base_indices = indices[:len(indices)//2]
            base_data = shadow_imgs[base_indices]
            base_labels = shadow_labs[base_indices]
            candidate_data = target_data[omeaga:omeaga+1]
            candidate_label = target_label[omeaga:omeaga+1]

            # H1: x* is a member. Both models contain the same data except
            # that the attacker-owned record is removed from the after world.
            h1_before_data = torch.cat((base_data,
                                        deletion_data,
                                        candidate_data), 0)
            h1_before_labels = torch.cat((base_labels,
                                          deletion_label,
                                          candidate_label), 0)
            h1_after_data = torch.cat((base_data, candidate_data), 0)
            h1_after_labels = torch.cat((base_labels, candidate_label), 0)

            # H0: x* is a non-member. Delete the same attacker-owned record.
            h0_before_data = torch.cat((base_data, deletion_data), 0)
            h0_before_labels = torch.cat((base_labels, deletion_label), 0)

            h1_before = _train_shadow_and_measure(h1_before_data,
                                                  h1_before_labels,
                                                  candidate_data,
                                                  candidate_label,
                                                  input_size,
                                                  hidden_size,
                                                  num_classes,
                                                  epochs,
                                                  lr,
                                                  device)
            h1_after = _train_shadow_and_measure(h1_after_data,
                                                 h1_after_labels,
                                                 candidate_data,
                                                 candidate_label,
                                                 input_size,
                                                 hidden_size,
                                                 num_classes,
                                                 epochs,
                                                 lr,
                                                 device)
            h0_before = _train_shadow_and_measure(h0_before_data,
                                                  h0_before_labels,
                                                  candidate_data,
                                                  candidate_label,
                                                  input_size,
                                                  hidden_size,
                                                  num_classes,
                                                  epochs,
                                                  lr,
                                                  device)
            h0_after = _train_shadow_and_measure(base_data,
                                                 base_labels,
                                                 candidate_data,
                                                 candidate_label,
                                                 input_size,
                                                 hidden_size,
                                                 num_classes,
                                                 epochs,
                                                 lr,
                                                 device)

            in_deltas[omeaga].append(h1_after - h1_before)
            out_deltas[omeaga].append(h0_after - h0_before)
            static_in_utilities[omeaga].append(h1_before)
            static_out_utilities[omeaga].append(h0_before)

    in_mean_list = [np.mean(deltas) for deltas in in_deltas]
    out_mean_list = [np.mean(deltas) for deltas in out_deltas]
    in_std_list = [np.std(deltas) for deltas in in_deltas]
    out_std_list = [np.std(deltas) for deltas in out_deltas]
    static_in_mean_list = [np.mean(values) for values in static_in_utilities]
    static_out_mean_list = [np.mean(values) for values in static_out_utilities]
    static_in_std_list = [np.std(values) for values in static_in_utilities]
    static_out_std_list = [np.std(values) for values in static_out_utilities]

    return (in_mean_list,
            in_std_list,
            out_mean_list,
            out_std_list,
            static_in_mean_list,
            static_in_std_list,
            static_out_mean_list,
            static_out_std_list)


# Static LiRA baseline: observe only the original model and use the deletion-before
# H1/H0 shadow utilities fitted above.
def static_membership_inference(model,
                                target_data,
                                target_label,
                                in_mean,
                                in_std,
                                out_mean,
                                out_std,
                                device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        output = model(target_data.to(device))
        utility = stable_phi(output,
                             target_label.to(device),
                             device=device).item()

    p_in = norm.logpdf(utility, loc=in_mean, scale=in_std)
    p_out = norm.logpdf(utility, loc=out_mean, scale=out_std)
    return p_in - p_out


def run_over_static_MIA(model,
                        target_data_col,
                        target_label_col,
                        in_mean_col,
                        in_std_col,
                        out_mean_col,
                        out_std_col,
                        device=None):
    result_col = []
    for i in range(len(target_data_col)):
        result = static_membership_inference(model,
                                             target_data_col[i:i+1],
                                             target_label_col[i:i+1],
                                             in_mean_col[i],
                                             in_std_col[i],
                                             out_mean_col[i],
                                             out_std_col[i],
                                             device=device)
        result_col.append(result)
    return np.array(result_col)


# Same LiRA likelihood score, evaluated on the observed deletion delta.
def membership_inference(model_before,
                         model_after,
                         target_data,
                         target_label,
                         in_mean,
                         in_std,
                         out_mean,
                         out_std,
                         device=None):
    if device is None:
        device = next(model_before.parameters()).device
    model_before.eval()
    model_after.eval()
    with torch.no_grad():
        output_before = model_before(target_data.to(device))
        output_after = model_after(target_data.to(device))
        utility_before = stable_phi(output_before,
                                    target_label.to(device),
                                    device=device).item()
        utility_after = stable_phi(output_after,
                                   target_label.to(device),
                                   device=device).item()
    delta = utility_after - utility_before

    p_in = norm.logpdf(delta, loc=in_mean, scale=in_std)
    p_out = norm.logpdf(delta, loc=out_mean, scale=out_std)
    return p_in - p_out


def run_over_MIA(model_before,
                 model_after,
                 target_data_col,
                 target_label_col,
                 in_mean_col,
                 in_std_col,
                 out_mean_col,
                 out_std_col,
                 device=None):
    result_col=[]
    for i in range(len(target_data_col)):
        result = membership_inference(model_before,
                                      model_after,
                                      target_data_col[i:i+1],
                                      target_label_col[i:i+1],
                                      in_mean_col[i],
                                      in_std_col[i],
                                      out_mean_col[i],
                                      out_std_col[i],
                                      device=device)
        result_col.append(result)
    return np.array(result_col)
