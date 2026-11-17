import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import multivariate_normal, norm
try:
    from .shadow import ShadowModel
except ImportError:
    from ADP_lira.shadow import ShadowModel






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


def _fit_scalar_gaussians(samples, minimum_std=1e-6):
    means = [float(np.mean(values)) for values in samples]
    stds = [max(float(np.std(values)), minimum_std) for values in samples]
    return means, stds


def _fit_trajectory_gaussians(samples, covariance_ridge=1e-6):
    means = []
    covariances = []
    for values in samples:
        values = np.asarray(values, dtype=np.float64)
        mean = values.mean(axis=0)
        centered = values - mean
        covariance = centered.T @ centered / values.shape[0]
        scale = max(float(np.trace(covariance) / covariance.shape[0]), 1e-12)
        covariance = covariance + covariance_ridge * scale * np.eye(
            covariance.shape[0]
        )
        means.append(mean)
        covariances.append(covariance)
    return means, covariances


def _train_shadow_trajectory(base_data,
                             base_labels,
                             owned_data,
                             owned_labels,
                             world_record,
                             world_label,
                             target_data,
                             target_label,
                             input_size,
                             hidden_size,
                             num_classes,
                             epochs,
                             lr,
                             device):
    utilities = []
    num_operations = owned_data.size(0)
    for deleted_count in range(num_operations + 1):
        remaining_owned_data = owned_data[deleted_count:]
        remaining_owned_labels = owned_labels[deleted_count:]
        train_data = torch.cat(
            (base_data, remaining_owned_data, world_record), dim=0
        )
        train_labels = torch.cat(
            (base_labels, remaining_owned_labels, world_label), dim=0
        )
        utilities.append(
            _train_shadow_and_measure(
                train_data,
                train_labels,
                target_data,
                target_label,
                input_size,
                hidden_size,
                num_classes,
                epochs,
                lr,
                device,
            )
        )
    return np.asarray(utilities, dtype=np.float64)


def estimate_sequential_distributions(target_data,
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
                                      covariance_ridge=1e-6,
                                      device=None):
    """Fit Static, one-deletion, and fixed-sequence trajectory LiRA models.

    H1 and H0 are matched in size and class composition. H1 contains the
    candidate target; H0 contains a same-class auxiliary replacement. Both
    worlds replay the same ordered deletion prefix.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if deletion_data.size(0) < 2:
        raise ValueError('Sequential ADP LiRA requires at least two deletions.')
    if deletion_data.size(0) != deletion_label.size(0):
        raise ValueError('Deletion data and labels must have the same length.')

    num_targets = target_data.size(0)
    static_in = [[] for _ in range(num_targets)]
    static_out = [[] for _ in range(num_targets)]
    trajectory_in = [[] for _ in range(num_targets)]
    trajectory_out = [[] for _ in range(num_targets)]
    total_shadow_worlds = num_targets * num_shadow_models

    for target_index in range(num_targets):
        candidate_data = target_data[target_index:target_index + 1]
        candidate_label = target_label[target_index:target_index + 1]
        for shadow_index in range(num_shadow_models):
            completed = target_index * num_shadow_models + shadow_index + 1
            print(
                f'[Sequential ADP LiRA] target {target_index + 1}/{num_targets} | '
                f'shadow world {shadow_index + 1}/{num_shadow_models} | '
                f'overall {completed}/{total_shadow_worlds} '
                f'({100.0 * completed / total_shadow_worlds:.1f}%)',
                flush=True,
            )

            indices = torch.randperm(shadow_imgs.size(0))
            base_count = len(indices) // 2
            replacement_candidates = indices[
                shadow_labs[indices] == candidate_label.item()
            ]
            if replacement_candidates.numel() == 0:
                raise ValueError(
                    f'No class-{candidate_label.item()} replacement in shadow data.'
                )

            replacement_index = int(replacement_candidates[0].item())
            base_indices = indices[indices != replacement_index][:base_count]
            base_data = shadow_imgs[base_indices]
            base_labels = shadow_labs[base_indices]
            replacement_data = shadow_imgs[
                replacement_index:replacement_index + 1
            ]
            replacement_label = shadow_labs[
                replacement_index:replacement_index + 1
            ]

            h1_utilities = _train_shadow_trajectory(
                base_data,
                base_labels,
                deletion_data,
                deletion_label,
                candidate_data,
                candidate_label,
                candidate_data,
                candidate_label,
                input_size,
                hidden_size,
                num_classes,
                epochs,
                lr,
                device,
            )
            h0_utilities = _train_shadow_trajectory(
                base_data,
                base_labels,
                deletion_data,
                deletion_label,
                replacement_data,
                replacement_label,
                candidate_data,
                candidate_label,
                input_size,
                hidden_size,
                num_classes,
                epochs,
                lr,
                device,
            )

            static_in[target_index].append(h1_utilities[0])
            static_out[target_index].append(h0_utilities[0])
            trajectory_in[target_index].append(np.diff(h1_utilities))
            trajectory_out[target_index].append(np.diff(h0_utilities))

    static_in_mean, static_in_std = _fit_scalar_gaussians(static_in)
    static_out_mean, static_out_std = _fit_scalar_gaussians(static_out)
    passive_in = [[values[0] for values in samples]
                  for samples in trajectory_in]
    passive_out = [[values[0] for values in samples]
                   for samples in trajectory_out]
    passive_in_mean, passive_in_std = _fit_scalar_gaussians(passive_in)
    passive_out_mean, passive_out_std = _fit_scalar_gaussians(passive_out)
    sequential_in_mean, sequential_in_cov = _fit_trajectory_gaussians(
        trajectory_in, covariance_ridge=covariance_ridge
    )
    sequential_out_mean, sequential_out_cov = _fit_trajectory_gaussians(
        trajectory_out, covariance_ridge=covariance_ridge
    )

    return {
        'static': {
            'in_mean': static_in_mean,
            'in_std': static_in_std,
            'out_mean': static_out_mean,
            'out_std': static_out_std,
        },
        'passive': {
            'in_mean': passive_in_mean,
            'in_std': passive_in_std,
            'out_mean': passive_out_mean,
            'out_std': passive_out_std,
        },
        'sequential': {
            'in_mean': sequential_in_mean,
            'in_cov': sequential_in_cov,
            'out_mean': sequential_out_mean,
            'out_cov': sequential_out_cov,
        },
    }


def _model_utility(model, target_data, target_label, device):
    model.eval()
    with torch.no_grad():
        output = model(target_data.to(device))
        return stable_phi(
            output,
            target_label.to(device),
            num_classes=output.size(1),
            device=device,
        ).item()


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
    utility = _model_utility(model, target_data, target_label, device)
    return (
        norm.logpdf(utility, loc=in_mean, scale=in_std)
        - norm.logpdf(utility, loc=out_mean, scale=out_std)
    )


def passive_membership_inference(model_before,
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
    before = _model_utility(model_before, target_data, target_label, device)
    after = _model_utility(model_after, target_data, target_label, device)
    delta = after - before
    return (
        norm.logpdf(delta, loc=in_mean, scale=in_std)
        - norm.logpdf(delta, loc=out_mean, scale=out_std)
    )


def sequential_membership_inference(models,
                                    target_data,
                                    target_label,
                                    in_mean,
                                    in_cov,
                                    out_mean,
                                    out_cov,
                                    device=None):
    if len(models) < 3:
        raise ValueError('Sequential ADP LiRA requires M_0 and at least M_1, M_2.')
    if device is None:
        device = next(models[0].parameters()).device
    utilities = np.asarray([
        _model_utility(model, target_data, target_label, device)
        for model in models
    ])
    trajectory = np.diff(utilities)
    return (
        multivariate_normal.logpdf(trajectory, mean=in_mean, cov=in_cov)
        - multivariate_normal.logpdf(trajectory, mean=out_mean, cov=out_cov)
    )


def run_over_static_MIA(model, target_data, target_label, statistics, device=None):
    scores = []
    for index in range(len(target_data)):
        scores.append(static_membership_inference(
            model,
            target_data[index:index + 1],
            target_label[index:index + 1],
            statistics['in_mean'][index],
            statistics['in_std'][index],
            statistics['out_mean'][index],
            statistics['out_std'][index],
            device=device,
        ))
    return np.asarray(scores)


def run_over_passive_MIA(models, target_data, target_label, statistics, device=None):
    scores = []
    for index in range(len(target_data)):
        scores.append(passive_membership_inference(
            models[0],
            models[1],
            target_data[index:index + 1],
            target_label[index:index + 1],
            statistics['in_mean'][index],
            statistics['in_std'][index],
            statistics['out_mean'][index],
            statistics['out_std'][index],
            device=device,
        ))
    return np.asarray(scores)


def run_over_sequential_MIA(models,
                            target_data,
                            target_label,
                            statistics,
                            device=None):
    scores = []
    for index in range(len(target_data)):
        scores.append(sequential_membership_inference(
            models,
            target_data[index:index + 1],
            target_label[index:index + 1],
            statistics['in_mean'][index],
            statistics['in_cov'][index],
            statistics['out_mean'][index],
            statistics['out_cov'][index],
            device=device,
        ))
    return np.asarray(scores)
