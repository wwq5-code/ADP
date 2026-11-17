import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import multivariate_normal, norm
try:
    from .shadow import ShadowModel
except ImportError:
    from ADP_active_lira.shadow import ShadowModel






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
                             deletion_schedule,
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
    num_operations = len(deletion_schedule)
    for deleted_count in range(num_operations + 1):
        deleted_positions = set(deletion_schedule[:deleted_count])
        remaining_positions = [
            index for index in range(owned_data.size(0))
            if index not in deleted_positions
        ]
        remaining_owned_data = owned_data[remaining_positions]
        remaining_owned_labels = owned_labels[remaining_positions]
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


def _load_salun_helpers():
    """Load the shared MNIST SalUn primitives without affecting retraining runs."""
    try:
        from ..salun_utils import (
            generate_salun_weight_mask,
            salun_unlearn,
            set_random_seed,
        )
    except (ImportError, ValueError):
        from salun_utils import (
            generate_salun_weight_mask,
            salun_unlearn,
            set_random_seed,
        )
    return generate_salun_weight_mask, salun_unlearn, set_random_seed


def _salun_seed(base_seed, *coordinates):
    """Return a stable per-world/per-request seed for paired H1/H0 updates."""
    seed = int(base_seed) % (2 ** 31 - 1)
    for coordinate in coordinates:
        seed = (seed * 1000003 + int(coordinate) + 9176) % (2 ** 31 - 1)
    return seed


def _make_salun_args(device, num_classes, mask_ratio, salun_lr,
                     unlearning_epochs, batch_size):
    return SimpleNamespace(
        device=device,
        num_classes=int(num_classes),
        salun_mask_ratio=float(mask_ratio),
        salun_lr=float(salun_lr),
        unlearning_epochs=int(unlearning_epochs),
        batch_size=int(batch_size),
    )


def _measure_shadow_utility(model, target_data, target_label, num_classes,
                            device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(target_data.to(device))
        utility = stable_phi(
            output,
            target_label.to(device),
            num_classes=num_classes,
            device=device,
        ).item()
    return float(utility)


def _train_salun_shadow_world(base_data,
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
                              device,
                              seed):
    """Train the pre-deletion shadow model once, as required by SalUn."""
    _, _, set_random_seed = _load_salun_helpers()
    set_random_seed(seed)
    train_data = torch.cat((base_data, owned_data, world_record), dim=0)
    train_labels = torch.cat((base_labels, owned_labels, world_label), dim=0)
    model = ShadowModel(input_size, hidden_size, num_classes).to(device)
    model = train_model_with_raw_tensors(
        model,
        train_data,
        train_labels,
        epochs=epochs,
        lr=lr,
        device=device,
    )
    utility = _measure_shadow_utility(
        model, target_data, target_label, num_classes, device
    )
    return model.to('cpu').eval(), utility


def _apply_salun_shadow_deletion(model,
                                 base_data,
                                 base_labels,
                                 owned_data,
                                 owned_labels,
                                 deleted_positions,
                                 probe_position,
                                 world_record,
                                 world_label,
                                 target_data,
                                 target_label,
                                 salun_args,
                                 seed):
    """Clone one shadow release and apply a single-record SalUn request."""
    generate_mask, salun_unlearn, set_random_seed = _load_salun_helpers()
    deleted_positions = set(int(index) for index in deleted_positions)
    probe_position = int(probe_position)
    if probe_position in deleted_positions:
        raise ValueError('A SalUn trajectory cannot delete a probe twice.')

    remaining_positions = [
        index for index in range(owned_data.size(0))
        if index not in deleted_positions and index != probe_position
    ]
    retain_data = torch.cat(
        (base_data, owned_data[remaining_positions], world_record), dim=0
    )
    retain_labels = torch.cat(
        (base_labels, owned_labels[remaining_positions], world_label), dim=0
    )
    forget_dataset = torch.utils.data.TensorDataset(
        owned_data[probe_position:probe_position + 1],
        owned_labels[probe_position:probe_position + 1],
    )
    retain_dataset = torch.utils.data.TensorDataset(retain_data, retain_labels)

    set_random_seed(seed)
    updated_model = copy.deepcopy(model).to(salun_args.device)
    forget_loader = torch.utils.data.DataLoader(
        forget_dataset, batch_size=1, shuffle=False
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_dataset,
        batch_size=salun_args.batch_size,
        shuffle=True,
    )
    loss_fn = nn.CrossEntropyLoss()
    mask = generate_mask(
        model=updated_model,
        forget_loader=forget_loader,
        loss_fn=loss_fn,
        args=salun_args,
        verbose=False,
    )
    salun_unlearn(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        model=updated_model,
        loss_fn=loss_fn,
        args=salun_args,
        mask=mask,
        verbose=False,
    )
    utility = _measure_shadow_utility(
        updated_model,
        target_data,
        target_label,
        salun_args.num_classes,
        salun_args.device,
    )
    return updated_model.to('cpu').eval(), utility


def _train_salun_shadow_trajectory(base_data,
                                   base_labels,
                                   owned_data,
                                   owned_labels,
                                   deletion_schedule,
                                   world_record,
                                   world_label,
                                   target_data,
                                   target_label,
                                   input_size,
                                   hidden_size,
                                   num_classes,
                                   epochs,
                                   lr,
                                   salun_args,
                                   seed):
    """Train M_0 once, then obtain M_1,...,M_k by sequential SalUn updates."""
    model, initial_utility = _train_salun_shadow_world(
        base_data,
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
        salun_args.device,
        seed,
    )
    utilities = [initial_utility]
    deleted_positions = []
    for deletion_step, probe_position in enumerate(deletion_schedule, start=1):
        model, utility = _apply_salun_shadow_deletion(
            model,
            base_data,
            base_labels,
            owned_data,
            owned_labels,
            deleted_positions,
            probe_position,
            world_record,
            world_label,
            target_data,
            target_label,
            salun_args,
            _salun_seed(seed, deletion_step, probe_position),
        )
        deleted_positions.append(int(probe_position))
        utilities.append(utility)
    return np.asarray(utilities, dtype=np.float64)


def _normalize_deletion_schedules(deletion_schedules,
                                  num_targets,
                                  num_owned_records):
    if deletion_schedules is None:
        deletion_schedules = [list(range(num_owned_records))]
    elif deletion_schedules and isinstance(deletion_schedules[0], int):
        deletion_schedules = [list(deletion_schedules)]
    if len(deletion_schedules) == 1:
        deletion_schedules = [
            list(deletion_schedules[0]) for _ in range(num_targets)
        ]
    if len(deletion_schedules) != num_targets:
        raise ValueError('Expected one deletion schedule per target.')

    normalized = []
    for schedule in deletion_schedules:
        schedule = [int(index) for index in schedule]
        if len(schedule) < 2:
            raise ValueError('Sequential ADP LiRA requires at least two deletions.')
        if len(set(schedule)) != len(schedule):
            raise ValueError('A deletion schedule cannot repeat a probe.')
        if min(schedule) < 0 or max(schedule) >= num_owned_records:
            raise ValueError('Deletion schedule index is outside the probe pool.')
        normalized.append(schedule)
    return normalized


def _build_matched_shadow_world(shadow_imgs,
                                shadow_labs,
                                candidate_label):
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
    return {
        'base_data': shadow_imgs[base_indices],
        'base_labels': shadow_labs[base_indices],
        'replacement_data': shadow_imgs[
            replacement_index:replacement_index + 1
        ],
        'replacement_label': shadow_labs[
            replacement_index:replacement_index + 1
        ],
    }


def _train_shadow_state(base_data,
                        base_labels,
                        owned_data,
                        owned_labels,
                        deleted_positions,
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
    deleted_positions = set(deleted_positions)
    remaining_positions = [
        index for index in range(owned_data.size(0))
        if index not in deleted_positions
    ]
    train_data = torch.cat(
        (base_data, owned_data[remaining_positions], world_record), dim=0
    )
    train_labels = torch.cat(
        (base_labels, owned_labels[remaining_positions], world_label), dim=0
    )
    return _train_shadow_and_measure(
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


def _conditional_probe_score(member_history,
                             member_response,
                             nonmember_history,
                             nonmember_response,
                             ridge):
    """Return Eq. (membership-discriminative score) for scalar responses."""
    member_history = np.asarray(member_history, dtype=np.float64)
    nonmember_history = np.asarray(nonmember_history, dtype=np.float64)
    member_response = np.asarray(member_response, dtype=np.float64)
    nonmember_response = np.asarray(nonmember_response, dtype=np.float64)

    all_history = np.concatenate((member_history, nonmember_history), axis=0)
    history_mean = all_history.mean(axis=0)
    history_std = np.maximum(all_history.std(axis=0), 1e-8)
    member_history = (member_history - history_mean) / history_std
    nonmember_history = (nonmember_history - history_mean) / history_std

    all_response = np.concatenate((member_response, nonmember_response))
    response_mean = all_response.mean()
    response_std = max(float(all_response.std()), 1e-8)
    member_response = (member_response - response_mean) / response_std
    nonmember_response = (nonmember_response - response_mean) / response_std

    centered_history = np.concatenate((
        member_history - member_history.mean(axis=0),
        nonmember_history - nonmember_history.mean(axis=0),
    ), axis=0)
    centered_response = np.concatenate((
        member_response - member_response.mean(),
        nonmember_response - nonmember_response.mean(),
    ))
    regression_matrix = (
        centered_history.T @ centered_history
        + ridge * np.eye(centered_history.shape[1])
    )
    regression_rhs = centered_history.T @ centered_response
    try:
        regression = np.linalg.solve(regression_matrix, regression_rhs)
    except np.linalg.LinAlgError:
        regression = np.linalg.pinv(regression_matrix) @ regression_rhs

    member_residual = member_response - member_history @ regression
    nonmember_residual = nonmember_response - nonmember_history @ regression
    member_mean = float(member_residual.mean())
    nonmember_mean = float(nonmember_residual.mean())
    centered_residual = np.concatenate((
        member_residual - member_mean,
        nonmember_residual - nonmember_mean,
    ))
    denominator = max(1, centered_residual.size - 2)
    pooled_variance = float(centered_residual @ centered_residual / denominator)
    separation = member_mean - nonmember_mean
    score = separation ** 2 / (pooled_variance + ridge)
    whitened_separation = separation / np.sqrt(pooled_variance + ridge)
    return float(score), float(whitened_separation)


def _select_conditional_probe_sequences_salun(
        target_data,
        target_label,
        design_shadow_imgs,
        design_shadow_labs,
        probe_data,
        probe_label,
        num_unlearning_operations,
        num_design_shadow_models,
        epochs,
        lr,
        input_size,
        hidden_size,
        num_classes,
        selection_ridge,
        device,
        salun_args,
        salun_base_seed):
    """Select probes by replaying candidate SalUn updates on design shadows."""
    _, _, set_random_seed = _load_salun_helpers()
    all_schedules = []
    all_diagnostics = []
    for target_index in range(target_data.size(0)):
        candidate_data = target_data[target_index:target_index + 1]
        candidate_label = target_label[target_index:target_index + 1]
        print(
            f'[Active ADP SalUn selection] target {target_index + 1}/'
            f'{target_data.size(0)}',
            flush=True,
        )
        set_random_seed(_salun_seed(salun_base_seed, 0, target_index))
        worlds = [
            _build_matched_shadow_world(
                design_shadow_imgs, design_shadow_labs, candidate_label
            )
            for _ in range(num_design_shadow_models)
        ]

        member_models = []
        nonmember_models = []
        member_static = []
        nonmember_static = []
        for world_index, world in enumerate(worlds):
            print(
                f'  initial SalUn design pair {world_index + 1}/'
                f'{num_design_shadow_models}',
                flush=True,
            )
            paired_seed = _salun_seed(
                salun_base_seed, 1, target_index, world_index
            )
            member_model, member_utility = _train_salun_shadow_world(
                world['base_data'],
                world['base_labels'],
                probe_data,
                probe_label,
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
                paired_seed,
            )
            nonmember_model, nonmember_utility = _train_salun_shadow_world(
                world['base_data'],
                world['base_labels'],
                probe_data,
                probe_label,
                world['replacement_data'],
                world['replacement_label'],
                candidate_data,
                candidate_label,
                input_size,
                hidden_size,
                num_classes,
                epochs,
                lr,
                device,
                paired_seed,
            )
            member_models.append(member_model)
            nonmember_models.append(nonmember_model)
            member_static.append(member_utility)
            nonmember_static.append(nonmember_utility)

        member_static = np.asarray(member_static, dtype=np.float64)
        nonmember_static = np.asarray(nonmember_static, dtype=np.float64)
        member_current = member_static.copy()
        nonmember_current = nonmember_static.copy()
        member_deltas = []
        nonmember_deltas = []
        selected = []
        target_diagnostics = []
        scalar_information = 1.0

        for round_index in range(num_unlearning_operations):
            member_history = np.column_stack([member_static] + member_deltas)
            nonmember_history = np.column_stack(
                [nonmember_static] + nonmember_deltas
            )
            candidate_results = {}
            for probe_index in range(probe_data.size(0)):
                if probe_index in selected:
                    continue
                print(
                    f'  round {round_index + 1}/{num_unlearning_operations} | '
                    f'evaluating SalUn probe_position={probe_index}',
                    flush=True,
                )
                candidate_member_models = []
                candidate_nonmember_models = []
                member_after = []
                nonmember_after = []
                for world_index, world in enumerate(worlds):
                    paired_seed = _salun_seed(
                        salun_base_seed,
                        2,
                        target_index,
                        round_index,
                        probe_index,
                        world_index,
                    )
                    member_after_model, member_utility = (
                        _apply_salun_shadow_deletion(
                            member_models[world_index],
                            world['base_data'],
                            world['base_labels'],
                            probe_data,
                            probe_label,
                            selected,
                            probe_index,
                            candidate_data,
                            candidate_label,
                            candidate_data,
                            candidate_label,
                            salun_args,
                            paired_seed,
                        )
                    )
                    nonmember_after_model, nonmember_utility = (
                        _apply_salun_shadow_deletion(
                            nonmember_models[world_index],
                            world['base_data'],
                            world['base_labels'],
                            probe_data,
                            probe_label,
                            selected,
                            probe_index,
                            world['replacement_data'],
                            world['replacement_label'],
                            candidate_data,
                            candidate_label,
                            salun_args,
                            paired_seed,
                        )
                    )
                    candidate_member_models.append(member_after_model)
                    candidate_nonmember_models.append(nonmember_after_model)
                    member_after.append(member_utility)
                    nonmember_after.append(nonmember_utility)

                member_after = np.asarray(member_after, dtype=np.float64)
                nonmember_after = np.asarray(nonmember_after, dtype=np.float64)
                member_response = member_after - member_current
                nonmember_response = nonmember_after - nonmember_current
                score, whitened = _conditional_probe_score(
                    member_history,
                    member_response,
                    nonmember_history,
                    nonmember_response,
                    selection_ridge,
                )
                logdet_gain = (
                    np.log(scalar_information + whitened ** 2)
                    - np.log(scalar_information)
                )
                candidate_results[probe_index] = {
                    'score': score,
                    'whitened_separation': whitened,
                    'logdet_gain': float(logdet_gain),
                    'member_after': member_after,
                    'nonmember_after': nonmember_after,
                    'member_response': member_response,
                    'nonmember_response': nonmember_response,
                    'member_models': candidate_member_models,
                    'nonmember_models': candidate_nonmember_models,
                }

            selected_probe = max(
                candidate_results,
                key=lambda index: (
                    candidate_results[index]['logdet_gain'], -index
                ),
            )
            selected_result = candidate_results[selected_probe]
            selected.append(selected_probe)
            member_models = selected_result['member_models']
            nonmember_models = selected_result['nonmember_models']
            member_current = selected_result['member_after']
            nonmember_current = selected_result['nonmember_after']
            member_deltas.append(selected_result['member_response'])
            nonmember_deltas.append(selected_result['nonmember_response'])
            scalar_information += selected_result['whitened_separation'] ** 2
            target_diagnostics.append({
                'round': round_index + 1,
                'selected_probe_position': int(selected_probe),
                'conditional_score': selected_result['score'],
                'logdet_gain': selected_result['logdet_gain'],
                'candidate_scores': {
                    str(index): result['score']
                    for index, result in candidate_results.items()
                },
            })
            print(
                f'  round {round_index + 1}/{num_unlearning_operations}: '
                f'SalUn probe_position={selected_probe}, '
                f'J={selected_result["score"]:.6f}',
                flush=True,
            )

        all_schedules.append(selected)
        all_diagnostics.append(target_diagnostics)
    return all_schedules, all_diagnostics


def select_conditional_probe_sequences(target_data,
                                       target_label,
                                       design_shadow_imgs,
                                       design_shadow_labs,
                                       probe_data,
                                       probe_label,
                                       num_unlearning_operations,
                                       num_design_shadow_models=5,
                                       epochs=100,
                                       lr=0.01,
                                       input_size=28 * 28,
                                       hidden_size=256,
                                       num_classes=10,
                                       selection_ridge=1e-4,
                                       device=None,
                                       unlearning_method='retrain',
                                       salun_mask_ratio=0.5,
                                       salun_lr=None,
                                       salun_unlearning_epochs=5,
                                       salun_batch_size=256,
                                       salun_seed=0):
    """Select a target-specific schedule using conditional design shadows.

    This implements the scalar black-box specialization of the paper's
    conditional Gaussian probe score. Selection never consumes target-model
    outputs; the returned complete schedule is precommitted before execution.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if num_unlearning_operations < 2:
        raise ValueError('Active ADP requires at least two deletion operations.')
    if num_unlearning_operations > probe_data.size(0):
        raise ValueError('The active probe pool is smaller than the deletion budget.')
    if probe_data.size(0) != probe_label.size(0):
        raise ValueError('Probe data and labels must have the same length.')
    if num_design_shadow_models < 2:
        raise ValueError('Conditional probe selection needs at least two design pairs.')
    if design_shadow_imgs.size(0) != design_shadow_labs.size(0):
        raise ValueError('Design shadow data and labels must have the same length.')
    if unlearning_method not in {'retrain', 'salun'}:
        raise ValueError(f'Unsupported unlearning method: {unlearning_method}')
    if unlearning_method == 'salun':
        if salun_lr is None:
            salun_lr = lr
        salun_args = _make_salun_args(
            device,
            num_classes,
            salun_mask_ratio,
            salun_lr,
            salun_unlearning_epochs,
            salun_batch_size,
        )
        return _select_conditional_probe_sequences_salun(
            target_data,
            target_label,
            design_shadow_imgs,
            design_shadow_labs,
            probe_data,
            probe_label,
            num_unlearning_operations,
            num_design_shadow_models,
            epochs,
            lr,
            input_size,
            hidden_size,
            num_classes,
            selection_ridge,
            device,
            salun_args,
            salun_seed,
        )

    all_schedules = []
    all_diagnostics = []
    for target_index in range(target_data.size(0)):
        candidate_data = target_data[target_index:target_index + 1]
        candidate_label = target_label[target_index:target_index + 1]
        print(
            f'[Active ADP selection] target {target_index + 1}/'
            f'{target_data.size(0)}',
            flush=True,
        )
        worlds = [
            _build_matched_shadow_world(
                design_shadow_imgs, design_shadow_labs, candidate_label
            )
            for _ in range(num_design_shadow_models)
        ]
        member_static = []
        nonmember_static = []
        for world_index, world in enumerate(worlds):
            print(
                f'  initial design pair {world_index + 1}/'
                f'{num_design_shadow_models}',
                flush=True,
            )
            member_static.append(_train_shadow_state(
                world['base_data'],
                world['base_labels'],
                probe_data,
                probe_label,
                [],
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
            ))
            nonmember_static.append(_train_shadow_state(
                world['base_data'],
                world['base_labels'],
                probe_data,
                probe_label,
                [],
                world['replacement_data'],
                world['replacement_label'],
                candidate_data,
                candidate_label,
                input_size,
                hidden_size,
                num_classes,
                epochs,
                lr,
                device,
            ))

        member_static = np.asarray(member_static, dtype=np.float64)
        nonmember_static = np.asarray(nonmember_static, dtype=np.float64)
        member_current = member_static.copy()
        nonmember_current = nonmember_static.copy()
        member_deltas = []
        nonmember_deltas = []
        selected = []
        target_diagnostics = []
        scalar_information = 1.0

        for round_index in range(num_unlearning_operations):
            member_history = np.column_stack(
                [member_static] + member_deltas
            )
            nonmember_history = np.column_stack(
                [nonmember_static] + nonmember_deltas
            )
            candidate_results = {}
            for probe_index in range(probe_data.size(0)):
                if probe_index in selected:
                    continue
                print(
                    f'  round {round_index + 1}/{num_unlearning_operations} | '
                    f'evaluating probe_position={probe_index}',
                    flush=True,
                )
                deleted_with_candidate = selected + [probe_index]
                member_after = []
                nonmember_after = []
                for world in worlds:
                    member_after.append(_train_shadow_state(
                        world['base_data'],
                        world['base_labels'],
                        probe_data,
                        probe_label,
                        deleted_with_candidate,
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
                    ))
                    nonmember_after.append(_train_shadow_state(
                        world['base_data'],
                        world['base_labels'],
                        probe_data,
                        probe_label,
                        deleted_with_candidate,
                        world['replacement_data'],
                        world['replacement_label'],
                        candidate_data,
                        candidate_label,
                        input_size,
                        hidden_size,
                        num_classes,
                        epochs,
                        lr,
                        device,
                    ))
                member_after = np.asarray(member_after, dtype=np.float64)
                nonmember_after = np.asarray(nonmember_after, dtype=np.float64)
                member_response = member_after - member_current
                nonmember_response = nonmember_after - nonmember_current
                score, whitened = _conditional_probe_score(
                    member_history,
                    member_response,
                    nonmember_history,
                    nonmember_response,
                    selection_ridge,
                )
                logdet_gain = np.log(scalar_information + whitened ** 2) - np.log(
                    scalar_information
                )
                candidate_results[probe_index] = {
                    'score': score,
                    'whitened_separation': whitened,
                    'logdet_gain': float(logdet_gain),
                    'member_after': member_after,
                    'nonmember_after': nonmember_after,
                    'member_response': member_response,
                    'nonmember_response': nonmember_response,
                }

            selected_probe = max(
                candidate_results,
                key=lambda index: (
                    candidate_results[index]['logdet_gain'], -index
                ),
            )
            selected_result = candidate_results[selected_probe]
            selected.append(selected_probe)
            member_current = selected_result['member_after']
            nonmember_current = selected_result['nonmember_after']
            member_deltas.append(selected_result['member_response'])
            nonmember_deltas.append(selected_result['nonmember_response'])
            scalar_information += selected_result['whitened_separation'] ** 2
            target_diagnostics.append({
                'round': round_index + 1,
                'selected_probe_position': int(selected_probe),
                'conditional_score': selected_result['score'],
                'logdet_gain': selected_result['logdet_gain'],
                'candidate_scores': {
                    str(index): result['score']
                    for index, result in candidate_results.items()
                },
            })
            print(
                f'  round {round_index + 1}/{num_unlearning_operations}: '
                f'probe_position={selected_probe}, '
                f'J={selected_result["score"]:.6f}',
                flush=True,
            )

        all_schedules.append(selected)
        all_diagnostics.append(target_diagnostics)
    return all_schedules, all_diagnostics


def estimate_sequential_distributions(target_data,
                                      target_label,
                                      shadow_imgs,
                                      shadow_labs,
                                      deletion_data,
                                      deletion_label,
                                      deletion_schedules=None,
                                      num_shadow_models=2,
                                      epochs=100,
                                      lr=0.01,
                                      input_size=28 * 28,
                                      hidden_size=256,
                                      num_classes=10,
                                      covariance_ridge=1e-6,
                                      progress_label='Sequential ADP LiRA',
                                      device=None,
                                      unlearning_method='retrain',
                                      salun_mask_ratio=0.5,
                                      salun_lr=None,
                                      salun_unlearning_epochs=5,
                                      salun_batch_size=256,
                                      salun_seed=0):
    """Fit Static, one-deletion, and ordered-trajectory LiRA models.

    H1 and H0 are matched in size and class composition. H1 contains the
    candidate target; H0 contains a same-class auxiliary replacement. Both
    worlds replay the same ordered deletion prefix. A common schedule or one
    target-specific schedule per measurement target may be supplied.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if deletion_data.size(0) != deletion_label.size(0):
        raise ValueError('Deletion data and labels must have the same length.')

    num_targets = target_data.size(0)
    deletion_schedules = _normalize_deletion_schedules(
        deletion_schedules,
        num_targets,
        deletion_data.size(0),
    )
    if unlearning_method not in {'retrain', 'salun'}:
        raise ValueError(f'Unsupported unlearning method: {unlearning_method}')
    salun_args = None
    if unlearning_method == 'salun':
        if salun_lr is None:
            salun_lr = lr
        salun_args = _make_salun_args(
            device,
            num_classes,
            salun_mask_ratio,
            salun_lr,
            salun_unlearning_epochs,
            salun_batch_size,
        )
    static_in = [[] for _ in range(num_targets)]
    static_out = [[] for _ in range(num_targets)]
    trajectory_in = [[] for _ in range(num_targets)]
    trajectory_out = [[] for _ in range(num_targets)]
    total_shadow_worlds = num_targets * num_shadow_models

    for target_index in range(num_targets):
        candidate_data = target_data[target_index:target_index + 1]
        candidate_label = target_label[target_index:target_index + 1]
        deletion_schedule = deletion_schedules[target_index]
        for shadow_index in range(num_shadow_models):
            completed = target_index * num_shadow_models + shadow_index + 1
            print(
                f'[{progress_label}] target {target_index + 1}/{num_targets} | '
                f'shadow world {shadow_index + 1}/{num_shadow_models} | '
                f'overall {completed}/{total_shadow_worlds} '
                f'({100.0 * completed / total_shadow_worlds:.1f}%)',
                flush=True,
            )

            if unlearning_method == 'salun':
                _, _, set_random_seed = _load_salun_helpers()
                set_random_seed(
                    _salun_seed(salun_seed, 30, target_index, shadow_index)
                )
            world = _build_matched_shadow_world(
                shadow_imgs, shadow_labs, candidate_label
            )

            if unlearning_method == 'salun':
                paired_seed = _salun_seed(
                    salun_seed, 3, target_index, shadow_index
                )
                h1_utilities = _train_salun_shadow_trajectory(
                    world['base_data'],
                    world['base_labels'],
                    deletion_data,
                    deletion_label,
                    deletion_schedule,
                    candidate_data,
                    candidate_label,
                    candidate_data,
                    candidate_label,
                    input_size,
                    hidden_size,
                    num_classes,
                    epochs,
                    lr,
                    salun_args,
                    paired_seed,
                )
                h0_utilities = _train_salun_shadow_trajectory(
                    world['base_data'],
                    world['base_labels'],
                    deletion_data,
                    deletion_label,
                    deletion_schedule,
                    world['replacement_data'],
                    world['replacement_label'],
                    candidate_data,
                    candidate_label,
                    input_size,
                    hidden_size,
                    num_classes,
                    epochs,
                    lr,
                    salun_args,
                    paired_seed,
                )
            else:
                h1_utilities = _train_shadow_trajectory(
                    world['base_data'],
                    world['base_labels'],
                    deletion_data,
                    deletion_label,
                    deletion_schedule,
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
                    world['base_data'],
                    world['base_labels'],
                    deletion_data,
                    deletion_label,
                    deletion_schedule,
                    world['replacement_data'],
                    world['replacement_label'],
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


def run_over_active_MIA(model_trajectories,
                        target_data,
                        target_label,
                        statistics,
                        device=None):
    if len(model_trajectories) != len(target_data):
        raise ValueError('Expected one active target-model trajectory per target.')
    scores = []
    for index, models in enumerate(model_trajectories):
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
