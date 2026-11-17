import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import ResNet18


def _measure_log_odds(model, data, labels, device, batch_size):
    loader = DataLoader(
        TensorDataset(data, labels), batch_size=batch_size, shuffle=False
    )
    values = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            batch_labels = batch_labels.to(device)
            true_logits = logits.gather(1, batch_labels[:, None]).squeeze(1)
            other_logits = logits.clone()
            other_logits.scatter_(1, batch_labels[:, None], float('-inf'))
            values.append(
                (true_logits - torch.logsumexp(other_logits, dim=1)).cpu()
            )
    return torch.cat(values).numpy().astype(np.float64, copy=False)


def _train_shadow_and_measure(
    train_data,
    train_labels,
    target_data,
    target_labels,
    num_classes,
    pretrained,
    epochs,
    lr,
    device,
    batch_size,
):
    model = ResNet18(
        num_classes=num_classes, pretrained=pretrained
    ).to(device)
    loader = DataLoader(
        TensorDataset(train_data, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    values = _measure_log_odds(
        model, target_data, target_labels, device, batch_size
    )
    del model
    return values


def _load_salun_helpers():
    """Load TinyImageNet SalUn primitives for SalUn-aware selection."""
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
    seed = int(base_seed) % (2 ** 31 - 1)
    for coordinate in coordinates:
        seed = (seed * 1000003 + int(coordinate) + 9176) % (2 ** 31 - 1)
    return seed


def _train_salun_shadow_world(
    world,
    probe_data,
    probe_labels,
    world_records,
    world_labels,
    target_data,
    target_labels,
    num_classes,
    pretrained,
    epochs,
    lr,
    device,
    batch_size,
    seed,
):
    """Train a pre-deletion design shadow once and keep it for SalUn."""
    _, _, set_random_seed = _load_salun_helpers()
    set_random_seed(seed)
    train_data = torch.cat(
        [world['base_data'], probe_data, world_records], dim=0
    )
    train_labels = torch.cat(
        [world['base_labels'], probe_labels, world_labels], dim=0
    )
    model = ResNet18(
        num_classes=num_classes, pretrained=pretrained
    ).to(device)
    loader = DataLoader(
        TensorDataset(train_data, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    values = _measure_log_odds(
        model, target_data, target_labels, device, batch_size
    )
    return model.to('cpu').eval(), values


def _apply_salun_shadow_deletion(
    model,
    world,
    probe_data,
    probe_labels,
    deleted_positions,
    probe_position,
    world_records,
    world_labels,
    target_data,
    target_labels,
    salun_args,
    measurement_batch_size,
    seed,
):
    """Clone one design-shadow release and apply one SalUn request."""
    generate_mask, salun_unlearn, set_random_seed = _load_salun_helpers()
    deleted_positions = set(int(index) for index in deleted_positions)
    probe_position = int(probe_position)
    if probe_position in deleted_positions:
        raise ValueError('A SalUn trajectory cannot delete a probe twice.')

    remaining_positions = [
        position for position in range(probe_data.size(0))
        if position not in deleted_positions and position != probe_position
    ]
    retain_data = torch.cat(
        [world['base_data'], probe_data[remaining_positions], world_records],
        dim=0,
    )
    retain_labels = torch.cat(
        [world['base_labels'], probe_labels[remaining_positions], world_labels],
        dim=0,
    )
    forget_loader = DataLoader(
        TensorDataset(
            probe_data[probe_position:probe_position + 1],
            probe_labels[probe_position:probe_position + 1],
        ),
        batch_size=1,
        shuffle=False,
    )
    retain_loader = DataLoader(
        TensorDataset(retain_data, retain_labels),
        batch_size=salun_args.batch_size,
        shuffle=True,
    )

    set_random_seed(seed)
    updated_model = copy.deepcopy(model).to(salun_args.device)
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
    values = _measure_log_odds(
        updated_model,
        target_data,
        target_labels,
        salun_args.device,
        measurement_batch_size,
    )
    return updated_model.to('cpu').eval(), values


def _build_matched_shadow_world(shadow_data, shadow_labels, target_labels):
    permutation = torch.randperm(shadow_data.size(0))
    replacement_indices = []
    used_indices = set()
    for label in target_labels.tolist():
        candidates = permutation[shadow_labels[permutation] == int(label)]
        selected = next(
            (
                int(index)
                for index in candidates.tolist()
                if int(index) not in used_indices
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                f'Not enough distinct class-{label} replacements in the '
                'active design-shadow split.'
            )
        replacement_indices.append(selected)
        used_indices.add(selected)

    base_count = shadow_data.size(0) // 2
    base_indices = [
        int(index) for index in permutation.tolist()
        if int(index) not in used_indices
    ][:base_count]
    replacement_indices = torch.tensor(replacement_indices, dtype=torch.long)
    return {
        'base_data': shadow_data[base_indices],
        'base_labels': shadow_labels[base_indices],
        'replacement_data': shadow_data[replacement_indices],
        'replacement_labels': shadow_labels[replacement_indices],
    }


def _train_shadow_state(
    world,
    probe_data,
    probe_labels,
    deleted_positions,
    world_records,
    world_labels,
    target_data,
    target_labels,
    num_classes,
    pretrained,
    epochs,
    lr,
    device,
    batch_size,
):
    deleted_positions = set(int(position) for position in deleted_positions)
    remaining_positions = [
        position for position in range(probe_data.size(0))
        if position not in deleted_positions
    ]
    train_data = torch.cat(
        [world['base_data'], probe_data[remaining_positions], world_records],
        dim=0,
    )
    train_labels = torch.cat(
        [world['base_labels'], probe_labels[remaining_positions], world_labels],
        dim=0,
    )
    return _train_shadow_and_measure(
        train_data,
        train_labels,
        target_data,
        target_labels,
        num_classes,
        pretrained,
        epochs,
        lr,
        device,
        batch_size,
    )


def _conditional_probe_score(
    member_history,
    member_response,
    nonmember_history,
    nonmember_response,
    ridge,
):
    member_history = np.asarray(member_history, dtype=np.float64)
    nonmember_history = np.asarray(nonmember_history, dtype=np.float64)
    member_response = np.asarray(member_response, dtype=np.float64)
    nonmember_response = np.asarray(nonmember_response, dtype=np.float64)

    all_history = np.concatenate([member_history, nonmember_history], axis=0)
    history_mean = all_history.mean(axis=0)
    history_std = np.maximum(all_history.std(axis=0), 1e-8)
    member_history = (member_history - history_mean) / history_std
    nonmember_history = (nonmember_history - history_mean) / history_std

    all_response = np.concatenate([member_response, nonmember_response])
    response_mean = all_response.mean()
    response_std = max(float(all_response.std()), 1e-8)
    member_response = (member_response - response_mean) / response_std
    nonmember_response = (nonmember_response - response_mean) / response_std

    centered_history = np.concatenate([
        member_history - member_history.mean(axis=0),
        nonmember_history - nonmember_history.mean(axis=0),
    ])
    centered_response = np.concatenate([
        member_response - member_response.mean(),
        nonmember_response - nonmember_response.mean(),
    ])
    matrix = (
        centered_history.T @ centered_history
        + ridge * np.eye(centered_history.shape[1])
    )
    rhs = centered_history.T @ centered_response
    try:
        regression = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        regression = np.linalg.pinv(matrix) @ rhs

    member_residual = member_response - member_history @ regression
    nonmember_residual = nonmember_response - nonmember_history @ regression
    member_mean = float(member_residual.mean())
    nonmember_mean = float(nonmember_residual.mean())
    centered_residual = np.concatenate([
        member_residual - member_mean,
        nonmember_residual - nonmember_mean,
    ])
    denominator = max(1, centered_residual.size - 2)
    pooled_variance = float(centered_residual @ centered_residual / denominator)
    separation = member_mean - nonmember_mean
    score = separation ** 2 / (pooled_variance + ridge)
    return float(score)


def _select_global_probe_sequence_salun(
    design_target_data,
    design_target_labels,
    design_shadow_data,
    design_shadow_labels,
    probe_data,
    probe_labels,
    num_unlearning_operations,
    num_design_shadow_models,
    epochs,
    lr,
    num_classes,
    pretrained,
    selection_ridge,
    device,
    batch_size,
    salun_args,
    salun_seed,
):
    """Select a shared-prefix SalUn schedule by mean target-wise score."""
    _, _, set_random_seed = _load_salun_helpers()
    set_random_seed(_salun_seed(salun_seed, 0))
    worlds = [
        _build_matched_shadow_world(
            design_shadow_data, design_shadow_labels, design_target_labels
        )
        for _ in range(num_design_shadow_models)
    ]

    member_models = []
    nonmember_models = []
    member_static = []
    nonmember_static = []
    for world_index, world in enumerate(worlds):
        print(
            f'[Global Active ADP SalUn] initial shadow pair '
            f'{world_index + 1}/{num_design_shadow_models}',
            flush=True,
        )
        paired_seed = _salun_seed(salun_seed, 1, world_index)
        member_model, member_values = _train_salun_shadow_world(
            world,
            probe_data,
            probe_labels,
            design_target_data,
            design_target_labels,
            design_target_data,
            design_target_labels,
            num_classes,
            pretrained,
            epochs,
            lr,
            device,
            batch_size,
            paired_seed,
        )
        nonmember_model, nonmember_values = _train_salun_shadow_world(
            world,
            probe_data,
            probe_labels,
            world['replacement_data'],
            world['replacement_labels'],
            design_target_data,
            design_target_labels,
            num_classes,
            pretrained,
            epochs,
            lr,
            device,
            batch_size,
            paired_seed,
        )
        member_models.append(member_model)
        nonmember_models.append(nonmember_model)
        member_static.append(member_values)
        nonmember_static.append(nonmember_values)

    member_static = np.asarray(member_static)
    nonmember_static = np.asarray(nonmember_static)
    member_current = member_static.copy()
    nonmember_current = nonmember_static.copy()
    member_deltas = []
    nonmember_deltas = []
    selected = []
    diagnostics = []

    for round_index in range(num_unlearning_operations):
        member_histories = [
            np.column_stack(
                [member_static[:, target_index]]
                + [delta[:, target_index] for delta in member_deltas]
            )
            for target_index in range(design_target_data.size(0))
        ]
        nonmember_histories = [
            np.column_stack(
                [nonmember_static[:, target_index]]
                + [delta[:, target_index] for delta in nonmember_deltas]
            )
            for target_index in range(design_target_data.size(0))
        ]
        candidate_mean_scores = {}
        candidate_scores_by_target = [
            {} for _ in range(design_target_data.size(0))
        ]
        best_key = None
        best_result = None
        for probe_position in range(probe_data.size(0)):
            if probe_position in selected:
                continue
            print(
                f'[Global Active ADP SalUn] round {round_index + 1}/'
                f'{num_unlearning_operations}, candidate={probe_position}',
                flush=True,
            )
            candidate_member_models = []
            candidate_nonmember_models = []
            member_after = []
            nonmember_after = []
            for world_index, world in enumerate(worlds):
                paired_seed = _salun_seed(
                    salun_seed,
                    2,
                    round_index,
                    probe_position,
                    world_index,
                )
                member_after_model, member_values = (
                    _apply_salun_shadow_deletion(
                        member_models[world_index],
                        world,
                        probe_data,
                        probe_labels,
                        selected,
                        probe_position,
                        design_target_data,
                        design_target_labels,
                        design_target_data,
                        design_target_labels,
                        salun_args,
                        batch_size,
                        paired_seed,
                    )
                )
                nonmember_after_model, nonmember_values = (
                    _apply_salun_shadow_deletion(
                        nonmember_models[world_index],
                        world,
                        probe_data,
                        probe_labels,
                        selected,
                        probe_position,
                        world['replacement_data'],
                        world['replacement_labels'],
                        design_target_data,
                        design_target_labels,
                        salun_args,
                        batch_size,
                        paired_seed,
                    )
                )
                candidate_member_models.append(member_after_model)
                candidate_nonmember_models.append(nonmember_after_model)
                member_after.append(member_values)
                nonmember_after.append(nonmember_values)

            member_after = np.asarray(member_after)
            nonmember_after = np.asarray(nonmember_after)
            member_response = member_after - member_current
            nonmember_response = nonmember_after - nonmember_current
            target_scores = []
            for target_index in range(design_target_data.size(0)):
                score = _conditional_probe_score(
                    member_histories[target_index],
                    member_response[:, target_index],
                    nonmember_histories[target_index],
                    nonmember_response[:, target_index],
                    selection_ridge,
                )
                target_scores.append(score)
                candidate_scores_by_target[target_index][
                    str(probe_position)
                ] = float(score)
            mean_score = float(np.mean(target_scores))
            candidate_mean_scores[str(probe_position)] = mean_score
            candidate_key = (mean_score, -probe_position)
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_result = {
                    'probe_position': probe_position,
                    'mean_score': mean_score,
                    'target_scores': target_scores,
                    'member_models': candidate_member_models,
                    'nonmember_models': candidate_nonmember_models,
                    'member_after': member_after,
                    'nonmember_after': nonmember_after,
                    'member_response': member_response,
                    'nonmember_response': nonmember_response,
                }

        selected_probe = best_result['probe_position']
        selected.append(selected_probe)
        member_models = best_result['member_models']
        nonmember_models = best_result['nonmember_models']
        member_current = best_result['member_after']
        nonmember_current = best_result['nonmember_after']
        member_deltas.append(best_result['member_response'])
        nonmember_deltas.append(best_result['nonmember_response'])
        diagnostics.append({
            'round': round_index + 1,
            'selected_probe_position': int(selected_probe),
            'mean_conditional_score': best_result['mean_score'],
            'selected_target_scores': [
                float(score) for score in best_result['target_scores']
            ],
            'candidate_mean_scores': candidate_mean_scores,
            'candidate_scores_by_design_target': candidate_scores_by_target,
        })
        print(
            f'[Global Active ADP SalUn] selected '
            f'probe_position={selected_probe}, '
            f'mean_J={best_result["mean_score"]:.6f}',
            flush=True,
        )

    return selected, diagnostics


def _select_global_probe_sequence_retrain(
    design_target_data,
    design_target_labels,
    design_shadow_data,
    design_shadow_labels,
    probe_data,
    probe_labels,
    num_unlearning_operations,
    num_design_shadow_models=2,
    epochs=5,
    lr=1e-4,
    num_classes=200,
    pretrained=False,
    selection_ridge=1e-4,
    device='cuda',
    batch_size=64,
):
    """Select a shared-prefix retraining schedule by mean target-wise score."""
    if num_unlearning_operations < 2:
        raise ValueError('Active ADP requires at least two deletions.')
    if num_unlearning_operations > probe_data.size(0):
        raise ValueError('The probe pool is smaller than the deletion budget.')
    if num_design_shadow_models < 2:
        raise ValueError('Active selection needs at least two shadow pairs.')
    if design_target_data.size(0) != design_target_labels.size(0):
        raise ValueError('Design target data/label sizes differ.')
    if design_shadow_data.size(0) != design_shadow_labels.size(0):
        raise ValueError('Design shadow data/label sizes differ.')

    worlds = [
        _build_matched_shadow_world(
            design_shadow_data, design_shadow_labels, design_target_labels
        )
        for _ in range(num_design_shadow_models)
    ]
    member_static = []
    nonmember_static = []
    for world_index, world in enumerate(worlds, start=1):
        print(
            f'[Global Active ADP] initial shadow pair '
            f'{world_index}/{num_design_shadow_models}',
            flush=True,
        )
        common = dict(
            world=world,
            probe_data=probe_data,
            probe_labels=probe_labels,
            deleted_positions=[],
            target_data=design_target_data,
            target_labels=design_target_labels,
            num_classes=num_classes,
            pretrained=pretrained,
            epochs=epochs,
            lr=lr,
            device=device,
            batch_size=batch_size,
        )
        member_static.append(_train_shadow_state(
            world_records=design_target_data,
            world_labels=design_target_labels,
            **common,
        ))
        nonmember_static.append(_train_shadow_state(
            world_records=world['replacement_data'],
            world_labels=world['replacement_labels'],
            **common,
        ))

    member_static = np.asarray(member_static)
    nonmember_static = np.asarray(nonmember_static)
    member_current = member_static.copy()
    nonmember_current = nonmember_static.copy()
    member_deltas = []
    nonmember_deltas = []
    selected = []
    diagnostics = []

    for round_index in range(num_unlearning_operations):
        member_histories = [
            np.column_stack(
                [member_static[:, target_index]]
                + [delta[:, target_index] for delta in member_deltas]
            )
            for target_index in range(design_target_data.size(0))
        ]
        nonmember_histories = [
            np.column_stack(
                [nonmember_static[:, target_index]]
                + [delta[:, target_index] for delta in nonmember_deltas]
            )
            for target_index in range(design_target_data.size(0))
        ]
        candidate_results = {}
        for probe_position in range(probe_data.size(0)):
            if probe_position in selected:
                continue
            print(
                f'[Global Active ADP] round {round_index + 1}/'
                f'{num_unlearning_operations}, candidate={probe_position}',
                flush=True,
            )
            deleted_positions = selected + [probe_position]
            member_after = []
            nonmember_after = []
            for world in worlds:
                common = dict(
                    world=world,
                    probe_data=probe_data,
                    probe_labels=probe_labels,
                    deleted_positions=deleted_positions,
                    target_data=design_target_data,
                    target_labels=design_target_labels,
                    num_classes=num_classes,
                    pretrained=pretrained,
                    epochs=epochs,
                    lr=lr,
                    device=device,
                    batch_size=batch_size,
                )
                member_after.append(_train_shadow_state(
                    world_records=design_target_data,
                    world_labels=design_target_labels,
                    **common,
                ))
                nonmember_after.append(_train_shadow_state(
                    world_records=world['replacement_data'],
                    world_labels=world['replacement_labels'],
                    **common,
                ))
            member_after = np.asarray(member_after)
            nonmember_after = np.asarray(nonmember_after)
            member_response = member_after - member_current
            nonmember_response = nonmember_after - nonmember_current
            target_scores = [
                _conditional_probe_score(
                    member_histories[target_index],
                    member_response[:, target_index],
                    nonmember_histories[target_index],
                    nonmember_response[:, target_index],
                    selection_ridge,
                )
                for target_index in range(design_target_data.size(0))
            ]
            candidate_results[probe_position] = {
                'mean_score': float(np.mean(target_scores)),
                'target_scores': target_scores,
                'member_after': member_after,
                'nonmember_after': nonmember_after,
                'member_response': member_response,
                'nonmember_response': nonmember_response,
            }

        selected_probe = max(
            candidate_results,
            key=lambda position: (
                candidate_results[position]['mean_score'], -position
            ),
        )
        result = candidate_results[selected_probe]
        selected.append(selected_probe)
        member_current = result['member_after']
        nonmember_current = result['nonmember_after']
        member_deltas.append(result['member_response'])
        nonmember_deltas.append(result['nonmember_response'])
        diagnostics.append({
            'round': round_index + 1,
            'selected_probe_position': int(selected_probe),
            'mean_conditional_score': result['mean_score'],
            'selected_target_scores': [
                float(score) for score in result['target_scores']
            ],
            'candidate_mean_scores': {
                str(position): candidate['mean_score']
                for position, candidate in candidate_results.items()
            },
            'candidate_scores_by_design_target': [
                {
                    str(position): float(
                        candidate_results[position]['target_scores'][
                            target_index
                        ]
                    )
                    for position in candidate_results
                }
                for target_index in range(design_target_data.size(0))
            ],
        })
        print(
            f'[Global Active ADP] selected probe_position={selected_probe}, '
            f'mean_J={result["mean_score"]:.6f}',
            flush=True,
        )

    return selected, diagnostics


def select_global_probe_sequence(
    design_target_data,
    design_target_labels,
    design_shadow_data,
    design_shadow_labels,
    probe_data,
    probe_labels,
    num_unlearning_operations,
    num_design_shadow_models=2,
    epochs=5,
    lr=1e-4,
    num_classes=200,
    pretrained=False,
    selection_ridge=1e-4,
    device='cuda',
    batch_size=64,
    unlearning_method='retrain',
    salun_mask_ratio=0.5,
    salun_lr=None,
    salun_unlearning_epochs=40,
    salun_batch_size=None,
    salun_seed=0,
):
    """Precommit one mean-score global Active ADP schedule."""
    if unlearning_method not in {'retrain', 'salun'}:
        raise ValueError(
            f'Unsupported active-selection unlearning method: '
            f'{unlearning_method}'
        )
    if unlearning_method == 'retrain':
        return _select_global_probe_sequence_retrain(
            design_target_data,
            design_target_labels,
            design_shadow_data,
            design_shadow_labels,
            probe_data,
            probe_labels,
            num_unlearning_operations,
            num_design_shadow_models=num_design_shadow_models,
            epochs=epochs,
            lr=lr,
            num_classes=num_classes,
            pretrained=pretrained,
            selection_ridge=selection_ridge,
            device=device,
            batch_size=batch_size,
        )

    if not 0.0 <= salun_mask_ratio <= 1.0:
        raise ValueError('salun_mask_ratio must lie in [0, 1].')
    if salun_lr is None:
        salun_lr = lr
    if salun_lr <= 0.0:
        raise ValueError('salun_lr must be positive.')
    if salun_unlearning_epochs < 1:
        raise ValueError('salun_unlearning_epochs must be at least 1.')
    if salun_batch_size is None:
        salun_batch_size = batch_size
    if salun_batch_size < 1:
        raise ValueError('salun_batch_size must be at least 1.')
    if num_unlearning_operations < 2:
        raise ValueError('Active ADP requires at least two deletions.')
    if num_unlearning_operations > probe_data.size(0):
        raise ValueError('The probe pool is smaller than the deletion budget.')
    if num_design_shadow_models < 2:
        raise ValueError('Active selection needs at least two shadow pairs.')
    if design_target_data.size(0) != design_target_labels.size(0):
        raise ValueError('Design target data/label sizes differ.')
    if design_shadow_data.size(0) != design_shadow_labels.size(0):
        raise ValueError('Design shadow data/label sizes differ.')

    salun_args = SimpleNamespace(
        device=device,
        num_classes=int(num_classes),
        salun_mask_ratio=float(salun_mask_ratio),
        salun_lr=float(salun_lr),
        unlearning_epochs=int(salun_unlearning_epochs),
        batch_size=int(salun_batch_size),
    )
    return _select_global_probe_sequence_salun(
        design_target_data,
        design_target_labels,
        design_shadow_data,
        design_shadow_labels,
        probe_data,
        probe_labels,
        num_unlearning_operations,
        num_design_shadow_models,
        epochs,
        lr,
        num_classes,
        pretrained,
        selection_ridge,
        device,
        batch_size,
        salun_args,
        salun_seed,
    )
