import argparse
import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset

try:
    from .target_specific_adp import run_target_specific_adp
except ImportError:
    from target_specific_adp import run_target_specific_adp



def args_parser():
    parser = argparse.ArgumentParser(
        description='Exact-retraining ADP with a MNIST MLP.'
    )
    parser.add_argument('--mia_max_samples', type=int, default=200,
                        help='Upper bound on the number of fixed-target ADP trials.')
    parser.add_argument('--adp_target_trials', type=int, default=4,
                        help='Number of independent fixed target records evaluated per seed.')
    parser.add_argument('--adp_shadow_pairs', type=int, default=4,
                        help='Matched H1/H0 shadow pairs per design, calibration, and validation fold.')
    parser.add_argument('--model_train_size', type=int, default=5000,
                        help='Number of samples used to train each original model.')
    parser.add_argument('--original_epochs', type=int, default=20,
                        help='Epochs used to train each target/shadow original model.')
    parser.add_argument('--erase_count', type=int, default=4,
                        help='Number of sequential ADP deletion requests.')
    parser.add_argument('--deletion_batch_size', type=int, default=1,
                        help='Must be 1: ADP issues one single-record deletion per release.')
    parser.add_argument('--adp_whitebox_layers', type=int, default=4,
                        help='Number of trainable leaf layers used by white-box ADP, selected from output to input.')
    parser.add_argument('--adp_probe_pool_size', type=int, default=64,
                        help='Number of attacker-owned candidate probes available for ranking.')
    parser.add_argument('--adp_shortlist_size', type=int, default=12,
                        help='Candidates retained by the auxiliary influence screen.')
    parser.add_argument('--adp_fisher_samples', type=int, default=256,
                        help='Auxiliary samples used for diagonal empirical-Fisher preconditioning.')
    parser.add_argument('--adp_fisher_damping', type=float, default=1e-3,
                        help='Damping added to the auxiliary diagonal empirical Fisher.')
    parser.add_argument('--adp_decoder_ridge', type=float, default=1e-3,
                        help='Ridge used by the conditional ADP decoder.')
    parser.add_argument('--adp_reference_fraction', type=float, default=0.2,
                        help='Fraction of MNIST training data reserved for the public reference domain.')
    parser.add_argument('--comparison_bootstrap_samples', type=int, default=10000,
                        help='Paired bootstrap replicates for 95% AUROC-difference intervals.')
    parser.add_argument('--data_root', type=str, default='/home/wwq/Data/data/mnist',
                        help='MNIST data directory.')
    parser.add_argument('--retrain_epochs', type=int, default=None,
                        help='Epochs for every exact-retraining release; defaults to original_epochs.')
    args = parser.parse_args()
    return args


class MNISTMLP(nn.Module):
    """Lightweight four-layer MLP used by every MNIST ADP world."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(28 * 28, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.linear_head = nn.Linear(64, num_classes)

    def forward_features(self, x):
        x = self.flatten(x)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return F.relu(self.hidden3(x))

    def forward(self, x):
        return self.linear_head(self.forward_features(x))


def init_classifier(args):
    """Construct the same fast MNIST MLP in every ADP world."""
    model = MNISTMLP(num_classes=args.num_classes)
    return model.to(args.device)


def train_classifier_epoch(dataloader, model, optimizer, loss_fn, args):
    """One ordinary supervised epoch for the fast MNIST MLP classifier."""
    model.train()
    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)
        if y.ndim == 2:
            y = y.argmax(dim=1)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model

def make_loader(ds, batch_size, shuffle=True, num_workers=1, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


def _rankdata_average(values):
    """Average ranks for ROC-AUC tie handling, implemented without sklearn/scipy."""
    values = np.asarray(values)
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j

    return ranks


def _binary_auc(member_scores, nonmember_scores):
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([
        np.ones(len(member_scores), dtype=np.int64),
        np.zeros(len(nonmember_scores), dtype=np.int64),
    ])
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _rankdata_average(scores)
    sum_pos_ranks = ranks[labels == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _tpr_at_fpr(member_scores, nonmember_scores, target_fpr=0.01):
    """TPR when the decision threshold is chosen to keep non-member FPR <= target_fpr."""
    member_scores = np.asarray(member_scores, dtype=np.float64)
    nonmember_scores = np.asarray(nonmember_scores, dtype=np.float64)
    if len(member_scores) == 0 or len(nonmember_scores) == 0:
        return float("nan")

    thresholds = np.unique(np.concatenate([member_scores, nonmember_scores]))
    thresholds = np.concatenate([[np.inf], thresholds[::-1], [-np.inf]])
    best_tpr = 0.0
    for threshold in thresholds:
        fpr = float(np.mean(nonmember_scores >= threshold))
        if fpr <= target_fpr + 1e-12:
            tpr = float(np.mean(member_scores >= threshold))
            best_tpr = max(best_tpr, tpr)
    return best_tpr


def _evaluate_member_scores(member_scores, nonmember_scores):
    labels = np.concatenate([
        np.ones(len(member_scores), dtype=np.int64),
        np.zeros(len(nonmember_scores), dtype=np.int64),
    ])
    scores = np.concatenate([member_scores, nonmember_scores])
    member_decisions = np.asarray(member_scores) >= 0.5
    nonmember_decisions = np.asarray(nonmember_scores) >= 0.5
    return {
        "attack_auc": _binary_auc(member_scores, nonmember_scores),
        "attack_acc": float(np.mean((scores >= 0.5) == labels)),
        # Adv = Pr[hat b=1 | b=1] - Pr[hat b=1 | b=0], with the
        # calibrated membership decision hat b = 1[score >= 0.5].
        "attack_advantage": float(
            np.mean(member_decisions) - np.mean(nonmember_decisions)
        ),
        "attack_tpr_at_1fpr": _tpr_at_fpr(
            member_scores, nonmember_scores, target_fpr=0.01
        ),
    }
def _regularized_solve(matrix, rhs, ridge):
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    regularized = matrix + ridge * identity
    try:
        return torch.linalg.solve(regularized, rhs)
    except RuntimeError:
        return torch.matmul(torch.linalg.pinv(regularized), rhs)


def _fit_conditional_response_model(member_history, member_response,
                                    nonmember_history, nonmember_response, ridge):
    """Fit one Gaussian ADP response model conditioned on the prior history."""
    dtype = torch.float64
    sm, sn = member_history.to(dtype), nonmember_history.to(dtype)
    rm, rn = member_response.to(dtype), nonmember_response.to(dtype)
    all_history = torch.cat([sm, sn], dim=0)
    all_response = torch.cat([rm, rn], dim=0)
    history_mean = all_history.mean(0)
    history_std = all_history.std(0, unbiased=False).clamp_min(1e-8)
    response_mean = all_response.mean(0)
    response_std = all_response.std(0, unbiased=False).clamp_min(1e-8)
    sm = (sm - history_mean) / history_std
    sn = (sn - history_mean) / history_std
    rm = (rm - response_mean) / response_std
    rn = (rn - response_mean) / response_std
    centered_s = torch.cat([sm - sm.mean(0), sn - sn.mean(0)], dim=0)
    centered_r = torch.cat([rm - rm.mean(0), rn - rn.mean(0)], dim=0)
    regression = _regularized_solve(
        torch.matmul(centered_s.t(), centered_s),
        torch.matmul(centered_s.t(), centered_r),
        ridge,
    )
    em = rm - torch.matmul(sm, regression)
    en = rn - torch.matmul(sn, regression)
    mu_member, mu_nonmember = em.mean(0), en.mean(0)
    centered_e = torch.cat([em - mu_member, en - mu_nonmember], dim=0)
    denominator = max(1, centered_e.shape[0] - 2)
    covariance = torch.matmul(centered_e.t(), centered_e) / denominator
    direction = _regularized_solve(
        covariance, (mu_member - mu_nonmember)[:, None], ridge
    ).squeeze(1)
    regularized_covariance = covariance + ridge * torch.eye(
        covariance.shape[0], dtype=dtype, device=covariance.device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(regularized_covariance)
    inverse_sqrt = torch.matmul(
        eigenvectors * eigenvalues.clamp_min(1e-12).rsqrt().unsqueeze(0),
        eigenvectors.t(),
    )
    separation = mu_member - mu_nonmember
    whitened_separation = torch.matmul(inverse_sqrt, separation)
    midpoint = 0.5 * (mu_member + mu_nonmember)
    return {
        "regression": regression,
        "direction": direction,
        "midpoint": midpoint,
        "history_mean": history_mean,
        "history_std": history_std,
        "response_mean": response_mean,
        "response_std": response_std,
        "whitened_separation": whitened_separation,
        "separation_score": float(torch.dot(separation, direction).item()),
    }


def _conditional_round_logits(decoder, history, response):
    dtype = decoder["direction"].dtype
    normalized_history = (
        history.to(dtype) - decoder["history_mean"]
    ) / decoder["history_std"]
    normalized_response = (
        response.to(dtype) - decoder["response_mean"]
    ) / decoder["response_std"]
    residual = normalized_response - torch.matmul(
        normalized_history, decoder["regression"]
    )
    return torch.matmul(residual - decoder["midpoint"], decoder["direction"])


def _adp_history(static, responses, round_index):
    """Return [s_0, delta_1, ..., delta_{t-1}] for round_index=t-1."""
    if round_index == 0:
        return static
    prefix = responses[:, :round_index, :].reshape(responses.shape[0], -1)
    return torch.cat([static, prefix], dim=1)


def _fit_sequential_conditional_decoder(member_static, member_responses,
                                        nonmember_static, nonmember_responses,
                                        ridge):
    if member_responses.ndim != 3 or nonmember_responses.ndim != 3:
        raise ValueError('ADP responses must have shape [samples, rounds, features].')
    if member_responses.shape[1:] != nonmember_responses.shape[1:]:
        raise ValueError('Member/non-member ADP response shapes must match.')
    decoders = []
    for round_index in range(member_responses.shape[1]):
        member_history = _adp_history(
            member_static, member_responses, round_index
        )
        nonmember_history = _adp_history(
            nonmember_static, nonmember_responses, round_index
        )
        decoders.append(_fit_conditional_response_model(
            member_history,
            member_responses[:, round_index, :],
            nonmember_history,
            nonmember_responses[:, round_index, :],
            ridge,
        ))
    return decoders


def _sequential_deletion_logits(decoders, static, responses):
    round_logits = []
    for round_index, decoder in enumerate(decoders):
        history = _adp_history(static, responses, round_index)
        round_logits.append(_conditional_round_logits(
            decoder, history, responses[:, round_index, :]
        ))
    stacked = torch.stack(round_logits, dim=1)
    return stacked.sum(dim=1), stacked


def _fit_logistic_fusion(member_static_logits, member_deletion_logits,
                         nonmember_static_logits, nonmember_deletion_logits,
                         ridge, max_iterations=50):
    """Calibrate static/deletion LLR scales on a held-out shadow fold."""
    dtype = torch.float64
    member_features = torch.stack([
        member_static_logits.to(dtype), member_deletion_logits.to(dtype)
    ], dim=1)
    nonmember_features = torch.stack([
        nonmember_static_logits.to(dtype), nonmember_deletion_logits.to(dtype)
    ], dim=1)
    features = torch.cat([member_features, nonmember_features], dim=0)
    labels = torch.cat([
        torch.ones(len(member_features), dtype=dtype),
        torch.zeros(len(nonmember_features), dtype=dtype),
    ])
    design = torch.cat([
        torch.ones((len(features), 1), dtype=dtype), features
    ], dim=1)
    coefficients = torch.zeros(design.shape[1], dtype=dtype)
    penalty = torch.diag(torch.tensor([0.0, ridge, ridge], dtype=dtype))
    for _ in range(max_iterations):
        probabilities = torch.sigmoid(torch.matmul(design, coefficients))
        weights = (probabilities * (1.0 - probabilities)).clamp_min(1e-6)
        gradient = torch.matmul(design.t(), probabilities - labels)
        gradient = gradient + torch.matmul(penalty, coefficients)
        hessian = torch.matmul(design.t(), weights[:, None] * design) + penalty
        step = _regularized_solve(hessian, gradient[:, None], 1e-8).squeeze(1)
        coefficients = coefficients - step
        if torch.norm(step).item() < 1e-8:
            break
    return coefficients


def _fused_logits(coefficients, static_logits, deletion_logits):
    dtype = coefficients.dtype
    return (
        coefficients[0]
        + coefficients[1] * static_logits.to(dtype)
        + coefficients[2] * deletion_logits.to(dtype)
    )


def _fit_gaussian_lda(member_features, nonmember_features, ridge):
    dtype = torch.float64
    member = member_features.to(dtype)
    nonmember = nonmember_features.to(dtype)
    all_features = torch.cat([member, nonmember], dim=0)
    feature_mean = all_features.mean(0)
    feature_std = all_features.std(0, unbiased=False).clamp_min(1e-8)
    member = (member - feature_mean) / feature_std
    nonmember = (nonmember - feature_mean) / feature_std
    mu_member, mu_nonmember = member.mean(0), nonmember.mean(0)
    centered = torch.cat([
        member - mu_member,
        nonmember - mu_nonmember,
    ], dim=0)
    denominator = max(1, centered.shape[0] - 2)
    covariance = torch.matmul(centered.t(), centered) / denominator
    direction = _regularized_solve(
        covariance, (mu_member - mu_nonmember)[:, None], ridge
    ).squeeze(1)
    return {
        "direction": direction,
        "midpoint": 0.5 * (mu_member + mu_nonmember),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


def _lda_logits(decoder, features):
    dtype = decoder["direction"].dtype
    normalized = (
        features.to(dtype) - decoder["feature_mean"]
    ) / decoder["feature_std"]
    return torch.matmul(
        normalized - decoder["midpoint"], decoder["direction"]
    )






def synchronize_device(args):
    if getattr(args, "device", None) is not None and args.device.type == "cuda":
        torch.cuda.synchronize(args.device)


def _stable_logit(probabilities, eps=1e-6):
    probabilities = probabilities.clamp(min=eps, max=1.0 - eps)
    return torch.log(probabilities) - torch.log1p(-probabilities)


@torch.no_grad()


@torch.no_grad()




@torch.no_grad()
def _collect_blackbox_adp_observables(model, dataloader, args):
    """Collect API-derived initial features and the target log-odds margin."""
    margin_batches = []
    static_batches = []
    model.to(args.device)
    model.eval()
    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)
        logits = model(x)
        probabilities = F.softmax(logits, dim=1)
        target_probability = probabilities.gather(1, y[:, None]).squeeze(1)
        margin = _stable_logit(target_probability)
        loss = -torch.log(target_probability.clamp_min(1e-6))
        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=1)
        static = torch.stack([margin, loss, entropy], dim=1)
        margin_batches.append(margin[:, None].cpu())
        static_batches.append(static.cpu())
    return torch.cat(margin_batches, dim=0), torch.cat(static_batches, dim=0)


@torch.no_grad()
def collect_blackbox_adp_step_response(previous_model, current_model,
                                       dataloader, args):
    """Compute the one-dimensional black-box response Delta log-odds."""
    previous_margin, _ = _collect_blackbox_adp_observables(
        previous_model, dataloader, args
    )
    current_margin, _ = _collect_blackbox_adp_observables(
        current_model, dataloader, args
    )
    previous_model.to('cpu')
    current_model.to('cpu')
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    return current_margin - previous_margin


def _last_trainable_leaf_layers(model, layer_count):
    """Select parameterized leaf layers in reverse (output-to-input) order."""
    if layer_count < 1:
        raise ValueError('adp_whitebox_layers must be at least 1.')
    parameterized_layers = []
    for module_name, module in model.named_modules():
        local_parameters = [
            local_name
            for local_name, parameter in module.named_parameters(recurse=False)
            if parameter.requires_grad
        ]
        if not local_parameters:
            continue
        full_parameter_names = [
            f"{module_name}.{local_name}" if module_name else local_name
            for local_name in local_parameters
        ]
        parameterized_layers.append({
            "name": module_name or "<root>",
            "parameter_names": full_parameter_names,
            "num_parameters": sum(
                dict(model.named_parameters())[name].numel()
                for name in full_parameter_names
            ),
        })
    if layer_count > len(parameterized_layers):
        raise ValueError(
            f"Requested {layer_count} white-box layers, but the model has only "
            f"{len(parameterized_layers)} trainable leaf layers."
        )
    return list(reversed(parameterized_layers[-layer_count:]))


def _target_log_odds(logits, labels):
    probabilities = F.softmax(logits, dim=1)
    target_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
    return _stable_logit(target_probability)


def _layer_signed_target_work(previous_model, x, y, layer, parameter_delta):
    """Evaluate <grad_layer loss_i, Delta layer>/sqrt(d_layer) for a batch."""
    if not hasattr(torch, 'func'):
        raise RuntimeError(
            'White-box ADP requires torch.func.jvp (PyTorch 2.0 or newer).'
        )
    previous_parameters = dict(previous_model.named_parameters())
    parameter_names = layer["parameter_names"]
    primal_parameters = tuple(previous_parameters[name] for name in parameter_names)

    def per_example_loss(*layer_parameters):
        replacements = dict(zip(parameter_names, layer_parameters))
        logits = torch.func.functional_call(
            previous_model, replacements, (x,), strict=False
        )
        return F.cross_entropy(logits, y, reduction='none')

    _, directional_loss = torch.func.jvp(
        per_example_loss,
        primal_parameters,
        tuple(parameter_delta[name] for name in parameter_names),
    )
    return directional_loss / math.sqrt(layer["num_parameters"])


def collect_whitebox_adp_step_response(previous_model, current_model,
                                       dataloader, args, selected_layers):
    """Compute [Delta margin, signed target work for the last L layers]."""
    previous_model.to(args.device).eval()
    current_model.to(args.device).eval()
    previous_parameters = dict(previous_model.named_parameters())
    current_parameters = dict(current_model.named_parameters())
    selected_parameter_names = {
        name
        for layer in selected_layers
        for name in layer["parameter_names"]
    }
    parameter_delta = {
        name: current_parameters[name].detach() - previous_parameters[name].detach()
        for name in selected_parameter_names
    }
    response_batches = []
    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            delta_margin = (
                _target_log_odds(current_model(x), y)
                - _target_log_odds(previous_model(x), y)
            )
        signed_work = [
            _layer_signed_target_work(
                previous_model, x, y, layer, parameter_delta
            )
            for layer in selected_layers
        ]
        response_batches.append(
            torch.stack([delta_margin] + signed_work, dim=1).detach().cpu()
        )
    previous_model.to('cpu')
    current_model.to('cpu')
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    return torch.cat(response_batches, dim=0)


def collect_blackbox_adp_trajectory_features(snapshots, dataloader, args):
    if len(snapshots) != args.erase_count + 1:
        raise ValueError('ADP requires exactly one released model per deletion round.')
    _, static_features = _collect_blackbox_adp_observables(
        snapshots[0], dataloader, args
    )
    responses = [
        collect_blackbox_adp_step_response(
            snapshots[t - 1], snapshots[t], dataloader, args
        )
        for t in range(1, len(snapshots))
    ]
    return torch.stack(responses, dim=1), static_features


def collect_whitebox_adp_trajectory_features(snapshots, dataloader, args,
                                             selected_layers):
    if len(snapshots) != args.erase_count + 1:
        raise ValueError('ADP requires exactly one released model per deletion round.')
    _, static_features = _collect_blackbox_adp_observables(
        snapshots[0], dataloader, args
    )
    responses = [
        collect_whitebox_adp_step_response(
            snapshots[t - 1], snapshots[t], dataloader, args, selected_layers
        )
        for t in range(1, len(snapshots))
    ]
    return torch.stack(responses, dim=1), static_features








def train_classifier_from_scratch(dataset, train_indices, args, seed, name,
                                  epochs, verbose=True, phase='training'):
    """Initialize a fresh MNIST MLP and train it on exactly train_indices."""
    if len(train_indices) == 0:
        raise ValueError(f'{name} cannot retrain on an empty retained set.')
    if epochs < 1:
        raise ValueError('Training epochs must be at least 1.')

    set_random_seed(int(seed))
    model = init_classifier(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(
        Subset(dataset, list(train_indices)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=False,
    )
    for epoch in range(epochs):
        model = train_classifier_epoch(loader, model, optimizer, loss_fn, args)
        if verbose:
            print(f"{name} {phase} epoch {epoch + 1}/{epochs}")
    model.eval()
    return model


def train_original_model(dataset, train_indices, args, seed, name):
    """Train one original MNIST MLP world on exactly the supplied indices."""
    return train_classifier_from_scratch(
        dataset=dataset,
        train_indices=train_indices,
        args=args,
        seed=seed,
        name=name,
        epochs=args.original_epochs,
        verbose=True,
        phase='original training',
    )


def retrain_unlearn_one_record(model, augmented_dataset, remaining_indices,
                               probe_index, args, name, deletion_step,
                               verbose=False, retrain_seed=None,
                               request_kind='active_probe'):
    """Delete one record and replace model with a fresh retained-set retrain.

    The incoming model is used only as the pre-deletion release when the ADP
    response is measured. Its parameters and optimizer state are never used to
    initialize the post-deletion model.
    """
    remaining_set = set(remaining_indices)
    if probe_index not in remaining_set:
        raise ValueError(f"Probe {probe_index} is not retained in {name}")
    next_remaining = [idx for idx in remaining_indices if idx != probe_index]
    if retrain_seed is None:
        retrain_seed = int(torch.initial_seed() % (2 ** 31 - 1))

    if verbose:
        print(
            f"\n{name}: single-record exact-retraining deletion "
            f"{deletion_step}/{args.erase_count}, probe_index={probe_index}, "
            f"remaining={len(next_remaining)}, retrain_seed={retrain_seed}"
        )

    retrained_model = train_classifier_from_scratch(
        dataset=augmented_dataset,
        train_indices=next_remaining,
        args=args,
        seed=retrain_seed,
        name=name,
        epochs=args.retrain_epochs,
        verbose=verbose,
        phase=f'retraining release {deletion_step}',
    )
    # Keep the in-place interface used by conditional probe design while
    # replacing every parameter with the independently retrained release.
    model.load_state_dict(retrained_model.state_dict())
    model.to(args.device).eval()
    retrained_model.to('cpu')
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()

    record = {
        "deletion_step": deletion_step,
        "request_kind": request_kind,
        "deleted_index": int(probe_index),
        # Retained for compatibility with existing result consumers.
        "probe_index": int(probe_index),
        "probe_indices": [int(probe_index)],
        "num_forget_records": 1,
        "num_retained_records": len(next_remaining),
        "retrain_seed": int(retrain_seed),
        "retrain_epochs": int(args.retrain_epochs),
        "fresh_initialization": True,
    }
    return next_remaining, [record]


def retrain_unlearn_sequential_records(original_model, augmented_dataset,
                                       train_indices, probe_schedule,
                                       args, name,
                                       request_kind='active_probe'):
    """Return [M_0,...,M_k]; every M_t is retrained from scratch."""
    if args.deletion_batch_size != 1:
        raise ValueError('ADP requires deletion_batch_size=1.')
    if len(probe_schedule) != args.erase_count:
        raise ValueError(
            f"Expected {args.erase_count} single-record probes, "
            f"received {len(probe_schedule)}"
        )
    if len(set(probe_schedule)) != len(probe_schedule):
        raise ValueError('Every ADP probe must be a distinct owned record.')

    model = copy.deepcopy(original_model).to(args.device)
    snapshots = [copy.deepcopy(original_model).to('cpu').eval()]
    remaining_indices = list(train_indices)
    history = []
    start_time = time.time()
    trajectory_seed = int(torch.initial_seed() % (2 ** 31 - 1))
    for deletion_step, probe_index in enumerate(probe_schedule, start=1):
        remaining_indices, request_history = retrain_unlearn_one_record(
            model,
            augmented_dataset,
            remaining_indices,
            int(probe_index),
            args,
            name,
            deletion_step,
            verbose=True,
            retrain_seed=trajectory_seed + deletion_step,
            request_kind=request_kind,
        )
        history.append(request_history)
        snapshots.append(copy.deepcopy(model).to('cpu').eval())
        model.to(args.device)

    synchronize_device(args)
    model.to('cpu')
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    return snapshots, time.time() - start_time, history

def configure_mnist_unlearning_args():
    args = args_parser()
    args.gpu = 0
    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu'
    )
    # This entry point executes the paper method; random-k is evaluated as an
    # explicit baseline inside target_specific_adp rather than as a run mode.
    args.adp_schedule = 'conditional_logdet'
    args.lr = 0.001
    args.batch_size = 256
    args.num_classes = 10
    args.seeds = [0, 1, 2, 3, 4]
    if args.retrain_epochs is None:
        args.retrain_epochs = args.original_epochs

    if args.deletion_batch_size != 1:
        raise ValueError(
            'This ADP protocol requires deletion_batch_size=1 so every '
            'release corresponds to one owned record.'
        )
    if args.erase_count > args.adp_probe_pool_size:
        raise ValueError(
            'adp_probe_pool_size must cover every distinct deletion record.'
        )
    if args.adp_shortlist_size < args.erase_count:
        raise ValueError('adp_shortlist_size must be at least erase_count.')
    if args.adp_target_trials < 2:
        raise ValueError('adp_target_trials must be at least 2.')
    if args.adp_shadow_pairs < 2:
        raise ValueError('adp_shadow_pairs must be at least 2.')
    if args.adp_fisher_samples < 1:
        raise ValueError('adp_fisher_samples must be at least 1.')
    if args.adp_fisher_damping <= 0.0:
        raise ValueError('adp_fisher_damping must be positive.')
    if args.adp_whitebox_layers < 1:
        raise ValueError('adp_whitebox_layers must be at least 1.')
    if args.original_epochs < 1:
        raise ValueError('original_epochs must be at least 1.')
    if args.retrain_epochs < 1:
        raise ValueError('retrain_epochs must be at least 1.')
    if args.comparison_bootstrap_samples < 1:
        raise ValueError('comparison_bootstrap_samples must be at least 1.')
    return args


def load_mnist_data(args):
    """Load the deterministic MNIST training view used by every ADP world."""
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set_train = MNIST(
        args.data_root,
        train=True,
        transform=mnist_transform,
        download=True,
    )
    train_set = MNIST(
        args.data_root,
        train=True,
        transform=mnist_transform,
        download=False,
    )
    return train_set, train_set_train


def set_random_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std


def paired_comparison_summary(left_values, right_values, seed,
                              bootstrap_samples=10000):
    """Summarize paired left-right differences across matched experiment seeds.

    The confidence interval is a deterministic paired percentile bootstrap.
    The p-value is a one-sided paired sign-flip test for H1: mean(left-right)>0,
    matching the paper's requirement that ADP exceed each baseline.
    """
    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError('Paired comparisons require arrays with identical shape.')
    finite = np.isfinite(left) & np.isfinite(right)
    differences = left[finite] - right[finite]
    if differences.size == 0:
        return {
            "n": 0,
            "mean_difference": float("nan"),
            "std_difference": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "sign_flip_p_greater": float("nan"),
        }

    mean_difference, std_difference = mean_std(differences)
    if bootstrap_samples < 1:
        raise ValueError('comparison_bootstrap_samples must be at least 1.')
    rng = np.random.RandomState(seed)
    sampled_positions = rng.randint(
        0, differences.size, size=(bootstrap_samples, differences.size)
    )
    bootstrap_means = differences[sampled_positions].mean(axis=1)
    ci95_low, ci95_high = np.quantile(bootstrap_means, [0.025, 0.975])

    if differences.size <= 20:
        assignments = np.arange(1 << differences.size, dtype=np.uint64)[:, None]
        bit_positions = np.arange(differences.size, dtype=np.uint64)[None, :]
        signs = 2.0 * ((assignments >> bit_positions) & 1).astype(np.float64) - 1.0
    else:
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(max(bootstrap_samples, 10000), differences.size),
        )
    null_means = (signs * differences[None, :]).mean(axis=1)
    p_greater = float(np.mean(null_means >= mean_difference - 1e-15))
    return {
        "n": int(differences.size),
        "mean_difference": mean_difference,
        "std_difference": std_difference,
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
        "sign_flip_p_greater": p_greater,
    }


def _target_specific_retrain_update_one(previous_model, dataset,
                                        remaining_indices, probe_index,
                                        args, name, deletion_step, seed):
    """Replay one exact-retraining candidate with a paired deterministic seed."""
    candidate = copy.deepcopy(previous_model).to(args.device)
    set_random_seed(seed)
    next_remaining, history = retrain_unlearn_one_record(
        candidate,
        dataset,
        remaining_indices,
        probe_index,
        args,
        name,
        deletion_step,
        verbose=False,
        retrain_seed=seed,
        request_kind='active_probe',
    )
    return candidate.to('cpu').eval(), next_remaining, history


def _target_specific_driver():
    """Bind the common paper-faithful protocol to exact retraining."""
    return {
        "train_original": train_original_model,
        "update_one": _target_specific_retrain_update_one,
        "unlearn_sequence": retrain_unlearn_sequential_records,
        "set_seed": set_random_seed,
        "collect_bb_observables": _collect_blackbox_adp_observables,
        "collect_bb_step": collect_blackbox_adp_step_response,
        "collect_wb_step": collect_whitebox_adp_step_response,
        "collect_bb_trajectory": collect_blackbox_adp_trajectory_features,
        "collect_wb_trajectory": collect_whitebox_adp_trajectory_features,
        "select_layers": _last_trainable_leaf_layers,
        "fit_conditional": _fit_conditional_response_model,
        "fit_sequential": _fit_sequential_conditional_decoder,
        "fit_static": _fit_gaussian_lda,
        "deletion_logits": _sequential_deletion_logits,
        "static_logits": _lda_logits,
        "fit_fusion": _fit_logistic_fusion,
        "fused_logits": _fused_logits,
        "evaluate_scores": _evaluate_member_scores,
    }


def run_adp_setting_for_seed(args, seed):
    """Run target-specific ADP with exact retraining as the state update."""
    set_random_seed(seed)
    print(
        f"\n===== Target-Specific Active Deletion Probing "
        f"with Exact-Retraining/MLP, seed={seed} ====="
    )
    train_set, train_set_train = load_mnist_data(args)
    return run_target_specific_adp(
        args,
        seed,
        train_set,
        train_set_train,
        _target_specific_driver(),
    )


def _summarize_access_setting_results(results, access, bootstrap_samples=10000):
    def attack_result(result):
        return result["access_results"][access]

    metric_paths = {
        "trajectory_auc": lambda r: attack_result(r)["mia"]["trajectory"]["attack_auc"],
        "static_auc": lambda r: attack_result(r)["mia"]["static"]["attack_auc"],
        "deletion_only_auc": lambda r: attack_result(r)["mia"]["deletion"]["attack_auc"],
        "auc_gain": lambda r: attack_result(r)["mia"]["auc_gain"],
        "trajectory_attack_advantage": lambda r: attack_result(r)["mia"]["trajectory"]["attack_advantage"],
        "static_attack_advantage": lambda r: attack_result(r)["mia"]["static"]["attack_advantage"],
        "passive_one_auc": lambda r: attack_result(r)["baselines"]["passive_one"]["mia"]["trajectory"]["attack_auc"],
        "passive_one_auc_gain": lambda r: attack_result(r)["baselines"]["passive_one"]["mia"]["auc_gain"],
        "passive_one_attack_advantage": lambda r: attack_result(r)["baselines"]["passive_one"]["mia"]["trajectory"]["attack_advantage"],
        "random_k_auc": lambda r: attack_result(r)["baselines"]["random_sequential"]["mia"]["trajectory"]["attack_auc"],
        "random_k_auc_gain": lambda r: attack_result(r)["baselines"]["random_sequential"]["mia"]["auc_gain"],
        "random_k_attack_advantage": lambda r: attack_result(r)["baselines"]["random_sequential"]["mia"]["trajectory"]["attack_advantage"],
        "adp_minus_passive_auc": lambda r: attack_result(r)["comparisons"]["adp_minus_passive_auc"],
        "adp_minus_random_auc": lambda r: attack_result(r)["comparisons"]["adp_minus_random_auc"],
        "deg_count": lambda r: attack_result(r)["mia"]["deg_count"],
        "deg_rate": lambda r: attack_result(r)["mia"]["deg_rate"],
        "trajectory_tpr_at_1fpr": lambda r: attack_result(r)["mia"]["trajectory"]["attack_tpr_at_1fpr"],
        "attacking_time": lambda r: attack_result(r)["attacking_time"],
    }
    print(
        f"\n===== Active Deletion Probing Exact-Retraining/MLP "
        f"{access} summary ====="
    )
    summary = {}
    for name, getter in metric_paths.items():
        value_mean, value_std = mean_std([getter(result) for result in results])
        summary[name] = {"mean": value_mean, "std": value_std}
        print(f"{name}={value_mean:.4f}±{value_std:.4f}")

    paired_specs = {
        "adp_vs_static_auc": (
            lambda r: attack_result(r)["mia"]["trajectory"]["attack_auc"],
            lambda r: attack_result(r)["mia"]["static"]["attack_auc"],
        ),
        "adp_vs_passive_auc": (
            lambda r: attack_result(r)["mia"]["trajectory"]["attack_auc"],
            lambda r: attack_result(r)["baselines"]["passive_one"]["mia"]["trajectory"]["attack_auc"],
        ),
        "adp_vs_random_auc": (
            lambda r: attack_result(r)["mia"]["trajectory"]["attack_auc"],
            lambda r: attack_result(r)["baselines"]["random_sequential"]["mia"]["trajectory"]["attack_auc"],
        ),
        "adp_vs_static_advantage": (
            lambda r: attack_result(r)["mia"]["trajectory"]["attack_advantage"],
            lambda r: attack_result(r)["mia"]["static"]["attack_advantage"],
        ),
        "adp_vs_passive_advantage": (
            lambda r: attack_result(r)["mia"]["trajectory"]["attack_advantage"],
            lambda r: attack_result(r)["baselines"]["passive_one"]["mia"]["trajectory"]["attack_advantage"],
        ),
        "adp_vs_random_advantage": (
            lambda r: attack_result(r)["mia"]["trajectory"]["attack_advantage"],
            lambda r: attack_result(r)["baselines"]["random_sequential"]["mia"]["trajectory"]["attack_advantage"],
        ),
    }
    paired = {}
    for comparison_index, (name, (left_getter, right_getter)) in enumerate(
        paired_specs.items()
    ):
        comparison = paired_comparison_summary(
            [left_getter(result) for result in results],
            [right_getter(result) for result in results],
            seed=91000 + comparison_index,
            bootstrap_samples=bootstrap_samples,
        )
        paired[name] = comparison
        print(
            f"{name}_delta={comparison['mean_difference']:.4f}"
            f"±{comparison['std_difference']:.4f}, "
            f"paired_bootstrap_95%CI=[{comparison['ci95_low']:.4f}, "
            f"{comparison['ci95_high']:.4f}], "
            f"sign_flip_p_greater={comparison['sign_flip_p_greater']:.6f}"
        )
    summary["paired_comparisons"] = paired
    auc_success_names = (
        "adp_vs_static_auc",
        "adp_vs_passive_auc",
        "adp_vs_random_auc",
    )
    auc_success_criterion_met = all(
        paired[name]["mean_difference"] > 0.0
        and paired[name]["ci95_low"] > 0.0
        and paired[name]["sign_flip_p_greater"] < 0.05
        for name in auc_success_names
    )
    advantage_success_names = (
        "adp_vs_static_advantage",
        "adp_vs_passive_advantage",
        "adp_vs_random_advantage",
    )
    advantage_success_criterion_met = all(
        paired[name]["mean_difference"] > 0.0
        and paired[name]["ci95_low"] > 0.0
        and paired[name]["sign_flip_p_greater"] < 0.05
        for name in advantage_success_names
    )
    success_criterion_met = (
        auc_success_criterion_met and advantage_success_criterion_met
    )
    summary["threat_model_auc_success_criterion_met"] = (
        auc_success_criterion_met
    )
    summary["threat_model_advantage_success_criterion_met"] = (
        advantage_success_criterion_met
    )
    summary["threat_model_success_criterion_met"] = success_criterion_met
    print(
        "threat_model_success_criterion_met="
        f"{success_criterion_met} "
        f"(auc={auc_success_criterion_met}, "
        f"advantage={advantage_success_criterion_met})"
    )
    return summary


def summarize_adp_results(results, bootstrap_samples=10000):
    """Report black-box and white-box experiments as separate settings."""
    return {
        access: _summarize_access_setting_results(
            results, access, bootstrap_samples=bootstrap_samples
        )
        for access in ("blackbox", "whitebox")
    }


def test_adp_remaining_sample_setting():
    args = configure_mnist_unlearning_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("device", args.device)
    print("Exact-Retraining MLP ADP evaluation seeds:", args.seeds)
    results = [run_adp_setting_for_seed(args, seed) for seed in args.seeds]
    return results, summarize_adp_results(
        results,
        bootstrap_samples=args.comparison_bootstrap_samples,
    )


# Backward-compatible name for existing launch scripts.
test_paper_remaining_sample_setting = test_adp_remaining_sample_setting


if __name__ == "__main__":
    test_adp_remaining_sample_setting()
