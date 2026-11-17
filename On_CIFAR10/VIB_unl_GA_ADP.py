import argparse
import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset



def args_parser():
    parser = argparse.ArgumentParser(
        description='Hessian-preconditioned GA ADP with a CIFAR-10 ResNet-18.'
    )
    parser.add_argument('--dataset', choices=['CIFAR10'], default='CIFAR10')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--mia_max_samples', type=int, default=200,
                        help='Maximum member/non-member samples used for membership inference attack.')
    parser.add_argument('--model_train_size', type=int, default=5000,
                        help='Number of samples used to train each original model.')
    parser.add_argument('--original_epochs', type=int, default=100,
                        help='Epochs used to train each target/shadow original model.')
    parser.add_argument('--erase_count', type=int, default=4,
                        help='Number of sequential ADP deletion requests.')
    parser.add_argument('--deletion_batch_size', type=int, default=1,
                        help='Must be 1: ADP issues one single-record deletion per release.')
    parser.add_argument('--adp_access', choices=['blackbox', 'whitebox'], default='whitebox',
                        help='ADP observation model. White-box adds signed classifier-head target work.')
    parser.add_argument(
        '--adp_schedule',
        choices=['conditional_logdet', 'coupling', 'gradient_cosine', 'random'],
        default='conditional_logdet',
        help='Precommitted probe design; conditional_logdet implements the paper method.',
    )
    parser.add_argument('--adp_probe_pool_size', type=int, default=64,
                        help='Number of attacker-owned candidate probes available for ranking.')
    parser.add_argument('--adp_shortlist_size', type=int, default=12,
                        help='Candidates retained by the auxiliary influence screen.')
    parser.add_argument('--adp_schedule_targets', type=int, default=128,
                        help='Maximum known candidate targets used by the auxiliary coupling screen.')
    parser.add_argument('--adp_fisher_samples', type=int, default=256,
                        help='Auxiliary samples used for diagonal empirical-Fisher preconditioning.')
    parser.add_argument('--adp_fisher_damping', type=float, default=1e-3,
                        help='Damping added to the auxiliary diagonal empirical Fisher.')
    parser.add_argument('--adp_decoder_ridge', type=float, default=1e-3,
                        help='Ridge used by the conditional ADP decoder.')
    parser.add_argument('--adp_reference_fraction', type=float, default=0.2,
                        help='Fraction of CIFAR-10 training data reserved for the public reference domain.')
    parser.add_argument('--adp_reference_epochs', type=int, default=None,
                        help='Reference-model epochs; defaults to original_epochs.')
    parser.add_argument('--data_root', type=str, default='/home/wwq/Data/data/cifar',
                        help='CIFAR-10 data directory.')
    parser.add_argument('--hessian_ga_steps', type=int, default=1,
                        help='Hessian-GA updates performed for each deletion request.')
    parser.add_argument('--hessian_ga_lr', type=float, default=1.0,
                        help='Multiplier for the inverse-Hessian ascent update.')
    parser.add_argument('--hessian_damping', type=float, default=0.1,
                        help='Initial damping added to the retain Hessian.')
    parser.add_argument('--hessian_cg_iters', type=int, default=10)
    parser.add_argument('--hessian_cg_tolerance', type=float, default=1e-5)
    parser.add_argument('--hessian_batches', type=int, default=8,
                        help='Deterministic retain batches used for Hessian-vector products.')
    parser.add_argument('--hessian_max_update_norm', type=float, default=1.0,
                        help='Maximum norm of one GA update; non-positive disables clipping.')
    args = parser.parse_args()
    return args


class CIFARResNet18(nn.Module):
    """Torchvision ResNet-18 with the standard CIFAR stem and one CE head."""
    def __init__(self, num_classes=10):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        self.backbone = backbone

    @property
    def linear_head(self):
        """Expose the only classification head without registering it twice."""
        return self.backbone.fc

    def forward_features(self, x):
        net = self.backbone
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.layer4(x)
        x = net.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.linear_head(self.forward_features(x))


def init_classifier(args):
    """Construct the single CIFAR-10 ResNet-18 used by every ADP world."""
    model = CIFARResNet18(num_classes=args.num_classes)
    return model.to(args.device)
def train_classifier_epoch(dataloader, model, optimizer, loss_fn, args):
    """One ordinary supervised epoch for the single ResNet-18 classifier."""
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


def build_attack_loader(dataset, indices, args):
    """Build an ordered black-box query loader for a fixed candidate set."""
    return make_loader(
        Subset(dataset, list(indices)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
    )
def _evaluate_member_scores(member_scores, nonmember_scores):
    labels = np.concatenate([
        np.ones(len(member_scores), dtype=np.int64),
        np.zeros(len(nonmember_scores), dtype=np.int64),
    ])
    scores = np.concatenate([member_scores, nonmember_scores])
    return {
        "attack_auc": _binary_auc(member_scores, nonmember_scores),
        "attack_acc": float(np.mean((scores >= 0.5) == labels)),
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


def _sigmoid_numpy(logits):
    return torch.sigmoid(logits).cpu().numpy()
def _take_class_matched_indices(pool, desired_labels, labels, rng):
    """Select distinct indices whose labels match desired_labels one-for-one."""
    by_class = {}
    for index in pool:
        by_class.setdefault(int(labels[index]), []).append(int(index))
    for values in by_class.values():
        rng.shuffle(values)
    selected = []
    offsets = {class_id: 0 for class_id in by_class}
    for label in desired_labels:
        label = int(label)
        candidates = by_class.get(label, [])
        offset = offsets.get(label, 0)
        if offset >= len(candidates):
            raise ValueError(f"Not enough class-{label} records for matched worlds")
        selected.append(candidates[offset])
        offsets[label] = offset + 1
    return selected


def build_adp_domain_splits(labels, args, seed):
    """Create independent reference, paired-shadow, and target domains.

    Shadow H1 and H0 have equal size and identical class composition.  They
    differ only by replacing every calibration target with a same-class record.
    The same attacker-owned probe pool is present in both shadow worlds and the
    target world, as required by the precommitted ADP threat model.
    """
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    all_indices = rng.permutation(len(labels))
    probe_pool = all_indices[:args.adp_probe_pool_size].tolist()
    remaining = all_indices[args.adp_probe_pool_size:]
    reference_size = int(round(args.adp_reference_fraction * len(labels)))
    if reference_size < args.model_train_size:
        raise ValueError(
            "adp_reference_fraction leaves fewer records than model_train_size"
        )
    if reference_size >= len(remaining):
        raise ValueError('Reference and owned-probe domains exhaust CIFAR-10.')
    reference_domain = remaining[:reference_size]
    private_domains = remaining[reference_size:]
    midpoint = len(private_domains) // 2
    shadow_domain = private_domains[:midpoint]
    target_domain = private_domains[midpoint:]

    reference_train_size = min(args.model_train_size, len(reference_domain))
    reference_train = rng.choice(
        reference_domain, size=reference_train_size, replace=False
    ).tolist()

    shadow_rng = np.random.RandomState(seed + 1009)
    shadow_domain = shadow_rng.permutation(shadow_domain)
    shadow_partitions = np.array_split(shadow_domain, 3)
    shadow_folds = {}
    for fold_offset, (fold_name, partition) in enumerate(zip(
        ('design', 'calibration', 'validation'), shadow_partitions
    )):
        fold_rng = np.random.RandomState(seed + 1100 + fold_offset)
        partition = fold_rng.permutation(partition).tolist()
        private_train_size = args.model_train_size - len(probe_pool)
        query_count = min(
            args.mia_max_samples,
            private_train_size // 2,
            len(partition) - private_train_size,
        )
        if query_count < 2:
            raise ValueError(
                f"Not enough records for independent {fold_name} shadow worlds"
            )
        shadow_targets = partition[:query_count]
        replacement_pool = partition[query_count:]
        replacements = _take_class_matched_indices(
            replacement_pool, labels[shadow_targets], labels, fold_rng
        )
        excluded = set(shadow_targets) | set(replacements)
        base_candidates = [idx for idx in partition if idx not in excluded]
        base_size = private_train_size - len(shadow_targets)
        if base_size < 1 or base_size > len(base_candidates):
            raise ValueError(
                f"Invalid model size for the {fold_name} paired shadow worlds"
            )
        shadow_base = fold_rng.choice(
            base_candidates, size=base_size, replace=False
        ).tolist()
        shadow_folds[fold_name] = {
            "member_train_indices": shadow_base + probe_pool + shadow_targets,
            "nonmember_train_indices": shadow_base + probe_pool + replacements,
            "query_indices": shadow_targets,
        }

    target_rng = np.random.RandomState(seed + 2018)
    target_domain = target_rng.permutation(target_domain)
    positive_cut = int(round(0.8 * len(target_domain)))
    positive_pool = target_domain[:positive_cut]
    negative_pool = target_domain[positive_cut:]
    target_private_size = args.model_train_size - len(probe_pool)
    if target_private_size < 1 or target_private_size > len(positive_pool):
        raise ValueError('Invalid target private-set size after adding owned probes.')
    target_private_train = target_rng.choice(
        positive_pool, size=target_private_size, replace=False
    ).tolist()
    target_train = probe_pool + target_private_train
    member_candidates = target_private_train
    member_count = min(args.mia_max_samples, len(member_candidates))
    target_members = target_rng.choice(
        member_candidates, size=member_count, replace=False
    ).tolist()
    target_nonmembers = _take_class_matched_indices(
        negative_pool.tolist(), labels[target_members], labels, target_rng
    )

    return {
        "reference": {"train_indices": reference_train},
        "shadow": {
            "probe_pool_indices": probe_pool,
            **shadow_folds,
        },
        "target": {
            "train_indices": target_train,
            "probe_pool_indices": probe_pool,
            "member_query_indices": target_members,
            "nonmember_query_indices": target_nonmembers,
        },
    }


def synchronize_device(args):
    if getattr(args, "device", None) is not None and args.device.type == "cuda":
        torch.cuda.synchronize(args.device)


def _stable_logit(probabilities, eps=1e-6):
    probabilities = probabilities.clamp(min=eps, max=1.0 - eps)
    return torch.log(probabilities) - torch.log1p(-probabilities)


@torch.no_grad()
def _resnet_classifier_quantities(model, x, y):
    """Deterministic logits and exact per-example final-head CE gradients."""
    model.eval()
    features = model.forward_features(x)
    logits = model.linear_head(features)
    probabilities = F.softmax(logits, dim=1)
    one_hot = F.one_hot(y, num_classes=logits.shape[1]).to(probabilities.dtype)
    output_error = probabilities - one_hot
    grad_weight = output_error.unsqueeze(2) * features.unsqueeze(1)
    gradient_embedding = torch.cat([grad_weight.flatten(1), output_error], dim=1)
    head_norm = torch.sqrt(
        output_error.square().sum(dim=1)
        * (features.square().sum(dim=1) + 1.0)
    )
    head_size = model.linear_head.weight.numel() + model.linear_head.bias.numel()
    head_energy = torch.log(head_norm / math.sqrt(head_size) + 1e-12).unsqueeze(1)
    return logits, gradient_embedding, head_energy


@torch.no_grad()
def _classifier_gradient_embeddings(model, dataset, indices, args, batch_size=32):
    loader = make_loader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    embeddings = []
    model.to(args.device)
    model.eval()
    for x, y in loader:
        x, y = x.to(args.device), y.to(args.device)
        _, gradients, _ = _resnet_classifier_quantities(model, x, y)
        embeddings.append(gradients.cpu())
    return torch.cat(embeddings, dim=0)


def screen_adp_probe_candidates(reference_model, dataset, reference_indices,
                                probe_indices, candidate_target_indices,
                                args, seed):
    """Auxiliary influence screen; this is not the final ADP design score."""
    screen_count = args.erase_count
    if args.adp_schedule == 'conditional_logdet':
        screen_count = max(screen_count, args.adp_shortlist_size)
    screen_count = min(screen_count, len(probe_indices))
    rng = np.random.RandomState(seed)
    targets = np.asarray(candidate_target_indices, dtype=np.int64).copy()
    rng.shuffle(targets)
    targets = targets[:args.adp_schedule_targets].tolist()

    if args.adp_schedule == 'random':
        order = rng.permutation(len(probe_indices))
        screened = [probe_indices[i] for i in order[:screen_count]]
        return screened, {
            "schedule": "random",
            "screened_scores": [float("nan")] * len(screened),
            "screen_count": len(screened),
        }

    probe_gradients = _classifier_gradient_embeddings(
        reference_model, dataset, probe_indices, args
    )
    target_gradients = _classifier_gradient_embeddings(
        reference_model, dataset, targets, args
    )
    fisher_indices = []
    if args.adp_schedule in ('coupling', 'conditional_logdet'):
        fisher_indices = np.asarray(reference_indices, dtype=np.int64).copy()
        rng.shuffle(fisher_indices)
        fisher_indices = fisher_indices[:args.adp_fisher_samples].tolist()
        fisher_gradients = _classifier_gradient_embeddings(
            reference_model, dataset, fisher_indices, args
        )
        fisher_diagonal = fisher_gradients.square().mean(dim=0)
        scale = torch.sqrt(fisher_diagonal + args.adp_fisher_damping)
        probe_gradients = probe_gradients / scale.unsqueeze(0)
        target_gradients = target_gradients / scale.unsqueeze(0)

    probe_gradients = F.normalize(probe_gradients, p=2, dim=1, eps=1e-12)
    target_gradients = F.normalize(target_gradients, p=2, dim=1, eps=1e-12)
    coupling = torch.matmul(target_gradients, probe_gradients.t())
    scores = torch.clamp(coupling, min=0.0).mean(dim=0)
    selected_positions = torch.argsort(
        scores, descending=True
    )[:screen_count].tolist()
    screened = [probe_indices[i] for i in selected_positions]
    screen_name = (
        'fisher_gradient_coupling'
        if args.adp_schedule in ('coupling', 'conditional_logdet')
        else 'gradient_cosine'
    )
    return screened, {
        "screen": screen_name,
        "screened_scores": [float(scores[i]) for i in selected_positions],
        "mean_positive_coupling": float(scores.mean()),
        "positive_coupling_fraction": float((coupling > 0).float().mean()),
        "num_schedule_targets": len(targets),
        "num_fisher_samples": len(fisher_indices),
        "screen_count": len(screened),
    }


@torch.no_grad()
def _collect_adp_observables(model, dataloader, args,
                             collect_head_gradients=False):
    """Collect initial features and target observables in a fixed order."""
    margin_batches = []
    static_batches = []
    gradient_batches = []
    model.to(args.device)
    model.eval()
    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)
        logits, gradients, head_energy = _resnet_classifier_quantities(model, x, y)
        probabilities = F.softmax(logits, dim=1)
        target_probability = probabilities.gather(1, y[:, None]).squeeze(1)
        margin = _stable_logit(target_probability)
        loss = -torch.log(target_probability.clamp_min(1e-6))
        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=1)
        static = torch.stack([margin, loss, entropy], dim=1)
        if args.adp_access == 'whitebox':
            static = torch.cat([static, head_energy], dim=1)
        margin_batches.append(margin[:, None].cpu())
        static_batches.append(static.cpu())
        if collect_head_gradients:
            gradient_batches.append(gradients.cpu())
    gradient_features = None
    if collect_head_gradients:
        gradient_features = torch.cat(gradient_batches, dim=0)
    return (
        torch.cat(margin_batches, dim=0),
        torch.cat(static_batches, dim=0),
        gradient_features,
    )


@torch.no_grad()
def collect_adp_step_response(previous_model, current_model, dataloader, args):
    """Compute Delta-margin and the signed target work of the classifier head."""
    previous_margin, _, previous_gradients = _collect_adp_observables(
        previous_model,
        dataloader,
        args,
        collect_head_gradients=(args.adp_access == 'whitebox'),
    )
    current_margin, _, _ = _collect_adp_observables(
        current_model, dataloader, args, collect_head_gradients=False
    )
    response = current_margin - previous_margin
    if args.adp_access == 'whitebox':
        previous_head = previous_model.linear_head
        current_head = current_model.linear_head
        parameter_delta = torch.cat([
            (current_head.weight.detach().cpu()
             - previous_head.weight.detach().cpu()).reshape(-1),
            (current_head.bias.detach().cpu()
             - previous_head.bias.detach().cpu()).reshape(-1),
        ])
        signed_target_work = torch.matmul(
            previous_gradients, parameter_delta
        ) / math.sqrt(parameter_delta.numel())
        response = torch.cat([response, signed_target_work[:, None]], dim=1)
    previous_model.to('cpu')
    current_model.to('cpu')
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    return response


def collect_adp_trajectory_features(snapshots, dataloader, args):
    if len(snapshots) != args.erase_count + 1:
        raise ValueError('ADP requires exactly one released model per deletion round.')
    _, static_features, _ = _collect_adp_observables(
        snapshots[0], dataloader, args, collect_head_gradients=False
    )
    responses = [
        collect_adp_step_response(
            snapshots[t - 1], snapshots[t], dataloader, args
        )
        for t in range(1, len(snapshots))
    ]
    return torch.stack(responses, dim=1), static_features


def adp_remaining_sample_mia(calibration_member_snapshots,
                             calibration_nonmember_snapshots,
                             validation_member_snapshots,
                             validation_nonmember_snapshots,
                             target_snapshots, shadow_calibration_loader,
                             shadow_validation_loader,
                             target_member_loader, target_nonmember_loader,
                             args, name='ADP'):
    """Fit per-release conditional LLRs and held-out static/deletion fusion."""
    shadow_mem_response, shadow_mem_static = collect_adp_trajectory_features(
        calibration_member_snapshots, shadow_calibration_loader, args
    )
    shadow_non_response, shadow_non_static = collect_adp_trajectory_features(
        calibration_nonmember_snapshots, shadow_calibration_loader, args
    )
    validation_mem_response, validation_mem_static = collect_adp_trajectory_features(
        validation_member_snapshots, shadow_validation_loader, args
    )
    validation_non_response, validation_non_static = collect_adp_trajectory_features(
        validation_nonmember_snapshots, shadow_validation_loader, args
    )
    target_mem_response, target_mem_static = collect_adp_trajectory_features(
        target_snapshots, target_member_loader, args
    )
    target_non_response, target_non_static = collect_adp_trajectory_features(
        target_snapshots, target_nonmember_loader, args
    )

    conditional_decoders = _fit_sequential_conditional_decoder(
        shadow_mem_static, shadow_mem_response,
        shadow_non_static, shadow_non_response,
        args.adp_decoder_ridge,
    )
    static_decoder = _fit_gaussian_lda(
        shadow_mem_static, shadow_non_static, args.adp_decoder_ridge
    )
    validation_deletion_mem_logits, _ = _sequential_deletion_logits(
        conditional_decoders, validation_mem_static, validation_mem_response
    )
    validation_deletion_non_logits, _ = _sequential_deletion_logits(
        conditional_decoders, validation_non_static, validation_non_response
    )
    validation_static_mem_logits = _lda_logits(
        static_decoder, validation_mem_static
    )
    validation_static_non_logits = _lda_logits(
        static_decoder, validation_non_static
    )
    fusion_coefficients = _fit_logistic_fusion(
        validation_static_mem_logits,
        validation_deletion_mem_logits,
        validation_static_non_logits,
        validation_deletion_non_logits,
        args.adp_decoder_ridge,
    )

    deletion_mem_logits, deletion_mem_round_logits = _sequential_deletion_logits(
        conditional_decoders, target_mem_static, target_mem_response
    )
    deletion_non_logits, deletion_non_round_logits = _sequential_deletion_logits(
        conditional_decoders, target_non_static, target_non_response
    )
    static_mem_logits = _lda_logits(static_decoder, target_mem_static)
    static_non_logits = _lda_logits(static_decoder, target_non_static)

    deletion_mem = _sigmoid_numpy(deletion_mem_logits)
    deletion_non = _sigmoid_numpy(deletion_non_logits)
    static_mem = _sigmoid_numpy(static_mem_logits)
    static_non = _sigmoid_numpy(static_non_logits)
    trajectory_mem = _sigmoid_numpy(_fused_logits(
        fusion_coefficients, static_mem_logits, deletion_mem_logits
    ))
    trajectory_non = _sigmoid_numpy(_fused_logits(
        fusion_coefficients, static_non_logits, deletion_non_logits
    ))

    deletion_metrics = _evaluate_member_scores(deletion_mem, deletion_non)
    static_metrics = _evaluate_member_scores(static_mem, static_non)
    trajectory_metrics = _evaluate_member_scores(trajectory_mem, trajectory_non)
    true_labels = np.concatenate([
        np.ones(len(trajectory_mem), dtype=np.int64),
        np.zeros(len(trajectory_non), dtype=np.int64),
    ])
    trajectory_confidence = np.concatenate([trajectory_mem, trajectory_non])
    static_confidence = np.concatenate([static_mem, static_non])
    true_trajectory_confidence = np.where(
        true_labels == 1, trajectory_confidence, 1.0 - trajectory_confidence
    )
    true_static_confidence = np.where(
        true_labels == 1, static_confidence, 1.0 - static_confidence
    )
    result = {
        "access": args.adp_access,
        "num_deletions": args.erase_count,
        "deletion_batch_size": args.deletion_batch_size,
        "num_deleted_records": args.erase_count * args.deletion_batch_size,
        "response_dim_per_request": int(shadow_mem_response.shape[2]),
        "response_dim": int(
            shadow_mem_response.shape[1] * shadow_mem_response.shape[2]
        ),
        "static_dim": int(shadow_mem_static.shape[1]),
        "deletion": deletion_metrics,
        "static": static_metrics,
        "trajectory": trajectory_metrics,
        "auc_gain": trajectory_metrics["attack_auc"] - static_metrics["attack_auc"],
        "deg_count": float(np.mean(true_trajectory_confidence > true_static_confidence)),
        "deg_rate": float(np.mean(true_trajectory_confidence - true_static_confidence)),
        "num_shadow_pairs": len(shadow_mem_response),
        "num_validation_pairs": len(validation_mem_response),
        "num_target_member": len(target_mem_response),
        "num_target_nonmember": len(target_non_response),
        "fusion_coefficients": fusion_coefficients.tolist(),
        "round_conditional_separation": [
            decoder["separation_score"] for decoder in conditional_decoders
        ],
        "target_member_round_logits": deletion_mem_round_logits.tolist(),
        "target_nonmember_round_logits": deletion_non_round_logits.tolist(),
    }
    print(f"\nActive Deletion Probing for {name}:")
    print(
        f"  model=resnet18, access={args.adp_access}, k={args.erase_count}, "
        f"records_per_request={args.deletion_batch_size}, "
        f"total_deleted={result['num_deleted_records']}, "
        f"response_dim={result['response_dim_per_request']}x{args.erase_count}, "
        f"trajectory_auc={trajectory_metrics['attack_auc']:.4f}, "
        f"static_auc={static_metrics['attack_auc']:.4f}, "
        f"deletion_only_auc={deletion_metrics['attack_auc']:.4f}, "
        f"auc_gain={result['auc_gain']:.4f}"
    )
    print(
        f"  trajectory_TPR@1%FPR={trajectory_metrics['attack_tpr_at_1fpr']:.4f}, "
        f"deletion_TPR@1%FPR={deletion_metrics['attack_tpr_at_1fpr']:.4f}, "
        f"DegCount={result['deg_count']:.4f}, DegRate={result['deg_rate']:.4f}"
    )
    print(
        f"  conditional_J={result['round_conditional_separation']}, "
        f"fusion_beta={result['fusion_coefficients']}"
    )
    return result


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def flatten_tensors(tensors):
    return torch.cat([tensor.contiguous().view(-1) for tensor in tensors])


def split_vector(vector, parameters):
    pieces = []
    offset = 0
    for parameter in parameters:
        next_offset = offset + parameter.numel()
        pieces.append(vector[offset:next_offset].view_as(parameter))
        offset = next_offset
    if offset != vector.numel():
        raise ValueError('Flat vector does not match the model parameter count.')
    return pieces


def materialize_batches(dataset, indices, args, max_batches=None):
    materialized_indices = list(indices)
    if max_batches is not None:
        max_records = max_batches * args.batch_size
        if len(materialized_indices) > max_records:
            # Cover the whole retained set deterministically instead of taking
            # only its first batches (whose construction order is structured).
            positions = np.linspace(
                0, len(materialized_indices) - 1, max_records, dtype=np.int64
            )
            materialized_indices = [
                materialized_indices[position] for position in positions
            ]
    dataloader = make_loader(
        Subset(dataset, materialized_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    batches = []
    for x, y in dataloader:
        batches.append((x, y))
    if not batches:
        raise ValueError('At least one batch is required for Hessian unlearning.')
    return batches


def mean_loss_and_gradient(model, parameters, batches, args):
    model.eval()
    loss_sum = None
    total = 0
    for x, y in batches:
        x, y = x.to(args.device), y.to(args.device)
        batch_loss = F.cross_entropy(model(x), y, reduction='sum')
        loss_sum = batch_loss if loss_sum is None else loss_sum + batch_loss
        total += len(x)
    mean_loss = loss_sum / float(total)
    gradients = torch.autograd.grad(mean_loss, parameters, create_graph=False)
    return mean_loss.detach(), flatten_tensors(gradients).detach()


@torch.no_grad()
def mean_loss_on_batches(model, batches, args):
    model.eval()
    loss_sum = 0.0
    total = 0
    for x, y in batches:
        x, y = x.to(args.device), y.to(args.device)
        loss_sum += F.cross_entropy(model(x), y, reduction='sum').item()
        total += len(x)
    return loss_sum / float(total)


def hessian_vector_product(model, parameters, retain_batches, vector, damping, args):
    """Compute (H_retain + damping I)v without materializing H."""
    model.eval()
    vector_parts = split_vector(vector, parameters)
    result = torch.zeros_like(vector)
    total = sum(len(y) for _, y in retain_batches)
    for x, y in retain_batches:
        x, y = x.to(args.device), y.to(args.device)
        loss = F.cross_entropy(model(x), y, reduction='sum')
        gradients = torch.autograd.grad(
            loss, parameters, create_graph=True, retain_graph=True
        )
        directional_derivative = sum(
            (gradient * direction).sum()
            for gradient, direction in zip(gradients, vector_parts)
        )
        products = torch.autograd.grad(
            directional_derivative, parameters, create_graph=False
        )
        result.add_(flatten_tensors(products).detach())
    return result / float(total) + damping * vector


def conjugate_gradient(operator, rhs, max_iterations, tolerance):
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    residual_sq = torch.dot(residual, residual)
    initial_sq = residual_sq.clone()
    if residual_sq.item() == 0.0:
        return solution, 0, 0.0, True
    positive_definite = True
    completed = 0
    for iteration in range(max_iterations):
        operator_direction = operator(direction)
        denominator = torch.dot(direction, operator_direction)
        if not torch.isfinite(denominator) or denominator.item() <= 1e-12:
            positive_definite = False
            break
        step_size = residual_sq / denominator
        solution = solution + step_size * direction
        residual = residual - step_size * operator_direction
        new_residual_sq = torch.dot(residual, residual)
        completed = iteration + 1
        if new_residual_sq <= (tolerance ** 2) * initial_sq:
            residual_sq = new_residual_sq
            break
        direction = residual + (new_residual_sq / residual_sq) * direction
        residual_sq = new_residual_sq
    relative_residual = float(
        torch.sqrt(residual_sq / initial_sq.clamp_min(1e-30)).item()
    )
    return solution, completed, relative_residual, positive_definite


def solve_inverse_hessian_gradient(
    model, parameters, retain_batches, forget_gradient, args
):
    damping = args.hessian_damping
    for _ in range(4):
        operator = lambda vector: hessian_vector_product(
            model, parameters, retain_batches, vector, damping, args
        )
        solution, iterations, residual, positive_definite = conjugate_gradient(
            operator,
            forget_gradient,
            max_iterations=args.hessian_cg_iters,
            tolerance=args.hessian_cg_tolerance,
        )
        info = (damping, iterations, residual, positive_definite)
        if positive_definite and torch.isfinite(solution).all():
            return solution, info
        damping *= 10.0
    return forget_gradient / damping, (damping, 0, 1.0, False)


def apply_flat_update(parameters, update):
    update_parts = split_vector(update, parameters)
    with torch.no_grad():
        for parameter, parameter_update in zip(parameters, update_parts):
            parameter.add_(parameter_update)


def train_original_model(dataset, train_indices, args, seed, name):
    """Train one ResNet-18 world on exactly the supplied indices."""
    set_random_seed(seed)
    model = init_classifier(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=False,
    )
    for epoch in range(args.original_epochs):
        model.train()
        model = train_classifier_epoch(loader, model, optimizer, loss_fn, args)
        print(f"{name} original training epoch {epoch + 1}/{args.original_epochs}")
    model.eval()
    return model


def hessian_ga_unlearn_one_record(model, dataset, remaining_indices,
                                  probe_index, args, name, deletion_step):
    """Apply one Hessian-GA request that forgets exactly one owned record."""
    if probe_index not in set(remaining_indices):
        raise ValueError(f"Probe {probe_index} is not retained in {name}")
    next_remaining = [idx for idx in remaining_indices if idx != probe_index]
    forget_batches = materialize_batches(
        dataset, [probe_index], args, max_batches=None
    )
    retain_batches = materialize_batches(
        dataset,
        next_remaining,
        args,
        max_batches=max(1, args.hessian_batches),
    )
    influence_scale = 1.0 / float(len(next_remaining))
    parameters = trainable_parameters(model)
    print(
        f"\n{name}: single-record Hessian-GA deletion "
        f"{deletion_step}/{args.erase_count}, probe_index={probe_index}, "
        f"remaining={len(next_remaining)}"
    )

    request_history = []
    for ga_step in range(args.hessian_ga_steps):
        forget_loss_before, forget_gradient = mean_loss_and_gradient(
            model, parameters, forget_batches, args
        )
        direction, solver_info = solve_inverse_hessian_gradient(
            model, parameters, retain_batches, forget_gradient, args
        )
        update = args.hessian_ga_lr * influence_scale * direction
        raw_norm = float(torch.norm(update, p=2).item())
        if (
            args.hessian_max_update_norm > 0
            and raw_norm > args.hessian_max_update_norm
        ):
            update = update * (
                args.hessian_max_update_norm / max(raw_norm, 1e-12)
            )
        update_norm = float(torch.norm(update, p=2).item())
        apply_flat_update(parameters, update)
        forget_loss_after = mean_loss_on_batches(model, forget_batches, args)
        damping, cg_iterations, cg_residual, positive_definite = solver_info
        record = {
            "deletion_step": deletion_step,
            "ga_step": ga_step + 1,
            "probe_index": int(probe_index),
            "probe_indices": [int(probe_index)],
            "num_forget_records": 1,
            "forget_loss_before": float(forget_loss_before.item()),
            "forget_loss_after": float(forget_loss_after),
            "update_norm": update_norm,
            "damping": damping,
            "cg_iterations": cg_iterations,
            "cg_relative_residual": cg_residual,
            "cg_positive_definite": positive_definite,
        }
        request_history.append(record)
        print(
            f"{name} Hessian-GA update {ga_step + 1}/"
            f"{args.hessian_ga_steps}: forget_loss="
            f"{record['forget_loss_before']:.6f}->"
            f"{record['forget_loss_after']:.6f}, "
            f"update_norm={update_norm:.6f}, damping={damping:.3g}, "
            f"cg_iters={cg_iterations}, cg_residual={cg_residual:.3e}"
        )
    model.eval()
    return next_remaining, request_history


def hessian_ga_unlearn_sequential_records(original_model, dataset,
                                           train_indices, probe_schedule,
                                           args, name):
    """Return [M_0,...,M_k], with one record removed at every release."""
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
    for deletion_step, probe_index in enumerate(probe_schedule, start=1):
        remaining_indices, request_history = hessian_ga_unlearn_one_record(
            model,
            dataset,
            remaining_indices,
            int(probe_index),
            args,
            name,
            deletion_step,
        )
        history.append(request_history)
        snapshots.append(copy.deepcopy(model).to('cpu').eval())
        model.to(args.device)

    synchronize_device(args)
    model.to('cpu')
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    return snapshots, time.time() - start_time, history


def select_conditional_logdet_schedule(
    shadow_member_original,
    shadow_nonmember_original,
    dataset,
    shadow_member_train_indices,
    shadow_nonmember_train_indices,
    screened_probes,
    design_loader,
    args,
):
    """Precommit ADP probes from conditional shadow separation and diversity.

    Candidate responses are generated by replaying the already selected prefix
    in paired shadow worlds.  No target-service observation is used here.
    """
    if len(screened_probes) < args.erase_count:
        raise ValueError('The influence shortlist is smaller than the ADP budget.')
    start_time = time.time()
    current_member = copy.deepcopy(shadow_member_original).to('cpu').eval()
    current_nonmember = copy.deepcopy(shadow_nonmember_original).to('cpu').eval()
    member_remaining = list(shadow_member_train_indices)
    nonmember_remaining = list(shadow_nonmember_train_indices)
    _, member_static, _ = _collect_adp_observables(
        current_member, design_loader, args, collect_head_gradients=False
    )
    _, nonmember_static, _ = _collect_adp_observables(
        current_nonmember, design_loader, args, collect_head_gradients=False
    )
    current_member.to('cpu')
    current_nonmember.to('cpu')
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    member_prefix_responses = []
    nonmember_prefix_responses = []
    selected = []
    selection_history = []
    member_history = []
    nonmember_history = []
    gram = None

    for round_index in range(args.erase_count):
        if member_prefix_responses:
            member_condition = torch.cat(
                [member_static] + member_prefix_responses, dim=1
            )
            nonmember_condition = torch.cat(
                [nonmember_static] + nonmember_prefix_responses, dim=1
            )
        else:
            member_condition = member_static
            nonmember_condition = nonmember_static

        best = None
        candidate_records = []
        for probe_index in screened_probes:
            if probe_index in selected:
                continue
            candidate_member = copy.deepcopy(current_member).to(args.device)
            next_member_remaining, candidate_member_history = hessian_ga_unlearn_one_record(
                candidate_member,
                dataset,
                member_remaining,
                int(probe_index),
                args,
                f"ADP design H1 probe={probe_index}",
                round_index + 1,
            )
            member_response = collect_adp_step_response(
                current_member, candidate_member, design_loader, args
            )
            candidate_nonmember = copy.deepcopy(current_nonmember).to(args.device)
            next_nonmember_remaining, candidate_nonmember_history = hessian_ga_unlearn_one_record(
                candidate_nonmember,
                dataset,
                nonmember_remaining,
                int(probe_index),
                args,
                f"ADP design H0 probe={probe_index}",
                round_index + 1,
            )
            nonmember_response = collect_adp_step_response(
                current_nonmember, candidate_nonmember, design_loader, args
            )
            response_model = _fit_conditional_response_model(
                member_condition,
                member_response,
                nonmember_condition,
                nonmember_response,
                args.adp_decoder_ridge,
            )
            signature = response_model["whitened_separation"]
            if gram is None:
                base_gram = torch.eye(len(signature), dtype=signature.dtype)
            else:
                base_gram = gram
            candidate_gram = base_gram + torch.outer(signature, signature)
            _, base_logdet = torch.linalg.slogdet(base_gram)
            _, candidate_logdet = torch.linalg.slogdet(candidate_gram)
            logdet_gain = float((candidate_logdet - base_logdet).item())
            record = {
                "probe_index": int(probe_index),
                "conditional_separation": response_model["separation_score"],
                "logdet_gain": logdet_gain,
            }
            candidate_records.append(record)
            if best is None or logdet_gain > best["logdet_gain"]:
                best = {
                    **record,
                    "member_model": candidate_member,
                    "nonmember_model": candidate_nonmember,
                    "member_remaining": next_member_remaining,
                    "nonmember_remaining": next_nonmember_remaining,
                    "member_response": member_response,
                    "nonmember_response": nonmember_response,
                    "signature": signature,
                    "member_history": candidate_member_history,
                    "nonmember_history": candidate_nonmember_history,
                }

        if best is None:
            raise RuntimeError('No unused ADP candidate remains during design.')
        selected.append(best["probe_index"])
        current_member = best["member_model"].to('cpu').eval()
        current_nonmember = best["nonmember_model"].to('cpu').eval()
        member_remaining = best["member_remaining"]
        nonmember_remaining = best["nonmember_remaining"]
        member_prefix_responses.append(best["member_response"])
        nonmember_prefix_responses.append(best["nonmember_response"])
        signature = best["signature"]
        if gram is None:
            gram = torch.eye(len(signature), dtype=signature.dtype)
        gram = gram + torch.outer(signature, signature)
        member_history.append(best["member_history"])
        nonmember_history.append(best["nonmember_history"])
        selection_history.append({
            "round": round_index + 1,
            "selected_probe": best["probe_index"],
            "selected_conditional_separation": best["conditional_separation"],
            "selected_logdet_gain": best["logdet_gain"],
            "candidates": candidate_records,
        })
        print(
            f"ADP design round {round_index + 1}/{args.erase_count}: "
            f"selected={best['probe_index']}, "
            f"conditional_J={best['conditional_separation']:.6f}, "
            f"logdet_gain={best['logdet_gain']:.6f}"
        )

    synchronize_device(args)
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    return {
        "schedule": selected,
        "member_history": member_history,
        "nonmember_history": nonmember_history,
        "selection_history": selection_history,
        "elapsed": time.time() - start_time,
    }


def configure_cifar10_unlearning_args():
    args = args_parser()
    args.gpu = 0
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.dataset = 'CIFAR10'
    args.lr = 0.0001
    args.batch_size = 200

    args.num_classes = 10
    args.seeds = [0, 1, 2, 3, 4]

    if args.hessian_ga_steps < 1:
        raise ValueError('hessian_ga_steps must be at least 1.')
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
    if args.hessian_damping <= 0:
        raise ValueError('hessian_damping must be positive.')
    if args.hessian_batches < 1:
        raise ValueError('hessian_batches must be at least 1.')
    if args.hessian_cg_iters < 1:
        raise ValueError('hessian_cg_iters must be at least 1.')

    return args


def load_cifar10_data(args):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    train_aug_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    deterministic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    train_set_aug = CIFAR10(args.data_root, train=True, transform=train_aug_transform, download=True)
    train_set = CIFAR10(args.data_root, train=True, transform=deterministic_transform, download=False)
    test_set = CIFAR10(args.data_root, train=False, transform=deterministic_transform, download=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    args.x_shape = (3, 32, 32)
    return train_set, train_set_aug, test_set, test_loader
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
def run_adp_setting_for_seed(args, seed):
    """Run the matched ADP protocol with Hessian-GA as the state update."""
    set_random_seed(seed)
    print(f"\n===== Active Deletion Probing with Hessian-GA/ResNet-18, seed={seed} =====")
    train_set, train_set_aug, _, _ = load_cifar10_data(args)
    splits = build_adp_domain_splits(train_set.targets, args, seed)

    reference_epochs = (
        args.original_epochs
        if args.adp_reference_epochs is None
        else args.adp_reference_epochs
    )
    saved_original_epochs = args.original_epochs
    args.original_epochs = reference_epochs
    reference_model = train_original_model(
        train_set_aug,
        splits["reference"]["train_indices"],
        args,
        seed + 5000,
        "public reference",
    ).to('cpu')
    args.original_epochs = saved_original_epochs

    # Design, decoder fitting, and fusion calibration use independent paired
    # shadow worlds. Within a pair, H1/H0 share initialization and differ only
    # by same-class replacement of the membership-query records.
    shadow_originals = {}
    for fold_offset, fold_name in enumerate(
        ('design', 'calibration', 'validation')
    ):
        fold = splits["shadow"][fold_name]
        fold_seed = seed + 10000 + 1000 * fold_offset
        shadow_originals[fold_name] = {
            "member": train_original_model(
                train_set_aug,
                fold["member_train_indices"],
                args,
                fold_seed,
                f"{fold_name} shadow H1",
            ).to('cpu'),
            "nonmember": train_original_model(
                train_set_aug,
                fold["nonmember_train_indices"],
                args,
                fold_seed,
                f"{fold_name} shadow H0",
            ).to('cpu'),
        }
    target_original = train_original_model(
        train_set_aug,
        splits["target"]["train_indices"],
        args,
        seed + 20000,
        "target",
    ).to('cpu')

    shadow_design_loader = build_attack_loader(
        train_set, splits["shadow"]["design"]["query_indices"], args
    )
    target_schedule_candidates = (
        splits["target"]["member_query_indices"]
        + splits["target"]["nonmember_query_indices"]
    )
    screened_probes, screen_info = screen_adp_probe_candidates(
        reference_model,
        train_set,
        splits["reference"]["train_indices"],
        splits["shadow"]["probe_pool_indices"],
        target_schedule_candidates,
        args,
        seed + 30000,
    )
    reference_model.to('cpu')
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()

    shadow_design_time = 0.0
    design_member_history = None
    design_nonmember_history = None
    if args.adp_schedule == 'conditional_logdet':
        set_random_seed(seed + 50000)
        design_result = select_conditional_logdet_schedule(
            shadow_originals["design"]["member"],
            shadow_originals["design"]["nonmember"],
            train_set,
            splits["shadow"]["design"]["member_train_indices"],
            splits["shadow"]["design"]["nonmember_train_indices"],
            screened_probes,
            shadow_design_loader,
            args,
        )
        probe_schedule = design_result["schedule"]
        design_member_history = design_result["member_history"]
        design_nonmember_history = design_result["nonmember_history"]
        shadow_design_time = design_result["elapsed"]
        schedule_info = {
            "schedule": "conditional_logdet",
            "screen": screen_info,
            "selection_history": design_result["selection_history"],
            "precommitted_before_target_M0": True,
        }
        design_result = None
    else:
        probe_schedule = screened_probes[:args.erase_count]
        schedule_info = {
            "schedule": args.adp_schedule,
            "screen": screen_info,
            "precommitted_before_target_M0": True,
        }
    shadow_originals["design"] = None
    print("Precommitted probe schedule:", probe_schedule, schedule_info)

    shadow_snapshots = {}
    shadow_histories = {}
    shadow_unlearning_times = {}
    for fold_offset, fold_name in enumerate(('calibration', 'validation')):
        fold = splits["shadow"][fold_name]
        set_random_seed(seed + 70000 + 1000 * fold_offset)
        member_snapshots, member_time, member_history = hessian_ga_unlearn_sequential_records(
            shadow_originals[fold_name]["member"],
            train_set,
            fold["member_train_indices"],
            probe_schedule,
            args,
            f"{fold_name} shadow H1",
        )
        set_random_seed(seed + 70000 + 1000 * fold_offset)
        nonmember_snapshots, nonmember_time, nonmember_history = hessian_ga_unlearn_sequential_records(
            shadow_originals[fold_name]["nonmember"],
            train_set,
            fold["nonmember_train_indices"],
            probe_schedule,
            args,
            f"{fold_name} shadow H0",
        )
        shadow_snapshots[fold_name] = {
            "member": member_snapshots,
            "nonmember": nonmember_snapshots,
        }
        shadow_histories[fold_name] = {
            "member": member_history,
            "nonmember": nonmember_history,
        }
        shadow_unlearning_times[fold_name] = {
            "member": member_time,
            "nonmember": nonmember_time,
        }

    set_random_seed(seed + 60000)
    target_snapshots, target_unlearning_time, target_history = hessian_ga_unlearn_sequential_records(
        target_original,
        train_set,
        splits["target"]["train_indices"],
        probe_schedule,
        args,
        "target",
    )

    shadow_calibration_loader = build_attack_loader(
        train_set, splits["shadow"]["calibration"]["query_indices"], args
    )
    shadow_validation_loader = build_attack_loader(
        train_set, splits["shadow"]["validation"]["query_indices"], args
    )
    target_member_loader = build_attack_loader(
        train_set, splits["target"]["member_query_indices"], args
    )
    target_nonmember_loader = build_attack_loader(
        train_set, splits["target"]["nonmember_query_indices"], args
    )

    synchronize_device(args)
    attack_start = time.time()
    mia = adp_remaining_sample_mia(
        shadow_snapshots["calibration"]["member"],
        shadow_snapshots["calibration"]["nonmember"],
        shadow_snapshots["validation"]["member"],
        shadow_snapshots["validation"]["nonmember"],
        target_snapshots,
        shadow_calibration_loader,
        shadow_validation_loader,
        target_member_loader,
        target_nonmember_loader,
        args,
        name=f"Hessian-GA seed={seed}",
    )
    synchronize_device(args)
    return {
        "seed": seed,
        "mia": mia,
        "shadow_member_unlearning_time": shadow_unlearning_times[
            "calibration"
        ]["member"],
        "shadow_nonmember_unlearning_time": shadow_unlearning_times[
            "calibration"
        ]["nonmember"],
        "target_unlearning_time": target_unlearning_time,
        "shadow_member_history": shadow_histories["calibration"]["member"],
        "shadow_nonmember_history": shadow_histories[
            "calibration"
        ]["nonmember"],
        "shadow_histories": shadow_histories,
        "shadow_unlearning_times": shadow_unlearning_times,
        "design_member_history": design_member_history,
        "design_nonmember_history": design_nonmember_history,
        "target_history": target_history,
        "attacking_time": time.time() - attack_start,
        "shadow_design_time": shadow_design_time,
        "probe_schedule": probe_schedule,
        "shadow_probe_schedule": probe_schedule,
        "target_probe_schedule": probe_schedule,
        "schedule_info": schedule_info,
        "shadow_schedule_info": schedule_info,
        "target_schedule_info": schedule_info,
        "shadow_query_folds": {
            name: splits["shadow"][name]["query_indices"]
            for name in ('design', 'calibration', 'validation')
        },
    }


def summarize_adp_results(results):
    metric_paths = {
        "trajectory_auc": lambda r: r["mia"]["trajectory"]["attack_auc"],
        "static_auc": lambda r: r["mia"]["static"]["attack_auc"],
        "deletion_only_auc": lambda r: r["mia"]["deletion"]["attack_auc"],
        "auc_gain": lambda r: r["mia"]["auc_gain"],
        "deg_count": lambda r: r["mia"]["deg_count"],
        "deg_rate": lambda r: r["mia"]["deg_rate"],
        "trajectory_tpr_at_1fpr": lambda r: r["mia"]["trajectory"]["attack_tpr_at_1fpr"],
    }
    print("\n===== Active Deletion Probing Hessian-GA/ResNet-18 summary =====")
    summary = {}
    for name, getter in metric_paths.items():
        value_mean, value_std = mean_std([getter(result) for result in results])
        summary[name] = {"mean": value_mean, "std": value_std}
        print(f"{name}={value_mean:.4f}±{value_std:.4f}")
    return summary


def test_adp_remaining_sample_setting():
    args = configure_cifar10_unlearning_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("device", args.device)
    print("Hessian-GA ResNet-18 ADP evaluation seeds:", args.seeds)
    results = [run_adp_setting_for_seed(args, seed) for seed in args.seeds]
    return results, summarize_adp_results(results)


# Backward-compatible name for existing launch scripts.
test_paper_remaining_sample_setting = test_adp_remaining_sample_setting


if __name__ == "__main__":
    test_adp_remaining_sample_setting()
