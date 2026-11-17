import sys

sys.argv = ['']
del sys

import os
os.environ["TQDM_DISABLE"] = "1"     # 放在导入 tqdm 前


import argparse
import math
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
import copy
import random
import time

from tqdm import tqdm



def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', choices=['CIFAR10'], default='CIFAR10')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs for VIBI.')
    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'],
                        default='ResNet_4x')
    parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=1)
    parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
    parser.add_argument('--beta', type=float, default=0, help='beta in objective J = I(y,t) - beta * I(x,t).')
    parser.add_argument('--unlearning_ratio', type=float, default=0.1)
    parser.add_argument('--mia_max_samples', type=int, default=1000,
                        help='Maximum member/non-member samples used for membership inference attack.')
    parser.add_argument('--mia_attack_epochs', type=int, default=200,
                        help='Training epochs for learned membership inference attack.')
    parser.add_argument('--mia_attack_lr', type=float, default=0.01,
                        help='Learning rate for learned membership inference attack.')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples used for estimating expectation over p(t|x).')
    args = parser.parse_args()
    return args


class CNNEncoder(nn.Module):
    """
    A small CNN encoder for VAEs:
      - Input: (B, in_channels, H, W)
      - Output: (B, 2*dim_z)   # concat(mu, logvar)
    """
    def __init__(self, in_channels=1, dim_z=32, base=32, hidden=256,
                 norm='bn', dropout=0.0):
        super().__init__()

        # norm: 'bn' | 'gn' | 'none'
        if norm == 'bn':
            Norm = lambda c: nn.BatchNorm2d(c)
        elif norm == 'gn':
            Norm = lambda c: nn.GroupNorm(8, c)
        else:
            Norm = None

        def block(cin, cout, stride=1):
            layers = [nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1,
                                bias=(Norm is None))]
            if Norm is not None:
                layers.append(Norm(cout))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # Feature extractor (downsamp ×2 three times)
        self.features = nn.Sequential(
            block(in_channels, base, 1),
            block(base, base, 1),

            block(base, base*2, 2),   # /2
            block(base*2, base*2, 1),

            block(base*2, base*4, 2), # /4
            block(base*4, base*4, 1),

            block(base*4, base*8, 2), # /8
            nn.AdaptiveAvgPool2d(1),  # -> (B, base*8, 1, 1)
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(base*8, hidden)
        self.fc_out = nn.Linear(hidden, dim_z)  # concat(mu, logvar)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.features(x).flatten(1)              # (B, base*8)
        h = self.dropout(F.relu(self.fc1(h)))        # (B, hidden)
        z_params = self.fc_out(h)                    # (B, 2*dim_z)
        return z_params



class LinearModel(nn.Module):
    # 定义神经网络
    def __init__(self, n_feature=192, h_dim=3 * 32, n_output=10):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, n_output)  # output



    def forward(self, x):
        # 定义向前传播函数
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None):
        super().__init__()
        stride = stride or (1 if in_channels >= out_channels else 2)
        self.block = conv_block(in_channels, out_channels, stride)
        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x))


class ResNet(nn.Module):
    def __init__(self, in_channels, block_features, num_classes=10, headless=False):
        super().__init__()
        block_features = [block_features[0]] + block_features + ([num_classes] if headless else [])
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, block_features[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(block_features[0]),
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(block_features[i], block_features[i + 1])
            for i in range(len(block_features) - 1)
        ])
        self.linear_head = None if headless else nn.Linear(block_features[-1], num_classes)

    def forward(self, x):
        x = self.expand(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        if self.linear_head is not None:
            x = F.avg_pool2d(x, x.shape[-1])  # completely reduce spatial dimension
            x = self.linear_head(x.reshape(x.shape[0], -1))
        return x


def resnet18(in_channels, num_classes):
    block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    return ResNet(in_channels, block_features, num_classes)


def resnet34(in_channels, num_classes):
    block_features = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    return ResNet(in_channels, block_features, num_classes)





class VIB(nn.Module):
    def __init__(self, encoder, approximator, decoder):
        super().__init__()

        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder
        self.fc3 = nn.Linear(3 * 32 * 32, 3 * 32 * 32)  # output

    def explain(self, x, mode='topk'):
        """Returns the relevance scores
        """
        double_logits_z = self.encoder(x)  # (B, C, h, w)
        if mode == 'distribution':  # return the distribution over explanation
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].to(x.device)
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).to(x.device)
            logits_z = self.reparametrize(mu, logvar)
            return logits_z, mu, logvar
        elif mode == 'test':  # return top k pixels from input
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].to(x.device)
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).to(x.device)
            logits_z = self.reparametrize(mu, logvar)
            return logits_z

    def forward(self, x, mode='topk'):
        B = x.size(0)
        #         print("B, C, H, W", B, C, H, W)
        if mode == 'distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            return logits_z, logits_y, mu, logvar
        elif mode == 'with_reconstruction':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.reconstruction(logits_z, x)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'VAE':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)

            x_hat = self.reconstruction(logits_z, x)
            return logits_z, x_hat, mu, logvar

    def reconstruction(self, logits_z, x):
        B, dimZ = logits_z.shape
        logits_z = logits_z.reshape((B, -1))
        output_x = self.decoder(logits_z)
        return torch.sigmoid(self.fc3(output_x))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.randn_like(std))
        return eps.mul(std).add_(mu)



def init_vib(args):
    if args.dataset == 'CIFAR10':

        approximator = LinearModel(n_feature=args.dimZ)

        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr

    elif args.dataset == 'CIFAR100':
        approximator = LinearModel(n_feature=args.dimZ, n_output=100)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr

    vib = VIB(encoder, approximator, decoder)
    vib.to(args.device)
    return vib, lr





def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def vib_train_original_IB(dataset, model,optimizer, loss_fn, reconstruction_function, args):


    with tqdm(dataset) as pbar:
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            x.requires_grad = True

            # logits_z, logits_y, x_hat, mu, logvar = model(x, mode='with_reconstruction')
            logits_z, logits_y, mu, logvar = model(x, mode='distribution')
            # VAE two loss: KLD + MSE
            H_p_q = loss_fn(logits_y, y)

            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).to(args.device)
            KLD = torch.sum(KLD_element).mul_(-0.5).to(args.device)
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).to(args.device)

            # x_hat = x_hat.view(x_hat.size(0), -1)
            # x = x.view(x.size(0), -1)
            # BCE = reconstruction_function(x_hat, x)

            # + args.mse_rate * BCE
            loss = args.beta * KLD_mean + H_p_q

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)

            optimizer.step()

            # acc = (logits_y.argmax(dim=1) == y).float().mean().item()
            sigma = torch.sqrt_(torch.exp(logvar)).mean().item()

            accuracy = (logits_y.argmax(dim=1) == y).float().mean()

            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), I_xz= KLD_mean.item(), lr=optimizer.param_groups[0]['lr'])

    return model

@torch.no_grad()
def eva_vib(vib, dataloader_erase, args, name='test', epoch=999):
    # first, generate x_hat from trained vae
    vib.eval()

    num_total = 0
    num_correct = 0
    for batch_idx, (x, y) in enumerate(dataloader_erase):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        logits_z, logits_y, x_hat, mu, logvar = vib(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)


        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (logits_y.argmax(dim=1) == y).sum().item()
        num_total += len(x)

    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}, total_num:{num_total}')
    return acc

def split_dataset_for_unlearning(dataset, erase_ratio: float, seed: int, erase_dataset=None, remain_dataset=None):
    """Split a standard training dataset into erase/remain subsets."""
    erase_dataset = erase_dataset or dataset
    remain_dataset = remain_dataset or dataset
    rng = np.random.RandomState(seed)
    indices = np.arange(len(dataset))
    n = len(indices)
    n_erase = max(1, int(round(n * erase_ratio)))

    perm = rng.permutation(n)
    erase_indices = indices[perm[:n_erase]].tolist()
    remain_indices = indices[perm[n_erase:]].tolist()

    erase_subset = Subset(erase_dataset, erase_indices)
    remain_subset = Subset(remain_dataset, remain_indices)

    return erase_subset, remain_subset, erase_indices, remain_indices


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


def build_learned_mia_loaders(train_set, test_set, unlearning_data_info, args, seed=0):
    retain_indices = []
    for info in unlearning_data_info.values():
        retain_indices.extend(info["remain_indices"])

    if len(retain_indices) == 0:
        return None, None

    rng = np.random.RandomState(seed)
    n_attack = min(len(retain_indices), len(test_set), args.mia_max_samples)

    retain_indices = rng.choice(retain_indices, size=n_attack, replace=False).tolist()
    nonmember_indices = rng.choice(np.arange(len(test_set)), size=n_attack, replace=False).tolist()

    retain_loader = make_loader(Subset(train_set, retain_indices), batch_size=args.batch_size, shuffle=False, num_workers=1)
    nonmember_loader = make_loader(Subset(test_set, nonmember_indices), batch_size=args.batch_size, shuffle=False, num_workers=1)

    return retain_loader, nonmember_loader


@torch.no_grad()
def _collect_learned_mia_features(vib, dataloader, args, max_samples=None):
    vib.eval()
    features = []
    collected = 0

    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)
        if y.ndim == 2:
            y = y.argmax(dim=1)

        _, logits_y, _, _, _ = vib(x, mode='with_reconstruction')
        probs = F.softmax(logits_y, dim=1)
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        true_conf = probs.gather(1, y.view(-1, 1)).squeeze(1)
        loss = F.cross_entropy(logits_y, y, reduction='none')
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        top1 = sorted_probs[:, 0]
        top2 = sorted_probs[:, 1]
        margin = top1 - top2
        correct = (logits_y.argmax(dim=1) == y).float()

        batch_features = torch.cat([
            sorted_probs,
            true_conf.unsqueeze(1),
            (-loss).unsqueeze(1),
            (-entropy).unsqueeze(1),
            margin.unsqueeze(1),
            correct.unsqueeze(1),
        ], dim=1)

        features.append(batch_features.detach().cpu())
        collected += x.size(0)
        if max_samples is not None and collected >= max_samples:
            break

    return torch.cat(features, dim=0)[:max_samples]


class LearnedMIAAttack(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(x).squeeze(1)


def _standardize_attack_features(train_features, *other_features):
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    scaled_train = (train_features - mean) / std
    scaled_others = [(features - mean) / std for features in other_features]
    return scaled_train, scaled_others


@torch.no_grad()
def _evaluate_attack_classifier(attack_model, features):
    attack_model.eval()
    logits = attack_model(features)
    probs = torch.sigmoid(logits)
    return probs.cpu().numpy()


def learned_membership_inference_attack(vib, retain_loader, nonmember_loader, args, name='model'):
    """Train a neural MIA on retained members vs held-out non-members."""
    retain_features = _collect_learned_mia_features(vib, retain_loader, args, max_samples=args.mia_max_samples)
    nonmember_features = _collect_learned_mia_features(vib, nonmember_loader, args, max_samples=args.mia_max_samples)

    n_attack = min(len(retain_features), len(nonmember_features))
    retain_features = retain_features[:n_attack]
    nonmember_features = nonmember_features[:n_attack]

    train_features = torch.cat([retain_features, nonmember_features], dim=0)
    train_labels = torch.cat([
        torch.ones(len(retain_features)),
        torch.zeros(len(nonmember_features)),
    ], dim=0)

    train_features, _ = _standardize_attack_features(train_features)
    train_labels = train_labels.float()

    attack_model = LearnedMIAAttack(train_features.size(1))
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=args.mia_attack_lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(args.mia_attack_epochs):
        attack_model.train()
        logits = attack_model(train_features)
        loss = loss_fn(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_probs = _evaluate_attack_classifier(attack_model, train_features)
    retain_probs = train_probs[:len(retain_features)]
    nonmember_probs = train_probs[len(retain_features):]
    attack_auc = _binary_auc(retain_probs, nonmember_probs)
    attack_tpr_at_1fpr, _, _ = _tpr_at_fpr(retain_probs, nonmember_probs, target_fpr=0.01)
    attack_labels = np.concatenate([
        np.ones(len(retain_probs), dtype=np.float32),
        np.zeros(len(nonmember_probs), dtype=np.float32),
    ])
    attack_acc = float(np.mean((train_probs >= 0.5).astype(np.float32) == attack_labels))

    result = {
        "attack_auc": attack_auc,
        "attack_acc": attack_acc,
        "attack_tpr_at_1fpr": attack_tpr_at_1fpr,
    }

    print(f"\nLearned MIA results for {name}:")
    print(f"  MIA AUC={result['attack_auc']:.4f}")
    print(f"  MIA ACC={result['attack_acc']:.4f}")
    print(f"  MIA TPR@1%FPR={result['attack_tpr_at_1fpr']:.4f}")

    return result


@torch.no_grad()
def _collect_dfd_outputs(model, dataset, indices, args, batch_size=None):
    model.eval()
    batch_size = batch_size or args.batch_size
    loader = make_loader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=1)
    reps_logits = []
    reps_mu = []
    probs = []
    labels = []

    for x, y in loader:
        x = x.to(args.device)
        _, logits_y, mu, _ = model(x, mode='distribution')
        logits_y = logits_y.reshape(x.size(0), -1)
        reps_logits.append(logits_y.detach().cpu())
        reps_mu.append(mu.detach().cpu())
        probs.append(F.softmax(logits_y, dim=1).detach().cpu())
        labels.append(y.detach().cpu())

    return {
        "indices": np.asarray(indices, dtype=np.int64),
        "reps_logits": torch.cat(reps_logits, dim=0),
        "reps_mu": torch.cat(reps_mu, dim=0),
        "probs": torch.cat(probs, dim=0),
        "labels": torch.cat(labels, dim=0).long(),
    }


def _probability_stats(probs):
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    confidence = sorted_probs[:, 0]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    return {
        "confidence": confidence.numpy(),
        "margin": margin.numpy(),
    }


def _mean_knn_distance(anchor_reps, reference_reps, k, exclude_self=False):
    k_eff = min(k + int(exclude_self), reference_reps.size(0))
    distances = torch.cdist(anchor_reps, reference_reps)
    knn_distances = torch.topk(distances, k=k_eff, largest=False).values
    if exclude_self and k_eff > 1:
        knn_distances = knn_distances[:, 1:]
    return knn_distances.mean(dim=1).numpy()


def _knn_reference_positions(anchor_reps, reference_reps, k, exclude_self=False):
    k_eff = min(k + int(exclude_self), reference_reps.size(0))
    distances = torch.cdist(anchor_reps, reference_reps)
    positions = torch.topk(distances, k=k_eff, largest=False).indices
    if exclude_self and k_eff > 1:
        positions = positions[:, 1:]
    return positions.numpy()


def _sample_indices(indices, max_count, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if max_count is None or len(indices) <= max_count:
        return indices.tolist()
    return rng.choice(indices, size=max_count, replace=False).tolist()


def _match_control_anchors(forget_stats, control_stats, forget_density, control_density):
    forget_labels = forget_stats["labels"].numpy()
    control_labels = control_stats["labels"].numpy()
    forget_prob_stats = _probability_stats(forget_stats["probs"])
    control_prob_stats = _probability_stats(control_stats["probs"])

    control_matrix = np.stack([
        control_prob_stats["confidence"],
        control_prob_stats["margin"],
        control_density,
    ], axis=1)
    scale = np.std(control_matrix, axis=0)
    scale[scale < 1e-6] = 1.0

    matched_positions = []
    used = set()
    for i, label in enumerate(forget_labels):
        candidates = np.where(control_labels == label)[0]
        if len(candidates) == 0:
            candidates = np.arange(len(control_labels))
        candidates = np.asarray([idx for idx in candidates if idx not in used])
        if len(candidates) == 0:
            candidates = np.arange(len(control_labels))

        forget_vector = np.asarray([
            forget_prob_stats["confidence"][i],
            forget_prob_stats["margin"][i],
            forget_density[i],
        ])
        scores = np.abs((control_matrix[candidates] - forget_vector) / scale).sum(axis=1)
        chosen = int(candidates[int(np.argmin(scores))])
        matched_positions.append(chosen)
        used.add(chosen)

    return np.asarray(matched_positions, dtype=np.int64)


def _neighborhood_reps_with_anchor(anchor_outputs, reference_outputs, rep_key, anchor_pos, neighbor_positions):
    positions = np.asarray(neighbor_positions, dtype=np.int64)
    return torch.cat([
        anchor_outputs[rep_key][int(anchor_pos)].unsqueeze(0),
        reference_outputs[rep_key][positions],
    ], dim=0)


def _upper_triangle_values(matrix):
    n = matrix.size(0)
    if n < 2:
        return np.asarray([], dtype=np.float64)
    rows, cols = torch.triu_indices(n, n, offset=1)
    return matrix[rows, cols].numpy().astype(np.float64)


def _one_minus_corr(x, y):
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0 if np.allclose(x, y) else 1.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return float(1.0 - corr)


def _js_divergence(p, q):
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def _linear_cka(x, y):
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = torch.norm(x.t() @ y, p="fro").pow(2)
    xx = torch.norm(x.t() @ x, p="fro")
    yy = torch.norm(y.t() @ y, p="fro")
    return float((xy / (xx * yy + 1e-12)).item())


def _neighborhood_probs_with_anchor(anchor_outputs, reference_outputs, anchor_pos, neighbor_positions):
    positions = np.asarray(neighbor_positions, dtype=np.int64)
    return torch.cat([
        anchor_outputs["probs"][int(anchor_pos)].unsqueeze(0),
        reference_outputs["probs"][positions],
    ], dim=0)


def _local_fingerprint_vectors(anchor_m0_outputs, anchor_mu_outputs, reference_m0, reference_mu, rep_key, neighbor_positions, anchor_positions=None):
    """Per-anchor features: JS, geometry shifts, anchor relation shift, and local CKA shift."""
    vectors = []
    if anchor_positions is None:
        anchor_positions = np.arange(len(neighbor_positions))

    for anchor_pos, positions in zip(anchor_positions, neighbor_positions):
        h0 = _neighborhood_reps_with_anchor(anchor_m0_outputs, reference_m0, rep_key, anchor_pos, positions)
        hu = _neighborhood_reps_with_anchor(anchor_mu_outputs, reference_mu, rep_key, anchor_pos, positions)
        p0 = _neighborhood_probs_with_anchor(anchor_m0_outputs, reference_m0, anchor_pos, positions)
        pu = _neighborhood_probs_with_anchor(anchor_mu_outputs, reference_mu, anchor_pos, positions)

        h0_norm = F.normalize(h0, dim=1)
        hu_norm = F.normalize(hu, dim=1)
        k0 = h0_norm @ h0_norm.t()
        ku = hu_norm @ hu_norm.t()
        cosine_shift = _one_minus_corr(_upper_triangle_values(k0), _upper_triangle_values(ku))
        anchor_relation_shift = _one_minus_corr(
            k0[0, 1:].numpy().astype(np.float64),
            ku[0, 1:].numpy().astype(np.float64),
        )

        d0 = torch.cdist(h0, h0)
        du = torch.cdist(hu, hu)
        distance_shift = torch.norm(du - d0, p="fro") / (torch.norm(d0, p="fro") + 1e-12)
        local_cka_shift = 1.0 - _linear_cka(h0, hu)
        js_shift = float(_js_divergence(p0, pu).mean().item())

        vectors.append([
            js_shift,
            cosine_shift,
            float(distance_shift.item()),
            anchor_relation_shift,
            local_cka_shift,
        ])

    return np.asarray(vectors, dtype=np.float64)


def _local_geometry_shift_vectors(anchor_m0_outputs, anchor_mu_outputs, reference_m0, reference_mu, rep_key, neighbor_positions, anchor_positions=None):
    fingerprint_vectors = _local_fingerprint_vectors(
        anchor_m0_outputs,
        anchor_mu_outputs,
        reference_m0,
        reference_mu,
        rep_key,
        neighbor_positions,
        anchor_positions=anchor_positions,
    )
    return fingerprint_vectors[:, 1:3]


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _mann_whitney_u_pvalue(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    ranks = _rankdata_average(np.concatenate([x, y]))
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0
    std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if std_u == 0:
        return float(u1), float("nan")
    z = (u1 - mean_u) / std_u
    p = 2.0 * min(_normal_cdf(z), 1.0 - _normal_cdf(z))
    return float(u1), float(max(min(p, 1.0), 0.0))


def _ks_2sample_stat_pvalue(x, y):
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    values = np.sort(np.concatenate([x, y]))
    cdf1 = np.searchsorted(x, values, side="right") / n1
    cdf2 = np.searchsorted(y, values, side="right") / n2
    d_stat = float(np.max(np.abs(cdf1 - cdf2)))
    en = math.sqrt(n1 * n2 / (n1 + n2))
    p = 2.0 * math.exp(-2.0 * (en * d_stat) ** 2)
    return d_stat, float(max(min(p, 1.0), 0.0))


def _cliffs_delta(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def _summarize_geometry_vectors(forget_vectors, control_vectors):
    names = ["geom_cosine_shift", "geom_distance_shift"]
    return _summarize_named_vectors(forget_vectors, control_vectors, names)


def _summarize_named_vectors(forget_vectors, control_vectors, names):
    summary = {}
    for col, name in enumerate(names):
        forget_values = forget_vectors[:, col]
        control_values = control_vectors[:, col]
        _, mw_p = _mann_whitney_u_pvalue(forget_values, control_values)
        ks_stat, ks_p = _ks_2sample_stat_pvalue(forget_values, control_values)
        summary[name] = {
            "forget_mean": float(np.mean(forget_values)),
            "control_mean": float(np.mean(control_values)),
            "mann_whitney_p": mw_p,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "cliffs_delta": _cliffs_delta(forget_values, control_values),
            "auc": _binary_auc(forget_values, control_values),
        }
    return summary


def _cross_entropy_from_probs(probs, labels):
    labels = labels.long()
    true_probs = probs.gather(1, labels.view(-1, 1)).squeeze(1).clamp_min(1e-12)
    return -torch.log(true_probs)


def _traditional_mia_signal_vectors(m0_outputs, mu_outputs, positions=None):
    if positions is None:
        positions = np.arange(len(m0_outputs["labels"]))
    positions = np.asarray(positions, dtype=np.int64)

    labels = m0_outputs["labels"][positions]
    probs0 = m0_outputs["probs"][positions]
    probsu = mu_outputs["probs"][positions]
    sorted0, _ = torch.sort(probs0, dim=1, descending=True)
    sortedu, _ = torch.sort(probsu, dim=1, descending=True)
    true_conf0 = probs0.gather(1, labels.view(-1, 1)).squeeze(1)
    true_confu = probsu.gather(1, labels.view(-1, 1)).squeeze(1)
    loss0 = _cross_entropy_from_probs(probs0, labels)
    lossu = _cross_entropy_from_probs(probsu, labels)
    entropy0 = -(probs0 * torch.log(probs0.clamp_min(1e-12))).sum(dim=1)
    entropyu = -(probsu * torch.log(probsu.clamp_min(1e-12))).sum(dim=1)
    margin0 = sorted0[:, 0] - sorted0[:, 1]
    marginu = sortedu[:, 0] - sortedu[:, 1]

    vectors = torch.stack([
        sortedu[:, 0] - sorted0[:, 0],
        true_confu - true_conf0,
        lossu - loss0,
        entropyu - entropy0,
        marginu - margin0,
    ], dim=1)
    return vectors.numpy().astype(np.float64)


def _summarize_traditional_mia_signals(forget_stats, forget_mu_stats, control_stats, control_mu_stats, control_positions):
    names = [
        "top1_conf_delta",
        "true_conf_delta",
        "loss_delta",
        "entropy_delta",
        "margin_delta",
    ]
    forget_vectors = _traditional_mia_signal_vectors(forget_stats, forget_mu_stats)
    control_vectors = _traditional_mia_signal_vectors(control_stats, control_mu_stats, positions=control_positions)
    summary = _summarize_named_vectors(forget_vectors, control_vectors, names)

    print("\nTraditional MIA signal before-after diagnostic:")
    for name, stats in summary.items():
        print(
            f"  {name}: forget={stats['forget_mean']:.6f}, "
            f"control={stats['control_mean']:.6f}, "
            f"AUC={stats['auc']:.4f}, "
            f"MWU_p={stats['mann_whitney_p']:.4g}, "
            f"cliff_delta={stats['cliffs_delta']:.4f}"
        )

    return {
        "metrics": summary,
        "vectors": {
            "forget": forget_vectors,
            "control": control_vectors,
        },
    }


def _summarize_fingerprint_vectors(forget_vectors, control_vectors, rep_name):
    names = [
        "fp_js_shift",
        "fp_geom_cosine_shift",
        "fp_geom_distance_shift",
        "fp_anchor_relation_shift",
        "fp_local_cka_shift",
    ]
    summary = _summarize_named_vectors(forget_vectors, control_vectors, names)

    print(f"\nDFD deletion fingerprint diagnostic ({rep_name} representation):")
    for name, stats in summary.items():
        print(
            f"  {name}: forget={stats['forget_mean']:.6f}, "
            f"control={stats['control_mean']:.6f}, "
            f"AUC={stats['auc']:.4f}, "
            f"MWU_p={stats['mann_whitney_p']:.4g}, "
            f"cliff_delta={stats['cliffs_delta']:.4f}"
        )

    return summary


def _stratified_attack_split(num_pos, num_neg, rng, train_fraction=0.7):
    pos_perm = rng.permutation(num_pos)
    neg_perm = rng.permutation(num_neg)
    n_pos_train = max(1, min(num_pos - 1, int(round(num_pos * train_fraction)))) if num_pos > 1 else num_pos
    n_neg_train = max(1, min(num_neg - 1, int(round(num_neg * train_fraction)))) if num_neg > 1 else num_neg
    return (
        pos_perm[:n_pos_train],
        neg_perm[:n_neg_train],
        pos_perm[n_pos_train:],
        neg_perm[n_neg_train:],
    )


def _tpr_at_fpr(pos_scores, neg_scores, target_fpr=0.01):
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan"), float("nan"), float("nan")
    threshold = float(np.quantile(neg_scores, 1.0 - target_fpr))
    tpr = float(np.mean(pos_scores >= threshold))
    fpr = float(np.mean(neg_scores >= threshold))
    return tpr, fpr, tpr - fpr


def _train_dfd_fingerprint_attack(forget_features, control_features, seed):
    forget_features = np.nan_to_num(np.asarray(forget_features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    control_features = np.nan_to_num(np.asarray(control_features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n_attack = min(len(forget_features), len(control_features))
    if n_attack < 4:
        return {
            "attack_acc": float("nan"),
            "attack_auc": float("nan"),
            "attack_tpr_at_1fpr": float("nan"),
            "attack_advantage": float("nan"),
        }

    forget_features = forget_features[:n_attack]
    control_features = control_features[:n_attack]
    rng = np.random.RandomState(seed)
    pos_train, neg_train, pos_test, neg_test = _stratified_attack_split(n_attack, n_attack, rng)
    if len(pos_test) == 0 or len(neg_test) == 0:
        return {
            "attack_acc": float("nan"),
            "attack_auc": float("nan"),
            "attack_tpr_at_1fpr": float("nan"),
            "attack_advantage": float("nan"),
        }

    train_x_np = np.concatenate([forget_features[pos_train], control_features[neg_train]], axis=0)
    train_y_np = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))], axis=0).astype(np.float32)
    test_x_np = np.concatenate([forget_features[pos_test], control_features[neg_test]], axis=0)
    test_y_np = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))], axis=0).astype(np.float32)

    train_x = torch.from_numpy(train_x_np)
    test_x = torch.from_numpy(test_x_np)
    train_x, (test_x,) = _standardize_attack_features(train_x, test_x)
    train_y = torch.from_numpy(train_y_np)

    torch.manual_seed(seed)
    attack_model = nn.Linear(train_x.size(1), 1)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.05, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(300):
        logits = attack_model(train_x).squeeze(1)
        loss = loss_fn(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_probs = torch.sigmoid(attack_model(test_x).squeeze(1)).numpy()

    pos_scores = test_probs[test_y_np == 1]
    neg_scores = test_probs[test_y_np == 0]
    attack_auc = _binary_auc(pos_scores, neg_scores)
    attack_acc = float(np.mean((test_probs >= 0.5).astype(np.float32) == test_y_np))
    tpr, _, advantage = _tpr_at_fpr(pos_scores, neg_scores, target_fpr=0.01)
    return {
        "attack_acc": attack_acc,
        "attack_auc": attack_auc,
        "attack_tpr_at_1fpr": tpr,
        "attack_advantage": advantage,
    }


def _build_dfd_attack_features(logits_vectors, mu_vectors):
    # Columns in each representation vector: JS, geom cosine, geom distance, anchor relation, local CKA.
    return np.column_stack([
        mu_vectors[:, 1],
        mu_vectors[:, 0],
        logits_vectors[:, 1],
        mu_vectors[:, 3],
        mu_vectors[:, 4],
    ]).astype(np.float32)


def _run_dfd_for_representation(
    rep_name,
    rep_key,
    forget_stats,
    forget_mu_stats,
    control_stats,
    control_mu_stats,
    reference_m0,
    reference_mu,
    args,
):
    forget_density = _mean_knn_distance(forget_stats[rep_key], reference_m0[rep_key], args.dfd_knn)
    control_density = _mean_knn_distance(control_stats[rep_key], reference_m0[rep_key], args.dfd_knn)
    matched_control_positions = _match_control_anchors(
        forget_stats,
        control_stats,
        forget_density,
        control_density,
    )

    forget_neighbor_positions = _knn_reference_positions(forget_stats[rep_key], reference_m0[rep_key], args.dfd_knn)
    matched_control_reps = control_stats[rep_key][matched_control_positions]
    control_neighbor_positions = _knn_reference_positions(matched_control_reps, reference_m0[rep_key], args.dfd_knn)

    forget_fingerprint_vectors = _local_fingerprint_vectors(
        forget_stats,
        forget_mu_stats,
        reference_m0,
        reference_mu,
        rep_key,
        forget_neighbor_positions,
    )
    control_fingerprint_vectors = _local_fingerprint_vectors(
        control_stats,
        control_mu_stats,
        reference_m0,
        reference_mu,
        rep_key,
        control_neighbor_positions,
        anchor_positions=matched_control_positions,
    )
    forget_geometry_vectors = forget_fingerprint_vectors[:, 1:3]
    control_geometry_vectors = control_fingerprint_vectors[:, 1:3]
    summary = _summarize_geometry_vectors(forget_geometry_vectors, control_geometry_vectors)

    print(f"\nDFD local geometry reconfiguration test ({rep_name} representation):")
    for name, stats in summary.items():
        print(
            f"  {name}: forget={stats['forget_mean']:.6f}, "
            f"control={stats['control_mean']:.6f}, "
            f"MWU_p={stats['mann_whitney_p']:.4g}, "
            f"KS_p={stats['ks_p']:.4g}, "
            f"cliff_delta={stats['cliffs_delta']:.4f}"
        )

    return {
        "metrics": summary,
        "fingerprint_vectors": {
            "forget": forget_fingerprint_vectors,
            "control": control_fingerprint_vectors,
        },
        "fingerprint_metrics": _summarize_fingerprint_vectors(
            forget_fingerprint_vectors,
            control_fingerprint_vectors,
            rep_name,
        ),
        "matched_control_positions": matched_control_positions,
        "num_controls": len(matched_control_positions),
    }


def run_dfd_statistical_test(m0, mu_model, train_set, unlearning_data_info, args, seed=0):
    erase_indices = []
    remain_indices = []
    for info in unlearning_data_info.values():
        erase_indices.extend(info["erase_indices"])
        remain_indices.extend(info["remain_indices"])
    if len(erase_indices) == 0 or len(remain_indices) == 0:
        return None

    rng = np.random.RandomState(seed)
    forget_anchor_indices = _sample_indices(erase_indices, args.dfd_max_anchors, rng)
    control_candidate_indices = _sample_indices(remain_indices, args.dfd_control_pool_size, rng)
    reference_indices = _sample_indices(remain_indices, args.dfd_reference_pool_size, rng)

    forget_stats = _collect_dfd_outputs(m0, train_set, forget_anchor_indices, args)
    forget_mu_stats = _collect_dfd_outputs(mu_model, train_set, forget_anchor_indices, args)
    control_stats = _collect_dfd_outputs(m0, train_set, control_candidate_indices, args)
    control_mu_stats = _collect_dfd_outputs(mu_model, train_set, control_candidate_indices, args)
    reference_m0 = _collect_dfd_outputs(m0, train_set, reference_indices, args)
    reference_mu = _collect_dfd_outputs(mu_model, train_set, reference_indices, args)

    print(
        "\nDFD setup: "
        f"anchors={len(forget_anchor_indices)}, "
        f"reference_pool={len(reference_indices)}, knn={args.dfd_knn}"
    )
    logits_result = _run_dfd_for_representation(
        "logits",
        "reps_logits",
        forget_stats,
        forget_mu_stats,
        control_stats,
        control_mu_stats,
        reference_m0,
        reference_mu,
        args,
    )
    traditional_mia_signal = _summarize_traditional_mia_signals(
        forget_stats,
        forget_mu_stats,
        control_stats,
        control_mu_stats,
        logits_result["matched_control_positions"],
    )
    mu_result = _run_dfd_for_representation(
        "mu",
        "reps_mu",
        forget_stats,
        forget_mu_stats,
        control_stats,
        control_mu_stats,
        reference_m0,
        reference_mu,
        args,
    )
    forget_attack_features = _build_dfd_attack_features(
        logits_result["fingerprint_vectors"]["forget"],
        mu_result["fingerprint_vectors"]["forget"],
    )
    control_attack_features = _build_dfd_attack_features(
        logits_result["fingerprint_vectors"]["control"],
        mu_result["fingerprint_vectors"]["control"],
    )
    attack_result = _train_dfd_fingerprint_attack(
        forget_attack_features,
        control_attack_features,
        seed=seed + 1777,
    )

    print("\nDFD fingerprint attack:")
    print(
        f"  attack_acc={attack_result['attack_acc']:.4f}, "
        f"attack_auc={attack_result['attack_auc']:.4f}, "
        f"tpr@1%fpr={attack_result['attack_tpr_at_1fpr']:.4f}, "
        f"advantage={attack_result['attack_advantage']:.4f}"
    )

    return {
        "num_anchors": len(forget_anchor_indices),
        "reference_pool": len(reference_indices),
        "knn": args.dfd_knn,
        "logits": logits_result,
        "mu": mu_result,
        "traditional_mia_signal": traditional_mia_signal,
        "attack": attack_result,
    }


def prepare_unl(erasing_dataset, dataloader_remaining_after_aux, model, loss_fn, args, noise_flag):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step = 0
    backdoor_acc_list = []

    print("erase samples:", len(erasing_dataset.dataset))
    for epoch in range(args.unlearning_epochs):
        model.train()
        for (x_e, y_e), (x_r, y_r) in zip(erasing_dataset, dataloader_remaining_after_aux):
            x_e, y_e = x_e.to(args.device), y_e.to(args.device)  # (B, C, H, W), (B, 10)
            x_r, y_r = x_r.to(args.device), y_r.to(args.device)  # (B, C, H, W), (B, 10)

            logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = model(x_e, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)

            logits_z_r, logits_y_r, x_hat_r, mu_r, logvar_r = model(x_r, mode='with_reconstruction')

            KLD_element = mu_e.pow(2).add_(logvar_e.exp()).mul_(-1).add_(1).add_(logvar_e).to(args.device)
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).to(args.device)
            H_p_q = loss_fn(logits_y_e, y_e)
            # loss = args.beta * KLD_mean - args.unlearn_learning_rate * H_p_q

            H_p_q2 = loss_fn(logits_y_r, y_r)
            KLD_element2 = mu_r.pow(2).add_(logvar_r.exp()).mul_(-1).add_(1).add_(logvar_r).to(args.device)
            KLD_mean2 = torch.mean(KLD_element2).mul_(-0.5).to(args.device)

            unlearning_item = (args.beta * KLD_mean - H_p_q).detach()
            learning_item = (args.beta * KLD_mean2 + H_p_q2).detach()

            unl_signal = torch.clamp(unlearning_item, min=0.0)
            total = unl_signal + learning_item.abs() + 1e-8

            unl_rate = torch.clamp(unl_signal / total, min=0.0, max=1.0)
            self_s_rate = 1.0 - unl_rate

            loss = unl_rate * (args.beta * KLD_mean - H_p_q) + self_s_rate * (args.beta * KLD_mean2 + H_p_q2)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()
            acc_back = (logits_y_e.argmax(dim=1) == y_e).float().mean().item()
            # backdoor_acc_list.append(acc_r)
            metrics = {
                'acc_back': acc_back,
                'loss1': loss.item(),
                'KLD_mean': KLD_mean.item(),
                # '1-JS(p,q)': JS_p_q,
                'mu': torch.mean(mu_e).item(),
                # 'KLD': KLD.item(),
                # 'KLD_mean': KLD_mean.item(),
            }
            # if epoch == args.unlearning_epochs - 1:
            #     mu_list.append(torch.mean(mu).item())
            #     sigma_list.append(sigma)
            if step % len(erasing_dataset) % 10000 == 0:
                print(f'[{epoch}/{args.unlearning_epochs}:{step % len(erasing_dataset):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
                x_cpu = x_e.cpu().data
                x_cpu = x_cpu.clamp(0, 1)
                x_cpu = x_cpu.view(x_cpu.size(0), 3, 32, 32)
                grid = torchvision.utils.make_grid(x_cpu, nrow=4)
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()

                print(acc_back)
                print("print x grad")
                # print(updated_x)
            backdoor_acc = eva_vib(model, erasing_dataset, args, name='on erased data', epoch=999)
            # if backdoor_acc < 0.8:
            #     break
            step += 1
    print("backdoor_acc_list", backdoor_acc_list)
    return model


def synchronize_device(args):
    if getattr(args, "device", None) is not None and args.device.type == "cuda":
        torch.cuda.synchronize(args.device)


def standard_unlearning(
    model: nn.Module,
    train_set,
    train_set_aug,
    args,
    erase_ratio=0.05,
    base_seed=0,
    loss_fn=None,
):
    """Run RFU with augmented erase/remain data; keep saved indices for deterministic evaluation."""
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    erase_subset, remain_subset, erase_idx, remain_idx = split_dataset_for_unlearning(
        train_set,
        erase_ratio=erase_ratio,
        seed=base_seed,
        erase_dataset=train_set_aug,
        remain_dataset=train_set_aug,
    )
    erase_loader = make_loader(erase_subset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    remain_loader = make_loader(remain_subset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)

    unlearning_data_info = {
        "standard_split": {
            "erase_indices": erase_idx,
            "remain_indices": remain_idx,
            "erase_loader": erase_loader,
            "remain_loader": remain_loader,
            "erase_size": len(erase_subset),
            "remain_size": len(remain_subset),
        }
    }

    synchronize_device(args)
    start_time = time.time()
    model = prepare_unl(
        erasing_dataset=erase_loader,
        dataloader_remaining_after_aux=remain_loader,
        model=model,
        loss_fn=loss_fn,
        args=args,
        noise_flag="no noise",
    )
    synchronize_device(args)
    running_time = time.time() - start_time

    model.to(args.device)
    model.eval()
    return model, unlearning_data_info, running_time


def resolve_existing_path(*candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0]


def build_erased_data_loader(train_set, unlearning_data_info, args):
    erase_indices = []
    for info in unlearning_data_info.values():
        erase_indices.extend(info["erase_indices"])
    if len(erase_indices) == 0:
        return None
    erase_subset = Subset(train_set, erase_indices)
    return make_loader(erase_subset, batch_size=args.batch_size, shuffle=False, num_workers=1)


def configure_cifar10_unlearning_args():
    args = args_parser()
    args.gpu = 0
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.dataset = 'CIFAR10'
    args.add_noise = False
    args.beta = 0.0001
    args.mse_rate = 0.1
    args.lr = 0.0001
    args.dimZ = 512
    args.batch_size = 200

    args.unlearning_ratio = 0.05
    args.unlearning_ratios = [0.01, 0.02, 0.03, 0.04, 0.05]
    args.unlearning_epochs = 40
    args.seeds = [0, 1, 2, 3, 4]
    args.dfd_knn = 20
    args.dfd_max_anchors = 200
    args.dfd_control_pool_size = 2000
    args.dfd_reference_pool_size = None

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
    train_set_aug = CIFAR10('/home/wwq/Data/data/cifar', train=True, transform=train_aug_transform, download=True)
    train_set = CIFAR10('/home/wwq/Data/data/cifar', train=True, transform=deterministic_transform, download=False)
    test_set = CIFAR10('/home/wwq/Data/data/cifar', train=False, transform=deterministic_transform, download=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    args.x_shape = (3, 32, 32)
    return train_set, train_set_aug, test_set, test_loader


def load_standard_cifar10_model(args):

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vib_cifar10_standard_test.pt")

    ckpt = torch.load(ckpt_path, map_location=args.device)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    vib, _ = init_vib(args)
    vib.load_state_dict(state_dict)
    vib.to(args.device)
    vib.eval()
    print("Loaded standard checkpoint:", ckpt_path)
    if isinstance(ckpt, dict):
        print("Loaded checkpoint from epoch:", ckpt.get("epoch", "unknown"))
    return vib, ckpt


def summarize_mia_metrics(prefix, mia_result):
    if mia_result is None:
        return f"{prefix}_mia_auc=nan, {prefix}_mia_acc=nan, {prefix}_mia_tpr_at_1fpr=nan"
    return (
        f"{prefix}_mia_auc={mia_result['attack_auc']:.4f}, "
        f"{prefix}_mia_acc={mia_result['attack_acc']:.4f}, "
        f"{prefix}_mia_tpr_at_1fpr={mia_result['attack_tpr_at_1fpr']:.4f}"
    )


def format_result_summary(result):
    parts = [
        f"unlearning_ratio={result['unlearning_ratio']:.4f}, "
        f"acc_after={result['acc_after']:.4f}, "
        f"erased_acc={result['erased_acc']:.4f}, "
        f"{summarize_mia_metrics('mia_before', result['mia_before'])}, "
        f"{summarize_mia_metrics('mia_after', result['mia_after'])}, "
        f"unlearning_running_time={result['unlearning_running_time']:.4f}s, "
        f"dfd_attacking_running_time={result['dfd_attacking_running_time']:.4f}s, "
        f"mia_attacking_running_time={result['mia_attacking_running_time']:.4f}s, "
        f"attacking_running_time={result['attacking_running_time']:.4f}s"
    ]
    dfd = result.get("dfd")
    if dfd is not None:
        attack_stats = dfd.get("attack")
        if attack_stats is not None:
            parts.append(
                f", dfd_attack_acc={attack_stats['attack_acc']:.4f}, "
                f"dfd_attack_auc={attack_stats['attack_auc']:.4f}, "
                f"dfd_attack_tpr_at_1fpr={attack_stats['attack_tpr_at_1fpr']:.4f}, "
                f"dfd_attack_advantage={attack_stats['attack_advantage']:.4f}"
            )
        traditional_stats = dfd.get("traditional_mia_signal", {}).get("metrics")
        if traditional_stats is not None:
            for name in ("top1_conf_delta", "true_conf_delta", "loss_delta"):
                stats = traditional_stats[name]
                parts.append(
                    f", mia_signal_{name}_forget={stats['forget_mean']:.6f}, "
                    f"mia_signal_{name}_control={stats['control_mean']:.6f}, "
                    f"mia_signal_{name}_auc={stats['auc']:.4f}, "
                    f"mia_signal_{name}_cliff_delta={stats['cliffs_delta']:.4f}"
                )
        for rep_name in ("logits", "mu"):
            geom_cosine_stats = dfd[rep_name]["metrics"]["geom_cosine_shift"]
            geom_distance_stats = dfd[rep_name]["metrics"]["geom_distance_shift"]
            parts.append(
                f", dfd_{rep_name}_geom_cos_forget={geom_cosine_stats['forget_mean']:.6f}, "
                f"dfd_{rep_name}_geom_cos_control={geom_cosine_stats['control_mean']:.6f}, "
                f"dfd_{rep_name}_geom_cos_mwu_p={geom_cosine_stats['mann_whitney_p']:.4g}, "
                f"dfd_{rep_name}_geom_cos_cliff_delta={geom_cosine_stats['cliffs_delta']:.4f}, "
                f"dfd_{rep_name}_geom_dist_forget={geom_distance_stats['forget_mean']:.6f}, "
                f"dfd_{rep_name}_geom_dist_control={geom_distance_stats['control_mean']:.6f}, "
                f"dfd_{rep_name}_geom_dist_mwu_p={geom_distance_stats['mann_whitney_p']:.4g}, "
                f"dfd_{rep_name}_geom_dist_cliff_delta={geom_distance_stats['cliffs_delta']:.4f}"
            )
            fingerprint_stats = dfd[rep_name].get("fingerprint_metrics")
            if fingerprint_stats is not None:
                for name in ("fp_js_shift", "fp_anchor_relation_shift", "fp_local_cka_shift"):
                    stats = fingerprint_stats[name]
                    parts.append(
                        f", dfd_{rep_name}_{name}_forget={stats['forget_mean']:.6f}, "
                        f"dfd_{rep_name}_{name}_control={stats['control_mean']:.6f}, "
                        f"dfd_{rep_name}_{name}_auc={stats['auc']:.4f}, "
                        f"dfd_{rep_name}_{name}_cliff_delta={stats['cliffs_delta']:.4f}"
                    )
    return "".join(parts)


def set_random_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_result_metrics(result):
    metrics = {
        "acc_after": result["acc_after"],
        "erased_acc": result["erased_acc"],
        "unlearning_running_time": result["unlearning_running_time"],
        "dfd_attacking_running_time": result["dfd_attacking_running_time"],
        "mia_attacking_running_time": result["mia_attacking_running_time"],
        "attacking_running_time": result["attacking_running_time"],
    }
    for prefix in ("mia_before", "mia_after"):
        mia_result = result.get(prefix)
        if mia_result is None:
            continue
        metrics[f"{prefix}_mia_auc"] = float(mia_result["attack_auc"])
        metrics[f"{prefix}_mia_acc"] = float(mia_result["attack_acc"])
        metrics[f"{prefix}_mia_tpr_at_1fpr"] = float(mia_result["attack_tpr_at_1fpr"])
    dfd_result = result.get("dfd")
    if dfd_result is not None:
        attack_stats = dfd_result.get("attack")
        if attack_stats is not None:
            metrics["dfd_attack_acc"] = float(attack_stats["attack_acc"])
            metrics["dfd_attack_auc"] = float(attack_stats["attack_auc"])
            metrics["dfd_attack_tpr_at_1fpr"] = float(attack_stats["attack_tpr_at_1fpr"])
            metrics["dfd_attack_advantage"] = float(attack_stats["attack_advantage"])
        traditional_stats = dfd_result.get("traditional_mia_signal", {}).get("metrics")
        if traditional_stats is not None:
            for name, stats in traditional_stats.items():
                metrics[f"mia_signal_{name}_forget_mean"] = float(stats["forget_mean"])
                metrics[f"mia_signal_{name}_control_mean"] = float(stats["control_mean"])
                metrics[f"mia_signal_{name}_mann_whitney_p"] = float(stats["mann_whitney_p"])
                metrics[f"mia_signal_{name}_ks_p"] = float(stats["ks_p"])
                metrics[f"mia_signal_{name}_cliffs_delta"] = float(stats["cliffs_delta"])
                metrics[f"mia_signal_{name}_auc"] = float(stats["auc"])
        for rep_name in ("logits", "mu"):
            for name, stats in dfd_result[rep_name]["metrics"].items():
                metrics[f"dfd_{rep_name}_{name}_forget_mean"] = float(stats["forget_mean"])
                metrics[f"dfd_{rep_name}_{name}_control_mean"] = float(stats["control_mean"])
                metrics[f"dfd_{rep_name}_{name}_mann_whitney_p"] = float(stats["mann_whitney_p"])
                metrics[f"dfd_{rep_name}_{name}_ks_p"] = float(stats["ks_p"])
                metrics[f"dfd_{rep_name}_{name}_cliffs_delta"] = float(stats["cliffs_delta"])
                metrics[f"dfd_{rep_name}_{name}_auc"] = float(stats["auc"])
            fingerprint_stats = dfd_result[rep_name].get("fingerprint_metrics")
            if fingerprint_stats is not None:
                for name, stats in fingerprint_stats.items():
                    metrics[f"dfd_{rep_name}_{name}_forget_mean"] = float(stats["forget_mean"])
                    metrics[f"dfd_{rep_name}_{name}_control_mean"] = float(stats["control_mean"])
                    metrics[f"dfd_{rep_name}_{name}_mann_whitney_p"] = float(stats["mann_whitney_p"])
                    metrics[f"dfd_{rep_name}_{name}_ks_p"] = float(stats["ks_p"])
                    metrics[f"dfd_{rep_name}_{name}_cliffs_delta"] = float(stats["cliffs_delta"])
                    metrics[f"dfd_{rep_name}_{name}_auc"] = float(stats["auc"])
    return metrics


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std


def summarize_seed_sweep(all_seed_results):
    grouped = {}
    for seed_result in all_seed_results:
        for result in seed_result["results"]:
            unlearning_ratio = result["unlearning_ratio"]
            grouped.setdefault(unlearning_ratio, []).append(flatten_result_metrics(result))

    summary = []
    for unlearning_ratio in sorted(grouped):
        metric_names = sorted({name for metrics in grouped[unlearning_ratio] for name in metrics})
        row = {
            "unlearning_ratio": unlearning_ratio,
        }
        for metric_name in metric_names:
            values = [metrics.get(metric_name, float("nan")) for metrics in grouped[unlearning_ratio]]
            metric_mean, metric_std = mean_std(values)
            row[f"{metric_name}_mean"] = metric_mean
            row[f"{metric_name}_std"] = metric_std
        summary.append(row)
    return summary


def print_seed_sweep_summary(summary):
    print("\n===== RFU final summary =====")
    metric_order = [
        "acc_after",
        "erased_acc",
        "mia_before_mia_auc",
        "mia_before_mia_acc",
        "mia_before_mia_tpr_at_1fpr",
        "mia_after_mia_auc",
        "mia_after_mia_acc",
        "mia_after_mia_tpr_at_1fpr",
        "unlearning_running_time",
        "dfd_attacking_running_time",
        "mia_attacking_running_time",
        "attacking_running_time",
        "dfd_attack_acc",
        "dfd_attack_auc",
        "dfd_attack_tpr_at_1fpr",
        "dfd_attack_advantage",
        "mia_signal_top1_conf_delta_forget_mean",
        "mia_signal_top1_conf_delta_control_mean",
        "mia_signal_top1_conf_delta_auc",
        "mia_signal_top1_conf_delta_cliffs_delta",
        "mia_signal_true_conf_delta_forget_mean",
        "mia_signal_true_conf_delta_control_mean",
        "mia_signal_true_conf_delta_auc",
        "mia_signal_true_conf_delta_cliffs_delta",
        "mia_signal_loss_delta_forget_mean",
        "mia_signal_loss_delta_control_mean",
        "mia_signal_loss_delta_auc",
        "mia_signal_loss_delta_cliffs_delta",
        "dfd_logits_geom_cosine_shift_forget_mean",
        "dfd_logits_geom_cosine_shift_control_mean",
        "dfd_logits_geom_cosine_shift_auc",
        "dfd_logits_geom_cosine_shift_mann_whitney_p",
        "dfd_logits_geom_cosine_shift_ks_p",
        "dfd_logits_geom_cosine_shift_cliffs_delta",
        "dfd_logits_geom_distance_shift_forget_mean",
        "dfd_logits_geom_distance_shift_control_mean",
        "dfd_logits_geom_distance_shift_auc",
        "dfd_logits_geom_distance_shift_mann_whitney_p",
        "dfd_logits_geom_distance_shift_ks_p",
        "dfd_logits_geom_distance_shift_cliffs_delta",
        "dfd_logits_fp_js_shift_forget_mean",
        "dfd_logits_fp_js_shift_control_mean",
        "dfd_logits_fp_js_shift_auc",
        "dfd_logits_fp_js_shift_cliffs_delta",
        "dfd_logits_fp_anchor_relation_shift_forget_mean",
        "dfd_logits_fp_anchor_relation_shift_control_mean",
        "dfd_logits_fp_anchor_relation_shift_auc",
        "dfd_logits_fp_anchor_relation_shift_cliffs_delta",
        "dfd_logits_fp_local_cka_shift_forget_mean",
        "dfd_logits_fp_local_cka_shift_control_mean",
        "dfd_logits_fp_local_cka_shift_auc",
        "dfd_logits_fp_local_cka_shift_cliffs_delta",
        "dfd_mu_geom_cosine_shift_forget_mean",
        "dfd_mu_geom_cosine_shift_control_mean",
        "dfd_mu_geom_cosine_shift_auc",
        "dfd_mu_geom_cosine_shift_mann_whitney_p",
        "dfd_mu_geom_cosine_shift_ks_p",
        "dfd_mu_geom_cosine_shift_cliffs_delta",
        "dfd_mu_geom_distance_shift_forget_mean",
        "dfd_mu_geom_distance_shift_control_mean",
        "dfd_mu_geom_distance_shift_auc",
        "dfd_mu_geom_distance_shift_mann_whitney_p",
        "dfd_mu_geom_distance_shift_ks_p",
        "dfd_mu_geom_distance_shift_cliffs_delta",
        "dfd_mu_fp_js_shift_forget_mean",
        "dfd_mu_fp_js_shift_control_mean",
        "dfd_mu_fp_js_shift_auc",
        "dfd_mu_fp_js_shift_cliffs_delta",
        "dfd_mu_fp_anchor_relation_shift_forget_mean",
        "dfd_mu_fp_anchor_relation_shift_control_mean",
        "dfd_mu_fp_anchor_relation_shift_auc",
        "dfd_mu_fp_anchor_relation_shift_cliffs_delta",
        "dfd_mu_fp_local_cka_shift_forget_mean",
        "dfd_mu_fp_local_cka_shift_control_mean",
        "dfd_mu_fp_local_cka_shift_auc",
        "dfd_mu_fp_local_cka_shift_cliffs_delta",
    ]
    for row in summary:
        parts = [
            f"unlearning_ratio={row['unlearning_ratio']:.4f}",
        ]
        for metric in metric_order:
            if f"{metric}_mean" not in row:
                continue
            parts.append(f"{metric}={row[f'{metric}_mean']:.4f}±{row[f'{metric}_std']:.4f}")
        print(", ".join(parts))


def run_unlearning_ratio_sweep_for_seed(args, seed):
    set_random_seed(seed)
    print(f"\n===== RFU seed={seed} =====")

    train_set, train_set_aug, test_set, test_loader = load_cifar10_data(args)
    base_vib, ckpt = load_standard_cifar10_model(args)
    loaded_epoch = ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0
    _ = eva_vib(base_vib, test_loader, args, name=f"loaded model (seed={seed})", epoch=loaded_epoch)

    loss_fn = nn.CrossEntropyLoss()
    sweep_results = []

    for unlearning_ratio in args.unlearning_ratios:
        print(f"\n===== RFU test: unlearning_ratio={unlearning_ratio:.4f} =====")
        args.unlearning_ratio = unlearning_ratio

        run_model = copy.deepcopy(base_vib)
        run_model.to(args.device)
        run_model.eval()
        pre_unlearn_vib = copy.deepcopy(run_model)
        pre_unlearn_vib.eval()

        new_vib, unlearning_data_info, unlearning_running_time = standard_unlearning(
            model=run_model,
            train_set=train_set,
            train_set_aug=train_set_aug,
            args=args,
            erase_ratio=args.unlearning_ratio,
            base_seed=seed,
            loss_fn=loss_fn,
        )
        print(f"RFU unlearning running time: {unlearning_running_time:.4f}s")

        acc_after = eva_vib(
            new_vib,
            test_loader,
            args,
            name=f"after unlearning (ratio={unlearning_ratio:.4f})",
            epoch=args.unlearning_epochs,
        )

        erased_loader = build_erased_data_loader(train_set, unlearning_data_info, args)
        erased_acc = float("nan")
        if erased_loader is not None:
            erased_acc = eva_vib(
                new_vib,
                erased_loader,
                args,
                name=f"erased data (ratio={unlearning_ratio:.4f})",
                epoch=args.unlearning_epochs,
            )

        synchronize_device(args)
        dfd_attack_start_time = time.time()
        dfd_result = run_dfd_statistical_test(
            m0=pre_unlearn_vib,
            mu_model=new_vib,
            train_set=train_set,
            unlearning_data_info=unlearning_data_info,
            args=args,
            seed=seed + 4048 + int(unlearning_ratio * 10000),
        )
        synchronize_device(args)
        dfd_attacking_running_time = time.time() - dfd_attack_start_time
        print(f"RFU DFD attacking running time: {dfd_attacking_running_time:.4f}s")

        synchronize_device(args)
        mia_attack_start_time = time.time()
        mia_before = None
        mia_after = None
        mia_retain_loader, mia_nonmember_loader = build_learned_mia_loaders(
            train_set=train_set,
            test_set=test_set,
            unlearning_data_info=unlearning_data_info,
            args=args,
            seed=seed + 2024 + int(unlearning_ratio * 10000),
        )
        if mia_retain_loader is not None:
            mia_before = learned_membership_inference_attack(
                pre_unlearn_vib,
                mia_retain_loader,
                mia_nonmember_loader,
                args,
                name=f"before unlearning (ratio={unlearning_ratio:.4f})",
            )
            mia_after = learned_membership_inference_attack(
                new_vib,
                mia_retain_loader,
                mia_nonmember_loader,
                args,
                name=f"after unlearning (ratio={unlearning_ratio:.4f})",
            )
        synchronize_device(args)
        mia_attacking_running_time = time.time() - mia_attack_start_time
        attacking_running_time = dfd_attacking_running_time + mia_attacking_running_time
        print(f"RFU MIA attacking running time: {mia_attacking_running_time:.4f}s")
        print(f"RFU attacking running time: {attacking_running_time:.4f}s")

        result = {
            "unlearning_ratio": unlearning_ratio,
            "seed": seed,
            "acc_after": acc_after,
            "erased_acc": erased_acc,
            "mia_before": mia_before,
            "mia_after": mia_after,
            "dfd": dfd_result,
            "unlearning_running_time": unlearning_running_time,
            "dfd_attacking_running_time": dfd_attacking_running_time,
            "mia_attacking_running_time": mia_attacking_running_time,
            "attacking_running_time": attacking_running_time,
        }
        sweep_results.append(result)

        print("RFU sweep result: " + format_result_summary(result))

        del run_model, pre_unlearn_vib, new_vib
        torch.cuda.empty_cache()

    print(f"\n===== RFU sweep summary (seed={seed}) =====")
    for result in sweep_results:
        print(format_result_summary(result))

    return sweep_results


def test_unlearning_ratio_sweep():
    args = configure_cifar10_unlearning_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("device", args.device)
    print("RFU evaluation seeds:", args.seeds)

    all_seed_results = []
    for seed in args.seeds:
        seed_results = run_unlearning_ratio_sweep_for_seed(args, seed)
        all_seed_results.append({
            "seed": seed,
            "results": seed_results,
        })

    summary = summarize_seed_sweep(all_seed_results)
    print_seed_sweep_summary(summary)

    return all_seed_results, summary


if __name__ == "__main__":
    test_unlearning_ratio_sweep()
