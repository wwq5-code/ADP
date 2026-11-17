import os
os.environ["TQDM_DISABLE"] = "1"     # 放在导入 tqdm 前


import argparse
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
    parser.add_argument('--mia_feature',
                        choices=['direct_concat', 'sorted_concat', 'direct_diff',
                                 'sorted_diff', 'euc_dist'],
                        default='sorted_diff',
                        help='Two-posterior feature from Chen et al. CCS 2021.')
    parser.add_argument('--model_train_size', type=int, default=5000,
                        help='Number of positive-pool samples used to train each original model.')
    parser.add_argument('--original_epochs', type=int, default=100,
                        help='Epochs used to train each target/shadow original model.')
    parser.add_argument('--erase_count', type=int, default=1,
                        help='Number of other records deleted for the remaining-sample experiment.')
    parser.add_argument('--data_root', type=str, default='/home/wwq/Data/data/cifar',
                        help='CIFAR-10 data directory.')
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


@torch.no_grad()
def _black_box_posteriors(vib, x, num_samples):
    """Estimate the posterior returned by a stochastic VIB prediction API."""
    vib.eval()
    posterior = None
    for _ in range(max(1, num_samples)):
        _, logits_y, _, _ = vib(x, mode='distribution')
        current = F.softmax(logits_y, dim=1)
        posterior = current if posterior is None else posterior + current
    return posterior / float(max(1, num_samples))


def _paper_pair_feature(probs_o, probs_u, feature_name):
    """Five posterior-pair constructions from Chen et al., Section 3.3."""
    if feature_name == 'direct_concat':
        return torch.cat([probs_o, probs_u], dim=1)
    if feature_name == 'direct_diff':
        return probs_o - probs_u
    if feature_name == 'euc_dist':
        return torch.norm(probs_o - probs_u, p=2, dim=1, keepdim=True)

    # Sort P_o and rearrange P_u with the same class order, as specified in
    # the paper.  Sorting P_u independently destroys class correspondence.
    sorted_o, order_o = torch.sort(probs_o, dim=1, descending=True)
    aligned_u = torch.gather(probs_u, 1, order_o)
    if feature_name == 'sorted_concat':
        return torch.cat([sorted_o, aligned_u], dim=1)
    if feature_name == 'sorted_diff':
        return sorted_o - aligned_u
    raise ValueError(f"Unsupported paper feature: {feature_name}")


@torch.no_grad()
def collect_paper_mia_features(original_vib, unlearned_vib, dataloader, args,
                               max_samples=None):
    """Collect paired-model and original-only features via black-box outputs."""
    pair_features = []
    static_features = []
    collected = 0

    for x, _ in dataloader:
        x = x.to(args.device)
        probs_o = _black_box_posteriors(original_vib, x, args.num_samples)
        probs_u = _black_box_posteriors(unlearned_vib, x, args.num_samples)
        pair_features.append(
            _paper_pair_feature(probs_o, probs_u, args.mia_feature).cpu()
        )
        static_features.append(torch.sort(probs_o, dim=1, descending=True).values.cpu())
        collected += x.size(0)
        if max_samples is not None and collected >= max_samples:
            break

    pair_features = torch.cat(pair_features, dim=0)
    static_features = torch.cat(static_features, dim=0)
    if max_samples is not None:
        pair_features = pair_features[:max_samples]
        static_features = static_features[:max_samples]
    return pair_features, static_features


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


def _fit_attack_model(member_features, nonmember_features, args):
    """Fit on shadow cases only and retain shadow normalization statistics."""
    n = min(len(member_features), len(nonmember_features))
    features = torch.cat([member_features[:n], nonmember_features[:n]], dim=0)
    labels = torch.cat([torch.ones(n), torch.zeros(n)], dim=0).float()
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
    features = (features - mean) / std

    model = LearnedMIAAttack(features.size(1))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.mia_attack_lr, weight_decay=1e-4
    )
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(args.mia_attack_epochs):
        model.train()
        logits = model(features)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, mean, std


def _attack_scores(model, features, mean, std):
    return _evaluate_attack_classifier(model, (features - mean) / std)


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


def paper_remaining_sample_mia(shadow_original, shadow_unlearned,
                               target_original, target_unlearned,
                               shadow_member_loader, shadow_nonmember_loader,
                               target_member_loader, target_nonmember_loader,
                               args, name='remaining-sample'):
    """Chen et al. Sec. 6.4: shadow-trained paired MIA on <in,in> records."""
    shadow_mem_pair, shadow_mem_static = collect_paper_mia_features(
        shadow_original, shadow_unlearned, shadow_member_loader, args,
        max_samples=args.mia_max_samples,
    )
    shadow_non_pair, shadow_non_static = collect_paper_mia_features(
        shadow_original, shadow_unlearned, shadow_nonmember_loader, args,
        max_samples=args.mia_max_samples,
    )
    target_mem_pair, target_mem_static = collect_paper_mia_features(
        target_original, target_unlearned, target_member_loader, args,
        max_samples=args.mia_max_samples,
    )
    target_non_pair, target_non_static = collect_paper_mia_features(
        target_original, target_unlearned, target_nonmember_loader, args,
        max_samples=args.mia_max_samples,
    )

    trajectory_model, trajectory_mean, trajectory_std = _fit_attack_model(
        shadow_mem_pair, shadow_non_pair, args
    )
    static_model, static_mean, static_std = _fit_attack_model(
        shadow_mem_static, shadow_non_static, args
    )

    traj_mem = _attack_scores(
        trajectory_model, target_mem_pair, trajectory_mean, trajectory_std
    )
    traj_non = _attack_scores(
        trajectory_model, target_non_pair, trajectory_mean, trajectory_std
    )
    static_mem = _attack_scores(static_model, target_mem_static, static_mean, static_std)
    static_non = _attack_scores(static_model, target_non_static, static_mean, static_std)

    trajectory_metrics = _evaluate_member_scores(traj_mem, traj_non)
    static_metrics = _evaluate_member_scores(static_mem, static_non)
    true_labels = np.concatenate([
        np.ones(len(traj_mem), dtype=np.int64),
        np.zeros(len(traj_non), dtype=np.int64),
    ])
    trajectory_conf = np.concatenate([traj_mem, traj_non])
    static_conf = np.concatenate([static_mem, static_non])
    true_trajectory_conf = np.where(true_labels == 1, trajectory_conf, 1.0 - trajectory_conf)
    true_static_conf = np.where(true_labels == 1, static_conf, 1.0 - static_conf)

    result = {
        "feature": args.mia_feature,
        "trajectory": trajectory_metrics,
        "static": static_metrics,
        "auc_gain": trajectory_metrics["attack_auc"] - static_metrics["attack_auc"],
        "deg_count": float(np.mean(true_trajectory_conf > true_static_conf)),
        "deg_rate": float(np.mean(true_trajectory_conf - true_static_conf)),
        "num_shadow_member": min(len(shadow_mem_pair), len(shadow_non_pair)),
        "num_target_member": len(target_mem_pair),
        "num_target_nonmember": len(target_non_pair),
    }
    print(f"\nChen et al. remaining-sample MIA for {name}:")
    print(
        f"  feature={result['feature']}, "
        f"trajectory_auc={trajectory_metrics['attack_auc']:.4f}, "
        f"static_auc={static_metrics['attack_auc']:.4f}, "
        f"auc_gain={result['auc_gain']:.4f}"
    )
    print(
        f"  DegCount={result['deg_count']:.4f}, "
        f"DegRate={result['deg_rate']:.4f}, "
        f"trajectory_TPR@1%FPR={trajectory_metrics['attack_tpr_at_1fpr']:.4f}"
    )
    return result

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


def build_paper_domain_splits(dataset_size, args, seed):
    """Paper Sec. 5.1: D -> disjoint target/shadow, then 80/20 positive/negative."""
    rng = np.random.RandomState(seed)
    all_indices = rng.permutation(dataset_size)
    midpoint = dataset_size // 2
    domains = {
        "shadow": all_indices[:midpoint],
        "target": all_indices[midpoint:],
    }

    result = {}
    for offset, (name, domain) in enumerate(domains.items()):
        domain_rng = np.random.RandomState(seed + 1009 * (offset + 1))
        domain = domain_rng.permutation(domain)
        positive_cut = int(round(0.8 * len(domain)))
        positive_pool = domain[:positive_cut]
        negative_pool = domain[positive_cut:]
        train_size = min(args.model_train_size, len(positive_pool))
        train_indices = domain_rng.choice(
            positive_pool, size=train_size, replace=False
        ).tolist()
        if args.erase_count < 1 or args.erase_count >= len(train_indices):
            raise ValueError(
                f"erase_count must be in [1, {len(train_indices) - 1}], "
                f"got {args.erase_count}"
            )
        erase_indices = train_indices[:args.erase_count]
        remain_indices = train_indices[args.erase_count:]
        member_count = min(args.mia_max_samples, len(remain_indices))
        nonmember_count = min(args.mia_max_samples, len(negative_pool))
        result[name] = {
            "train_indices": train_indices,
            "erase_indices": erase_indices,
            "remain_indices": remain_indices,
            "member_query_indices": domain_rng.choice(
                remain_indices, size=member_count, replace=False
            ).tolist(),
            "nonmember_query_indices": domain_rng.choice(
                negative_pool, size=nonmember_count, replace=False
            ).tolist(),
        }
    return result


def train_original_model(dataset, train_indices, args, seed, name):
    """Train a target or shadow VIB only on its paper-positive training subset."""
    set_random_seed(seed)
    model, _ = init_vib(args)
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
        model = vib_train_original_IB(
            loader, model, optimizer, loss_fn, None, args
        )
        print(f"{name} original training epoch {epoch + 1}/{args.original_epochs}")
    model.eval()
    return model


def rfu_unlearn_fixed_records(original_model, augmented_dataset, split, args):
    """Apply RFU to the explicitly selected deletion records in one domain."""
    model = copy.deepcopy(original_model)
    erase_loader = make_loader(
        Subset(augmented_dataset, split["erase_indices"]),
        batch_size=min(args.batch_size, len(split["erase_indices"])),
        shuffle=True,
        num_workers=1,
        drop_last=False,
    )
    remain_loader = make_loader(
        Subset(augmented_dataset, split["remain_indices"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=False,
    )
    synchronize_device(args)
    start_time = time.time()
    model = prepare_unl(
        erasing_dataset=erase_loader,
        dataloader_remaining_after_aux=remain_loader,
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        args=args,
        noise_flag="no noise",
    )
    synchronize_device(args)
    model.eval()
    return model, time.time() - start_time


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

    args.unlearning_epochs = 40
    args.seeds = [0, 1, 2, 3, 4]

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


def load_standard_cifar10_model(args):

    ckpt_path = "vib_cifar10_standard_test.pt"

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


def run_paper_setting_for_seed(args, seed):
    set_random_seed(seed)
    print(f"\n===== Chen et al. remaining-sample setting, seed={seed} =====")
    train_set, train_set_aug, _, _ = load_cifar10_data(args)
    splits = build_paper_domain_splits(len(train_set), args, seed)

    shadow_original = train_original_model(
        train_set_aug, splits["shadow"]["train_indices"], args,
        seed + 10000, "shadow",
    )
    target_original = train_original_model(
        train_set_aug, splits["target"]["train_indices"], args,
        seed + 20000, "target",
    )
    shadow_unlearned, shadow_unlearning_time = rfu_unlearn_fixed_records(
        shadow_original, train_set_aug, splits["shadow"], args
    )
    target_unlearned, target_unlearning_time = rfu_unlearn_fixed_records(
        target_original, train_set_aug, splits["target"], args
    )

    shadow_member_loader = build_attack_loader(
        train_set, splits["shadow"]["member_query_indices"], args
    )
    shadow_nonmember_loader = build_attack_loader(
        train_set, splits["shadow"]["nonmember_query_indices"], args
    )
    target_member_loader = build_attack_loader(
        train_set, splits["target"]["member_query_indices"], args
    )
    target_nonmember_loader = build_attack_loader(
        train_set, splits["target"]["nonmember_query_indices"], args
    )

    synchronize_device(args)
    attack_start = time.time()
    mia = paper_remaining_sample_mia(
        shadow_original,
        shadow_unlearned,
        target_original,
        target_unlearned,
        shadow_member_loader,
        shadow_nonmember_loader,
        target_member_loader,
        target_nonmember_loader,
        args,
        name=f"RFU seed={seed}",
    )
    synchronize_device(args)
    return {
        "seed": seed,
        "mia": mia,
        "shadow_unlearning_time": shadow_unlearning_time,
        "target_unlearning_time": target_unlearning_time,
        "attacking_time": time.time() - attack_start,
    }


def summarize_paper_results(results):
    metric_paths = {
        "trajectory_auc": lambda r: r["mia"]["trajectory"]["attack_auc"],
        "static_auc": lambda r: r["mia"]["static"]["attack_auc"],
        "auc_gain": lambda r: r["mia"]["auc_gain"],
        "deg_count": lambda r: r["mia"]["deg_count"],
        "deg_rate": lambda r: r["mia"]["deg_rate"],
    }
    print("\n===== Chen et al. remaining-sample RFU summary =====")
    summary = {}
    for name, getter in metric_paths.items():
        value_mean, value_std = mean_std([getter(result) for result in results])
        summary[name] = {"mean": value_mean, "std": value_std}
        print(f"{name}={value_mean:.4f}±{value_std:.4f}")
    return summary


def test_paper_remaining_sample_setting():
    args = configure_cifar10_unlearning_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("device", args.device)
    print("RFU paper-setting evaluation seeds:", args.seeds)
    results = [run_paper_setting_for_seed(args, seed) for seed in args.seeds]
    return results, summarize_paper_results(results)


if __name__ == "__main__":
    test_paper_remaining_sample_setting()
