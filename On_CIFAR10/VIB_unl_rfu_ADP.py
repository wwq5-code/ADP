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
    parser.add_argument('--mia_max_samples', type=int, default=200,
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
    parser.add_argument('--erase_count', type=int, default=4,
                        help='Number of attacker-owned records deleted sequentially by ADP.')
    parser.add_argument('--adp_access', choices=['blackbox', 'whitebox'], default='whitebox',
                        help='ADP observation model. White-box adds classifier-layer gradient energies.')
    parser.add_argument('--adp_schedule', choices=['coupling', 'gradient_cosine', 'random'],
                        default='coupling', help='Precommitted probe-ranking rule.')
    parser.add_argument('--adp_probe_pool_size', type=int, default=32,
                        help='Number of attacker-owned candidate probes available for ranking.')
    parser.add_argument('--adp_schedule_targets', type=int, default=128,
                        help='Maximum unlabeled candidate targets used to construct one batch ADP schedule.')
    parser.add_argument('--adp_fisher_samples', type=int, default=256,
                        help='Auxiliary samples used for diagonal empirical-Fisher preconditioning.')
    parser.add_argument('--adp_fisher_damping', type=float, default=1e-3,
                        help='Damping added to the auxiliary diagonal empirical Fisher.')
    parser.add_argument('--adp_decoder_ridge', type=float, default=1e-3,
                        help='Ridge used by conditional regression and shared-covariance matched filters.')
    parser.add_argument('--adp_reference_fraction', type=float, default=0.2,
                        help='Fraction of CIFAR-10 training data reserved for the public reference domain.')
    parser.add_argument('--adp_reference_epochs', type=int, default=None,
                        help='Reference-model epochs; defaults to original_epochs.')
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


def _stable_logit(probabilities, eps=1e-6):
    probabilities = probabilities.clamp(min=eps, max=1.0 - eps)
    return torch.log(probabilities) - torch.log1p(-probabilities)


@torch.no_grad()
def _deterministic_classifier_quantities(vib, x, y):
    """Per-example gradients for VIB's two-layer prediction head.

    The white-box feature and coupling rule need gradients of the target loss.
    For the VIB classifier these gradients have a closed form, avoiding one
    backward pass per candidate and never using server-side optimizer state.
    The encoder mean is used instead of a stochastic latent draw so that the
    public coupling schedule is reproducible.
    """
    vib.eval()
    encoded = vib.encoder(x)
    dim_z = encoded.shape[1] // 2
    latent = encoded[:, :dim_z]
    preactivation = vib.approximator.fc1(latent)
    hidden = F.relu(preactivation)
    logits = vib.approximator.fc2(hidden)
    probabilities = F.softmax(logits, dim=1)
    one_hot = F.one_hot(y, num_classes=logits.shape[1]).to(probabilities.dtype)
    output_error = probabilities - one_hot
    hidden_error = torch.matmul(output_error, vib.approximator.fc2.weight)
    hidden_error = hidden_error * (preactivation > 0).to(hidden_error.dtype)

    # Exact per-example gradients of cross entropy for the two linear layers.
    grad_fc1_w = hidden_error.unsqueeze(2) * latent.unsqueeze(1)
    grad_fc2_w = output_error.unsqueeze(2) * hidden.unsqueeze(1)
    gradient_embedding = torch.cat([
        grad_fc1_w.flatten(1),
        hidden_error,
        grad_fc2_w.flatten(1),
        output_error,
    ], dim=1)

    fc1_norm = torch.sqrt(
        hidden_error.square().sum(dim=1)
        * (latent.square().sum(dim=1) + 1.0)
    )
    fc2_norm = torch.sqrt(
        output_error.square().sum(dim=1)
        * (hidden.square().sum(dim=1) + 1.0)
    )
    layer_energies = torch.stack([fc1_norm, fc2_norm], dim=1)
    layer_sizes = torch.tensor([
        vib.approximator.fc1.weight.numel() + vib.approximator.fc1.bias.numel(),
        vib.approximator.fc2.weight.numel() + vib.approximator.fc2.bias.numel(),
    ], device=x.device, dtype=layer_energies.dtype)
    layer_energies = torch.log(
        layer_energies / torch.sqrt(layer_sizes).unsqueeze(0) + 1e-12
    )
    return logits, gradient_embedding, layer_energies


@torch.no_grad()
def _classifier_gradient_embeddings(vib, dataset, indices, args, batch_size=32):
    loader = make_loader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    embeddings = []
    vib.to(args.device)
    vib.eval()
    for x, y in loader:
        x, y = x.to(args.device), y.to(args.device)
        _, gradients, _ = _deterministic_classifier_quantities(vib, x, y)
        embeddings.append(gradients.cpu())
    return torch.cat(embeddings, dim=0)


def select_adp_probe_schedule(reference_model, dataset, reference_indices,
                              probe_indices, candidate_target_indices, args, seed):
    """Rank owned probes without consulting the target model or membership labels.

    A diagonal empirical Fisher on the independent reference domain implements
    the tractable H^{-1/2} preconditioner in Eq. (18).  The schedule is optimized
    for a label-blind batch of candidate targets, as Nasr et al.'s active attack
    likewise supports a target batch.  This lets one released deletion path be
    reused for every candidate while remaining precommitted.
    """
    if args.erase_count > len(probe_indices):
        raise ValueError(
            f"erase_count={args.erase_count} exceeds probe pool size={len(probe_indices)}"
        )
    rng = np.random.RandomState(seed)
    shuffled_targets = np.asarray(candidate_target_indices, dtype=np.int64).copy()
    rng.shuffle(shuffled_targets)
    shuffled_targets = shuffled_targets[:args.adp_schedule_targets].tolist()

    if args.adp_schedule == 'random':
        order = rng.permutation(len(probe_indices))
        selected = [probe_indices[i] for i in order[:args.erase_count]]
        return selected, {
            "schedule": "random",
            "selected_scores": [float("nan")] * len(selected),
        }

    probe_gradients = _classifier_gradient_embeddings(
        reference_model, dataset, probe_indices, args
    )
    target_gradients = _classifier_gradient_embeddings(
        reference_model, dataset, shuffled_targets, args
    )
    fisher_indices = []
    if args.adp_schedule == 'coupling':
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
    # A probe is useful for a target only when deletion is predicted to move in
    # the target-loss ascent direction.  Average positive coupling implements
    # a single schedule for a batch without using any membership labels.
    scores = torch.clamp(coupling, min=0.0).mean(dim=0)
    order = torch.argsort(scores, descending=True).tolist()
    selected_positions = order[:args.erase_count]
    selected = [probe_indices[i] for i in selected_positions]
    selected_scores = [float(scores[i]) for i in selected_positions]
    return selected, {
        "schedule": args.adp_schedule,
        "selected_scores": selected_scores,
        "mean_positive_coupling": float(scores.mean()),
        "positive_coupling_fraction": float((coupling > 0).float().mean()),
        "num_schedule_targets": len(shuffled_targets),
        "num_fisher_samples": len(fisher_indices),
    }


@torch.no_grad()
def _collect_adp_features_for_model(vib, dataloader, args):
    """Collect observable phi_t and static features in a fixed candidate order."""
    phi_batches = []
    static_batches = []
    vib.to(args.device)
    vib.eval()
    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)
        probabilities = _black_box_posteriors(vib, x, args.num_samples)
        target_probability = probabilities.gather(1, y[:, None]).squeeze(1)
        margin = _stable_logit(target_probability)
        loss = -torch.log(target_probability.clamp_min(1e-6))
        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=1)
        static = torch.stack([margin, loss, entropy], dim=1)
        phi = margin[:, None]

        if args.adp_access == 'whitebox':
            _, _, layer_energies = _deterministic_classifier_quantities(vib, x, y)
            phi = torch.cat([phi, layer_energies], dim=1)
            static = torch.cat([static, layer_energies], dim=1)
        phi_batches.append(phi.cpu())
        static_batches.append(static.cpu())
    return torch.cat(phi_batches, dim=0), torch.cat(static_batches, dim=0)


def collect_adp_trajectory_features(snapshots, dataloader, args):
    """Vectorize consecutive responses from M_0,...,M_k for each target."""
    snapshot_features = []
    static_features = None
    for position, snapshot in enumerate(snapshots):
        phi, static = _collect_adp_features_for_model(snapshot, dataloader, args)
        snapshot_features.append(phi)
        if position == 0:
            static_features = static
        snapshot.to('cpu')
        if args.device.type == 'cuda':
            torch.cuda.empty_cache()
    responses = [
        snapshot_features[t] - snapshot_features[t - 1]
        for t in range(1, len(snapshot_features))
    ]
    return torch.cat(responses, dim=1), static_features


def _regularized_solve(matrix, rhs, ridge):
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    regularized = matrix + ridge * identity
    try:
        return torch.linalg.solve(regularized, rhs)
    except RuntimeError:
        return torch.matmul(torch.linalg.pinv(regularized), rhs)


def _fit_conditional_matched_filter(member_static, member_response,
                                    nonmember_static, nonmember_response, ridge):
    """Fit r | s_0,b ~ N(B s_0 + mu_b, Sigma) on paired shadow traces."""
    dtype = torch.float64
    sm, sn = member_static.to(dtype), nonmember_static.to(dtype)
    rm, rn = member_response.to(dtype), nonmember_response.to(dtype)
    all_static = torch.cat([sm, sn], dim=0)
    all_response = torch.cat([rm, rn], dim=0)
    static_mean = all_static.mean(0)
    static_std = all_static.std(0, unbiased=False).clamp_min(1e-8)
    response_mean = all_response.mean(0)
    response_std = all_response.std(0, unbiased=False).clamp_min(1e-8)
    sm = (sm - static_mean) / static_std
    sn = (sn - static_mean) / static_std
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
    midpoint = 0.5 * (mu_member + mu_nonmember)
    return {
        "regression": regression,
        "direction": direction,
        "midpoint": midpoint,
        "static_mean": static_mean,
        "static_std": static_std,
        "response_mean": response_mean,
        "response_std": response_std,
    }


def _conditional_deletion_logits(decoder, static, response):
    dtype = decoder["direction"].dtype
    normalized_static = (
        static.to(dtype) - decoder["static_mean"]
    ) / decoder["static_std"]
    normalized_response = (
        response.to(dtype) - decoder["response_mean"]
    ) / decoder["response_std"]
    residual = normalized_response - torch.matmul(
        normalized_static, decoder["regression"]
    )
    return torch.matmul(residual - decoder["midpoint"], decoder["direction"])


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


def adp_remaining_sample_mia(shadow_member_snapshots, shadow_nonmember_snapshots,
                             target_snapshots, shadow_query_loader,
                             target_member_loader, target_nonmember_loader,
                             args, name='ADP'):
    """Calibrate on matched shadow worlds and evaluate conditional trajectory LLR."""
    shadow_mem_response, shadow_mem_static = collect_adp_trajectory_features(
        shadow_member_snapshots, shadow_query_loader, args
    )
    shadow_non_response, shadow_non_static = collect_adp_trajectory_features(
        shadow_nonmember_snapshots, shadow_query_loader, args
    )
    target_mem_response, target_mem_static = collect_adp_trajectory_features(
        target_snapshots, target_member_loader, args
    )
    target_non_response, target_non_static = collect_adp_trajectory_features(
        target_snapshots, target_nonmember_loader, args
    )

    conditional_decoder = _fit_conditional_matched_filter(
        shadow_mem_static, shadow_mem_response,
        shadow_non_static, shadow_non_response,
        args.adp_decoder_ridge,
    )
    static_decoder = _fit_gaussian_lda(
        shadow_mem_static, shadow_non_static, args.adp_decoder_ridge
    )
    deletion_mem_logits = _conditional_deletion_logits(
        conditional_decoder, target_mem_static, target_mem_response
    )
    deletion_non_logits = _conditional_deletion_logits(
        conditional_decoder, target_non_static, target_non_response
    )
    static_mem_logits = _lda_logits(static_decoder, target_mem_static)
    static_non_logits = _lda_logits(static_decoder, target_non_static)

    deletion_mem = _sigmoid_numpy(deletion_mem_logits)
    deletion_non = _sigmoid_numpy(deletion_non_logits)
    static_mem = _sigmoid_numpy(static_mem_logits)
    static_non = _sigmoid_numpy(static_non_logits)
    trajectory_mem = _sigmoid_numpy(static_mem_logits + deletion_mem_logits)
    trajectory_non = _sigmoid_numpy(static_non_logits + deletion_non_logits)

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
        "response_dim": int(shadow_mem_response.shape[1]),
        "static_dim": int(shadow_mem_static.shape[1]),
        "deletion": deletion_metrics,
        "static": static_metrics,
        "trajectory": trajectory_metrics,
        "auc_gain": trajectory_metrics["attack_auc"] - static_metrics["attack_auc"],
        "deg_count": float(np.mean(
            true_trajectory_confidence > true_static_confidence
        )),
        "deg_rate": float(np.mean(
            true_trajectory_confidence - true_static_confidence
        )),
        "num_shadow_pairs": len(shadow_mem_response),
        "num_target_member": len(target_mem_response),
        "num_target_nonmember": len(target_non_response),
    }
    print(f"\nActive Deletion Probing for {name}:")
    print(
        f"  access={args.adp_access}, k={args.erase_count}, "
        f"response_dim={result['response_dim']}, "
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
            #self_s_rate = 1.0 - unl_rate
            self_s_rate = 1
            unl_rate = 1
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
    The attacker-owned probe pool is present in both worlds.
    """
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    all_indices = rng.permutation(len(labels))
    reference_size = int(round(args.adp_reference_fraction * len(labels)))
    if reference_size < args.model_train_size:
        raise ValueError(
            "adp_reference_fraction leaves fewer records than model_train_size"
        )
    reference_domain = all_indices[:reference_size]
    remaining = all_indices[reference_size:]
    midpoint = len(remaining) // 2
    shadow_domain = remaining[:midpoint]
    target_domain = remaining[midpoint:]

    reference_train_size = min(args.model_train_size, len(reference_domain))
    reference_train = rng.choice(
        reference_domain, size=reference_train_size, replace=False
    ).tolist()

    shadow_rng = np.random.RandomState(seed + 1009)
    shadow_domain = shadow_rng.permutation(shadow_domain).tolist()
    calibration_count = min(
        args.mia_max_samples,
        (args.model_train_size - args.adp_probe_pool_size) // 2,
    )
    if calibration_count < 2:
        raise ValueError("Not enough room for paired shadow calibration targets")
    probe_pool = shadow_domain[:args.adp_probe_pool_size]
    cursor = args.adp_probe_pool_size
    shadow_targets = shadow_domain[cursor:cursor + calibration_count]
    cursor += calibration_count
    replacement_pool = shadow_domain[cursor:]
    replacements = _take_class_matched_indices(
        replacement_pool, labels[shadow_targets], labels, shadow_rng
    )
    excluded = set(probe_pool) | set(shadow_targets) | set(replacements)
    base_candidates = [idx for idx in shadow_domain if idx not in excluded]
    base_size = args.model_train_size - len(probe_pool) - len(shadow_targets)
    if base_size < 1 or base_size > len(base_candidates):
        raise ValueError("Invalid model_train_size for paired ADP shadow worlds")
    shadow_base = shadow_rng.choice(
        base_candidates, size=base_size, replace=False
    ).tolist()
    shadow_member_train = shadow_base + probe_pool + shadow_targets
    shadow_nonmember_train = shadow_base + probe_pool + replacements

    target_rng = np.random.RandomState(seed + 2018)
    target_domain = target_rng.permutation(target_domain)
    positive_cut = int(round(0.8 * len(target_domain)))
    positive_pool = target_domain[:positive_cut]
    negative_pool = target_domain[positive_cut:]
    target_train = target_rng.choice(
        positive_pool, size=args.model_train_size, replace=False
    ).tolist()
    target_probe_pool = target_train[:args.adp_probe_pool_size]
    member_candidates = target_train[args.adp_probe_pool_size:]
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
            "member_train_indices": shadow_member_train,
            "nonmember_train_indices": shadow_nonmember_train,
            "probe_pool_indices": probe_pool,
            "query_indices": shadow_targets,
        },
        "target": {
            "train_indices": target_train,
            "probe_pool_indices": target_probe_pool,
            "member_query_indices": target_members,
            "nonmember_query_indices": target_nonmembers,
        },
    }


def train_original_model(dataset, train_indices, args, seed, name):
    """Train one VIB world on exactly the supplied indices."""
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


def rfu_unlearn_sequential_records(original_model, augmented_dataset,
                                   train_indices, probe_schedule, args, name):
    """Return CPU snapshots [M_0,...,M_k] after one RFU request per probe."""
    model = copy.deepcopy(original_model).to(args.device)
    snapshots = [copy.deepcopy(original_model).to('cpu').eval()]
    deleted = set()
    start_time = time.time()
    for step, probe_index in enumerate(probe_schedule, start=1):
        if probe_index not in train_indices:
            raise ValueError(f"Probe {probe_index} is not in {name}'s training set")
        deleted.add(probe_index)
        remain_indices = [idx for idx in train_indices if idx not in deleted]
        erase_loader = make_loader(
            Subset(augmented_dataset, [probe_index]),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )
        remain_loader = make_loader(
            Subset(augmented_dataset, remain_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=False,
        )
        print(
            f"\n{name}: sequential RFU deletion {step}/{len(probe_schedule)}, "
            f"probe_index={probe_index}, remaining={len(remain_indices)}"
        )
        model = prepare_unl(
            erasing_dataset=erase_loader,
            dataloader_remaining_after_aux=remain_loader,
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            args=args,
            noise_flag="no noise",
        )
        model.eval()
        snapshots.append(copy.deepcopy(model).to('cpu').eval())
    synchronize_device(args)
    model.to('cpu')
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    return snapshots, time.time() - start_time


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


def run_adp_setting_for_seed(args, seed):
    set_random_seed(seed)
    print(f"\n===== Active Deletion Probing, seed={seed} =====")
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
        train_set_aug, splits["reference"]["train_indices"], args,
        seed + 5000, "public reference",
    ).to('cpu')
    args.original_epochs = saved_original_epochs

    shadow_member_original = train_original_model(
        train_set_aug, splits["shadow"]["member_train_indices"], args,
        seed + 10000, "shadow H1",
    ).to('cpu')
    shadow_nonmember_original = train_original_model(
        train_set_aug, splits["shadow"]["nonmember_train_indices"], args,
        seed + 10000, "shadow H0",
    ).to('cpu')
    target_original = train_original_model(
        train_set_aug, splits["target"]["train_indices"], args,
        seed + 20000, "target",
    ).to('cpu')

    shadow_schedule, shadow_schedule_info = select_adp_probe_schedule(
        reference_model,
        train_set,
        splits["reference"]["train_indices"],
        splits["shadow"]["probe_pool_indices"],
        splits["shadow"]["query_indices"],
        args,
        seed + 30000,
    )
    target_schedule_candidates = (
        splits["target"]["member_query_indices"]
        + splits["target"]["nonmember_query_indices"]
    )
    target_schedule, target_schedule_info = select_adp_probe_schedule(
        reference_model,
        train_set,
        splits["reference"]["train_indices"],
        splits["target"]["probe_pool_indices"],
        target_schedule_candidates,
        args,
        seed + 40000,
    )
    reference_model.to('cpu')
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    print("Shadow probe schedule:", shadow_schedule, shadow_schedule_info)
    print("Target probe schedule:", target_schedule, target_schedule_info)

    # Couple all stochastic RFU choices across H1/H0.  The two shadow paths
    # then differ in their matched target/replacement records, not in RNG.
    set_random_seed(seed + 50000)
    shadow_member_snapshots, shadow_member_unlearning_time = rfu_unlearn_sequential_records(
        shadow_member_original,
        train_set_aug,
        splits["shadow"]["member_train_indices"],
        shadow_schedule,
        args,
        "shadow H1",
    )
    set_random_seed(seed + 50000)
    shadow_nonmember_snapshots, shadow_nonmember_unlearning_time = rfu_unlearn_sequential_records(
        shadow_nonmember_original,
        train_set_aug,
        splits["shadow"]["nonmember_train_indices"],
        shadow_schedule,
        args,
        "shadow H0",
    )
    set_random_seed(seed + 60000)
    target_snapshots, target_unlearning_time = rfu_unlearn_sequential_records(
        target_original,
        train_set_aug,
        splits["target"]["train_indices"],
        target_schedule,
        args,
        "target",
    )

    shadow_query_loader = build_attack_loader(
        train_set, splits["shadow"]["query_indices"], args
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
        shadow_member_snapshots,
        shadow_nonmember_snapshots,
        target_snapshots,
        shadow_query_loader,
        target_member_loader,
        target_nonmember_loader,
        args,
        name=f"RFU seed={seed}",
    )
    synchronize_device(args)
    return {
        "seed": seed,
        "mia": mia,
        "shadow_member_unlearning_time": shadow_member_unlearning_time,
        "shadow_nonmember_unlearning_time": shadow_nonmember_unlearning_time,
        "target_unlearning_time": target_unlearning_time,
        "attacking_time": time.time() - attack_start,
        "shadow_probe_schedule": shadow_schedule,
        "target_probe_schedule": target_schedule,
        "shadow_schedule_info": shadow_schedule_info,
        "target_schedule_info": target_schedule_info,
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
    print("\n===== Active Deletion Probing RFU summary =====")
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
    print("RFU ADP evaluation seeds:", args.seeds)
    results = [run_adp_setting_for_seed(args, seed) for seed in args.seeds]
    return results, summarize_adp_results(results)


# Backward-compatible name for existing launch scripts.
test_paper_remaining_sample_setting = test_adp_remaining_sample_setting


if __name__ == "__main__":
    test_adp_remaining_sample_setting()
