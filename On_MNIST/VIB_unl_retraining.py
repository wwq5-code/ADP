import sys

sys.argv = ['']
del sys

import os
os.environ["TQDM_DISABLE"] = "1"     # 放在导入 tqdm 前


import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
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
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='MNIST')
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
        self.fc3 = nn.Linear(28 * 28, 28 * 28)  # output

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
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)



def init_vib(args):
    if args.dataset == 'MNIST':
        approximator = LinearModel(n_feature=args.dimZ)
        decoder = LinearModel(n_feature=args.dimZ, n_output=28 * 28)
        encoder = CNNEncoder(in_channels=1, dim_z=args.dimZ * 2)

        lr = args.lr

    elif args.dataset == 'CIFAR10':

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

def split_dataset_for_unlearning(dataset, erase_ratio: float, seed: int):
    """Split a standard training dataset into erase/remain subsets."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(dataset))
    n = len(indices)
    n_erase = max(1, int(round(n * erase_ratio)))

    perm = rng.permutation(n)
    erase_indices = indices[perm[:n_erase]].tolist()
    remain_indices = indices[perm[n_erase:]].tolist()

    erase_subset = Subset(dataset, erase_indices)
    remain_subset = Subset(dataset, remain_indices)

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
def _evaluate_attack_classifier(attack_model, features, labels):
    attack_model.eval()
    logits = attack_model(features)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = (preds == labels).float().mean().item()
    return probs.cpu().numpy(), acc


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

    train_probs, train_acc = _evaluate_attack_classifier(attack_model, train_features, train_labels)
    retain_probs = train_probs[:len(retain_features)]
    nonmember_probs = train_probs[len(retain_features):]
    attack_auc = _binary_auc(retain_probs, nonmember_probs)

    result = {
        "attack_train_acc": train_acc,
        "attack_auc": attack_auc,
    }

    print(f"\nLearned MIA results for {name}:")
    print(
        f"  MIA acc={result['attack_train_acc']:.4f}, "
        f"MIA AUC={result['attack_auc']:.4f}"
    )

    return result


def synchronize_device(args):
    if getattr(args, "device", None) is not None and args.device.type == "cuda":
        torch.cuda.synchronize(args.device)


def retrain_on_remaining_dataset(
    train_set,
    args,
    erase_ratio=0.05,
    base_seed=0,
    loss_fn=None,
):
    """Retrain a fresh VIB model only on the remaining training subset."""
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    reconstruction_function = nn.MSELoss(reduction='mean')

    erase_subset, remain_subset, erase_idx, remain_idx = split_dataset_for_unlearning(
        train_set,
        erase_ratio=erase_ratio,
        seed=base_seed,
    )
    remain_loader = make_loader(remain_subset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    unlearning_data_info = {
        "standard_split": {
            "erase_indices": erase_idx,
            "remain_indices": remain_idx,
            "remain_loader": remain_loader,
            "erase_size": len(erase_subset),
            "remain_size": len(remain_subset),
        }
    }

    model, _ = init_vib(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    synchronize_device(args)
    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        print(f"Retraining epoch {epoch}/{args.num_epochs}")
        model.train()
        model = vib_train_original_IB(
            remain_loader,
            model,
            optimizer,
            loss_fn,
            reconstruction_function,
            args,
        )
    synchronize_device(args)
    running_time = time.time() - start_time

    model.to(args.device)
    model.eval()
    return model, unlearning_data_info, running_time


def build_erased_data_loader(train_set, unlearning_data_info, args):
    erase_indices = []
    for info in unlearning_data_info.values():
        erase_indices.extend(info["erase_indices"])
    if len(erase_indices) == 0:
        return None
    erase_subset = Subset(train_set, erase_indices)
    return make_loader(erase_subset, batch_size=args.batch_size, shuffle=False, num_workers=1)


def configure_mnist_unlearning_args():
    args = args_parser()
    args.gpu = 0
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.dataset = 'MNIST'
    args.add_noise = False
    args.beta = 0.0001
    args.mse_rate = 0.1
    args.lr = 0.0001
    args.dimZ = 512
    args.batch_size = 100

    args.num_epochs = 40
    args.unlearning_ratio = 0.05
    args.unlearning_ratios = [0.01, 0.02, 0.03, 0.04, 0.05]
    args.seeds = [0, 1, 2, 3, 4]

    return args


def load_mnist_data(args):
    trans_mnist = transforms.Compose([transforms.ToTensor(), ])
    train_set = MNIST('/home/wwq/Data/data/mnist', train=True, transform=trans_mnist, download=True)
    test_set = MNIST('/home/wwq/Data/data/mnist', train=False, transform=trans_mnist, download=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    args.x_shape = (1, 28, 28)
    return train_set, test_set, test_loader


def load_standard_mnist_model(args):

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vib_mnist_standard.pt")
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


def summarize_mia(prefix, mia_result):
    if mia_result is None:
        return f"{prefix}_mia_acc=nan, {prefix}_mia_auc=nan"
    return (
        f"{prefix}_mia_acc={mia_result['attack_train_acc']:.4f}, "
        f"{prefix}_mia_auc={mia_result['attack_auc']:.4f}"
    )


def format_result_summary(result):
    return (
        f"unlearning_ratio={result['unlearning_ratio']:.4f}, "
        f"acc_after={result['acc_after']:.4f}, "
        f"erased_acc={result['erased_acc']:.4f}, "
        f"{summarize_mia('mia_before', result['mia_before'])}, "
        f"{summarize_mia('mia_after', result['mia_after'])}, "
        f"running_time={result['running_time']:.4f}s"
    )


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
        "running_time": result["running_time"],
    }
    for prefix in ("mia_before", "mia_after"):
        mia_result = result.get(prefix)
        if mia_result is None:
            continue
        metrics[f"{prefix}_mia_acc"] = float(mia_result["attack_train_acc"])
        metrics[f"{prefix}_mia_auc"] = float(mia_result["attack_auc"])
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
    print("\n===== Retraining final summary =====")
    metric_order = [
        "acc_after",
        "erased_acc",
        "mia_before_mia_acc",
        "mia_before_mia_auc",
        "mia_after_mia_acc",
        "mia_after_mia_auc",
        "running_time",
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


def run_retraining_ratio_sweep_for_seed(args, seed):
    set_random_seed(seed)
    print(f"\n===== Retraining seed={seed} =====")

    train_set, test_set, test_loader = load_mnist_data(args)
    base_vib, ckpt = load_standard_mnist_model(args)
    loaded_epoch = ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0
    _ = eva_vib(base_vib, test_loader, args, name=f"loaded model (seed={seed})", epoch=loaded_epoch)

    loss_fn = nn.CrossEntropyLoss()
    sweep_results = []

    for unlearning_ratio in args.unlearning_ratios:
        print(f"\n===== Retraining test: unlearning_ratio={unlearning_ratio:.4f} =====")
        args.unlearning_ratio = unlearning_ratio

        pre_unlearn_vib = copy.deepcopy(base_vib)
        pre_unlearn_vib.to(args.device)
        pre_unlearn_vib.eval()

        new_vib, unlearning_data_info, running_time = retrain_on_remaining_dataset(
            train_set=train_set,
            args=args,
            erase_ratio=args.unlearning_ratio,
            base_seed=seed,
            loss_fn=loss_fn,
        )
        print(f"Retraining running time: {running_time:.4f}s")

        acc_after = eva_vib(
            new_vib,
            test_loader,
            args,
            name=f"after retraining (ratio={unlearning_ratio:.4f})",
            epoch=args.num_epochs,
        )

        erased_loader = build_erased_data_loader(train_set, unlearning_data_info, args)
        erased_acc = float("nan")
        if erased_loader is not None:
            erased_acc = eva_vib(
                new_vib,
                erased_loader,
                args,
                name=f"erased data (ratio={unlearning_ratio:.4f})",
                epoch=args.num_epochs,
            )

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
                name=f"before retraining (ratio={unlearning_ratio:.4f})",
            )
            mia_after = learned_membership_inference_attack(
                new_vib,
                mia_retain_loader,
                mia_nonmember_loader,
                args,
                name=f"after retraining (ratio={unlearning_ratio:.4f})",
            )

        result = {
            "unlearning_ratio": unlearning_ratio,
            "seed": seed,
            "acc_after": acc_after,
            "erased_acc": erased_acc,
            "mia_before": mia_before,
            "mia_after": mia_after,
            "running_time": running_time,
        }
        sweep_results.append(result)

        print("Retraining sweep result: " + format_result_summary(result))

        del pre_unlearn_vib, new_vib
        torch.cuda.empty_cache()

    print(f"\n===== Retraining sweep summary (seed={seed}) =====")
    for result in sweep_results:
        print(format_result_summary(result))

    return sweep_results


def test_retraining_ratio_sweep():
    args = configure_mnist_unlearning_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("device", args.device)
    print("Retraining evaluation seeds:", args.seeds)

    all_seed_results = []
    for seed in args.seeds:
        seed_results = run_retraining_ratio_sweep_for_seed(args, seed)
        all_seed_results.append({
            "seed": seed,
            "results": seed_results,
        })

    summary = summarize_seed_sweep(all_seed_results)
    print_seed_sweep_summary(summary)

    return all_seed_results, summary


if __name__ == "__main__":
    test_retraining_ratio_sweep()
