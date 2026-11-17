import sys

sys.argv = ['']
del sys

import os
os.environ["TQDM_DISABLE"] = "1"     # 放在导入 tqdm 前


import argparse
import pandas as pd
from PIL import Image
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import copy
import random
import time

from tqdm import tqdm



def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', choices=['tinyIMAGENET'], default='tinyIMAGENET')
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
    def __init__(self, n_feature=192, h_dim=3 * 32, n_output=200):
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
            logits_y = self.approximator(logits_z)
            logits_y = logits_y.view(B, -1)
            return logits_z, logits_y, mu, logvar
        elif mode == 'with_reconstruction':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            logits_y = self.approximator(logits_z)
            logits_y = logits_y.view(B, -1)
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
        return torch.sigmoid(output_x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.randn_like(std))
        return eps.mul(std).add_(mu)



def init_vib(args):
    if args.dataset == 'tinyIMAGENET':
        approximator = LinearModel(n_feature=args.dimZ, n_output=200)
        model_ft = models.resnet18(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.dimZ * 2)
        encoder = model_ft
        decoder = LinearModel(n_feature=args.dimZ, n_output=3 * args.image_size * args.image_size)
        lr = args.lr

    vib = VIB(encoder, approximator, decoder)
    vib.to(args.device)
    return vib, lr





def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def vib_train_original_IB(dataset, model,optimizer, loss_fn, reconstruction_function, args):


    with tqdm(dataset) as pbar:
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 100)
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
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 100)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        logits_z, logits_y, x_hat, mu, logvar = vib(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 100)


        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (logits_y.argmax(dim=1) == y).sum().item()
        num_total += len(x)

    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}, total_num:{num_total}')
    return acc


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.data = []
        self.labels = []

        if self.train:
            classes = sorted(os.listdir(os.path.join(root_dir, 'train')))
            for label, cls in enumerate(classes):
                cls_dir = os.path.join(root_dir, 'train', cls, 'images')
                for img_name in os.listdir(cls_dir):
                    self.data.append(os.path.join(cls_dir, img_name))
                    self.labels.append(label)
        else:
            val_dir = os.path.join(root_dir, 'val', 'images')
            val_annotations = pd.read_csv(
                os.path.join(root_dir, 'val', 'val_annotations.txt'),
                sep='\t',
                header=None,
                names=['file_name', 'class', 'x1', 'y1', 'x2', 'y2'],
            )
            class_to_idx = {
                cls: idx
                for idx, cls in enumerate(sorted(os.listdir(os.path.join(root_dir, 'train'))))
            }
            for _, row in val_annotations.iterrows():
                self.data.append(os.path.join(val_dir, row['file_name']))
                self.labels.append(class_to_idx[row['class']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


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
    attack_tpr_at_1fpr = _tpr_at_fpr(retain_probs, nonmember_probs, target_fpr=0.01)

    result = {
        "attack_auc": attack_auc,
        "attack_tpr_at_1fpr": attack_tpr_at_1fpr,
    }

    print(f"\nLearned MIA results for {name}:")
    print(
        f"  MIA AUC={result['attack_auc']:.4f}, "
        f"TPR@1%FPR={result['attack_tpr_at_1fpr']:.4f}"
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
            x_e, y_e = x_e.to(args.device), y_e.to(args.device)  # (B, C, H, W), (B, 100)
            x_r, y_r = x_r.to(args.device), y_r.to(args.device)  # (B, C, H, W), (B, 100)

            logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = model(x_e, mode='with_reconstruction')  # (B, C* h* w), (B, N, 100)

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
                x_cpu = x_cpu.view(x_cpu.size(0), 3, args.image_size, args.image_size)
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


def configure_tinyimagenet_unlearning_args():
    args = args_parser()
    args.gpu = 1
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.dataset = 'tinyIMAGENET'
    args.add_noise = False
    args.beta = 0.0001
    args.mse_rate = 0.1
    args.lr = 0.0001
    args.dimZ = 512
    args.batch_size = 200
    args.image_size = 128

    args.unlearning_ratio = 0.05
    args.unlearning_ratios = [0.01, 0.02, 0.03, 0.04, 0.05]
    args.unlearning_epochs = 1
    args.seeds = [0, 1, 2, 3, 4]

    return args


def load_tinyimagenet_data(args):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_dir = "/home/wwq/Data/data/tiny-imagenet-200/tiny-imagenet-200"
    train_set_aug = TinyImageNetDataset(root_dir=data_dir, transform=train_transform, train=True)
    train_set = TinyImageNetDataset(root_dir=data_dir, transform=test_transform, train=True)
    test_set = TinyImageNetDataset(root_dir=data_dir, transform=test_transform, train=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    args.x_shape = (3, args.image_size, args.image_size)
    return train_set, train_set_aug, test_set, test_loader


def load_standard_tinyimagenet_model(args):

    ckpt_name = "vib_tinyimagenet_standard.pt"
    ckpt_path = resolve_existing_path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_name),
        ckpt_name,
    )

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
        return f"{prefix}_mia_auc=nan, {prefix}_mia_tpr_at_1fpr=nan"
    return (
        f"{prefix}_mia_auc={mia_result['attack_auc']:.4f}, "
        f"{prefix}_mia_tpr_at_1fpr={mia_result['attack_tpr_at_1fpr']:.4f}"
    )


def format_result_summary(result):
    return "".join([
        f"unlearning_ratio={result['unlearning_ratio']:.4f}, "
        f"acc_after={result['acc_after']:.4f}, "
        f"erased_acc={result['erased_acc']:.4f}, "
        f"{summarize_mia_metrics('mia_before', result['mia_before'])}, "
        f"{summarize_mia_metrics('mia_after', result['mia_after'])}, "
        f"unlearning_running_time={result['unlearning_running_time']:.4f}s, "
        f"attacking_running_time={result['attacking_running_time']:.4f}s"
    ])


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
        "attacking_running_time": result["attacking_running_time"],
    }
    for prefix in ("mia_before", "mia_after"):
        mia_result = result.get(prefix)
        if mia_result is None:
            continue
        metrics[f"{prefix}_mia_auc"] = float(mia_result["attack_auc"])
        metrics[f"{prefix}_mia_tpr_at_1fpr"] = float(mia_result["attack_tpr_at_1fpr"])
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
        "mia_before_mia_tpr_at_1fpr",
        "mia_after_mia_auc",
        "mia_after_mia_tpr_at_1fpr",
        "unlearning_running_time",
        "attacking_running_time",
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

    train_set, train_set_aug, test_set, test_loader = load_tinyimagenet_data(args)
    base_vib, ckpt = load_standard_tinyimagenet_model(args)
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
        attack_start_time = time.time()

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
        attacking_running_time = time.time() - attack_start_time
        print(f"RFU attacking running time: {attacking_running_time:.4f}s")

        result = {
            "unlearning_ratio": unlearning_ratio,
            "seed": seed,
            "acc_after": acc_after,
            "erased_acc": erased_acc,
            "mia_before": mia_before,
            "mia_after": mia_after,
            "unlearning_running_time": unlearning_running_time,
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
    args = configure_tinyimagenet_unlearning_args()
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
