"""Minimal MNIST exact-retraining unlearning experiment.

This module intentionally contains no VIB, ADP, membership inference, shadow
models, or approximate unlearning.  It trains one MLP on D0, deletes a small
set of training records, and trains a fresh MLP on the retained set.
"""

import argparse
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST


def parse_args():
    """Initialize the experiment argument namespace only."""
    return argparse.Namespace()


class MNISTMLP(nn.Module):
    """Small classifier used by both initial training and exact retraining."""

    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(28 * 28, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.classifier(x)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, indices, batch_size, shuffle, args, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.type == "cuda",
        generator=generator,
    )


def train_from_scratch(dataset, train_indices, epochs, args, seed, name):
    """Create a fresh model and train it only on train_indices."""
    set_seed(seed)
    model = MNISTMLP().to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(
        dataset,
        train_indices,
        args.batch_size,
        shuffle=True,
        args=args,
        seed=seed,
    )

    start = time.time()
    for epoch in range(epochs):
        model.train()
        loss_sum = torch.zeros((), device=args.device)
        correct = torch.zeros((), dtype=torch.long, device=args.device)
        count = 0
        for x, y in loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            batch_count = len(y)
            loss_sum += loss.detach() * batch_count
            correct += (logits.argmax(1) == y).sum()
            count += batch_count

        print(
            f"{name}: epoch {epoch + 1:02d}/{epochs}, "
            f"loss={loss_sum.item() / count:.4f}, "
            f"accuracy={correct.item() / count:.4f}"
        )

    model.eval()
    return model, time.time() - start


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    confidence_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probabilities = F.softmax(logits, dim=1)
        batch_count = len(y)
        loss_sum += F.cross_entropy(logits, y, reduction="sum")
        confidence_sum += probabilities.gather(1, y[:, None]).sum()
        correct += (logits.argmax(1) == y).sum()
        count += batch_count
    return {
        "count": count,
        "loss": loss_sum.item() / count,
        "accuracy": correct.item() / count,
        "mean_true_label_confidence": confidence_sum.item() / count,
    }


def choose_indices(dataset_size, train_size, delete_count, seed):
    if train_size < 2 or train_size > dataset_size:
        raise ValueError(
            f"train_size must be in [2, {dataset_size}], got {train_size}."
        )
    if delete_count < 1 or delete_count >= train_size:
        raise ValueError(
            "delete_count must be positive and leave at least one retained record."
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_indices = torch.randperm(
        dataset_size, generator=generator
    )[:train_size].tolist()
    deleted_indices = train_indices[:delete_count]
    return train_indices, deleted_indices


def exact_retraining_unlearn(
    dataset,
    initial_train_indices,
    deleted_indices,
    args,
):
    """Delete records and train a new model from scratch on the retained set."""
    deleted_set = set(int(index) for index in deleted_indices)
    initial_set = set(int(index) for index in initial_train_indices)
    missing = deleted_set - initial_set
    if missing:
        raise ValueError(
            f"Deletion records are not in D0: {sorted(missing)}"
        )
    retained_indices = [
        int(index)
        for index in initial_train_indices
        if int(index) not in deleted_set
    ]
    retrained_model, elapsed = train_from_scratch(
        dataset,
        retained_indices,
        args.retrain_epochs,
        args,
        # The same seed couples the two training runs for a clean comparison,
        # but this is still a newly initialized model trained from scratch.
        args.seed,
        "retrained model",
    )
    return retrained_model, retained_indices, elapsed


def load_mnist(data_root):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = MNIST(
        data_root, train=True, transform=transform, download=True
    )
    test_set = MNIST(
        data_root, train=False, transform=transform, download=True
    )
    return train_set, test_set


def main():
    args = parse_args()

    # Set every experiment parameter here before the program runs.
    args.data_root = "/home/wwq/Data/data/mnist"
    args.train_size = 5000
    args.delete_count = 1
    args.epochs = 10
    args.retrain_epochs = 10
    args.batch_size = 256
    args.learning_rate = 1e-3
    args.seed = 0
    args.num_workers = 0
    args.gpu = 0  # Set to -1 to force CPU, or use 0, 1, ... for a CUDA GPU.
    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu'
    )

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    if args.epochs < 1:
        raise ValueError("epochs must be positive.")
    if args.retrain_epochs is None:
        args.retrain_epochs = args.epochs
    if args.retrain_epochs < 1:
        raise ValueError("retrain_epochs must be positive.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    set_seed(args.seed)
    train_set, test_set = load_mnist(args.data_root)
    initial_indices, deleted_indices = choose_indices(
        len(train_set),
        args.train_size,
        args.delete_count,
        args.seed + 1000,
    )

    print(
        f"device={args.device}, D0={len(initial_indices)}, "
        f"delete={len(deleted_indices)}, "
        f"retained={len(initial_indices) - len(deleted_indices)}"
    )
    print(f"deleted_indices={deleted_indices}")

    initial_model, initial_time = train_from_scratch(
        train_set,
        initial_indices,
        args.epochs,
        args,
        args.seed,
        "initial model",
    )
    retrained_model, retained_indices, retrain_time = (
        exact_retraining_unlearn(
            train_set,
            initial_indices,
            deleted_indices,
            args,
        )
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.type == "cuda",
    )
    deleted_loader = make_loader(
        train_set,
        deleted_indices,
        args.batch_size,
        shuffle=False,
        args=args,
        seed=args.seed,
    )
    retained_loader = make_loader(
        train_set,
        retained_indices,
        args.batch_size,
        shuffle=False,
        args=args,
        seed=args.seed,
    )

    results = {
        "configuration": {
            "device": str(args.device),
            "seed": args.seed,
            "initial_train_size": len(initial_indices),
            "deleted_count": len(deleted_indices),
            "retained_count": len(retained_indices),
            "initial_epochs": args.epochs,
            "retrain_epochs": args.retrain_epochs,
            "fresh_retraining": True,
        },
        "deleted_indices": deleted_indices,
        "training_seconds": {
            "initial": initial_time,
            "retraining": retrain_time,
        },
        "initial_model": {
            "test": evaluate(initial_model, test_loader, args.device),
            "deleted": evaluate(initial_model, deleted_loader, args.device),
        },
        "retrained_model": {
            "test": evaluate(retrained_model, test_loader, args.device),
            "deleted": evaluate(retrained_model, deleted_loader, args.device),
            "retained": evaluate(retrained_model, retained_loader, args.device),
        },
    }
    print("\n===== Exact-retraining unlearning result =====")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
