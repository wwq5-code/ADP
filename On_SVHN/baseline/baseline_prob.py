import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def calculate_accuracy(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_model_with_raw_tensors(
    model,
    train_data,
    train_labels,
    epochs=100,
    lr=0.01,
    bs=128 * 2,
    device='cuda',
):
    dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    return train_model_with_loader(model, train_loader, epochs, lr, device)


def train_model_with_loader(
    model,
    train_loader,
    training_epochs=100,
    lr=0.01,
    device='cuda',
    verbose=True,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(training_epochs):
        started = time.monotonic()
        if verbose:
            print(
                f'[SVHN target training] epoch {epoch + 1}/{training_epochs} '
                f'| batches={len(train_loader)} | device={device}',
                flush=True,
            )
        total_loss = 0.0
        sample_count = 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            sample_count += labels.size(0)
        if verbose:
            print(
                f'[SVHN target training] epoch {epoch + 1}/{training_epochs} '
                f'complete | loss={total_loss / max(sample_count, 1):.4f} '
                f'| elapsed={time.monotonic() - started:.1f}s',
                flush=True,
            )
    return model


def get_probability_vectors(model, dataloader, device):
    """Return full posterior vectors for every sample in a dataloader."""
    probabilities = []
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            outputs = model(images.to(device))
            probabilities.append(F.softmax(outputs, dim=1).cpu())
    if not probabilities:
        raise ValueError('Probability feature loader must not be empty.')
    return torch.cat(probabilities, dim=0)


def get_tensor_probability_vectors(model, data, device, batch_size=256):
    """Batched posterior extraction for an in-memory target tensor."""
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for (images,) in loader:
            outputs = model(images.to(device))
            probabilities.append(F.softmax(outputs, dim=1).cpu())
    if not probabilities:
        raise ValueError('Target data must not be empty.')
    return torch.cat(probabilities, dim=0)


def _fit_probability_attack(
    member_features,
    nonmember_features,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device,
):
    if len(member_features) != len(nonmember_features):
        raise ValueError('Attack member and non-member features must be balanced.')

    features = torch.cat([member_features, nonmember_features], dim=0)
    labels = torch.tensor(
        [1] * len(member_features) + [0] * len(nonmember_features),
        dtype=torch.float32,
    ).unsqueeze(1)
    loader = DataLoader(
        TensorDataset(features, labels), batch_size=128, shuffle=True
    )

    classifier = nn.Sequential(
        nn.Linear(features.shape[1], attack_hidden_size),
        nn.ReLU(),
        nn.Linear(attack_hidden_size, 1),
        nn.Sigmoid(),
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=attack_lr)

    for epoch in range(attack_epochs):
        classifier.train()
        running_loss = 0.0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            predictions = classifier(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == attack_epochs - 1:
            print(
                f'[Probability MIA] epoch {epoch + 1}/{attack_epochs} | '
                f'loss={running_loss / len(loader):.4f}',
                flush=True,
            )
    return classifier


def MIA(
    target_model,
    member_loader,
    nonmember_loader,
    target_data,
    target_labels,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device='cuda',
):
    """Static probability MIA using only the original target model."""
    del target_labels  # The full posterior vector is the attack feature.
    member_features = get_probability_vectors(target_model, member_loader, device)
    nonmember_features = get_probability_vectors(
        target_model, nonmember_loader, device
    )
    classifier = _fit_probability_attack(
        member_features,
        nonmember_features,
        attack_hidden_size,
        attack_epochs,
        attack_lr,
        device,
    )
    target_features = get_tensor_probability_vectors(
        target_model, target_data, device
    )
    classifier.eval()
    with torch.no_grad():
        return classifier(target_features.to(device)).squeeze(1).cpu().numpy()


def _passive_features(before_probabilities, after_probabilities):
    """Concatenate deletion-before, deletion-after, and posterior delta."""
    return torch.cat(
        [
            before_probabilities,
            after_probabilities,
            after_probabilities - before_probabilities,
        ],
        dim=1,
    )


def PassiveMIA(
    model_before,
    model_after,
    member_loader,
    nonmember_loader,
    target_data,
    target_labels,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device='cuda',
):
    """Probability MIA using an exact-retraining deletion response."""
    del target_labels
    member_features = _passive_features(
        get_probability_vectors(model_before, member_loader, device),
        get_probability_vectors(model_after, member_loader, device),
    )
    nonmember_features = _passive_features(
        get_probability_vectors(model_before, nonmember_loader, device),
        get_probability_vectors(model_after, nonmember_loader, device),
    )
    classifier = _fit_probability_attack(
        member_features,
        nonmember_features,
        attack_hidden_size,
        attack_epochs,
        attack_lr,
        device,
    )
    target_features = _passive_features(
        get_tensor_probability_vectors(model_before, target_data, device),
        get_tensor_probability_vectors(model_after, target_data, device),
    )
    classifier.eval()
    with torch.no_grad():
        scores = classifier(target_features.to(device)).squeeze(1).cpu().numpy()
    return np.asarray(scores)


def _sequential_features(probability_trajectory):
    """Encode all released posteriors and every consecutive response delta."""
    if len(probability_trajectory) < 3:
        raise ValueError(
            'Sequential ADP requires M_0 and at least two deletion releases.'
        )
    deltas = [
        probability_trajectory[index] - probability_trajectory[index - 1]
        for index in range(1, len(probability_trajectory))
    ]
    return torch.cat([*probability_trajectory, *deltas], dim=1)


def SequentialADP(
    models,
    member_loader,
    nonmember_loader,
    target_data,
    target_labels,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device='cuda',
):
    """Probability MIA over a fixed, precommitted deletion trajectory."""
    del target_labels
    member_features = _sequential_features([
        get_probability_vectors(model, member_loader, device)
        for model in models
    ])
    nonmember_features = _sequential_features([
        get_probability_vectors(model, nonmember_loader, device)
        for model in models
    ])
    classifier = _fit_probability_attack(
        member_features,
        nonmember_features,
        attack_hidden_size,
        attack_epochs,
        attack_lr,
        device,
    )
    target_features = _sequential_features([
        get_tensor_probability_vectors(model, target_data, device)
        for model in models
    ])
    classifier.eval()
    with torch.no_grad():
        scores = classifier(target_features.to(device)).squeeze(1).cpu().numpy()
    return np.asarray(scores)


def ActiveADP(
    models,
    member_loader,
    nonmember_loader,
    target_data,
    target_labels,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device='cuda',
):
    """Probability decoder for a design-shadow-selected ADP trajectory.

    Probe selection is intentionally kept outside this decoder.  Once the
    complete active schedule has been precommitted, its released-model
    trajectory has the same fixed-width representation as Sequential ADP.
    """
    return SequentialADP(
        models,
        member_loader,
        nonmember_loader,
        target_data,
        target_labels,
        attack_hidden_size,
        attack_epochs,
        attack_lr,
        device,
    )
