import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """MLP classifier for 28 x 28 grayscale MNIST images."""

    def __init__(self, input_size=28 * 28, hidden_size=256, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_size // 2, num_classes)

    def feature(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def forward(self, x):
        x = self.feature(x)
        return self.classifier(self.dropout(x))
