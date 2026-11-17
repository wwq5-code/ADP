import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Small CNN for 28 x 28 grayscale Fashion-MNIST images."""

    def __init__(self, num_classes=10, feature_size=256, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, feature_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_size, num_classes)

    def feature(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return F.relu(self.fc(self.flatten(x)))

    def forward(self, x):
        return self.classifier(self.dropout(self.feature(x)))


class MLP(nn.Module):
    """MLP classifier for 28 x 28 grayscale Fashion-MNIST images."""

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
