import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """CNN classifier for 3 x 32 x 32 CIFAR-10 images.

    This matches the target architecture used by ``On_CIFAR10/main_lira.py``.
    """

    def __init__(self, channel=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def feature(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.feature(x)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
