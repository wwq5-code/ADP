import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    """Torchvision ResNet-18 classifier for 200-class TinyImageNet."""

    def __init__(self, num_classes=200, pretrained=False):
        super().__init__()
        self.pretrained = bool(pretrained)
        weights_enum = getattr(models, 'ResNet18_Weights', None)
        if weights_enum is not None:
            weights = weights_enum.DEFAULT if self.pretrained else None
            self.backbone = models.resnet18(weights=weights)
        else:  # torchvision < 0.13
            self.backbone = models.resnet18(pretrained=self.pretrained)
        self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def feature(self, x):
        model = self.backbone
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return x.flatten(1)

    def forward(self, x):
        return self.backbone.fc(self.feature(x))


def resnet18(num_classes=200, pretrained=False):
    return ResNet18(num_classes=num_classes, pretrained=pretrained)
