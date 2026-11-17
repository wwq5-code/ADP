try:
    from .model import ResNet18
except ImportError:
    from model import ResNet18


class ShadowModel(ResNet18):
    """ResNet-18 shadow classifier for 200-class TinyImageNet."""

    def __init__(self, num_classes=200, pretrained=False, **_):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
