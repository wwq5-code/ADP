if __package__:
    from .model import CNN
else:
    from baseline.model import CNN


class ShadowModel(CNN):
    """Fashion-MNIST shadow model matching the target CNN architecture."""

    def __init__(self, num_classes=10, feature_size=256, dropout=0.1):
        super().__init__(num_classes, feature_size, dropout)
