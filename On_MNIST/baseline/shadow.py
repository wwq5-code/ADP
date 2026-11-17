if __package__:
    from .model import MLP
else:
    from baseline.model import MLP


class ShadowModel(MLP):
    """MNIST shadow model with the same architecture as the target MLP."""

    def __init__(self, input_size=28 * 28, hidden_size=256, num_classes=10):
        super().__init__(input_size, hidden_size, num_classes)
