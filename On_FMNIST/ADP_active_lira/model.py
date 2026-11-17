try:
    from ..baseline.model import CNN as _BaselineCNN
except ImportError:  # Support direct execution from On_FMNIST.
    from baseline.model import CNN as _BaselineCNN


class CNN(_BaselineCNN):
    """Active-ADP CNN matching the Fashion-MNIST baseline architecture."""

    def __init__(
        self,
        input_size=28 * 28,
        hidden_size=256,
        num_classes=10,
        dropout=0.1,
    ):
        if input_size != 28 * 28:
            raise ValueError('Fashion-MNIST CNN expects 28 x 28 inputs.')
        super().__init__(
            num_classes=num_classes,
            feature_size=hidden_size,
            dropout=dropout,
        )
