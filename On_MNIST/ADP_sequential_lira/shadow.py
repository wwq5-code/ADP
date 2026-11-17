try:
    from .model import MLP
except ImportError:  # Support running main_lira.py directly from On_MNIST.
    from ADP_lira.model import MLP


class ShadowModel(MLP):
    """MNIST shadow model with the same MLP architecture as the target."""

    pass
