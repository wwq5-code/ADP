try:
    from .model import CNN
except ImportError:  # Support running main_ADP_active_lira.py directly.
    from ADP_active_lira.model import CNN


class ShadowModel(CNN):
    """CIFAR-10 shadow model matching the target CNN architecture."""

    pass
