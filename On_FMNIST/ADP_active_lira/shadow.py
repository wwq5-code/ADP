try:
    from .model import CNN
except ImportError:  # Support running entry points directly from On_FMNIST.
    from ADP_active_lira.model import CNN


class ShadowModel(CNN):
    """Design-shadow CNN matching the Active ADP target architecture."""

    pass
