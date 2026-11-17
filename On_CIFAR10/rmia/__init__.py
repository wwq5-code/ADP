"""RMIA implementation used by the CIFAR-10 experiment entry point."""

from .model import CNN
from .rmia import RMIA

__all__ = ["CNN", "RMIA"]
