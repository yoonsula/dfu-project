from .dfu_trainer import train as train_dfu
from .foot_trainer import train as train_foot
from .wound_trainer import train as train_wound

__all__ = ["train_dfu", "train_foot", "train_wound"]
