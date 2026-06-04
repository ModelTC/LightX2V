from lightx2v_train.utils.registry import build_trainer

from .lora import LoraTrainer
from .dopsd import DopsdTrainer

__all__ = ["build_trainer", "LoraTrainer", "DopsdTrainer"]
