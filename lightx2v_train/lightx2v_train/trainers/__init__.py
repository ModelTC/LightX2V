from lightx2v_train.utils.registry import build_trainer

from .dmd_lora import DmdLoraTrainer
from .full import FullTrainer
from .lora import LoraTrainer

__all__ = ["build_trainer", "DmdLoraTrainer", "FullTrainer", "LoraTrainer"]
