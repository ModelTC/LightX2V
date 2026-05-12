from lightx2v_train.utils.registry import build_trainer

from .lora import LoraTrainer

__all__ = ["build_trainer", "LoraTrainer"]
