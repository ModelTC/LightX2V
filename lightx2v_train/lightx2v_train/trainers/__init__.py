from lightx2v_train.utils.registry import TRAINER_REGISTER, build_trainer

from .full_finetune import FullFinetuneTrainer
from .lora import LoraTrainer

__all__ = ["FullFinetuneTrainer", "LoraTrainer", "TRAINER_REGISTER", "build_trainer"]
