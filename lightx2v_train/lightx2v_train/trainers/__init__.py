from lightx2v_train.utils.registry import TRAINER_REGISTER, build_trainer

from .distill import DistillTrainer
from .full_finetune import FullFinetuneTrainer
from .lora import LoraTrainer
from .rl import RLTrainer

__all__ = ["DistillTrainer", "FullFinetuneTrainer", "LoraTrainer", "RLTrainer", "TRAINER_REGISTER", "build_trainer"]
