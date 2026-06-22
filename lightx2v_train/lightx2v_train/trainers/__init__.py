from lightx2v_train.utils.registry import build_trainer

from .dmd import DmdTrainer
from .flow import FlowMatchingTrainer
from .tf import TFTrainer

__all__ = ["build_trainer", "DmdTrainer", "FlowMatchingTrainer", "TFTrainer"]
