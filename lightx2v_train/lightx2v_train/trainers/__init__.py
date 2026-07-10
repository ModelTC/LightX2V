from lightx2v_train.utils.registry import build_trainer

from .dmd import DmdTrainer, VideoArDmdTrainer, VideoDmdTrainer
from .dopsd import DopsdTrainer
from .flow import FlowMatchingTrainer
from .one_forcing import OneForcingTrainer
from .tf import TFTrainer

ARDmdTrainer = VideoArDmdTrainer

__all__ = ["build_trainer", "ARDmdTrainer", "DmdTrainer", "FlowMatchingTrainer", "OneForcingTrainer", "TFTrainer", "VideoArDmdTrainer", "VideoDmdTrainer", "DopsdTrainer"]
