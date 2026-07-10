from lightx2v_train.utils.registry import build_trainer

from .dmd import DmdTrainer, VideoArDmdTrainer, VideoDmdTrainer
from .dfd import VideoDfdTrainer
from .dopsd import DopsdTrainer
from .flow import FlowMatchingTrainer
from .tf import TFTrainer

ARDmdTrainer = VideoArDmdTrainer

__all__ = [
    "build_trainer",
    "ARDmdTrainer",
    "DmdTrainer",
    "FlowMatchingTrainer",
    "TFTrainer",
    "VideoArDmdTrainer",
    "VideoDfdTrainer",
    "VideoDmdTrainer",
    "DopsdTrainer",
]
