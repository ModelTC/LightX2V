from lightx2v_train.utils.registry import build_trainer

from .dmd import DmdTrainer, VideoArDmdTrainer, VideoDmdTrainer
from .dopsd import DopsdTrainer
from .fastwam import FastWAMTrainer
from .flow import FlowMatchingTrainer
from .tf import TFTrainer

ARDmdTrainer = VideoArDmdTrainer

__all__ = ["build_trainer", "ARDmdTrainer", "DmdTrainer", "FlowMatchingTrainer", "TFTrainer", "VideoArDmdTrainer", "VideoDmdTrainer", "DopsdTrainer", "FastWAMTrainer"]
