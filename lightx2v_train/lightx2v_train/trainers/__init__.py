from lightx2v_train.utils.registry import build_trainer

from .dmd import DmdTrainer, LTX2T2AVArDmdTrainer, LTX2T2AVDmdTrainer, LingBotVideoDmdTrainer, VideoArDmdTrainer, VideoDmdTrainer
from .dopsd import DopsdTrainer
from .flow import FlowMatchingTrainer, LTX2T2AVFlowTrainer
from .tf import LTX2T2AVTeacherForcingTrainer, TFTrainer
from .fastwam import FastWAMTrainer

ARDmdTrainer = VideoArDmdTrainer

__all__ = [
    "build_trainer",
    "ARDmdTrainer",
    "DmdTrainer",
    "FlowMatchingTrainer",
    "LTX2T2AVArDmdTrainer",
    "LTX2T2AVDmdTrainer",
    "LTX2T2AVFlowTrainer",
    "LTX2T2AVTeacherForcingTrainer",
    "LingBotVideoDmdTrainer",
    "TFTrainer",
    "VideoArDmdTrainer",
    "VideoDmdTrainer",
    "DopsdTrainer",
    "FastWAMTrainer",
]
