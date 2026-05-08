from __future__ import annotations

from lightx2v_train.utils.registry import TRAINER_REGISTER

from .base import BaseTrainer


@TRAINER_REGISTER("distill")
class DistillTrainer(BaseTrainer):
    def train(self, dataloader):
        raise NotImplementedError("DistillTrainer is reserved for the next step.")
