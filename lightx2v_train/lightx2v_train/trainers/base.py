from __future__ import annotations


class BaseTrainer:
    def __init__(self, config):
        self.config = config

    def set_model(self, model):
        self.model = model

    def set_data(self, dataloader, dataloader_eval=None):
        self.dataloader = dataloader
        self.dataloader_eval = dataloader_eval

    def train(self):
        raise NotImplementedError
