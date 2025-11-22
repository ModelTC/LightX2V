from abc import ABCMeta, abstractmethod


class AttnWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name):
        self.weight_name = weight_name
        self.config = {}

    def load(self, weight_dict):
        pass

    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config

    def to_cpu(self, non_blocking=False):
        pass

    def to_cuda(self, non_blocking=False):
        pass

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_inde=None):
        return {}
