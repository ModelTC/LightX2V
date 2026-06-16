from lightx2v.common.modules.weight_module import WeightModule


class HidreamO1ImagePostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
