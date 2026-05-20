from lightx2v_train.schedulers.flow_matching import RectifiedFlowMatchingScheduler


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.model_config = self.config["model"]
        self.training_config = self.config["training"]
        self.infer_config = self.config["inference"]

        self.noise_scheduler = RectifiedFlowMatchingScheduler(config)

    def set_model(self, model):
        self.model = model

    def set_data(self, dataloader_train, dataloader_eval=None):
        self.dataloader_train = dataloader_train
        self.dataloader_eval = dataloader_eval

    def train(self):
        raise NotImplementedError
