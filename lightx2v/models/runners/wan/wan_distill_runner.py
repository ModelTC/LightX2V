from loguru import logger

from lightx2v.models.networks.wan.distill_model import WanDistillModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.step_distill.scheduler import WanStepDistillScheduler
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("wan2.1_distill")
class WanDistillRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_transformer(self):
        if self.config.get("lora_configs") and self.config.lora_configs:
            model = WanModel(
                self.config.model_path,
                self.config,
                self.init_device,
            )
            lora_wrapper = WanLoraWrapper(model)
            for lora_config in self.config.lora_configs:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, strength)
                logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")
        else:
            model = WanDistillModel(self.config.model_path, self.config, self.init_device)
        return model

    def init_scheduler(self):
        if self.config.feature_caching == "NoCaching":
            scheduler = WanStepDistillScheduler(self.config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")
        self.model.set_scheduler(scheduler)
