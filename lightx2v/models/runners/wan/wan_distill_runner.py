import os

from loguru import logger

from lightx2v.models.networks.wan.distill_model import WanDistillModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct, WanRunner
from lightx2v.models.schedulers.wan.step_distill.scheduler import Wan22StepDistillScheduler, WanStepDistillScheduler
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("wan2.1_distill")
class WanDistillRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_transformer(self):
        if self.config.get("lora_configs") and self.config["lora_configs"]:
            model = WanModel(
                self.config["model_path"],
                self.config,
                self.init_device,
            )
            lora_wrapper = WanLoraWrapper(model)
            for lora_config in self.config["lora_configs"]:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, strength)
                logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")
        else:
            model = WanDistillModel(self.config["model_path"], self.config, self.init_device)
        return model

    def init_scheduler(self):
        if self.config["feature_caching"] == "NoCaching":
            self.scheduler = WanStepDistillScheduler(self.config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")


class MultiDistillModelStruct(MultiModelStruct):
    def __init__(self, model_list, config, boundary_step_index=2):
        self.model = model_list  # [high_noise_model, low_noise_model]
        assert len(self.model) == 2, "MultiModelStruct only supports 2 models now."
        self.config = config
        self.boundary_step_index = boundary_step_index
        self.cur_model_index = -1
        logger.info(f"boundary step index: {self.boundary_step_index}")

    @ProfilingContext4DebugL2("Swtich models in infer_main costs")
    def get_current_model_index(self):
        if self.scheduler.step_index < self.boundary_step_index:
            logger.info(f"using - HIGH - noise model at step_index {self.scheduler.step_index + 1}")
            self.scheduler.sample_guide_scale = self.config["sample_guide_scale"][0]
            if self.config.get("cpu_offload", False) and self.config.get("offload_granularity", "block") == "model":
                if self.cur_model_index == -1:
                    self.to_cuda(model_index=0)
                elif self.cur_model_index == 1:  # 1 -> 0
                    self.offload_cpu(model_index=1)
                    self.to_cuda(model_index=0)
            self.cur_model_index = 0
        else:
            logger.info(f"using - LOW - noise model at step_index {self.scheduler.step_index + 1}")
            self.scheduler.sample_guide_scale = self.config["sample_guide_scale"][1]
            if self.config.get("cpu_offload", False) and self.config.get("offload_granularity", "block") == "model":
                if self.cur_model_index == -1:
                    self.to_cuda(model_index=1)
                elif self.cur_model_index == 0:  # 0 -> 1
                    self.offload_cpu(model_index=0)
                    self.to_cuda(model_index=1)
            self.cur_model_index = 1


@RUNNER_REGISTER("wan2.2_moe_distill")
class Wan22MoeDistillRunner(WanDistillRunner):
    def __init__(self, config):
        super().__init__(config)
        if self.config.get("dit_quantized", False) and self.config.get("high_noise_quantized_ckpt", None):
            self.high_noise_model_path = self.config["high_noise_quantized_ckpt"]
        elif self.config.get("high_noise_original_ckpt", None):
            self.high_noise_model_path = self.config["high_noise_original_ckpt"]
        else:
            self.high_noise_model_path = os.path.join(self.config["model_path"], "high_noise_model")
            if not os.path.isdir(self.high_noise_model_path):
                self.high_noise_model_path = os.path.join(self.config["model_path"], "distill_models", "high_noise_model")
                if not os.path.isdir(self.high_noise_model_path):
                    raise FileNotFoundError(f"High Noise Model does not find")

        if self.config.get("dit_quantized", False) and self.config.get("low_noise_quantized_ckpt", None):
            self.low_noise_model_path = self.config["low_noise_quantized_ckpt"]
        elif not self.config.get("dit_quantized", False) and self.config.get("low_noise_original_ckpt", None):
            self.low_noise_model_path = self.config["low_noise_original_ckpt"]
        else:
            self.low_noise_model_path = os.path.join(self.config["model_path"], "low_noise_model")
            if not os.path.isdir(self.low_noise_model_path):
                self.low_noise_model_path = os.path.join(self.config["model_path"], "distill_models", "low_noise_model")
                if not os.path.isdir(self.high_noise_model_path):
                    raise FileNotFoundError(f"Low Noise Model does not find")

    def load_transformer(self):
        use_high_lora, use_low_lora = False, False
        if self.config.get("lora_configs") and self.config["lora_configs"]:
            for lora_config in self.config["lora_configs"]:
                if lora_config.get("name", "") == "high_noise_model":
                    use_high_lora = True
                elif lora_config.get("name", "") == "low_noise_model":
                    use_low_lora = True

        if use_high_lora:
            high_noise_model = WanModel(
                self.high_noise_model_path,
                self.config,
                self.init_device,
                model_type="wan2.2_moe_high_noise",
            )
            high_lora_wrapper = WanLoraWrapper(high_noise_model)
            for lora_config in self.config["lora_configs"]:
                if lora_config.get("name", "") == "high_noise_model":
                    lora_path = lora_config["path"]
                    strength = lora_config.get("strength", 1.0)
                    lora_name = high_lora_wrapper.load_lora(lora_path)
                    high_lora_wrapper.apply_lora(lora_name, strength)
                    logger.info(f"High noise model loaded LoRA: {lora_name} with strength: {strength}")
        else:
            high_noise_model = WanDistillModel(
                self.high_noise_model_path,
                self.config,
                self.init_device,
                model_type="wan2.2_moe_high_noise",
            )

        if use_low_lora:
            low_noise_model = WanModel(
                self.low_noise_model_path,
                self.config,
                self.init_device,
                model_type="wan2.2_moe_low_noise",
            )
            low_lora_wrapper = WanLoraWrapper(low_noise_model)
            for lora_config in self.config["lora_configs"]:
                if lora_config.get("name", "") == "low_noise_model":
                    lora_path = lora_config["path"]
                    strength = lora_config.get("strength", 1.0)
                    lora_name = low_lora_wrapper.load_lora(lora_path)
                    low_lora_wrapper.apply_lora(lora_name, strength)
                    logger.info(f"Low noise model loaded LoRA: {lora_name} with strength: {strength}")
        else:
            low_noise_model = WanDistillModel(
                self.low_noise_model_path,
                self.config,
                self.init_device,
                model_type="wan2.2_moe_low_noise",
            )

        return MultiDistillModelStruct([high_noise_model, low_noise_model], self.config, self.config["boundary_step_index"])

    def init_scheduler(self):
        if self.config["feature_caching"] == "NoCaching":
            self.scheduler = Wan22StepDistillScheduler(self.config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")
