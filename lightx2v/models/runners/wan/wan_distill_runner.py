import os

from loguru import logger

from lightx2v.models.networks.wan.distill_model import WanDistillModel
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct, WanRunner, build_wan_model_with_lora
from lightx2v.models.schedulers.wan.step_distill.scheduler import Wan21MeanFlowStepDistillScheduler, Wan22StepDistillScheduler, WanStepDistillScheduler
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("wan2.1_distill")
class WanDistillRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_transformer(self):
        wan_model_kwargs = {"model_path": self.config["model_path"], "config": self.config, "device": self.init_device}
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = WanDistillModel(**wan_model_kwargs)
        else:
            model = build_wan_model_with_lora(WanModel, self.config, wan_model_kwargs, lora_configs, model_type="wan2.1")
        return model

    def init_scheduler(self):
        if self.config["feature_caching"] == "NoCaching":
            self.scheduler = WanStepDistillScheduler(self.config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")


@RUNNER_REGISTER("wan2.1_mean_flow_distill")
class Wan21MeanFlowDistillRunner(WanDistillRunner):
    def __init__(self, config):
        super().__init__(config)

    def init_scheduler(self):
        if self.config["feature_caching"] == "NoCaching":
            self.scheduler = Wan21MeanFlowStepDistillScheduler(self.config)
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
            #  self.scheduler.sample_guide_scale = self.config["sample_guide_scale"][0]
            if self.config.get("cpu_offload", False) and self.config.get("offload_granularity", "block") == "model":
                if self.cur_model_index == -1:
                    self.to_cuda(model_index=0)
                elif self.cur_model_index == 1:  # 1 -> 0
                    self.offload_cpu(model_index=1)
                    self.to_cuda(model_index=0)
            self.cur_model_index = 0
        else:
            logger.info(f"using - LOW - noise model at step_index {self.scheduler.step_index + 1}")
            # self.scheduler.sample_guide_scale = self.config["sample_guide_scale"][1]
            if self.config.get("cpu_offload", False) and self.config.get("offload_granularity", "block") == "model":
                if self.cur_model_index == -1:
                    self.to_cuda(model_index=1)
                elif self.cur_model_index == 0:  # 0 -> 1
                    self.offload_cpu(model_index=0)
                    self.to_cuda(model_index=1)
            self.cur_model_index = 1

    def infer(self, inputs):
        self.get_current_model_index()
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.model[self.cur_model_index].infer(inputs)
        else:
            if self.model[self.cur_model_index] is not None:
                self.model[self.cur_model_index].infer(inputs)
            else:
                if self.cur_model_index == 0:
                    lora_configs = self.config.get("lora_configs")
                    high_model_kwargs = {
                        "model_path": self.high_noise_model_path,
                        "config": self.config,
                        "device": self.init_device,
                        "model_type": "wan2.2_moe_high_noise",
                    }
                    if not lora_configs:
                        high_noise_model = WanDistillModel(**high_model_kwargs)
                    else:
                        assert self.config.get("lora_dynamic_apply", False)
                        high_noise_model = build_wan_model_with_lora(WanModel, self.config, high_model_kwargs, lora_configs, model_type="high_noise_model")
                    high_noise_model.set_scheduler(self.scheduler)
                    self.model[0] = high_noise_model
                    self.model[0].infer(inputs)
                elif self.cur_model_index == 1:
                    lora_configs = self.config.get("lora_configs")
                    low_model_kwargs = {
                        "model_path": self.low_noise_model_path,
                        "config": self.config,
                        "device": self.init_device,
                        "model_type": "wan2.2_moe_low_noise",
                    }
                    if not lora_configs:
                        low_noise_model = WanDistillModel(**low_model_kwargs)
                    else:
                        assert self.config.get("lora_dynamic_apply", False)
                        low_noise_model = build_wan_model_with_lora(WanModel, self.config, low_model_kwargs, lora_configs, model_type="low_noise_model")
                    low_noise_model.set_scheduler(self.scheduler)
                    self.model[1] = low_noise_model
                    self.model[1].infer(inputs)


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
                raise FileNotFoundError(f"High Noise Model does not find")

        if self.config.get("dit_quantized", False) and self.config.get("low_noise_quantized_ckpt", None):
            self.low_noise_model_path = self.config["low_noise_quantized_ckpt"]
        elif not self.config.get("dit_quantized", False) and self.config.get("low_noise_original_ckpt", None):
            self.low_noise_model_path = self.config["low_noise_original_ckpt"]
        else:
            self.low_noise_model_path = os.path.join(self.config["model_path"], "low_noise_model")
            if not os.path.isdir(self.low_noise_model_path):
                raise FileNotFoundError(f"Low Noise Model does not find")

    def load_transformer(self):
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            lora_configs = self.config.get("lora_configs")
            high_model_kwargs = {
                "model_path": self.high_noise_model_path,
                "config": self.config,
                "device": self.init_device,
                "model_type": "wan2.2_moe_high_noise",
            }
            low_model_kwargs = {
                "model_path": self.low_noise_model_path,
                "config": self.config,
                "device": self.init_device,
                "model_type": "wan2.2_moe_low_noise",
            }
            if not lora_configs:
                high_noise_model = WanDistillModel(**high_model_kwargs)
                low_noise_model = WanDistillModel(**low_model_kwargs)
            else:
                high_noise_model = build_wan_model_with_lora(WanModel, self.config, high_model_kwargs, lora_configs, model_type="high_noise_model")
                low_noise_model = build_wan_model_with_lora(WanModel, self.config, low_model_kwargs, lora_configs, model_type="low_noise_model")
            return MultiDistillModelStruct([high_noise_model, low_noise_model], self.config, self.config["boundary_step_index"])
        else:
            model_struct = MultiDistillModelStruct([None, None], self.config, self.config["boundary_step_index"])
            model_struct.low_noise_model_path = self.low_noise_model_path
            model_struct.high_noise_model_path = self.high_noise_model_path
            model_struct.init_device = self.init_device
            return model_struct

    def init_scheduler(self):
        if self.config["feature_caching"] == "NoCaching":
            self.scheduler = Wan22StepDistillScheduler(self.config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")

    def switch_lora(self, high_lora_path: str = None, high_lora_strength: float = 1.0, low_lora_path: str = None, low_lora_strength: float = 1.0):
        """
        Switch LoRA weights dynamically for Wan2.2 MoE Distill models.
        This method handles both high_noise_model and low_noise_model separately.

        Args:
            lora_path: Path to the LoRA safetensors file (for backward compatibility)
            strength: LoRA strength (default: 1.0) (for backward compatibility)
            high_lora_path: Path to the high_noise_model LoRA safetensors file
            high_lora_strength: High noise model LoRA strength (default: 1.0)
            low_lora_path: Path to the low_noise_model LoRA safetensors file
            low_lora_strength: Low noise model LoRA strength (default: 1.0)

        Returns:
            bool: True if LoRA was successfully switched, False otherwise
        """
        if not hasattr(self, "model") or self.model is None:
            logger.error("Model not loaded. Please load model first.")
            return False

        if high_lora_path is not None:
            if self.model.model[0] is not None and hasattr(self.model.model[0], "_update_lora"):
                logger.info(f"Switching high_noise_model LoRA to: {high_lora_path} with strength={high_lora_strength}")
                self.model.model[0]._update_lora(high_lora_path, high_lora_strength)

        if low_lora_path is not None:
            if self.model.model[1] is not None and hasattr(self.model.model[1], "_update_lora"):
                logger.info(f"Switching low_noise_model LoRA to: {low_lora_path} with strength={low_lora_strength}")
                self.model.model[1]._update_lora(low_lora_path, low_lora_strength)

        logger.info("LoRA switched successfully for Wan2.2 MoE Distill models")
        return True
