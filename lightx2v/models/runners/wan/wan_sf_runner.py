from lightx2v.models.networks.wan.sf_model import WanSFModel
from lightx2v.models.runners.wan.autoregressive_runner import WanAutoregressiveRunnerMixin
from lightx2v.models.runners.wan.wan_runner import WanRunner, build_wan_model_with_lora
from lightx2v.models.schedulers.wan.self_forcing.scheduler import WanSFScheduler
from lightx2v.utils.input_info import T2VInputInfo
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("wan2.1_sf")
class WanSFRunner(WanAutoregressiveRunnerMixin, WanRunner):
    _SUPPORTS_AUTOREGRESSIVE_WARMUP = True
    _WARMUP_TASKS = ("t2v",)

    supported_request_fields_by_task = {
        "t2v": WanRunner.supported_request_fields_by_task["t2v"] - {"num_frames"},
    }

    def __init__(self, config):
        super().__init__(config)

    def load_transformer(self):
        wan_model_kwargs = {"model_path": self.config["model_path"], "config": self.config, "device": self.init_device}
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = WanSFModel(**wan_model_kwargs)
        else:
            model = build_wan_model_with_lora(WanSFModel, self.config, wan_model_kwargs, lora_configs, model_type="wan2.1")
        return model

    def init_scheduler(self):
        self.scheduler = WanSFScheduler(self.config)

    def prepare_warmup_inputs(self, height, width, latent_shape, text_output):
        self.input_info = T2VInputInfo(task="t2v", prompt="warmup", size=[height, width])
        self.input_info.latent_shape = latent_shape
        self.inputs = {"text_encoder_output": text_output, "image_encoder_output": None}
