import torch

from lightx2v.models.networks.wan.lingbot_fast_model import WanLingbotFastModel
from lightx2v.models.runners.wan.autoregressive_runner import WanAutoregressiveRunnerMixin
from lightx2v.models.runners.wan.wan_runner import LingbotRunner, WanRunner, build_wan_model_with_lora
from lightx2v.models.schedulers.wan.self_forcing.scheduler import WanSFScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.input_info import ActionI2VInputInfo
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("lingbot_world_fast")
class LingbotFastRunner(WanAutoregressiveRunnerMixin, LingbotRunner):
    """Share Wan AR scheduling while retaining Lingbot camera input handling."""

    _SUPPORTS_AUTOREGRESSIVE_WARMUP = True
    _WARMUP_TASKS = ("i2v",)
    supported_request_fields_by_task = {
        "i2v": LingbotRunner.supported_request_fields_by_task["i2v"] - {"num_frames"},
    }

    def __init__(self, config):
        WanRunner.__init__(self, config)
        self.control_type = config.get("control_type", "cam")

    def load_transformer(self):
        wan_model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
        }
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = WanLingbotFastModel(**wan_model_kwargs)
        else:
            model = build_wan_model_with_lora(WanLingbotFastModel, self.config, wan_model_kwargs, lora_configs, model_type="wan2.1")
        return model

    def init_scheduler(self):
        self.scheduler = WanSFScheduler(self.config)

    def validate_warmup(self):
        super().validate_warmup()
        if self.control_type != "cam":
            raise NotImplementedError("Lingbot Fast warmup supports camera-conditioned i2v")

    def get_target_latent_size(self, target_height, target_width):
        _, stride_h, stride_w = self.config["vae_stride"]
        _, patch_h, patch_w = self.config.get("patch_size", (1, 2, 2))
        latent_h = target_height // stride_h // patch_h * patch_h
        latent_w = target_width // stride_w // patch_w * patch_w
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            latent_h, latent_w, _, _ = self._adjust_latent_for_grid_splitting(latent_h, latent_w, torch.distributed.get_world_size())
        return latent_h, latent_w

    def get_warmup_latent_shape(self, target_height, target_width):
        return self.get_latent_shape_with_lat_hw(*self.get_target_latent_size(target_height, target_width))

    def run_vae_encoder(self, first_frame, last_frame=None):
        # Use the camera-calibrated output grid for both real requests and warmup.
        latent_h, latent_w = self.get_target_latent_size(*self.get_target_size())
        encoded = self.get_vae_encoder_output(first_frame, latent_h, latent_w, last_frame)
        return encoded, self.get_latent_shape_with_lat_hw(latent_h, latent_w)

    def prepare_warmup_inputs(self, height, width, latent_shape, text_output):
        frame_count = (latent_shape[1] - 1) * self.config["vae_stride"][0] + 1
        self.input_info = ActionI2VInputInfo(task="i2v", prompt="warmup", size=[height, width], num_frames=frame_count)
        self.input_info.latent_shape = latent_shape
        image_output = self.get_warmup_image_encoder_output(latent_shape)
        camera_channels = 6 * self.config["vae_stride"][1] * self.config["vae_stride"][2]
        image_output["dit_cond_dict"] = {"c2ws_plucker_emb": torch.zeros(camera_channels, *latent_shape[1:], device=self.init_device, dtype=GET_DTYPE())}
        self.inputs = {"text_encoder_output": text_output, "image_encoder_output": image_output}
