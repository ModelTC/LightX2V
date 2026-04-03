import torch
from loguru import logger

from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.feature_caching.scheduler import (
    WanSchedulerCaching,
    WanSchedulerTaylorCaching,
)
from lightx2v.models.schedulers.wan.ltx2_bridge_resolution.scheduler import (
    WanScheduler4LTX2BridgeInterface,
)
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.video_encoders.hf.ltx2.model import LTX2Upsampler, LTX2VideoVAE
from lightx2v.models.video_encoders.hf.wan.upsampler.ltx2_pixel_bridge import (
    LTX2PixelBridgeResizer,
)
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("wan2.1_ltx2_bridge")
class WanLTX2BridgeRunner(WanRunner):
    """独立的 WAN + LTX2 upsample 实验链路。"""

    def __init__(self, config):
        super().__init__(config)
        self._validate_bridge_config()

    def _validate_bridge_config(self):
        if self.config["task"] != "t2v":
            raise NotImplementedError("wan2.1_ltx2_bridge currently only supports t2v.")
        if self.config.get("use_tae", False):
            raise ValueError("wan2.1_ltx2_bridge requires the full WAN VAE encoder/decoder, not TAE.")
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            raise ValueError("wan2.1_ltx2_bridge does not support lazy_load/unload_modules yet.")

        resolution_rate = list(self.config.get("resolution_rate", []))
        if resolution_rate != [0.5]:
            raise ValueError(
                "wan2.1_ltx2_bridge expects a single lowres->fullres stage with resolution_rate=[0.5]. "
                f"Got {resolution_rate}"
            )
        if len(self.config.get("changing_resolution_steps", [])) != 1:
            raise ValueError("wan2.1_ltx2_bridge expects exactly one changing_resolution step.")
        if self.config["target_height"] % 64 != 0 or self.config["target_width"] % 64 != 0:
            raise ValueError(
                "wan2.1_ltx2_bridge expects final target height/width to be divisible by 64 "
                "so the 0.5 stage stays aligned with LTX2's 32x spatial grid."
            )
        if self.config.get("ltx2_vae_ckpt") is None or self.config.get("ltx2_upsampler_ckpt") is None:
            raise ValueError("wan2.1_ltx2_bridge requires both ltx2_vae_ckpt and ltx2_upsampler_ckpt in config.")

    def init_scheduler(self):
        if self.config["feature_caching"] == "NoCaching":
            scheduler_class = WanScheduler
        elif self.config["feature_caching"] == "TaylorSeer":
            scheduler_class = WanSchedulerTaylorCaching
        elif self.config.feature_caching in ["Tea", "Ada", "Custom", "FirstBlock", "DualBlock", "DynamicBlock", "Mag"]:
            scheduler_class = WanSchedulerCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")

        self.scheduler = WanScheduler4LTX2BridgeInterface(scheduler_class, self.config)

    def load_vae_encoder(self):
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        vae_device = torch.device("cpu") if vae_offload else torch.device(AI_DEVICE)
        vae_config = {
            "vae_path": self.config["vae_path"] if self.config.get("vae_path") is not None else None,
            "device": vae_device,
            "parallel": self.get_vae_parallel(),
            "use_tiling": self.config.get("use_tiling_vae", False),
            "cpu_offload": vae_offload,
            "dtype": GET_DTYPE(),
            "load_from_rank0": self.config.get("load_from_rank0", False),
            "use_lightvae": self.config.get("use_lightvae", False),
        }
        if vae_config["vae_path"] is None:
            from lightx2v.utils.utils import find_torch_model_path

            vae_config["vae_path"] = find_torch_model_path(self.config, "vae_path", self.vae_name)
        return self.vae_cls(**vae_config)

    def load_ltx2_video_vae(self):
        vae_offload = self.config.get("ltx2_cpu_offload", self.config.get("cpu_offload", False))
        vae_device = torch.device("cpu") if vae_offload else torch.device(AI_DEVICE)
        return LTX2VideoVAE(
            checkpoint_path=self.config["ltx2_vae_ckpt"],
            device=vae_device,
            dtype=GET_DTYPE(),
            load_encoder=True,
            use_tiling=False,
            cpu_offload=vae_offload,
        )

    def load_ltx2_upsampler(self):
        upsampler_offload = self.config.get("ltx2_cpu_offload", self.config.get("cpu_offload", False))
        upsampler = LTX2Upsampler(
            checkpoint_path=self.config["ltx2_upsampler_ckpt"],
            device=self.init_device,
            dtype=GET_DTYPE(),
            cpu_offload=upsampler_offload,
        )
        return upsampler

    def load_model(self):
        super().load_model()
        self.ltx2_video_vae = self.load_ltx2_video_vae()
        self.ltx2_upsampler = self.load_ltx2_upsampler()
        self.clean_latent_resizer = LTX2PixelBridgeResizer(
            wan_vae_encoder=self.vae_encoder,
            wan_vae_decoder=self.vae_decoder,
            ltx2_video_vae=self.ltx2_video_vae,
            ltx2_upsampler=self.ltx2_upsampler,
            config=self.config,
        )
        self.scheduler.set_clean_latent_resizer(self.clean_latent_resizer)
        logger.info("Initialized WAN + LTX2 bridge resizer.")

    def init_run(self):
        super().init_run()
        if hasattr(self.scheduler, "set_clean_latent_resizer"):
            self.scheduler.set_clean_latent_resizer(self.clean_latent_resizer)
