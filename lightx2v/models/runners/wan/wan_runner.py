import gc
import os

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from loguru import logger

from lightx2v.models.input_encoders.hf.wan.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.wan.xlm_roberta.model import CLIPModel
from lightx2v.models.networks.lora_adapter import LoraAdapter
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.wan.changing_resolution.scheduler import (
    WanScheduler4ChangingResolutionInterface,
)
from lightx2v.models.schedulers.wan.feature_caching.scheduler import (
    WanSchedulerCaching,
    WanSchedulerTaylorCaching,
)
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE
from lightx2v.models.video_encoders.hf.wan.vae_tiny import Wan2_2_VAE_tiny, WanVAE_tiny
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import *
from lightx2v_platform.base.global_var import AI_DEVICE


def build_wan_model_with_lora(wan_module, config, model_kwargs, lora_configs, model_type="high_noise_model"):
    lora_dynamic_apply = config.get("lora_dynamic_apply", False)

    if lora_dynamic_apply:
        if model_type in ["high_noise_model", "low_noise_model"]:
            # For wan2.2
            lora_name_to_info = {item["name"]: item for item in lora_configs}
            lora_path = lora_name_to_info[model_type]["path"]
            lora_strength = lora_name_to_info[model_type]["strength"]
        else:
            # For wan2.1
            lora_path = lora_configs[0]["path"]
            lora_strength = lora_configs[0]["strength"]

        model_kwargs["lora_path"] = lora_path
        model_kwargs["lora_strength"] = lora_strength
        model = wan_module(**model_kwargs)
    else:
        assert not config.get("dit_quantized", False), "Online LoRA only for quantized models; merging LoRA is unsupported."
        assert not config.get("lazy_load", False), "Lazy load mode does not support LoRA merging."
        model = wan_module(**model_kwargs)
        lora_adapter = LoraAdapter(model)
        if model_type in ["high_noise_model", "low_noise_model"]:
            lora_configs = [lora_config for lora_config in lora_configs if lora_config["name"] == model_type]
        lora_adapter.apply_lora(lora_configs, model_type=model_type)
    return model


@RUNNER_REGISTER("wan2.1")
class WanRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)
        self.vae_cls = WanVAE
        self.tiny_vae_cls = WanVAE_tiny
        self.vae_name = config.get("vae_name", "Wan2.1_VAE.pth")
        self.tiny_vae_name = "taew2_1.pth"

    def load_transformer(self):
        wan_model_kwargs = {"model_path": self.config["model_path"], "config": self.config, "device": self.init_device}
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = WanModel(**wan_model_kwargs)
        else:
            model = build_wan_model_with_lora(WanModel, self.config, wan_model_kwargs, lora_configs, model_type="wan2.1")
        return model

    def load_image_encoder(self):
        image_encoder = None
        if self.config["task"] in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
            # offload config
            clip_offload = self.config.get("clip_cpu_offload", self.config.get("cpu_offload", False))
            if clip_offload:
                clip_device = torch.device("cpu")
            else:
                clip_device = torch.device(AI_DEVICE)
            # quant_config
            clip_quantized = self.config.get("clip_quantized", False)
            if clip_quantized:
                clip_quant_scheme = self.config.get("clip_quant_scheme", None)
                assert clip_quant_scheme is not None
                tmp_clip_quant_scheme = clip_quant_scheme.split("-")[0]
                clip_model_name = f"models_clip_open-clip-xlm-roberta-large-vit-huge-14-{tmp_clip_quant_scheme}.pth"
                clip_quantized_ckpt = find_torch_model_path(self.config, "clip_quantized_ckpt", clip_model_name)
                clip_original_ckpt = None
            else:
                clip_quantized_ckpt = None
                clip_quant_scheme = None
                clip_model_name = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
                clip_original_ckpt = find_torch_model_path(self.config, "clip_original_ckpt", clip_model_name)

            image_encoder = CLIPModel(
                dtype=torch.float16,
                device=clip_device,
                checkpoint_path=clip_original_ckpt,
                clip_quantized=clip_quantized,
                clip_quantized_ckpt=clip_quantized_ckpt,
                quant_scheme=clip_quant_scheme,
                cpu_offload=clip_offload,
                use_31_block=self.config.get("use_31_block", True),
                load_from_rank0=self.config.get("load_from_rank0", False),
            )

        return image_encoder

    def load_text_encoder(self):
        # offload config
        t5_offload = self.config.get("t5_cpu_offload", self.config.get("cpu_offload"))
        if t5_offload:
            t5_device = torch.device("cpu")
        else:
            t5_device = torch.device(AI_DEVICE)
        tokenizer_path = os.path.join(self.config["model_path"], "google/umt5-xxl")
        # quant_config
        t5_quantized = self.config.get("t5_quantized", False)
        if t5_quantized:
            t5_quant_scheme = self.config.get("t5_quant_scheme", None)
            assert t5_quant_scheme is not None
            tmp_t5_quant_scheme = t5_quant_scheme.split("-")[0]
            t5_model_name = f"models_t5_umt5-xxl-enc-{tmp_t5_quant_scheme}.pth"
            t5_quantized_ckpt = find_torch_model_path(self.config, "t5_quantized_ckpt", t5_model_name)
            t5_original_ckpt = None
        else:
            t5_quant_scheme = None
            t5_quantized_ckpt = None
            t5_model_name = "models_t5_umt5-xxl-enc-bf16.pth"
            t5_original_ckpt = find_torch_model_path(self.config, "t5_original_ckpt", t5_model_name)

        text_encoder = T5EncoderModel(
            text_len=self.config["text_len"],
            dtype=torch.bfloat16,
            device=t5_device,
            checkpoint_path=t5_original_ckpt,
            tokenizer_path=tokenizer_path,
            shard_fn=None,
            cpu_offload=t5_offload,
            t5_quantized=t5_quantized,
            t5_quantized_ckpt=t5_quantized_ckpt,
            quant_scheme=t5_quant_scheme,
            load_from_rank0=self.config.get("load_from_rank0", False),
            lazy_load=self.config.get("t5_lazy_load", False),
        )
        text_encoders = [text_encoder]
        return text_encoders

    def get_vae_parallel(self):
        if isinstance(self.config.get("parallel", False), bool):
            return self.config.get("parallel", False)
        if isinstance(self.config.get("parallel", False), dict):
            return self.config.get("parallel", {}).get("vae_parallel", True)
        return False

    def load_vae_encoder(self):
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device(AI_DEVICE)

        vae_config = {
            "vae_path": find_torch_model_path(self.config, "vae_path", self.vae_name),
            "device": vae_device,
            "parallel": self.get_vae_parallel(),
            "use_tiling": self.config.get("use_tiling_vae", False),
            "cpu_offload": vae_offload,
            "dtype": GET_DTYPE(),
            "load_from_rank0": self.config.get("load_from_rank0", False),
            "use_lightvae": self.config.get("use_lightvae", False),
        }
        if self.config["task"] not in ["i2v", "flf2v", "animate", "vace", "s2v", "rs2v"]:
            return None
        else:
            return self.vae_cls(**vae_config)

    def load_vae_decoder(self):
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device(AI_DEVICE)

        vae_config = {
            "vae_path": find_torch_model_path(self.config, "vae_path", self.vae_name),
            "device": vae_device,
            "parallel": self.get_vae_parallel(),
            "use_tiling": self.config.get("use_tiling_vae", False),
            "cpu_offload": vae_offload,
            "use_lightvae": self.config.get("use_lightvae", False),
            "dtype": GET_DTYPE(),
            "load_from_rank0": self.config.get("load_from_rank0", False),
        }
        if self.config.get("use_tae", False):
            tae_path = find_torch_model_path(self.config, "tae_path", self.tiny_vae_name)
            vae_decoder = self.tiny_vae_cls(vae_path=tae_path, device=self.init_device, need_scaled=self.config.get("need_scaled", False)).to(AI_DEVICE)
        else:
            vae_decoder = self.vae_cls(**vae_config)
        return vae_decoder

    def load_vae(self):
        vae_encoder = self.load_vae_encoder()
        if vae_encoder is None or self.config.get("use_tae", False):
            vae_decoder = self.load_vae_decoder()
        else:
            vae_decoder = vae_encoder
        return vae_encoder, vae_decoder

    def init_scheduler(self):
        if self.config["feature_caching"] == "NoCaching":
            scheduler_class = WanScheduler
        elif self.config["feature_caching"] == "TaylorSeer":
            scheduler_class = WanSchedulerTaylorCaching
        elif self.config.feature_caching in ["Tea", "Ada", "Custom", "FirstBlock", "DualBlock", "DynamicBlock", "Mag"]:
            scheduler_class = WanSchedulerCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")

        if self.config.get("changing_resolution", False):
            self.scheduler = WanScheduler4ChangingResolutionInterface(scheduler_class, self.config)
        else:
            self.scheduler = scheduler_class(self.config)

    @ProfilingContext4DebugL1(
        "Run Text Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_text_encode_duration,
        metrics_labels=["WanRunner"],
    )
    def run_text_encoder(self, input_info):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()

        prompt = input_info.prompt_enhanced if self.config["use_prompt_enhancer"] else input_info.prompt
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_prompt_len.observe(len(prompt))
        neg_prompt = input_info.negative_prompt

        if self.config.get("enable_cfg", False) and self.config["cfg_parallel"]:
            cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
            cfg_p_rank = dist.get_rank(cfg_p_group)
            if cfg_p_rank == 0:
                context = self.text_encoders[0].infer([prompt])
                context = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context])
                text_encoder_output = {"context": context}
            else:
                context_null = self.text_encoders[0].infer([neg_prompt])
                context_null = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context_null])
                text_encoder_output = {"context_null": context_null}
        else:
            context = self.text_encoders[0].infer([prompt])
            context = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context])
            if self.config.get("enable_cfg", False):
                context_null = self.text_encoders[0].infer([neg_prompt])
                context_null = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context_null])
            else:
                context_null = None
            text_encoder_output = {
                "context": context,
                "context_null": context_null,
            }

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]
            torch_device_module.empty_cache()
            gc.collect()

        return text_encoder_output

    @ProfilingContext4DebugL1(
        "Run Image Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_img_encode_duration,
        metrics_labels=["WanRunner"],
    )
    def run_image_encoder(self, first_frame, last_frame=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.image_encoder = self.load_image_encoder()
        if last_frame is None:
            clip_encoder_out = self.image_encoder.visual([first_frame]).squeeze(0).to(GET_DTYPE())
        else:
            clip_encoder_out = self.image_encoder.visual([first_frame, last_frame]).squeeze(0).to(GET_DTYPE())
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.image_encoder
            torch_device_module.empty_cache()
            gc.collect()
        return clip_encoder_out

    def _adjust_latent_for_grid_splitting(self, latent_h, latent_w, world_size):
        """
        Adjust latent dimensions for optimal 2D grid splitting.
        Prefers balanced grids like 2x4 or 4x2 over 1x8 or 8x1.
        """
        world_size_h, world_size_w = 1, 1
        if world_size <= 1:
            return latent_h, latent_w, world_size_h, world_size_w

        # Define priority grids for different world sizes
        priority_grids = []
        if world_size == 8:
            # For 8 cards, prefer 2x4 and 4x2 over 1x8 and 8x1
            priority_grids = [(2, 4), (4, 2), (1, 8), (8, 1)]
        elif world_size == 4:
            priority_grids = [(2, 2), (1, 4), (4, 1)]
        elif world_size == 2:
            priority_grids = [(1, 2), (2, 1)]
        else:
            # For other sizes, try factor pairs
            for h in range(1, int(np.sqrt(world_size)) + 1):
                if world_size % h == 0:
                    w = world_size // h
                    priority_grids.append((h, w))

        # Try priority grids first
        for world_size_h, world_size_w in priority_grids:
            if latent_h % world_size_h == 0 and latent_w % world_size_w == 0:
                return latent_h, latent_w, world_size_h, world_size_w

        # If no perfect fit, find minimal padding solution
        best_grid = (1, world_size)  # fallback
        min_total_padding = float("inf")

        for world_size_h, world_size_w in priority_grids:
            # Calculate required padding
            pad_h = (world_size_h - (latent_h % world_size_h)) % world_size_h
            pad_w = (world_size_w - (latent_w % world_size_w)) % world_size_w
            total_padding = pad_h + pad_w

            # Prefer grids with minimal total padding
            if total_padding < min_total_padding:
                min_total_padding = total_padding
                best_grid = (world_size_h, world_size_w)

        # Apply padding
        world_size_h, world_size_w = best_grid
        pad_h = (world_size_h - (latent_h % world_size_h)) % world_size_h
        pad_w = (world_size_w - (latent_w % world_size_w)) % world_size_w

        return latent_h + pad_h, latent_w + pad_w, world_size_h, world_size_w

    @ProfilingContext4DebugL1(
        "Run VAE Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration,
        metrics_labels=["WanRunner"],
    )
    def run_vae_encoder(self, first_frame, last_frame=None):
        if self.config.get("resize_mode", None) is None:
            h, w = first_frame.shape[2:]
            aspect_ratio = h / w
            max_area = self.config["target_height"] * self.config["target_width"]

            # Calculate initial latent dimensions
            ori_latent_h = round(np.sqrt(max_area * aspect_ratio) // self.config["vae_stride"][1] // self.config["patch_size"][1] * self.config["patch_size"][1])
            ori_latent_w = round(np.sqrt(max_area / aspect_ratio) // self.config["vae_stride"][2] // self.config["patch_size"][2] * self.config["patch_size"][2])

            # Adjust latent dimensions for optimal 2D grid splitting when using distributed processing
            if dist.is_initialized() and dist.get_world_size() > 1:
                latent_h, latent_w, world_size_h, world_size_w = self._adjust_latent_for_grid_splitting(ori_latent_h, ori_latent_w, dist.get_world_size())
                logger.info(f"ori latent: {ori_latent_h}x{ori_latent_w}, adjust_latent: {latent_h}x{latent_w}, grid: {world_size_h}x{world_size_w}")
            else:
                latent_h, latent_w = ori_latent_h, ori_latent_w
                world_size_h, world_size_w = None, None

            latent_shape = self.get_latent_shape_with_lat_hw(latent_h, latent_w)  # Important: latent_shape is used to set the input_info
        else:
            latent_shape = self.input_info.latent_shape
            latent_h, latent_w = self.input_info.latent_shape[-2], self.input_info.latent_shape[-1]
            world_size_h, world_size_w = None, None

        if self.config.get("changing_resolution", False):
            assert last_frame is None
            vae_encode_out_list = []
            for i in range(len(self.config["resolution_rate"])):
                latent_h_tmp, latent_w_tmp = (
                    int(latent_h * self.config["resolution_rate"][i]) // 2 * 2,
                    int(latent_w * self.config["resolution_rate"][i]) // 2 * 2,
                )
                vae_encode_out_list.append(self.get_vae_encoder_output(first_frame, latent_h_tmp, latent_w_tmp, world_size_h=world_size_h, world_size_w=world_size_w))
            vae_encode_out_list.append(self.get_vae_encoder_output(first_frame, latent_h, latent_w, world_size_h=world_size_h, world_size_w=world_size_w))
            return vae_encode_out_list, latent_shape
        else:
            if last_frame is not None:
                first_frame_size = first_frame.shape[2:]
                last_frame_size = last_frame.shape[2:]
                if first_frame_size != last_frame_size:
                    last_frame_resize_ratio = max(first_frame_size[0] / last_frame_size[0], first_frame_size[1] / last_frame_size[1])
                    last_frame_size = [
                        round(last_frame_size[0] * last_frame_resize_ratio),
                        round(last_frame_size[1] * last_frame_resize_ratio),
                    ]
                    last_frame = TF.center_crop(last_frame, last_frame_size)
            vae_encoder_out = self.get_vae_encoder_output(first_frame, latent_h, latent_w, last_frame, world_size_h=world_size_h, world_size_w=world_size_w)
            return vae_encoder_out, latent_shape

    def get_vae_encoder_output(self, first_frame, lat_h, lat_w, last_frame=None, world_size_h=None, world_size_w=None):
        h = lat_h * self.config["vae_stride"][1]
        w = lat_w * self.config["vae_stride"][2]
        msk = torch.ones(
            1,
            self.config["target_video_length"],
            lat_h,
            lat_w,
            device=torch.device(AI_DEVICE),
        )
        if last_frame is not None:
            msk[:, 1:-1] = 0
        else:
            msk[:, 1:] = 0

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder = self.load_vae_encoder()

        if last_frame is not None:
            vae_input = torch.concat(
                [
                    torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, self.config["target_video_length"] - 2, h, w),
                    torch.nn.functional.interpolate(last_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                ],
                dim=1,
            ).to(AI_DEVICE)
        else:
            vae_input = torch.concat(
                [
                    torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, self.config["target_video_length"] - 1, h, w),
                ],
                dim=1,
            ).to(AI_DEVICE)

        vae_encoder_out = self.vae_encoder.encode(vae_input.unsqueeze(0).to(GET_DTYPE()), world_size_h=world_size_h, world_size_w=world_size_w)

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_encoder
            torch_device_module.empty_cache()
            gc.collect()
        vae_encoder_out = torch.concat([msk, vae_encoder_out]).to(GET_DTYPE())
        return vae_encoder_out

    def get_encoder_output_i2v(self, clip_encoder_out, vae_encoder_out, text_encoder_output, img=None):
        image_encoder_output = {
            "clip_encoder_out": clip_encoder_out,
            "vae_encoder_out": vae_encoder_out,
        }
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }

    def get_latent_shape_with_lat_hw(self, latent_h, latent_w):
        latent_shape = [
            self.config.get("num_channels_latents", 16),
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            latent_h,
            latent_w,
        ]
        return latent_shape

    def get_latent_shape_with_target_hw(self):
        target_height = self.input_info.target_shape[0] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_height"]
        target_width = self.input_info.target_shape[1] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_width"]

        latent_shape = [
            self.config.get("num_channels_latents", 16),
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            int(target_height) // self.config["vae_stride"][1],
            int(target_width) // self.config["vae_stride"][2],
        ]
        return latent_shape


class MultiModelStruct:
    def __init__(self, model_list, config, boundary=0.875, num_train_timesteps=1000):
        self.model = model_list  # [high_noise_model, low_noise_model]
        assert len(self.model) == 2, "MultiModelStruct only supports 2 models now."
        self.config = config
        self.boundary = boundary
        self.boundary_timestep = self.boundary * num_train_timesteps
        self.cur_model_index = -1
        logger.info(f"boundary: {self.boundary}, boundary_timestep: {self.boundary_timestep}")

    @property
    def device(self):
        return self.model[self.cur_model_index].device

    def set_scheduler(self, shared_scheduler):
        self.scheduler = shared_scheduler
        for model in self.model:
            if model is not None:
                model.set_scheduler(shared_scheduler)

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
                        high_noise_model = WanModel(**high_model_kwargs)
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
                        low_noise_model = WanModel(**low_model_kwargs)
                    else:
                        assert self.config.get("lora_dynamic_apply", False)
                        low_noise_model = build_wan_model_with_lora(WanModel, self.config, low_model_kwargs, lora_configs, model_type="low_noise_model")
                    low_noise_model.set_scheduler(self.scheduler)
                    self.model[1] = low_noise_model
                    self.model[1].infer(inputs)

    @ProfilingContext4DebugL2("Swtich models in infer_main costs")
    def get_current_model_index(self):
        if self.scheduler.timesteps[self.scheduler.step_index] >= self.boundary_timestep:
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

    def offload_cpu(self, model_index):
        self.model[model_index].to_cpu()

    def to_cuda(self, model_index):
        self.model[model_index].to_cuda()


@RUNNER_REGISTER("wan2.2_moe")
class Wan22MoeRunner(WanRunner):
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
        # encoder -> high_noise_model -> low_noise_model -> vae -> video_output
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
                high_noise_model = WanModel(**high_model_kwargs)
                low_noise_model = WanModel(**low_model_kwargs)
            else:
                high_noise_model = build_wan_model_with_lora(WanModel, self.config, high_model_kwargs, lora_configs, model_type="high_noise_model")
                low_noise_model = build_wan_model_with_lora(WanModel, self.config, low_model_kwargs, lora_configs, model_type="low_noise_model")

            return MultiModelStruct([high_noise_model, low_noise_model], self.config, self.config["boundary"])
        else:
            model_struct = MultiModelStruct([None, None], self.config, self.config["boundary"])
            model_struct.low_noise_model_path = self.low_noise_model_path
            model_struct.high_noise_model_path = self.high_noise_model_path
            model_struct.init_device = self.init_device
            return model_struct

    def switch_lora(self, high_lora_path: str = None, high_lora_strength: float = 1.0, low_lora_path: str = None, low_lora_strength: float = 1.0):
        """
        Switch LoRA weights dynamically for Wan2.2 MoE models.
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

        logger.info("LoRA switched successfully for Wan2.2 MoE models")
        return True


@RUNNER_REGISTER("wan2.2")
class Wan22DenseRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)
        self.vae_encoder_need_img_original = True
        self.vae_cls = Wan2_2_VAE
        self.tiny_vae_cls = Wan2_2_VAE_tiny
        self.vae_name = "Wan2.2_VAE.pth"
        self.tiny_vae_name = "taew2_2.pth"

    @ProfilingContext4DebugL1(
        "Run VAE Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration,
        metrics_labels=["Wan22DenseRunner"],
    )
    def run_vae_encoder(self, img):
        max_area = self.config.target_height * self.config.target_width
        ih, iw = img.height, img.width
        dh, dw = self.config.patch_size[1] * self.config.vae_stride[1], self.config.patch_size[2] * self.config.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(AI_DEVICE).unsqueeze(1)
        vae_encoder_out = self.get_vae_encoder_output(img)
        latent_w, latent_h = ow // self.config["vae_stride"][2], oh // self.config["vae_stride"][1]
        latent_shape = self.get_latent_shape_with_lat_hw(latent_h, latent_w)
        return vae_encoder_out, latent_shape

    def get_vae_encoder_output(self, img):
        z = self.vae_encoder.encode(img.unsqueeze(0).to(GET_DTYPE()))
        return z
