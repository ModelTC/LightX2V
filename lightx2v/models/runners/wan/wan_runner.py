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
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
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


@RUNNER_REGISTER("wan2.1")
class WanRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)
        self.vae_cls = WanVAE
        self.tiny_vae_cls = WanVAE_tiny
        self.vae_name = config.get("vae_name", "Wan2.1_VAE.pth")
        self.tiny_vae_name = "taew2_1.pth"

    def load_transformer(self):
        model = WanModel(
            self.config["model_path"],
            self.config,
            self.init_device,
        )
        if self.config.get("lora_configs") and self.config.lora_configs:
            assert not self.config.get("dit_quantized", False)
            lora_wrapper = WanLoraWrapper(model)
            for lora_config in self.config.lora_configs:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, strength)
                logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")
        return model

    def load_image_encoder(self):
        image_encoder = None
        if self.config["task"] in ["i2v", "flf2v", "animate", "s2v"] and self.config.get("use_image_encoder", True):
            # offload config
            clip_offload = self.config.get("clip_cpu_offload", self.config.get("cpu_offload", False))
            if clip_offload:
                clip_device = torch.device("cpu")
            else:
                clip_device = torch.device("cuda")
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
            t5_device = torch.device("cuda")

        # quant_config
        t5_quantized = self.config.get("t5_quantized", False)
        if t5_quantized:
            t5_quant_scheme = self.config.get("t5_quant_scheme", None)
            assert t5_quant_scheme is not None
            tmp_t5_quant_scheme = t5_quant_scheme.split("-")[0]
            t5_model_name = f"models_t5_umt5-xxl-enc-{tmp_t5_quant_scheme}.pth"
            t5_quantized_ckpt = find_torch_model_path(self.config, "t5_quantized_ckpt", t5_model_name)
            t5_original_ckpt = None
            tokenizer_path = os.path.join(os.path.dirname(t5_quantized_ckpt), "google/umt5-xxl")
        else:
            t5_quant_scheme = None
            t5_quantized_ckpt = None
            t5_model_name = "models_t5_umt5-xxl-enc-bf16.pth"
            t5_original_ckpt = find_torch_model_path(self.config, "t5_original_ckpt", t5_model_name)
            tokenizer_path = os.path.join(os.path.dirname(t5_original_ckpt), "google/umt5-xxl")

        text_encoder = T5EncoderModel(
            text_len=self.config["text_len"],
            dtype=torch.bfloat16,
            device=t5_device,
            checkpoint_path=t5_original_ckpt,
            tokenizer_path=tokenizer_path,
            shard_fn=None,
            cpu_offload=t5_offload,
            offload_granularity=self.config.get("t5_offload_granularity", "model"),  # support ['model', 'block']
            t5_quantized=t5_quantized,
            t5_quantized_ckpt=t5_quantized_ckpt,
            quant_scheme=t5_quant_scheme,
            load_from_rank0=self.config.get("load_from_rank0", False),
        )
        text_encoders = [text_encoder]
        return text_encoders

    def load_vae_encoder(self):
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device("cuda")

        vae_config = {
            "vae_path": find_torch_model_path(self.config, "vae_path", self.vae_name),
            "device": vae_device,
            "parallel": self.config["parallel"],
            "use_tiling": self.config.get("use_tiling_vae", False),
            "cpu_offload": vae_offload,
            "dtype": GET_DTYPE(),
            "load_from_rank0": self.config.get("load_from_rank0", False),
            "use_lightvae": self.config.get("use_lightvae", False),
        }
        if self.config["task"] not in ["i2v", "flf2v", "animate", "vace", "s2v"]:
            return None
        else:
            return self.vae_cls(**vae_config)

    def load_vae_decoder(self):
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device("cuda")

        vae_config = {
            "vae_path": find_torch_model_path(self.config, "vae_path", self.vae_name),
            "device": vae_device,
            "parallel": self.config["parallel"],
            "use_tiling": self.config.get("use_tiling_vae", False),
            "cpu_offload": vae_offload,
            "use_lightvae": self.config.get("use_lightvae", False),
            "dtype": GET_DTYPE(),
            "load_from_rank0": self.config.get("load_from_rank0", False),
        }
        if self.config.get("use_tae", False):
            tae_path = find_torch_model_path(self.config, "tae_path", self.tiny_vae_name)
            vae_decoder = self.tiny_vae_cls(vae_path=tae_path, device=self.init_device, need_scaled=self.config.get("need_scaled", False)).to("cuda")
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

        if self.config["cfg_parallel"]:
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
            context_null = self.text_encoders[0].infer([neg_prompt])
            context_null = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context_null])
            text_encoder_output = {
                "context": context,
                "context_null": context_null,
            }

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]
            torch.cuda.empty_cache()
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
            torch.cuda.empty_cache()
            gc.collect()
        return clip_encoder_out

    @ProfilingContext4DebugL1(
        "Run VAE Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration,
        metrics_labels=["WanRunner"],
    )
    def run_vae_encoder(self, first_frame, last_frame=None):
        h, w = first_frame.shape[2:]
        aspect_ratio = h / w
        max_area = self.config["target_height"] * self.config["target_width"]
        latent_h = round(np.sqrt(max_area * aspect_ratio) // self.config["vae_stride"][1] // self.config["patch_size"][1] * self.config["patch_size"][1])
        latent_w = round(np.sqrt(max_area / aspect_ratio) // self.config["vae_stride"][2] // self.config["patch_size"][2] * self.config["patch_size"][2])
        latent_shape = self.get_latent_shape_with_lat_hw(latent_h, latent_w)  # Important: latent_shape is used to set the input_info

        if self.config.get("changing_resolution", False):
            assert last_frame is None
            vae_encode_out_list = []
            for i in range(len(self.config["resolution_rate"])):
                latent_h_tmp, latent_w_tmp = (
                    int(latent_h * self.config["resolution_rate"][i]) // 2 * 2,
                    int(latent_w * self.config["resolution_rate"][i]) // 2 * 2,
                )
                vae_encode_out_list.append(self.get_vae_encoder_output(first_frame, latent_h_tmp, latent_w_tmp))
            vae_encode_out_list.append(self.get_vae_encoder_output(first_frame, latent_h, latent_w))
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
            vae_encoder_out = self.get_vae_encoder_output(first_frame, latent_h, latent_w, last_frame)
            return vae_encoder_out, latent_shape

    def get_vae_encoder_output(self, first_frame, lat_h, lat_w, last_frame=None):
        h = lat_h * self.config["vae_stride"][1]
        w = lat_w * self.config["vae_stride"][2]
        msk = torch.ones(
            1,
            self.config["target_video_length"],
            lat_h,
            lat_w,
            device=torch.device("cuda"),
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
            ).cuda()
        else:
            vae_input = torch.concat(
                [
                    torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, self.config["target_video_length"] - 1, h, w),
                ],
                dim=1,
            ).cuda()

        vae_encoder_out = self.vae_encoder.encode(vae_input.unsqueeze(0).to(GET_DTYPE()))

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_encoder
            torch.cuda.empty_cache()
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

    def get_latent_shape_with_target_hw(self, target_h, target_w):
        latent_shape = [
            self.config.get("num_channels_latents", 16),
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            int(target_h) // self.config["vae_stride"][1],
            int(target_w) // self.config["vae_stride"][2],
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
            model.set_scheduler(shared_scheduler)

    def infer(self, inputs):
        self.get_current_model_index()
        self.model[self.cur_model_index].infer(inputs)

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
        self.high_noise_model_path = os.path.join(self.config["model_path"], "high_noise_model")
        if not os.path.isdir(self.high_noise_model_path):
            self.high_noise_model_path = os.path.join(self.config["model_path"], "distill_models", "high_noise_model")
        if self.config.get("dit_quantized", False) and self.config.get("high_noise_quantized_ckpt", None):
            self.high_noise_model_path = self.config["high_noise_quantized_ckpt"]
        elif self.config.get("high_noise_original_ckpt", None):
            self.high_noise_model_path = self.config["high_noise_original_ckpt"]

        self.low_noise_model_path = os.path.join(self.config["model_path"], "low_noise_model")
        if not os.path.isdir(self.low_noise_model_path):
            self.low_noise_model_path = os.path.join(self.config["model_path"], "distill_models", "low_noise_model")
        if self.config.get("dit_quantized", False) and self.config.get("low_noise_quantized_ckpt", None):
            self.low_noise_model_path = self.config["low_noise_quantized_ckpt"]
        elif not self.config.get("dit_quantized", False) and self.config.get("low_noise_original_ckpt", None):
            self.low_noise_model_path = self.config["low_noise_original_ckpt"]

    def load_transformer(self):
        # encoder -> high_noise_model -> low_noise_model -> vae -> video_output
        high_noise_model = WanModel(
            self.high_noise_model_path,
            self.config,
            self.init_device,
            model_type="wan2.2_moe_high_noise",
        )
        low_noise_model = WanModel(
            self.low_noise_model_path,
            self.config,
            self.init_device,
            model_type="wan2.2_moe_low_noise",
        )

        if self.config.get("lora_configs") and self.config["lora_configs"]:
            assert not self.config.get("dit_quantized", False)

            for lora_config in self.config["lora_configs"]:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                base_name = os.path.basename(lora_path)
                if base_name.startswith("high"):
                    lora_wrapper = WanLoraWrapper(high_noise_model)
                    lora_name = lora_wrapper.load_lora(lora_path)
                    lora_wrapper.apply_lora(lora_name, strength)
                    logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")
                elif base_name.startswith("low"):
                    lora_wrapper = WanLoraWrapper(low_noise_model)
                    lora_name = lora_wrapper.load_lora(lora_path)
                    lora_wrapper.apply_lora(lora_name, strength)
                    logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")
                else:
                    raise ValueError(f"Unsupported LoRA path: {lora_path}")

        return MultiModelStruct([high_noise_model, low_noise_model], self.config, self.config["boundary"])


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
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda().unsqueeze(1)
        vae_encoder_out = self.get_vae_encoder_output(img)
        self.config.lat_w, self.config.lat_h = ow // self.config.vae_stride[2], oh // self.config.vae_stride[1]

        return vae_encoder_out

    def get_vae_encoder_output(self, img):
        z = self.vae_encoder.encode(img.to(GET_DTYPE()))
        return z
