import gc
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from loguru import logger

from lightx2v.models.input_encoders.hf.hunyuan15.byt5.model import ByT5TextEncoder
from lightx2v.models.input_encoders.hf.hunyuan15.qwen25.model import Qwen25VL_TextEncoder
from lightx2v.models.input_encoders.hf.hunyuan15.siglip.model import SiglipVisionEncoder
from lightx2v.models.networks.hunyuan_video.model import HunyuanVideo15Model
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.hunyuan_video.scheduler import HunyuanVideo15Scheduler
from lightx2v.models.schedulers.hunyuan_video.feature_caching.scheduler import HunyuanVideo15SchedulerCaching
from lightx2v.models.video_encoders.hf.hunyuanvideo15.hunyuanvideo_15_vae import HunyuanVideo15VAE
from lightx2v.models.video_encoders.hf.hunyuanvideo15.lighttae_hy15 import LightTaeHy15
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import *


@RUNNER_REGISTER("hunyuan_video_1.5")
class HunyuanVideo15Runner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)
        self.target_size_config = {
            "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
            "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
            "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
            "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
        }
        self.vision_num_semantic_tokens = 729
        self.vision_states_dim = 1152
        self.vae_cls = HunyuanVideo15VAE
        self.tae_cls = LightTaeHy15

    def init_scheduler(self):
        if self.config["feature_caching"] == "NoCaching":
            scheduler_class = HunyuanVideo15Scheduler
        elif self.config.feature_caching in ["Mag"]:
            scheduler_class = HunyuanVideo15SchedulerCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")
        self.scheduler = scheduler_class(self.config)

    def load_text_encoder(self):
        qwen25vl_offload = self.config.get("qwen25vl_cpu_offload", self.config.get("cpu_offload"))
        if qwen25vl_offload:
            qwen25vl_device = torch.device("cpu")
        else:
            qwen25vl_device = torch.device("cuda")

        text_encoder_path = os.path.join(self.config["model_path"], "text_encoder/llm")
        logger.info(f"Loading text encoder from {text_encoder_path}")
        text_encoder = Qwen25VL_TextEncoder(dtype=torch.float16, device=qwen25vl_device, checkpoint_path=text_encoder_path, cpu_offload=qwen25vl_offload)

        byt5_offload = self.config.get("byt5_cpu_offload", self.config.get("cpu_offload"))
        if byt5_offload:
            byt5_device = torch.device("cpu")
        else:
            byt5_device = torch.device("cuda")

        byt5 = ByT5TextEncoder(config=self.config, device=byt5_device, checkpoint_path=self.config["model_path"], cpu_offload=byt5_offload)
        text_encoders = [text_encoder, byt5]
        return text_encoders

    def load_transformer(self):
        model = HunyuanVideo15Model(self.config["model_path"], self.config, self.init_device)
        return model

    def get_latent_shape_with_target_hw(self, origin_size=None):
        if origin_size is None:
            width, height = self.config["aspect_ratio"].split(":")
        else:
            width, height = origin_size
        target_size = self.config["transformer_model_name"].split("_")[0]
        self.target_height, self.target_width = self.get_closest_resolution_given_original_size((int(width), int(height)), target_size)
        latent_shape = [
            self.config.get("in_channels", 32),
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            self.target_height // self.config["vae_stride"][1],
            self.target_width // self.config["vae_stride"][2],
        ]
        return latent_shape

    def get_closest_resolution_given_original_size(self, origin_size, target_size):
        bucket_hw_base_size = self.target_size_config[target_size]["bucket_hw_base_size"]
        bucket_hw_bucket_stride = self.target_size_config[target_size]["bucket_hw_bucket_stride"]

        assert bucket_hw_base_size in [128, 256, 480, 512, 640, 720, 960, 1440], f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960], but got {bucket_hw_base_size}"

        crop_size_list = self.generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
        aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
        closest_size, closest_ratio = self.get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)

        height = closest_size[0]
        width = closest_size[1]

        return height, width

    def generate_crop_size_list(self, base_size=256, patch_size=16, max_ratio=4.0):
        num_patches = round((base_size / patch_size) ** 2)
        assert max_ratio >= 1.0
        crop_size_list = []
        wp, hp = num_patches, 1
        while wp > 0:
            if max(wp, hp) / min(wp, hp) <= max_ratio:
                crop_size_list.append((wp * patch_size, hp * patch_size))
            if (hp + 1) * wp <= num_patches:
                hp += 1
            else:
                wp -= 1
        return crop_size_list

    def get_closest_ratio(self, height: float, width: float, ratios: list, buckets: list):
        aspect_ratio = float(height) / float(width)
        diff_ratios = ratios - aspect_ratio

        if aspect_ratio >= 1:
            indices = [(index, x) for index, x in enumerate(diff_ratios) if x <= 0]
        else:
            indices = [(index, x) for index, x in enumerate(diff_ratios) if x > 0]

        closest_ratio_id = min(indices, key=lambda pair: abs(pair[1]))[0]
        closest_size = buckets[closest_ratio_id]
        closest_ratio = ratios[closest_ratio_id]

        return closest_size, closest_ratio

    def run_text_encoder(self, input_info):
        prompt = input_info.prompt_enhanced if self.config["use_prompt_enhancer"] else input_info.prompt
        neg_prompt = input_info.negative_prompt

        # run qwen25vl
        if self.config.get("enable_cfg", False) and self.config["cfg_parallel"]:
            cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
            cfg_p_rank = dist.get_rank(cfg_p_group)
            if cfg_p_rank == 0:
                context = self.text_encoders[0].infer([prompt])
                text_encoder_output = {"context": context}
            else:
                context_null = self.text_encoders[0].infer([neg_prompt])
                text_encoder_output = {"context_null": context_null}
        else:
            context = self.text_encoders[0].infer([prompt])
            context_null = self.text_encoders[0].infer([neg_prompt]) if self.config.get("enable_cfg", False) else None
            text_encoder_output = {
                "context": context,
                "context_null": context_null,
            }

        # run byt5
        byt5_features, byt5_masks = self.text_encoders[1].infer([prompt])
        text_encoder_output.update({"byt5_features": byt5_features, "byt5_masks": byt5_masks})

        return text_encoder_output

    def load_image_encoder(self):
        siglip_offload = self.config.get("siglip_cpu_offload", self.config.get("cpu_offload"))
        if siglip_offload:
            siglip_device = torch.device("cpu")
        else:
            siglip_device = torch.device("cuda")
        image_encoder = SiglipVisionEncoder(
            config=self.config,
            device=siglip_device,
            checkpoint_path=self.config["model_path"],
            cpu_offload=siglip_offload,
        )
        return image_encoder

    def load_vae_encoder(self):
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device("cuda")

        vae_config = {
            "checkpoint_path": self.config["model_path"],
            "device": vae_device,
            "cpu_offload": vae_offload,
            "dtype": GET_DTYPE(),
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
            "checkpoint_path": self.config["model_path"],
            "device": vae_device,
            "cpu_offload": vae_offload,
            "dtype": GET_DTYPE(),
        }
        if self.config.get("use_tae", False):
            tae_path = self.config["tae_path"]
            vae_decoder = self.tae_cls(vae_path=tae_path, dtype=GET_DTYPE()).to("cuda")
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

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2v(self):
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw()  # Important: set latent_shape in input_info
        text_encoder_output = self.run_text_encoder(self.input_info)

        # vision_states is all zero, because we don't have any image input
        siglip_output = torch.zeros(1, self.vision_num_semantic_tokens, self.config["hidden_size"], dtype=torch.bfloat16).cuda()
        siglip_mask = torch.zeros(1, self.vision_num_semantic_tokens, dtype=torch.bfloat16, device=torch.device("cuda"))

        torch.cuda.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {
                "siglip_output": siglip_output,
                "siglip_mask": siglip_mask,
                "cond_latents": None,
            },
        }

    def read_image_input(self, img_path):
        if isinstance(img_path, Image.Image):
            img_ori = img_path
        else:
            img_ori = Image.open(img_path).convert("RGB")
        return img_ori

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        img_ori = self.read_image_input(self.input_info.image_path)
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw(origin_size=img_ori.size)  # Important: set latent_shape in input_info
        siglip_output, siglip_mask = self.run_image_encoder(img_ori) if self.config.get("use_image_encoder", True) else None
        cond_latents = self.run_vae_encoder(img_ori)
        text_encoder_output = self.run_text_encoder(self.input_info)
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {
                "siglip_output": siglip_output,
                "siglip_mask": siglip_mask,
                "cond_latents": cond_latents,
            },
        }

    @ProfilingContext4DebugL1(
        "Run Image Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_img_encode_duration,
        metrics_labels=["WanRunner"],
    )
    def run_image_encoder(self, first_frame, last_frame=None):
        input_image_np = self.resize_and_center_crop(first_frame, target_width=self.target_width, target_height=self.target_height)
        vision_states = self.image_encoder.encode_images(input_image_np).last_hidden_state.to(device=torch.device("cuda"), dtype=torch.bfloat16)
        image_encoder_output = self.image_encoder.infer(vision_states)
        image_encoder_mask = torch.ones((1, image_encoder_output.shape[1]), dtype=torch.bfloat16, device=torch.device("cuda"))
        return image_encoder_output, image_encoder_mask

    def resize_and_center_crop(self, image, target_width, target_height):
        image = np.array(image)
        if target_height == image.shape[0] and target_width == image.shape[1]:
            return image

        pil_image = Image.fromarray(image)
        original_width, original_height = pil_image.size
        scale_factor = max(target_width / original_width, target_height / original_height)
        resized_width = int(round(original_width * scale_factor))
        resized_height = int(round(original_height * scale_factor))
        resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
        left = (resized_width - target_width) / 2
        top = (resized_height - target_height) / 2
        right = (resized_width + target_width) / 2
        bottom = (resized_height + target_height) / 2
        cropped_image = resized_image.crop((left, top, right, bottom))
        return np.array(cropped_image)

    @ProfilingContext4DebugL1(
        "Run VAE Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration,
        metrics_labels=["WanRunner"],
    )
    def run_vae_encoder(self, first_frame):
        origin_size = first_frame.size
        original_width, original_height = origin_size

        scale_factor = max(self.target_width / original_width, self.target_height / original_height)
        resize_width = int(round(original_width * scale_factor))
        resize_height = int(round(original_height * scale_factor))

        ref_image_transform = transforms.Compose(
            [
                transforms.Resize((resize_height, resize_width), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop((self.target_height, self.target_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        ref_images_pixel_values = ref_image_transform(first_frame).unsqueeze(0).unsqueeze(2).cuda()

        cond_latents = self.vae_encoder.encode(ref_images_pixel_values.to(GET_DTYPE()))
        return cond_latents
