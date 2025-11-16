import gc
import os

import numpy as np
import torch
from loguru import logger

from lightx2v.models.input_encoders.hf.hunyuan15.byt5.model import ByT5TextEncoder
from lightx2v.models.input_encoders.hf.hunyuan15.qwen25.model import Qwen25VL_TextEncoder
from lightx2v.models.input_encoders.hf.hunyuan15.siglip.model import SiglipVisionEncoder
from lightx2v.models.networks.hunyuan_video.model import HunyuanVideo15Model
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.hunyuan_video.scheduler import HunyuanVideo15Scheduler
from lightx2v.models.video_encoders.hf.hunyuanvideo15.hunyuanvideo_15_vae import HunyuanVideo15VAE
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER


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

    def init_scheduler(self):
        self.scheduler = HunyuanVideo15Scheduler(self.config)

    def load_text_encoder(self):
        qwen25vl_device = torch.device("cuda")
        text_encoder_path = os.path.join(self.config["model_path"], "text_encoder/llm")
        logger.info(f"Loading text encoder from {text_encoder_path}")
        text_encoder = Qwen25VL_TextEncoder(
            dtype=torch.float16,
            device=qwen25vl_device,
            checkpoint_path=text_encoder_path,
        )

        byt5_device = torch.device("cuda")
        byt5 = ByT5TextEncoder(config=self.config, device=byt5_device, checkpoint_path=self.config["model_path"])
        text_encoders = [text_encoder, byt5]
        return text_encoders

    def load_transformer(self):
        model = HunyuanVideo15Model(self.config["model_path"], self.config, self.init_device)
        return model

    def load_vae(self):
        logger.info("Loading vae encoder")
        return None, None

    def get_latent_shape_with_target_hw(self):
        width, height = self.config["aspect_ratio"].split(":")
        target_size = self.config["transformer_model_name"].split("_")[0]
        target_height, target_width = self.get_closest_resolution_given_original_size((int(width), int(height)), target_size)
        latent_shape = [
            self.config.get("in_channels", 32),
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            target_height // self.config["vae_stride"][1],
            target_width // self.config["vae_stride"][2],
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
        image_encoder = SiglipVisionEncoder(
            config=self.config,
            device=torch.device("cuda"),
            checkpoint_path=self.config["model_path"],
        )
        return image_encoder

    def load_vae(self):
        vae_encoder = HunyuanVideo15VAE(checkpoint_path=self.config["model_path"], dtype=torch.bfloat16, device="cuda")
        vae_decoder = vae_encoder
        return vae_encoder, vae_decoder

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2v(self):
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw()  # Important: set latent_shape in input_info
        text_encoder_output = self.run_text_encoder(self.input_info)

        # vision_states is all zero, because we don't have any image input
        image_encoder_output = torch.zeros(1, self.vision_num_semantic_tokens, self.vision_states_dim, dtype=torch.bfloat16).cuda()
        image_encoder_output = self.image_encoder.infer(image_encoder_output) * 0.0
        image_encoder_mask = torch.zeros((1, image_encoder_output.shape[1]), dtype=torch.bfloat16, device=torch.device("cuda"))

        torch.cuda.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
            "image_encoder_mask": image_encoder_mask,
        }
