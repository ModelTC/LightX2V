import os

import numpy as np
import torch
import torchvision
from PIL import Image

from lightx2v.models.input_encoders.hf.clip.model import TextEncoderHFClipModel
from lightx2v.models.input_encoders.hf.llama.model import TextEncoderHFLlamaModel
from lightx2v.models.input_encoders.hf.llava.model import TextEncoderHFLlavaModel
from lightx2v.models.networks.hunyuan.model import HunyuanModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.hunyuan.feature_caching.scheduler import HunyuanSchedulerAdaCaching, HunyuanSchedulerCustomCaching, HunyuanSchedulerTaylorCaching, HunyuanSchedulerTeaCaching
from lightx2v.models.schedulers.hunyuan.scheduler import HunyuanScheduler
from lightx2v.models.video_encoders.hf.autoencoder_kl_causal_3d.model import VideoEncoderKLCausal3DModel
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import save_videos_grid


@RUNNER_REGISTER("hunyuan")
class HunyuanRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_transformer(self):
        return HunyuanModel(self.config.model_path, self.config, self.init_device, self.config)

    def load_image_encoder(self):
        return None

    def load_text_encoder(self):
        if self.config.task == "t2v":
            text_encoder_1 = TextEncoderHFLlamaModel(os.path.join(self.config.model_path, "text_encoder"), self.init_device)
        else:
            text_encoder_1 = TextEncoderHFLlavaModel(os.path.join(self.config.model_path, "text_encoder_i2v"), self.init_device)
        text_encoder_2 = TextEncoderHFClipModel(os.path.join(self.config.model_path, "text_encoder_2"), self.init_device)
        text_encoders = [text_encoder_1, text_encoder_2]
        return text_encoders

    def load_vae(self):
        vae_model = VideoEncoderKLCausal3DModel(self.config.model_path, dtype=torch.float16, device=self.init_device, config=self.config)
        return vae_model, vae_model

    def init_scheduler(self):
        if self.config.feature_caching == "NoCaching":
            scheduler = HunyuanScheduler(self.config)
        elif self.config.feature_caching == "Tea":
            scheduler = HunyuanSchedulerTeaCaching(self.config)
        elif self.config.feature_caching == "TaylorSeer":
            scheduler = HunyuanSchedulerTaylorCaching(self.config)
        elif self.config.feature_caching == "Ada":
            scheduler = HunyuanSchedulerAdaCaching(self.config)
        elif self.config.feature_caching == "Custom":
            scheduler = HunyuanSchedulerCustomCaching(self.config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")
        self.model.set_scheduler(scheduler)

    def run_text_encoder(self, text, img):
        text_encoder_output = {}
        for i, encoder in enumerate(self.text_encoders):
            if self.config.task == "i2v" and i == 0:
                text_state, attention_mask = encoder.infer(text, img, self.config)
            else:
                text_state, attention_mask = encoder.infer(text, self.config)
            text_encoder_output[f"text_encoder_{i + 1}_text_states"] = text_state.to(dtype=torch.bfloat16)
            text_encoder_output[f"text_encoder_{i + 1}_attention_mask"] = attention_mask
        return text_encoder_output

    @staticmethod
    def get_closest_ratio(height: float, width: float, ratios: list, buckets: list):
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

    @staticmethod
    def generate_crop_size_list(base_size=256, patch_size=32, max_ratio=4.0):
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

    def run_image_encoder(self, img):
        return None

    def run_vae_encoder(self, img):
        kwargs = {}
        if self.config.i2v_resolution == "720p":
            bucket_hw_base_size = 960
        elif self.config.i2v_resolution == "540p":
            bucket_hw_base_size = 720
        elif self.config.i2v_resolution == "360p":
            bucket_hw_base_size = 480
        else:
            raise ValueError(f"self.config.i2v_resolution: {self.config.i2v_resolution} must be in [360p, 540p, 720p]")

        origin_size = img.size

        crop_size_list = self.generate_crop_size_list(bucket_hw_base_size, 32)
        aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
        closest_size, closest_ratio = self.get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)

        self.config.target_height, self.config.target_width = closest_size
        kwargs["target_height"], kwargs["target_width"] = closest_size

        resize_param = min(closest_size)
        center_crop_param = closest_size

        ref_image_transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(resize_param), torchvision.transforms.CenterCrop(center_crop_param), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5], [0.5])]
        )

        semantic_image_pixel_values = [ref_image_transform(img)]
        semantic_image_pixel_values = torch.cat(semantic_image_pixel_values).unsqueeze(0).unsqueeze(2).to(torch.float16).to(torch.device("cuda"))

        img_latents = self.vae_encoder.encode(semantic_image_pixel_values, self.config).mode()

        scaling_factor = 0.476986
        img_latents.mul_(scaling_factor)

        return img_latents, kwargs

    def get_encoder_output_i2v(self, clip_encoder_out, vae_encoder_out, text_encoder_output, img):
        image_encoder_output = {"img": img, "img_latents": vae_encoder_out}
        return {"text_encoder_output": text_encoder_output, "image_encoder_output": image_encoder_output}

    def set_target_shape(self):
        vae_scale_factor = 2 ** (4 - 1)
        self.config.target_shape = (
            1,
            16,
            (self.config.target_video_length - 1) // 4 + 1,
            int(self.config.target_height) // vae_scale_factor,
            int(self.config.target_width) // vae_scale_factor,
        )
        return {"target_height": self.config.target_height, "target_width": self.config.target_width, "target_shape": self.config.target_shape}

    def save_video_func(self, images):
        save_videos_grid(images, self.config.save_video_path, fps=self.config.get("fps", 24))
