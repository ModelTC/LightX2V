import os
import gc
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.schedulers.wan.feature_caching.scheduler import (
    WanSchedulerTeaCaching,
)
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.xlm_roberta.model import CLIPModel
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.models.video_encoders.hf.wan.vae_tiny import WanVAE_tiny
from lightx2v.utils.utils import cache_video
from loguru import logger


@RUNNER_REGISTER("wan2.1")
class WanRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_transformer(self):
        model = WanModel(
            self.config.model_path,
            self.config,
            self.init_device,
        )
        if self.config.lora_path:
            assert not self.config.get("dit_quantized", False) or self.config.mm_config.get("weight_auto_quant", False)
            lora_wrapper = WanLoraWrapper(model)
            lora_name = lora_wrapper.load_lora(self.config.lora_path)
            lora_wrapper.apply_lora(lora_name, self.config.strength_model)
            logger.info(f"Loaded LoRA: {lora_name}")
        return model

    def load_image_encoder(self):
        image_encoder = None
        if self.config.task == "i2v":
            image_encoder = CLIPModel(
                dtype=torch.float16,
                device=self.init_device,
                checkpoint_path=os.path.join(
                    self.config.model_path,
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                ),
                clip_quantized=self.config.get("clip_quantized", False),
                clip_quantized_ckpt=self.config.get("clip_quantized_ckpt", None),
                quant_scheme=self.config.get("clip_quant_scheme", None),
            )
        return image_encoder

    def load_text_encoder(self):
        text_encoder = T5EncoderModel(
            text_len=self.config["text_len"],
            dtype=torch.bfloat16,
            device=self.init_device,
            checkpoint_path=os.path.join(self.config.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self.config.model_path, "google/umt5-xxl"),
            shard_fn=None,
            cpu_offload=self.config.cpu_offload,
            offload_granularity=self.config.get("t5_offload_granularity", "model"),
            t5_quantized=self.config.get("t5_quantized", False),
            t5_quantized_ckpt=self.config.get("t5_quantized_ckpt", None),
            quant_scheme=self.config.get("t5_quant_scheme", None),
        )
        text_encoders = [text_encoder]
        return text_encoders

    def load_vae_encoder(self):
        vae_config = {
            "vae_pth": os.path.join(self.config.model_path, "Wan2.1_VAE.pth"),
            "device": self.init_device,
            "parallel": self.config.parallel_vae,
            "use_tiling": self.config.get("use_tiling_vae", False),
        }
        if self.config.task != "i2v":
            return None
        else:
            return WanVAE(**vae_config)

    def load_vae_decoder(self):
        vae_config = {
            "vae_pth": os.path.join(self.config.model_path, "Wan2.1_VAE.pth"),
            "device": self.init_device,
            "parallel": self.config.parallel_vae,
            "use_tiling": self.config.get("use_tiling_vae", False),
        }
        if self.config.get("tiny_vae", False):
            vae_decoder = WanVAE_tiny(
                vae_pth=self.config.tiny_vae_path,
                device=self.init_device,
            ).to("cuda")
        else:
            vae_decoder = WanVAE(**vae_config)
        return vae_decoder

    def load_vae(self):
        return self.load_vae_encoder(), self.load_vae_decoder()

    def init_scheduler(self):
        if self.config.feature_caching == "NoCaching":
            scheduler = WanScheduler(self.config)
        elif self.config.feature_caching == "Tea":
            scheduler = WanSchedulerTeaCaching(self.config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")
        self.model.set_scheduler(scheduler)

    def run_text_encoder(self, text, img):
        if self.config.get("lazy_load", False):
            self.text_encoders = self.load_text_encoder()
        text_encoder_output = {}
        n_prompt = self.config.get("negative_prompt", "")
        context = self.text_encoders[0].infer([text])
        context_null = self.text_encoders[0].infer([n_prompt if n_prompt else ""])
        if self.config.get("lazy_load", False):
            del self.text_encoders[0]
            torch.cuda.empty_cache()
            gc.collect()
        text_encoder_output["context"] = context
        text_encoder_output["context_null"] = context_null
        return text_encoder_output

    def run_image_encoder(self, img):
        if self.config.get("lazy_load", False):
            self.image_encoder = self.load_image_encoder()
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        clip_encoder_out = self.image_encoder.visual([img[:, None, :, :]], self.config).squeeze(0).to(torch.bfloat16)
        if self.config.get("lazy_load", False):
            del self.image_encoder
            torch.cuda.empty_cache()
            gc.collect()
        return clip_encoder_out

    def run_vae_encoder(self, img):
        kwargs = {}
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        h, w = img.shape[1:]
        aspect_ratio = h / w
        max_area = self.config.target_height * self.config.target_width
        lat_h = round(np.sqrt(max_area * aspect_ratio) // self.config.vae_stride[1] // self.config.patch_size[1] * self.config.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // self.config.vae_stride[2] // self.config.patch_size[2] * self.config.patch_size[2])
        h = lat_h * self.config.vae_stride[1]
        w = lat_w * self.config.vae_stride[2]

        self.config.lat_h, kwargs["lat_h"] = lat_h, lat_h
        self.config.lat_w, kwargs["lat_w"] = lat_w, lat_w

        msk = torch.ones(
            1,
            self.config.target_video_length,
            lat_h,
            lat_w,
            device=torch.device("cuda"),
        )
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        if self.config.get("lazy_load", False):
            self.vae_encoder = self.load_vae_encoder()
        
        with ProfilingContext("get videos"):
            videos = [
                torch.concat(
                    [
                        torch.nn.functional.interpolate(img[None].cuda(), size=(h, w), mode="bicubic").transpose(0, 1),
                        torch.zeros(3, self.config.target_video_length - 1, h, w, device="cuda"),
                    ],
                    dim=1,
                ).cuda()
            ]
        with ProfilingContext("vae_encode_out"):
            torch.cuda.nvtx.range_push(f"vae_encode_out")
            vae_encode_out = self.vae_encoder.encode(
                videos,
                self.config,
            )[0]
            torch.cuda.nvtx.range_pop()
        exit()
        if self.config.get("lazy_load", False):
            del self.vae_encoder
            torch.cuda.empty_cache()
            gc.collect()
        vae_encode_out = torch.concat([msk, vae_encode_out]).to(torch.bfloat16)
        return vae_encode_out, kwargs

    def get_encoder_output_i2v(self, clip_encoder_out, vae_encode_out, text_encoder_output, img):
        image_encoder_output = {
            "clip_encoder_out": clip_encoder_out,
            "vae_encode_out": vae_encode_out,
        }
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }

    def set_target_shape(self):
        ret = {}
        num_channels_latents = self.config.get("num_channels_latents", 16)
        if self.config.task == "i2v":
            self.config.target_shape = (
                num_channels_latents,
                (self.config.target_video_length - 1) // self.config.vae_stride[0] + 1,
                self.config.lat_h,
                self.config.lat_w,
            )
            ret["lat_h"] = self.config.lat_h
            ret["lat_w"] = self.config.lat_w
        elif self.config.task == "t2v":
            self.config.target_shape = (
                num_channels_latents,
                (self.config.target_video_length - 1) // self.config.vae_stride[0] + 1,
                int(self.config.target_height) // self.config.vae_stride[1],
                int(self.config.target_width) // self.config.vae_stride[2],
            )
        ret["target_shape"] = self.config.target_shape
        return ret

    def save_video_func(self, images):
        cache_video(
            tensor=images,
            save_file=self.config.save_video_path,
            fps=self.config.get("fps", 16),
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
