import os
import gc
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.distributed as dist
from loguru import logger
from PIL import Image
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.schedulers.wan.changing_resolution.scheduler import (
    WanScheduler4ChangingResolutionInterface,
)
from lightx2v.models.schedulers.wan.feature_caching.scheduler import (
    WanSchedulerCaching,
    WanSchedulerTaylorCaching,
)
from lightx2v.utils.utils import *
from lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.xlm_roberta.model import CLIPModel
from lightx2v.models.networks.wan.model import WanModel, Wan22MoeModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE
from lightx2v.models.video_encoders.hf.wan.vae_tiny import WanVAE_tiny
from lightx2v.utils.utils import cache_video, best_output_size
from lightx2v.utils.profiler import ProfilingContext


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
        if self.config.get("lora_configs") and self.config.lora_configs:
            assert not self.config.get("dit_quantized", False) or self.config.mm_config.get("weight_auto_quant", False)
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
        if self.config.task == "i2v" and self.config.get("use_image_encoder", True):
            # quant_config
            clip_quantized = self.config.get("clip_quantized", False)
            if clip_quantized:
                clip_quant_scheme = self.config.get("clip_quant_scheme", None)
                assert clip_quant_scheme is not None
                tmp_clip_quant_scheme = clip_quant_scheme.split("-")[0]
                clip_model_name = f"clip-{tmp_clip_quant_scheme}.pth"
                clip_quantized_ckpt = find_torch_model_path(self.config, "clip_quantized_ckpt", clip_model_name, tmp_clip_quant_scheme)
                clip_original_ckpt = None
            else:
                clip_quantized_ckpt = None
                clip_quant_scheme = None
                clip_model_name = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
                clip_original_ckpt = find_torch_model_path(self.config, "clip_original_ckpt", clip_model_name, "original")

            image_encoder = CLIPModel(
                dtype=torch.float16,
                device=self.init_device,
                checkpoint_path=clip_original_ckpt,
                clip_quantized=clip_quantized,
                clip_quantized_ckpt=clip_quantized_ckpt,
                quant_scheme=clip_quant_scheme,
            )

        return image_encoder

    def load_text_encoder(self):
        # offload config
        t5_offload = self.config.get("t5_cpu_offload", False)
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
            t5_quantized_ckpt = find_torch_model_path(self.config, "t5_quantized_ckpt", t5_model_name, tmp_t5_quant_scheme)
            t5_original_ckpt = None
            tokenizer_path = os.path.join(os.path.dirname(t5_quantized_ckpt), "google/umt5-xxl")
        else:
            t5_quant_scheme = None
            t5_quantized_ckpt = None
            t5_model_name = "models_t5_umt5-xxl-enc-bf16.pth"
            t5_original_ckpt = find_torch_model_path(self.config, "t5_original_ckpt", t5_model_name, "original")
            tokenizer_path = os.path.join(os.path.dirname(t5_original_ckpt), "google/umt5-xxl")

        text_encoder = T5EncoderModel(
            text_len=self.config["text_len"],
            dtype=torch.bfloat16,
            device=t5_device,
            checkpoint_path=t5_original_ckpt,
            tokenizer_path=tokenizer_path,
            shard_fn=None,
            cpu_offload=t5_offload,
            offload_granularity=self.config.get("t5_offload_granularity", "model"),
            t5_quantized=t5_quantized,
            t5_quantized_ckpt=t5_quantized_ckpt,
            quant_scheme=t5_quant_scheme,
        )
        text_encoders = [text_encoder]
        return text_encoders

    def load_vae_encoder(self):
        vae_config = {
            "vae_pth": find_torch_model_path(self.config, "vae_pth", "Wan2.1_VAE.pth"),
            "device": self.init_device,
            "parallel": self.config.parallel and self.config.parallel.get("vae_p_size", False) and self.config.parallel.vae_p_size > 1,
            "use_tiling": self.config.get("use_tiling_vae", False),
        }
        if self.config.task != "i2v":
            return None
        else:
            return WanVAE(**vae_config)

    def load_vae_decoder(self):
        vae_config = {
            "vae_pth": find_torch_model_path(self.config, "vae_pth", "Wan2.1_VAE.pth"),
            "device": self.init_device,
            "parallel": self.config.parallel and self.config.parallel.get("vae_p_size", False) and self.config.parallel.vae_p_size > 1,
            "use_tiling": self.config.get("use_tiling_vae", False),
        }
        if self.config.get("use_tiny_vae", False):
            tiny_vae_path = find_torch_model_path(self.config, "tiny_vae_path", "taew2_1.pth")
            vae_decoder = WanVAE_tiny(
                vae_pth=tiny_vae_path,
                device=self.init_device,
            ).to("cuda")
        else:
            vae_decoder = WanVAE(**vae_config)
        return vae_decoder

    def load_vae(self):
        vae_encoder = self.load_vae_encoder()
        if vae_encoder is None or self.config.get("use_tiny_vae", False):
            vae_decoder = self.load_vae_decoder()
        else:
            vae_decoder = vae_encoder
        return vae_encoder, vae_decoder

    def init_scheduler(self):
        if self.config.feature_caching == "NoCaching":
            scheduler_class = WanScheduler
        elif self.config.feature_caching == "TaylorSeer":
            scheduler_class = WanSchedulerTaylorCaching
        elif self.config.feature_caching in ["Tea", "Ada", "Custom", "FirstBlock", "DualBlock", "DynamicBlock"]:
            scheduler_class = WanSchedulerCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")

        if self.config.get("changing_resolution", False):
            scheduler = WanScheduler4ChangingResolutionInterface(scheduler_class, self.config)
        else:
            scheduler = scheduler_class(self.config)
        self.model.set_scheduler(scheduler)

    def run_text_encoder(self, text, img):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()
        n_prompt = self.config.get("negative_prompt", "")

        if self.config["cfg_parallel"]:
            cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
            cfg_p_rank = dist.get_rank(cfg_p_group)
            if cfg_p_rank == 0:
                context = self.text_encoders[0].infer([text])
                text_encoder_output = {"context": context}
            else:
                context_null = self.text_encoders[0].infer([n_prompt])
                text_encoder_output = {"context_null": context_null}
        else:
            context = self.text_encoders[0].infer([text])
            context_null = self.text_encoders[0].infer([n_prompt])
            text_encoder_output = {
                "context": context,
                "context_null": context_null,
            }

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]
            torch.cuda.empty_cache()
            gc.collect()

        return text_encoder_output

    def run_image_encoder(self, img):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.image_encoder = self.load_image_encoder()
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        clip_encoder_out = self.image_encoder.visual([img[None, :, :, :]], self.config).squeeze(0).to(torch.bfloat16)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.image_encoder
            torch.cuda.empty_cache()
            gc.collect()
        return clip_encoder_out

    def run_vae_encoder(self, img):
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        h, w = img.shape[1:]
        aspect_ratio = h / w
        max_area = self.config.target_height * self.config.target_width
        lat_h = round(np.sqrt(max_area * aspect_ratio) // self.config.vae_stride[1] // self.config.patch_size[1] * self.config.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // self.config.vae_stride[2] // self.config.patch_size[2] * self.config.patch_size[2])

        if self.config.get("changing_resolution", False):
            self.config.lat_h, self.config.lat_w = lat_h, lat_w
            vae_encode_out_list = []
            for i in range(len(self.config["resolution_rate"])):
                lat_h, lat_w = (
                    int(self.config.lat_h * self.config.resolution_rate[i]) // 2 * 2,
                    int(self.config.lat_w * self.config.resolution_rate[i]) // 2 * 2,
                )
                vae_encode_out_list.append(self.get_vae_encoder_output(img, lat_h, lat_w))
            vae_encode_out_list.append(self.get_vae_encoder_output(img, self.config.lat_h, self.config.lat_w))
            return vae_encode_out_list
        else:
            self.config.lat_h, self.config.lat_w = lat_h, lat_w
            vae_encoder_out = self.get_vae_encoder_output(img, lat_h, lat_w)
            return vae_encoder_out

    def get_vae_encoder_output(self, img, lat_h, lat_w):
        h = lat_h * self.config.vae_stride[1]
        w = lat_w * self.config.vae_stride[2]

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
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder = self.load_vae_encoder()
        vae_encoder_out = self.vae_encoder.encode(
            [
                torch.concat(
                    [
                        torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                        torch.zeros(3, self.config.target_video_length - 1, h, w),
                    ],
                    dim=1,
                ).cuda()
            ],
            self.config,
        )[0]
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_encoder
            torch.cuda.empty_cache()
            gc.collect()
        vae_encoder_out = torch.concat([msk, vae_encoder_out]).to(torch.bfloat16)
        return vae_encoder_out

    def get_encoder_output_i2v(self, clip_encoder_out, vae_encoder_out, text_encoder_output, img):
        image_encoder_output = {
            "clip_encoder_out": clip_encoder_out,
            "vae_encoder_out": vae_encoder_out,
        }
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }

    def set_target_shape(self):
        num_channels_latents = self.config.get("num_channels_latents", 16)
        if self.config.task == "i2v":
            self.config.target_shape = (
                num_channels_latents,
                (self.config.target_video_length - 1) // self.config.vae_stride[0] + 1,
                self.config.lat_h,
                self.config.lat_w,
            )
        elif self.config.task == "t2v":
            self.config.target_shape = (
                num_channels_latents,
                (self.config.target_video_length - 1) // self.config.vae_stride[0] + 1,
                int(self.config.target_height) // self.config.vae_stride[1],
                int(self.config.target_width) // self.config.vae_stride[2],
            )

    def save_video_func(self, images):
        cache_video(
            tensor=images,
            save_file=self.config.save_video_path,
            fps=self.config.get("fps", 16),
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )


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

    def get_current_model_index(self):
        if self.scheduler.timesteps[self.scheduler.step_index] >= self.boundary_timestep:
            logger.info(f"using - HIGH - noise model at step_index {self.scheduler.step_index + 1}")
            self.scheduler.sample_guide_scale = self.config.sample_guide_scale[0]
            if self.cur_model_index == -1:
                self.to_cuda(model_index=0)
            elif self.cur_model_index == 1:  # 1 -> 0
                self.offload_cpu(model_index=1)
                self.to_cuda(model_index=0)
            self.cur_model_index = 0
        else:
            logger.info(f"using - LOW - noise model at step_index {self.scheduler.step_index + 1}")
            self.scheduler.sample_guide_scale = self.config.sample_guide_scale[1]
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

    def load_transformer(self):
        # encoder -> high_noise_model -> low_noise_model -> vae -> video_output
        high_noise_model = Wan22MoeModel(
            os.path.join(self.config.model_path, "high_noise_model"),
            self.config,
            self.init_device,
        )
        low_noise_model = Wan22MoeModel(
            os.path.join(self.config.model_path, "low_noise_model"),
            self.config,
            self.init_device,
        )
        return MultiModelStruct([high_noise_model, low_noise_model], self.config, self.config.boundary)


@RUNNER_REGISTER("wan2.2")
class Wan22DenseRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_vae_decoder(self):
        vae_config = {
            "vae_pth": find_torch_model_path(self.config, "vae_pth", "Wan2.2_VAE.pth"),
            "device": self.init_device,
        }
        vae_decoder = Wan2_2_VAE(**vae_config)
        return vae_decoder

    def load_vae_encoder(self):
        vae_config = {
            "vae_pth": find_torch_model_path(self.config, "vae_pth", "Wan2.2_VAE.pth"),
            "device": self.init_device,
        }
        if self.config.task != "i2v":
            return None
        else:
            return Wan2_2_VAE(**vae_config)

    def load_vae(self):
        vae_encoder = self.load_vae_encoder()
        vae_decoder = self.load_vae_decoder()
        return vae_encoder, vae_decoder

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
        z = self.vae_encoder.encode(img, self.config)
        return z
