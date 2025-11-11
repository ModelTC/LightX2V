import os

import torch
from diffusers.utils import load_image
from torchvision.transforms import v2

from lightx2v.models.input_encoders.hf.wan.matrix_game2.conditions import Bench_actions_gta_drive, Bench_actions_templerun, Bench_actions_universal
from lightx2v.models.networks.wan.matrix_game2_model import WanSFMtxg2Model
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner
from lightx2v.models.video_encoders.hf.wan.wanx_vae_src import CLIPModel, WanVAE
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER


class VAEWrapper:
    def __init__(self, vae):
        self.vae = vae

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self.vae, name)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, latents):
        return NotImplementedError


class WanxVAEWrapper(VAEWrapper):
    def __init__(self, vae, clip):
        # super().__init__()
        self.vae = vae
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.clip = clip
        if clip is not None:
            self.clip.requires_grad_(False)
            self.clip.eval()

    def encode(self, x, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        x = self.vae.encode(x, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)  # already scaled
        return x  # torch.stack(x, dim=0)

    def clip_img(self, x):
        x = self.clip(x)
        return x

    def decode(self, latents, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        videos = self.vae.decode(latents, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return videos  # self.vae.decode(videos, dim=0) # already scaled

    def to(self, device, dtype):
        # 移动 vae 到指定设备
        self.vae = self.vae.to(device, dtype)

        # 如果 clip 存在，也移动到指定设备
        if self.clip is not None:
            self.clip = self.clip.to(device, dtype)

        return self


def get_wanx_vae_wrapper(model_path, weight_dtype):
    vae = WanVAE(pretrained_path=os.path.join(model_path, "Wan2.1_VAE.pth")).to(weight_dtype)
    clip = CLIPModel(checkpoint_path=os.path.join(model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"), tokenizer_path=os.path.join(model_path, "xlm-roberta-large"))
    return WanxVAEWrapper(vae, clip)


@RUNNER_REGISTER("wan2.1_sf_mtxg2")
class WanSFMtxg2Runner(WanSFRunner):
    def __init__(self, config):
        super().__init__(config)
        self.frame_process = v2.Compose(
            [
                v2.Resize(size=(352, 640), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

    def load_text_encoder(self):
        from lightx2v.models.input_encoders.hf.wan.matrix_game2.conditions import MatrixGame2_Bench

        return MatrixGame2_Bench()

    def load_image_encoder(self):
        wrapper = get_wanx_vae_wrapper(self.config["model_path"], torch.float16)
        wrapper.requires_grad_(False)
        wrapper.eval()
        return wrapper.to(self.device, self.weight_dtype)

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        # image
        image = load_image(self.input_info.image_path)
        image = self._resizecrop(image, 352, 640)

        image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.config["num_output_frames"] - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.image_encoder.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
        visual_context = self.image_encoder.clip.encode_video(image)
        image_encoder_output = {"cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype), "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)}

        # text
        text_encoder_output = {}
        num_frames = (self.config["num_output_frames"] - 1) * 4 + 1
        if self.config["mode"] == "universal":
            cond_data = Bench_actions_universal(num_frames)
            mouse_condition = cond_data["mouse_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            text_encoder_output["mouse_cond"] = mouse_condition
        elif self.config["mode"] == "gta_drive":
            cond_data = Bench_actions_gta_drive(num_frames)
            mouse_condition = cond_data["mouse_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            text_encoder_output["mouse_cond"] = mouse_condition
        else:
            cond_data = Bench_actions_templerun(num_frames)
        keyboard_condition = cond_data["keyboard_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        text_encoder_output["keyboard_cond"] = keyboard_condition

        # set shape
        self.input_info.latent_shape = [16, self.config["num_output_frames"], 44, 80]

        return {"text_encoder_output": text_encoder_output, "image_encoder_output": image_encoder_output}

    def load_transformer(self):
        model = WanSFMtxg2Model(
            self.config["model_path"],
            self.config,
            self.init_device,
        )
        return model
