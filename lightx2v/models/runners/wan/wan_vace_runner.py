import gc

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from lightx2v.models.input_encoders.hf.vace.vace_processor import VaceVideoProcessor
from lightx2v.models.networks.wan.vace_model import WanVaceModel
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("wan2.1_vace")
class WanVaceRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)
        assert self.config["task"] == "vace"
        self.vid_proc = VaceVideoProcessor(
            downsample=tuple([x * y for x, y in zip(self.config["vae_stride"], self.config["patch_size"])]),
            min_area=720 * 1280,
            max_area=720 * 1280,
            min_fps=self.config["fps"] if "fps" in self.config else 16,
            max_fps=self.config["fps"] if "fps" in self.config else 16,
            zero_start=True,
            seq_len=75600,
            keep_last=True,
        )

    def load_transformer(self):
        model = WanVaceModel(
            self.config["model_path"],
            self.config,
            self.init_device,
        )
        return model

    def prepare_source(self, src_video, src_mask, src_ref_images, image_size, device=torch.device("cuda")):
        area = image_size[0] * image_size[1]
        self.vid_proc.set_area(area)
        if area == 720 * 1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480 * 832:
            self.vid_proc.set_seq_len(32760)
        else:
            raise NotImplementedError(f"image_size {image_size} is not supported")

        image_size = (image_size[1], image_size[0])
        image_sizes = []
        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i], src_mask[i], _, _, _ = self.vid_proc.load_video_pair(sub_src_video, sub_src_mask)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                src_mask[i] = torch.clamp((src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                src_video[i] = torch.zeros((3, self.config["target_video_length"], image_size[0], image_size[1]), device=device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(src_video[i].shape[2:])

        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        ref_img = Image.open(ref_img).convert("RGB")
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
                        if ref_img.shape[-2:] != image_size:
                            canvas_height, canvas_width = image_size
                            ref_height, ref_width = ref_img.shape[-2:]
                            white_canvas = torch.ones((3, 1, canvas_height, canvas_width), device=device)  # [-1, 1]
                            scale = min(canvas_height / ref_height, canvas_width / ref_width)
                            new_height = int(ref_height * scale)
                            new_width = int(ref_width * scale)
                            resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False).squeeze(0).unsqueeze(1)
                            top = (canvas_height - new_height) // 2
                            left = (canvas_width - new_width) // 2
                            white_canvas[:, :, top : top + new_height, left : left + new_width] = resized_image
                            ref_img = white_canvas
                        src_ref_images[i][j] = ref_img.to(device)
        return src_video, src_mask, src_ref_images

    @ProfilingContext4DebugL1(
        "Run VAE Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_encode_duration,
        metrics_labels=["WanVaceRunner"],
    )
    def run_vae_encoder(self, frames, ref_images, masks):
        if (self.config["lazy_load"] if "lazy_load" in self.config else False) or (self.config["unload_modules"] if "unload_modules" in self.config else False):
            self.vae_encoder = self.load_vae_encoder()
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = [self.vae_encoder.encode(frame.unsqueeze(0).to(GET_DTYPE())) for frame in frames]
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = [self.vae_encoder.encode(inact.unsqueeze(0).to(GET_DTYPE())) for inact in inactive]
            reactive = [self.vae_encoder.encode(react.unsqueeze(0).to(GET_DTYPE())) for react in reactive]
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = [self.vae_encoder.encode(ref.unsqueeze(0).to(GET_DTYPE())) for ref in refs]
                else:
                    ref_latent = [self.vae_encoder.encode(ref.unsqueeze(0).to(GET_DTYPE())) for ref in refs]
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        self.latent_shape = list(cat_latents[0].shape)
        if (self.config["lazy_load"] if "lazy_load" in self.config else False) or (self.config["unload_modules"] if "unload_modules" in self.config else False):
            del self.vae_encoder
            torch.cuda.empty_cache()
            gc.collect()
        return self.get_vae_encoder_output(cat_latents, masks, ref_images), self.set_input_info_latent_shape()

    def get_vae_encoder_output(self, cat_latents, masks, ref_images):
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // self.config["vae_stride"][0])
            height = 2 * (int(height) // (self.config["vae_stride"][1] * 2))
            width = 2 * (int(width) // (self.config["vae_stride"][2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(depth, height, self.config["vae_stride"][1], width, self.config["vae_stride"][1])  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(self.config["vae_stride"][1] * self.config["vae_stride"][2], depth, height, width)  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode="nearest-exact").squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)

        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(cat_latents, result_masks)]

    def set_input_info_latent_shape(self):
        latent_shape = self.latent_shape
        latent_shape[0] = int(latent_shape[0] / 2)
        return latent_shape

    @ProfilingContext4DebugL1(
        "Run VAE Decoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_decode_duration,
        metrics_labels=["WanVaceRunner"],
    )
    def run_vae_decoder(self, latents):
        if (self.config["lazy_load"] if "lazy_load" in self.config else False) or (self.config["unload_modules"] if "unload_modules" in self.config else False):
            self.vae_decoder = self.load_vae_decoder()

        if self.src_ref_images is not None:
            assert len(self.src_ref_images) == 1
            refs = self.src_ref_images[0]
            if refs is not None:
                latents = latents[:, len(refs) :, :, :]

        images = self.vae_decoder.decode(latents.to(GET_DTYPE()))

        if (self.config["lazy_load"] if "lazy_load" in self.config else False) or (self.config["unload_modules"] if "unload_modules" in self.config else False):
            del self.vae_decoder
            torch.cuda.empty_cache()
            gc.collect()

        return images
