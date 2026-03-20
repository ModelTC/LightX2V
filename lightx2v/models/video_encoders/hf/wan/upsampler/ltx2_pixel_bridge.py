import math

import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class LTX2PixelBridgeResizer:
    """把 WAN clean latent 借道 RGB / LTX2 latent 做一次无训练的 x2 升分。"""

    def __init__(self, wan_vae_encoder, wan_vae_decoder, ltx2_video_vae, ltx2_upsampler, config):
        self.wan_vae_encoder = wan_vae_encoder
        self.wan_vae_decoder = wan_vae_decoder
        self.ltx2_video_vae = ltx2_video_vae
        self.ltx2_upsampler = ltx2_upsampler
        self.config = config
        self.dtype = GET_DTYPE()
        self.device = torch.device(AI_DEVICE)
        self.ltx2_spatial_multiple = 32
        self.wan_vae_stride = config["vae_stride"]

    @torch.no_grad()
    def resize(self, latent, target_latent_shape, step_index=None, changing_resolution_index=None):
        if latent.dim() != 4:
            raise ValueError(f"Expected WAN latent shape [C, T, H, W], got {tuple(latent.shape)}")

        target_latent_shape = tuple(target_latent_shape)
        current_shape = tuple(latent.shape)
        if target_latent_shape[0] != current_shape[0] or target_latent_shape[1] != current_shape[1]:
            raise ValueError(
                "LTX2 bridge expects channel/time to stay unchanged. "
                f"Current={current_shape}, target={target_latent_shape}"
            )
        if target_latent_shape[2] != current_shape[2] * 2 or target_latent_shape[3] != current_shape[3] * 2:
            raise ValueError(
                "LTX2 bridge currently only supports a single spatial x2 upscale. "
                f"Current={current_shape}, target={target_latent_shape}"
            )

        target_pixel_h = target_latent_shape[2] * self.wan_vae_stride[1]
        target_pixel_w = target_latent_shape[3] * self.wan_vae_stride[2]
        logger.info(
            "LTX2 pixel bridge resize: "
            f"step={step_index}, stage={changing_resolution_index}, "
            f"wan_latent={current_shape} -> {target_latent_shape}, "
            f"target_rgb=({target_pixel_h}, {target_pixel_w})"
        )

        lowres_rgb = self.wan_vae_decoder.decode(latent.to(self.dtype)).to(device=self.device, dtype=self.dtype)
        padded_lowres_rgb, pad_info = self._pad_video_for_ltx2(lowres_rgb)

        ltx2_latent = self.ltx2_video_vae.encode(padded_lowres_rgb.to(device=self.device, dtype=self.dtype))
        upsampled_ltx2_latent = self.ltx2_upsampler.upsample(
            ltx2_latent.to(device=self.device, dtype=self.dtype),
            self.ltx2_video_vae.encoder,
        )
        highres_rgb = self._decode_ltx2_latent_to_rgb(upsampled_ltx2_latent)
        highres_rgb = self._remove_scaled_padding(highres_rgb, pad_info)
        highres_rgb = self._center_crop_video(highres_rgb, target_pixel_h, target_pixel_w)

        wan_latent = self.wan_vae_encoder.encode(highres_rgb.to(device=self.device, dtype=self.dtype))
        if tuple(wan_latent.shape) != target_latent_shape:
            logger.warning(
                "WAN VAE re-encode shape mismatch after bridge, fallback to latent trilinear resize: "
                f"{tuple(wan_latent.shape)} -> {target_latent_shape}"
            )
            wan_latent = F.interpolate(
                wan_latent.unsqueeze(0),
                size=target_latent_shape[1:],
                mode="trilinear",
            ).squeeze(0)

        return wan_latent.to(dtype=latent.dtype, device=latent.device)

    def _pad_video_for_ltx2(self, video):
        _, _, _, height, width = video.shape
        padded_h = math.ceil(height / self.ltx2_spatial_multiple) * self.ltx2_spatial_multiple
        padded_w = math.ceil(width / self.ltx2_spatial_multiple) * self.ltx2_spatial_multiple
        pad_h = padded_h - height
        pad_w = padded_w - width

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if pad_h == 0 and pad_w == 0:
            return video, (0, 0, 0, 0)

        logger.info(
            "Pad low-res RGB for LTX2 VAE: "
            f"({height}, {width}) -> ({padded_h}, {padded_w}), "
            f"pad=(left={left}, right={right}, top={top}, bottom={bottom})"
        )
        padded = F.pad(video, (left, right, top, bottom), mode="replicate")
        return padded, (left, right, top, bottom)

    def _decode_ltx2_latent_to_rgb(self, latent):
        latent_5d = latent.unsqueeze(0) if latent.dim() == 4 else latent
        latent_5d = latent_5d.to(device=self.device, dtype=self.dtype)

        if self.ltx2_video_vae.cpu_offload:
            self.ltx2_video_vae.decoder = self.ltx2_video_vae.decoder.to(AI_DEVICE)

        try:
            decoded = self.ltx2_video_vae.decoder(latent_5d)
        finally:
            if self.ltx2_video_vae.cpu_offload:
                self.ltx2_video_vae.decoder = self.ltx2_video_vae.decoder.to("cpu")

        return decoded.to(device=self.device, dtype=self.dtype)

    def _remove_scaled_padding(self, video, pad_info):
        left, right, top, bottom = pad_info
        if left == 0 and right == 0 and top == 0 and bottom == 0:
            return video

        left *= 2
        right *= 2
        top *= 2
        bottom *= 2
        h_end = video.shape[-2] - bottom if bottom > 0 else video.shape[-2]
        w_end = video.shape[-1] - right if right > 0 else video.shape[-1]
        return video[:, :, :, top:h_end, left:w_end].contiguous()

    def _center_crop_video(self, video, target_h, target_w):
        _, _, _, height, width = video.shape
        if height == target_h and width == target_w:
            return video

        if height < target_h or width < target_w:
            raise ValueError(
                "Bridge output is smaller than the WAN target size. "
                f"Current=({height}, {width}), target=({target_h}, {target_w})"
            )

        top = (height - target_h) // 2
        left = (width - target_w) // 2
        return video[:, :, :, top : top + target_h, left : left + target_w].contiguous()
