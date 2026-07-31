# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from PIL import Image

from lightx2v.models.video_encoders.hf.bagel.vae import BagelVae


class SenseNovaVisionVae(BagelVae):
    """Bagel VAE decoder with SenseNova's multi-view/raw-pointmap outputs."""

    def decode(self, latents, decode_info):
        packed_seqlens = decode_info["packed_seqlens"]
        unpacked_latents = latents.split((packed_seqlens - 2).tolist())
        image_shapes = decode_info.get("image_shapes")
        if image_shapes is None:
            image_shapes = [decode_info["image_shape"]] * len(unpacked_latents)
        if len(image_shapes) != len(unpacked_latents):
            raise ValueError(f"SenseNova decode received {len(image_shapes)} image shapes for {len(unpacked_latents)} latent sequences.")

        outputs = []
        for latent, (height, width) in zip(unpacked_latents, image_shapes):
            h = height // decode_info["latent_downsample"]
            w = width // decode_info["latent_downsample"]
            patch_size = decode_info["latent_patch_size"]
            latent_channel = decode_info["latent_channel"]
            latent = latent.reshape(1, h, w, patch_size, patch_size, latent_channel)
            latent = torch.einsum("nhwpqc->nchpwq", latent)
            latent = latent.reshape(1, latent_channel, h * patch_size, w * patch_size)
            latent = latent.to(device=self.device, dtype=torch.float32)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                decoded = self.vae_model.decode(latent)

            if decode_info.get("output_raw_tensor", False):
                outputs.append(decoded[0].permute(1, 2, 0).float().detach().cpu().numpy())
                continue

            image = (decoded * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
            image = image.to(torch.uint8).cpu()
            if decode_info.get("return_result_tensor", False):
                outputs.append(image)
            else:
                outputs.append(Image.fromarray(np.asarray(image)))
        return outputs
