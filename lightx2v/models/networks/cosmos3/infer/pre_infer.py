import math

import torch
import torch.nn.functional as F

from lightx2v.models.networks.cosmos3.infer.module_io import Cosmos3PreInferModuleOutput
from lightx2v.models.networks.cosmos3.infer.utils import (
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
    get_timestep_embedding,
)


class Cosmos3PreInfer:
    def __init__(self, config):
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.latent_channel = config["latent_channel"]
        self.latent_patch_size = config["latent_patch_size"]
        self.timestep_scale = config.get("timestep_scale", 0.001)
        self.enable_fps_modulation = config.get("enable_fps_modulation", True)
        self.base_fps = float(config.get("base_fps", 24))
        self.temporal_margin = config.get("unified_3d_mrope_temporal_modality_margin", 15000)
        self.reset_spatial_ids = config.get("unified_3d_mrope_reset_spatial_ids", True)

    def _patchify_and_pack_latents(self, latents):
        p = self.latent_patch_size
        packed_latent = []
        original_latent_shapes = []
        for latent in [latents]:
            latent = latent.squeeze(0)
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros((self.latent_channel, t_actual, h_padded, w_padded), device=latent.device, dtype=latent.dtype)
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded
            h_patches = h_padded // p
            w_patches = w_padded // p
            latent = latent.reshape(self.latent_channel, t_actual, h_patches, p, w_patches, p)
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        return torch.cat(packed_latent, dim=0), original_latent_shapes

    def _apply_timestep_embeds_to_noisy_tokens(self, packed_tokens, packed_timestep_embeds, noisy_frame_indexes, token_shapes):
        start_noisy_index = 0
        flattened_noisy_frame_indexes = []
        for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
            spatial_numel_i = math.prod(token_shape_i[1:])
            spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)
            frame_offsets = (noisy_indexes_i * spatial_numel_i).unsqueeze(-1) + spatial_indexes_i + start_noisy_index
            flattened_noisy_frame_indexes.append(frame_offsets.flatten())
            start_noisy_index += token_shape_i[0] * spatial_numel_i
        flattened = torch.cat(flattened_noisy_frame_indexes, dim=0).unsqueeze(-1).expand(-1, packed_tokens.shape[1])
        return packed_tokens.scatter_add(dim=0, index=flattened, src=packed_timestep_embeds)

    def _prepare_text_segment(self, input_ids, device):
        und_len = len(input_ids)
        text_mrope_ids, next_mrope_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=und_len,
            temporal_offset=0,
            use_float_positions=self.enable_fps_modulation,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "text_indexes": torch.arange(und_len, dtype=torch.long, device=device),
            "und_len": und_len,
            "text_mrope_ids": text_mrope_ids.to(device),
            "vision_start_temporal_offset": next_mrope_offset + self.temporal_margin,
        }

    def _prepare_vision_segment(self, latents, text_segment, device):
        p = self.latent_patch_size
        _, _, latent_t, latent_h, latent_w = latents.shape
        patch_h = math.ceil(latent_h / p)
        patch_w = math.ceil(latent_w / p)
        num_vision_tokens = latent_t * patch_h * patch_w
        noisy_frame_indexes = torch.arange(latent_t, device=device, dtype=torch.long)
        frame_token_stride = patch_h * patch_w
        curr = text_segment["und_len"]
        mse_loss_indexes = []
        for frame_idx in noisy_frame_indexes.tolist():
            frame_start = curr + frame_idx * frame_token_stride
            mse_loss_indexes.extend(range(frame_start, frame_start + frame_token_stride))

        effective_fps = self.config.get("target_fps", 24.0) if self.enable_fps_modulation else None
        vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=text_segment["vision_start_temporal_offset"],
            reset_spatial_indices=self.reset_spatial_ids,
            fps=effective_fps,
            base_fps=self.base_fps,
            temporal_compression_factor=self.config.get("vae_scale_factor_temporal", 4),
        )
        return {
            "vision_token_shapes": [(latent_t, patch_h, patch_w)],
            "vision_sequence_indexes": torch.arange(curr, curr + num_vision_tokens, dtype=torch.long, device=device),
            "vision_mse_loss_indexes": torch.tensor(mse_loss_indexes, dtype=torch.long, device=device),
            "vision_noisy_frame_indexes": [noisy_frame_indexes],
            "vision_mrope_ids": vision_mrope_ids.to(device),
            "num_vision_tokens": num_vision_tokens,
            "num_noisy_vision_tokens": len(noisy_frame_indexes) * frame_token_stride,
        }

    def infer(self, weights, input_ids, latents, timestep):
        device = latents.device
        text_segment = self._prepare_text_segment(input_ids, device)
        vision_segment = self._prepare_vision_segment(latents, text_segment, device)
        sequence_length = text_segment["und_len"] + vision_segment["num_vision_tokens"]

        packed_text_embedding = weights.embed_tokens.apply(text_segment["input_ids"])
        target_dtype = packed_text_embedding.dtype
        hidden_states = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        hidden_states[text_segment["text_indexes"]] = packed_text_embedding

        packed_tokens_vision, original_latent_shapes = self._patchify_and_pack_latents(latents.to(dtype=target_dtype))
        packed_tokens_vision = weights.proj_in.apply(packed_tokens_vision)
        timestep = timestep.to(device=device, dtype=torch.float32) if isinstance(timestep, torch.Tensor) else torch.tensor(float(timestep), device=device, dtype=torch.float32)
        vision_timesteps = timestep.expand(vision_segment["num_noisy_vision_tokens"]) * self.timestep_scale
        timestep_proj = get_timestep_embedding(
            vision_timesteps,
            256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1,
        ).to(device=packed_tokens_vision.device, dtype=packed_tokens_vision.dtype)
        packed_timestep_embeds_vision = weights.time_embedder_linear_2.apply(F.silu(weights.time_embedder_linear_1.apply(timestep_proj))).to(target_dtype)
        packed_tokens_vision = self._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens_vision,
            packed_timestep_embeds_vision,
            vision_segment["vision_noisy_frame_indexes"],
            vision_segment["vision_token_shapes"],
        )
        hidden_states[vision_segment["vision_sequence_indexes"]] = packed_tokens_vision

        position_ids = torch.cat([text_segment["text_mrope_ids"], vision_segment["vision_mrope_ids"]], dim=1)
        return Cosmos3PreInferModuleOutput(
            hidden_states=hidden_states,
            und_len=text_segment["und_len"],
            position_ids=position_ids,
            vision_mse_loss_indexes=vision_segment["vision_mse_loss_indexes"],
            vision_token_shapes=vision_segment["vision_token_shapes"],
            vision_noisy_frame_indexes=vision_segment["vision_noisy_frame_indexes"],
            original_latent_shapes=original_latent_shapes,
        )
