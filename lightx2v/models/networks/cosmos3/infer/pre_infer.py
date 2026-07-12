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
        if packed_timestep_embeds.numel() == 0:
            return packed_tokens
        start_noisy_index = 0
        flattened_noisy_frame_indexes = []
        for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
            spatial_numel_i = math.prod(token_shape_i[1:])
            if len(noisy_indexes_i) > 0:
                spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)
                frame_offsets = (noisy_indexes_i * spatial_numel_i).unsqueeze(-1) + spatial_indexes_i + start_noisy_index
                flattened_noisy_frame_indexes.append(frame_offsets.flatten())
            start_noisy_index += token_shape_i[0] * spatial_numel_i
        if not flattened_noisy_frame_indexes:
            return packed_tokens
        flattened = torch.cat(flattened_noisy_frame_indexes, dim=0).unsqueeze(-1).expand(-1, packed_tokens.shape[1])
        return packed_tokens.scatter_add(dim=0, index=flattened, src=packed_timestep_embeds)

    def _embed_timestep(self, weights, timestep, length, device, dtype):
        if length == 0:
            return torch.empty((0, self.hidden_size), device=device, dtype=dtype)
        timestep = timestep.to(device=device, dtype=torch.float32) if isinstance(timestep, torch.Tensor) else torch.tensor(float(timestep), device=device, dtype=torch.float32)
        timesteps = timestep.expand(length) * self.timestep_scale
        timestep_proj = get_timestep_embedding(timesteps, 256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1).to(device=device, dtype=dtype)
        return weights.time_embedder_linear_2.apply(F.silu(weights.time_embedder_linear_1.apply(timestep_proj))).to(dtype)

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

    def _prepare_vision_segment(self, latents, text_segment, device, condition_frame_indexes=None):
        p = self.latent_patch_size
        _, _, latent_t, latent_h, latent_w = latents.shape
        patch_h = math.ceil(latent_h / p)
        patch_w = math.ceil(latent_w / p)
        num_vision_tokens = latent_t * patch_h * patch_w
        condition_frame_indexes = [] if condition_frame_indexes is None else condition_frame_indexes
        condition_frame_set = {int(idx) for idx in condition_frame_indexes if 0 <= int(idx) < latent_t}
        noisy_frame_indexes = torch.tensor([idx for idx in range(latent_t) if idx not in condition_frame_set], device=device, dtype=torch.long)
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

    def _prepare_sound_segment(self, sound_latents, text_segment, vision_segment, device):
        sound_len = int(sound_latents.shape[1])
        curr = text_segment["und_len"] + vision_segment["num_vision_tokens"]
        effective_fps = self.config.get("sound_latent_fps", 25.0) if self.enable_fps_modulation else None
        sound_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=sound_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=text_segment["vision_start_temporal_offset"],
            reset_spatial_indices=self.reset_spatial_ids,
            fps=effective_fps,
            base_fps=self.base_fps,
            temporal_compression_factor=1,
        )
        indexes = torch.arange(sound_len, device=device, dtype=torch.long)
        return {
            "sound_token_shapes": [(sound_len, 1, 1)],
            "sound_sequence_indexes": torch.arange(curr, curr + sound_len, dtype=torch.long, device=device),
            "sound_mse_loss_indexes": torch.arange(curr, curr + sound_len, dtype=torch.long, device=device),
            "sound_noisy_frame_indexes": [indexes],
            "sound_mrope_ids": sound_mrope_ids.to(device),
            "sound_len": sound_len,
        }

    def _prepare_action_segment(
        self,
        action_latents,
        text_segment,
        vision_segment,
        sound_segment,
        device,
        condition_frame_indexes=None,
        start_frame_offset=1,
    ):
        action_len = int(action_latents.shape[0])
        condition_frame_indexes = [] if condition_frame_indexes is None else condition_frame_indexes
        cond_set = {int(idx) for idx in condition_frame_indexes if 0 <= int(idx) < action_len}
        noisy_frame_indexes = torch.tensor([idx for idx in range(action_len) if idx not in cond_set], device=device, dtype=torch.long)
        curr = text_segment["und_len"] + vision_segment["num_vision_tokens"] + sound_segment.get("sound_len", 0)
        effective_fps = self.config.get("target_fps", 24.0) if self.enable_fps_modulation else None
        action_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=action_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=text_segment["vision_start_temporal_offset"],
            reset_spatial_indices=self.reset_spatial_ids,
            fps=effective_fps,
            base_fps=self.base_fps,
            temporal_compression_factor=1,
            base_temporal_compression_factor=self.config.get("vae_scale_factor_temporal", 4),
            start_frame_offset=int(start_frame_offset),
        )
        sequence_indexes = torch.arange(curr, curr + action_len, dtype=torch.long, device=device)
        return {
            "action_token_shapes": [(action_len, 1, 1)],
            "action_sequence_indexes": sequence_indexes,
            "action_mse_loss_indexes": sequence_indexes[noisy_frame_indexes],
            "action_noisy_frame_indexes": [noisy_frame_indexes],
            "action_mrope_ids": action_mrope_ids.to(device),
            "action_len": action_len,
            "num_noisy_action_tokens": len(noisy_frame_indexes),
        }

    def infer(
        self,
        weights,
        input_ids,
        latents,
        timestep,
        condition_frame_indexes=None,
        sound_latents=None,
        action_latents=None,
        action_domain_id=None,
        action_condition_frame_indexes=None,
        action_start_frame_offset=1,
        raw_action_dim=None,
    ):
        device = latents.device
        text_segment = self._prepare_text_segment(input_ids, device)
        vision_segment = self._prepare_vision_segment(latents, text_segment, device, condition_frame_indexes=condition_frame_indexes)
        sound_segment = {}
        action_segment = {}
        if sound_latents is not None:
            sound_segment = self._prepare_sound_segment(sound_latents, text_segment, vision_segment, device)
        if action_latents is not None:
            action_segment = self._prepare_action_segment(
                action_latents,
                text_segment,
                vision_segment,
                sound_segment,
                device,
                condition_frame_indexes=action_condition_frame_indexes,
                start_frame_offset=action_start_frame_offset,
            )
        sequence_length = text_segment["und_len"] + vision_segment["num_vision_tokens"] + sound_segment.get("sound_len", 0) + action_segment.get("action_len", 0)

        packed_text_embedding = weights.embed_tokens.apply(text_segment["input_ids"])
        target_dtype = packed_text_embedding.dtype
        hidden_states = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        hidden_states[text_segment["text_indexes"]] = packed_text_embedding

        packed_tokens_vision, original_latent_shapes = self._patchify_and_pack_latents(latents.to(dtype=target_dtype))
        packed_tokens_vision = weights.proj_in.apply(packed_tokens_vision)
        packed_timestep_embeds_vision = self._embed_timestep(weights, timestep, vision_segment["num_noisy_vision_tokens"], packed_tokens_vision.device, packed_tokens_vision.dtype)
        packed_tokens_vision = self._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens_vision,
            packed_timestep_embeds_vision,
            vision_segment["vision_noisy_frame_indexes"],
            vision_segment["vision_token_shapes"],
        )
        hidden_states[vision_segment["vision_sequence_indexes"]] = packed_tokens_vision

        if sound_latents is not None:
            packed_tokens_sound = sound_latents[:, : sound_segment["sound_len"]].permute(1, 0).to(dtype=target_dtype)
            packed_tokens_sound = weights.audio_proj_in.apply(packed_tokens_sound) + weights.audio_modality_embed.tensor.to(device=device, dtype=target_dtype)
            packed_timestep_embeds_sound = self._embed_timestep(weights, timestep, sound_segment["sound_len"], packed_tokens_sound.device, packed_tokens_sound.dtype)
            packed_tokens_sound = self._apply_timestep_embeds_to_noisy_tokens(
                packed_tokens_sound,
                packed_timestep_embeds_sound,
                sound_segment["sound_noisy_frame_indexes"],
                sound_segment["sound_token_shapes"],
            )
            hidden_states[sound_segment["sound_sequence_indexes"]] = packed_tokens_sound

        action_domain_ids = None
        if action_latents is not None:
            action_domain_id = torch.as_tensor(action_domain_id, device=device, dtype=torch.long).reshape(1)
            action_domain_ids = action_domain_id.expand(action_segment["action_len"])
            packed_tokens_action = action_latents[: action_segment["action_len"]].to(dtype=target_dtype)
            packed_tokens_action = weights.action_proj_in.apply(packed_tokens_action, action_domain_ids)
            packed_tokens_action = packed_tokens_action + weights.action_modality_embed.tensor.to(device=device, dtype=target_dtype)
            packed_timestep_embeds_action = self._embed_timestep(weights, timestep, action_segment["num_noisy_action_tokens"], packed_tokens_action.device, packed_tokens_action.dtype)
            packed_tokens_action = self._apply_timestep_embeds_to_noisy_tokens(
                packed_tokens_action,
                packed_timestep_embeds_action,
                action_segment["action_noisy_frame_indexes"],
                action_segment["action_token_shapes"],
            )
            hidden_states[action_segment["action_sequence_indexes"]] = packed_tokens_action

        mrope_segments = [text_segment["text_mrope_ids"], vision_segment["vision_mrope_ids"]]
        if sound_segment:
            mrope_segments.append(sound_segment["sound_mrope_ids"])
        if action_segment:
            mrope_segments.append(action_segment["action_mrope_ids"])
        position_ids = torch.cat(mrope_segments, dim=1)
        return Cosmos3PreInferModuleOutput(
            hidden_states=hidden_states,
            und_len=text_segment["und_len"],
            position_ids=position_ids,
            vision_mse_loss_indexes=vision_segment["vision_mse_loss_indexes"],
            vision_token_shapes=vision_segment["vision_token_shapes"],
            vision_noisy_frame_indexes=vision_segment["vision_noisy_frame_indexes"],
            original_latent_shapes=original_latent_shapes,
            sound_mse_loss_indexes=sound_segment.get("sound_mse_loss_indexes"),
            sound_token_shapes=sound_segment.get("sound_token_shapes"),
            sound_noisy_frame_indexes=sound_segment.get("sound_noisy_frame_indexes"),
            action_mse_loss_indexes=action_segment.get("action_mse_loss_indexes"),
            action_token_shapes=action_segment.get("action_token_shapes"),
            action_noisy_frame_indexes=action_segment.get("action_noisy_frame_indexes"),
            action_domain_ids=action_domain_ids,
            raw_action_dim=raw_action_dim,
        )
