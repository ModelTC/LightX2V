import torch
import torch.nn.functional as F

from lightx2v.utils.envs import *

from .module_io import ZPreInferModuleOutput
from .utils import patchify

# Official Z-Image uses SEQ_MULTI_OF = 32 for padding
SEQ_MULTI_OF = 32


class ZImagePreInfer:
    def __init__(self, config):
        self.config = config
        self.attention_kwargs = {}
        self.cpu_offload = config.get("cpu_offload", False)
        self.zero_cond_t = config.get("zero_cond_t", False)
        self.rope = None
        self._rope_cache = {}

    def set_rope(self, rope):
        self.rope = rope
        self._rope_cache.clear()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self._rope_cache.clear()

    @staticmethod
    def _device_key(device):
        return device.type, device.index

    def _prepare_rope_cache(
        self,
        device,
        f_tokens,
        h_tokens,
        w_tokens,
        x_ori_len,
        x_padded_len,
        cap_padded_len,
    ):
        if self.rope is None:
            raise RuntimeError("ZImagePreInfer RoPE is not initialized.")

        world_size = 1
        rank = 0
        if self.config["seq_parallel"]:
            seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
            world_size = torch.distributed.get_world_size(seq_p_group)
            rank = torch.distributed.get_rank(seq_p_group)

        cache_key = (
            self._device_key(device),
            f_tokens,
            h_tokens,
            w_tokens,
            x_ori_len,
            x_padded_len,
            cap_padded_len,
            world_size,
            rank,
        )
        cached = self._rope_cache.get(cache_key)
        if cached is not None:
            return cached

        cap_pos_ids = self.scheduler.create_coordinate_grid(
            size=(cap_padded_len, 1, 1),
            start=(1, 0, 0),
            device=device,
        ).flatten(0, 2)
        image_pos_ids = self.scheduler.create_coordinate_grid(
            size=(f_tokens, h_tokens, w_tokens),
            start=(cap_padded_len + 1, 0, 0),
            device=device,
        ).flatten(0, 2)
        if x_padded_len > x_ori_len:
            padding_pos_ids = torch.zeros(
                (x_padded_len - x_ori_len, image_pos_ids.shape[-1]),
                dtype=image_pos_ids.dtype,
                device=device,
            )
            image_pos_ids = torch.cat([image_pos_ids, padding_pos_ids], dim=0)

        x_freqs_cis = self.scheduler.generate_freqs_cis_from_position_ids(image_pos_ids, device=device)
        cap_freqs_cis = self.scheduler.generate_freqs_cis_from_position_ids(cap_pos_ids, device=device)

        if world_size > 1:
            padding_size = (-x_freqs_cis.shape[0]) % world_size
            if padding_size:
                x_freqs_cis = F.pad(x_freqs_cis, (0, 0, 0, padding_size))
            x_freqs_cis = torch.chunk(x_freqs_cis, world_size, dim=0)[rank]

        rotary_dim = sum(self.config.get("axes_dims", [32, 48, 48]))
        x_freqs_cis = self.rope.prepare_freqs(x_freqs_cis, rotary_dim=rotary_dim)
        cap_freqs_cis = self.rope.prepare_freqs(cap_freqs_cis, rotary_dim=rotary_dim)
        unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=0)
        cached = (
            x_freqs_cis,
            cap_freqs_cis,
            unified_freqs_cis,
            self.rope.prepare_positions(x_freqs_cis),
            self.rope.prepare_positions(cap_freqs_cis),
            self.rope.prepare_positions(unified_freqs_cis),
        )
        self._rope_cache[cache_key] = cached
        return cached

    def infer(self, weights, hidden_states, encoder_hidden_states):
        patch_size = self.config.get("patch_size", 2)
        f_patch_size = self.config.get("f_patch_size", 1)

        hidden_states = patchify(hidden_states, patch_size=patch_size, f_patch_size=f_patch_size).squeeze(0)

        num_tokens, patch_dim = hidden_states.shape

        original_shape = self.scheduler.input_info.target_shape
        if len(original_shape) >= 2:
            original_height = original_shape[-2]
            original_width = original_shape[-1]
            original_frames = 1

            F_tokens = original_frames // f_patch_size
            H_tokens = original_height // patch_size
            W_tokens = original_width // patch_size

        # Process image tokens (single sample, no batch)
        x_ori_len = hidden_states.shape[0]
        x_padding_len = (-x_ori_len) % SEQ_MULTI_OF

        if x_padding_len > 0:
            x_pad_mask = torch.cat(
                [
                    torch.zeros((x_ori_len,), dtype=torch.bool, device=hidden_states.device),
                    torch.ones((x_padding_len,), dtype=torch.bool, device=hidden_states.device),
                ],
                dim=0,
            )
            x_padded = torch.cat([hidden_states, hidden_states[-1:].repeat(x_padding_len, 1)], dim=0)
        else:
            x_pad_mask = torch.zeros((x_ori_len,), dtype=torch.bool, device=hidden_states.device)
            x_padded = hidden_states

        x_padded_len = x_padded.shape[0]
        hidden_states = weights.img_in.apply(x_padded)  # [L, D]

        if hasattr(weights, "x_pad_token") and hasattr(weights.x_pad_token, "tensor"):
            x_pad_token = weights.x_pad_token.tensor
            # Handle both [1, D] and [D] formats
            if x_pad_token.dim() == 2:
                x_pad_token = x_pad_token.squeeze(0)  # [D]
            hidden_states[x_pad_mask] = x_pad_token

        # Process encoder hidden states (single sample, no batch)
        # Remove batch dimension if present: [B, L, D] -> [L, D]
        if encoder_hidden_states.dim() == 3:
            encoder_hidden_states = encoder_hidden_states.squeeze(0)
        elif encoder_hidden_states.dim() != 2:
            raise ValueError(f"encoder_hidden_states must be 2D [L, D] or 3D [B, L, D], got {encoder_hidden_states.shape}")

        cap_ori_len = encoder_hidden_states.shape[0]
        cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF

        if cap_padding_len > 0:
            cap_pad_mask = torch.cat(
                [
                    torch.zeros((cap_ori_len,), dtype=torch.bool, device=encoder_hidden_states.device),
                    torch.ones((cap_padding_len,), dtype=torch.bool, device=encoder_hidden_states.device),
                ],
                dim=0,
            )
            cap_padded = torch.cat([encoder_hidden_states, encoder_hidden_states[-1:].repeat(cap_padding_len, 1)], dim=0)
        else:
            cap_pad_mask = torch.zeros((cap_ori_len,), dtype=torch.bool, device=encoder_hidden_states.device)
            cap_padded = encoder_hidden_states

        cap_padded_len = cap_padded.shape[0]
        encoder_hidden_states = weights.txt_norm.apply(cap_padded)  # [L, D]
        encoder_hidden_states = weights.txt_in.apply(encoder_hidden_states)  # [L, D]

        if hasattr(weights, "cap_pad_token") and hasattr(weights.cap_pad_token, "tensor"):
            cap_pad_token = weights.cap_pad_token.tensor
            # Handle both [1, D] and [D] formats
            if cap_pad_token.dim() == 2:
                cap_pad_token = cap_pad_token.squeeze(0)  # [D]
            encoder_hidden_states[cap_pad_mask] = cap_pad_token

        (
            x_freqs_cis,
            cap_freqs_cis,
            unified_freqs_cis,
            x_rope_positions,
            cap_rope_positions,
            unified_rope_positions,
        ) = self._prepare_rope_cache(
            hidden_states.device,
            F_tokens,
            H_tokens,
            W_tokens,
            x_ori_len,
            x_padded_len,
            cap_padded_len,
        )

        embed0 = weights.time_text_embed_timestep_embedder_linear_1.apply(self.scheduler.timesteps_proj)
        embed0 = F.silu(embed0)
        temb_img_silu = weights.time_text_embed_timestep_embedder_linear_2.apply(embed0)

        if self.zero_cond_t:
            temb_txt_silu = torch.zeros_like(temb_img_silu)  # [D]
        else:
            # encoder_hidden_states is [L, D], mean over sequence length
            pooled_text = encoder_hidden_states.mean(dim=0)  # [D]

            if pooled_text.shape[-1] != temb_img_silu.shape[-1]:
                target_dim = temb_img_silu.shape[-1]
                if pooled_text.shape[-1] > target_dim:
                    pooled_text = pooled_text[..., :target_dim]
                else:
                    padding = torch.zeros(target_dim - pooled_text.shape[-1], device=pooled_text.device, dtype=pooled_text.dtype)
                    pooled_text = torch.cat([pooled_text, padding], dim=-1)

            temb_txt_silu = F.silu(pooled_text)  # [D]

        image_tokens_len = x_ori_len

        return ZPreInferModuleOutput(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_img_silu=temb_img_silu,
            temb_txt_silu=temb_txt_silu,
            x_freqs_cis=x_freqs_cis,
            cap_freqs_cis=cap_freqs_cis,
            unified_freqs_cis=unified_freqs_cis,
            x_rope_positions=x_rope_positions,
            cap_rope_positions=cap_rope_positions,
            unified_rope_positions=unified_rope_positions,
            image_tokens_len=image_tokens_len,
            x_item_seqlens=[x_padded_len],
            cap_item_seqlens=[cap_padded_len],
        )
