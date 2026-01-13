import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, encoder_hidden_states):
        patch_size = self.config.get("patch_size", 2)
        f_patch_size = self.config.get("f_patch_size", 1)

        if hidden_states.dim() == 4:
            patch_size = self.config.get("patch_size", 2)
            hidden_states = patchify(hidden_states, patch_size=patch_size, f_patch_size=f_patch_size)

        batch_size, num_tokens, patch_dim = hidden_states.shape

        original_shape = self.scheduler.input_info.target_shape
        if len(original_shape) >= 2:
            original_height = original_shape[-2]
            original_width = original_shape[-1]
            original_frames = 1

            F_tokens = original_frames // f_patch_size
            H_tokens = original_height // patch_size
            W_tokens = original_width // patch_size

        padded_list = []
        image_pad_masks = []
        x_item_seqlens_ori = []
        for b in range(batch_size):
            x_item = hidden_states[b]
            x_ori_len = x_item.shape[0]
            x_item_seqlens_ori.append(x_ori_len)
            x_padding_len = (-x_ori_len) % SEQ_MULTI_OF

            if x_padding_len > 0:
                pad_mask = torch.cat(
                    [
                        torch.zeros((x_ori_len,), dtype=torch.bool, device=x_item.device),
                        torch.ones((x_padding_len,), dtype=torch.bool, device=x_item.device),
                    ],
                    dim=0,
                )
                x_padded = torch.cat([x_item, x_item[-1:].repeat(x_padding_len, 1)], dim=0)
                padded_list.append(x_padded)
                image_pad_masks.append(pad_mask)
            else:
                pad_mask = torch.zeros((x_ori_len,), dtype=torch.bool, device=x_item.device)
                padded_list.append(x_item)
                image_pad_masks.append(pad_mask)

        x_item_seqlens = [x.shape[0] for x in padded_list]
        x_cat = torch.cat(padded_list, dim=0)

        hidden_states_2d = weights.img_in.apply(x_cat)

        if hasattr(weights, "x_pad_token") and hasattr(weights.x_pad_token, "tensor"):
            x_pad_token = weights.x_pad_token.tensor
            x_inner_pad_mask = torch.cat(image_pad_masks, dim=0)
            hidden_states_2d[x_inner_pad_mask] = x_pad_token.squeeze(0)  # Broadcast to [D]

        hidden_states_list = list(hidden_states_2d.split(x_item_seqlens, dim=0))
        hidden_states = pad_sequence(hidden_states_list, batch_first=True, padding_value=0.0)

        if encoder_hidden_states.dim() == 3:
            pass
        elif encoder_hidden_states.dim() == 2:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
        else:
            raise ValueError(f"encoder_hidden_states must be 2D [L, D] or 3D [B, L, D], got {encoder_hidden_states.shape}")

        cap_padded_list = []
        cap_pad_masks = []
        cap_item_seqlens_ori = []
        for b in range(batch_size):
            cap_item = encoder_hidden_states[b]
            cap_ori_len = cap_item.shape[0]
            cap_item_seqlens_ori.append(cap_ori_len)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF

            if cap_padding_len > 0:
                cap_pad_mask = torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=cap_item.device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=cap_item.device),
                    ],
                    dim=0,
                )
                cap_padded = torch.cat([cap_item, cap_item[-1:].repeat(cap_padding_len, 1)], dim=0)
                cap_padded_list.append(cap_padded)
                cap_pad_masks.append(cap_pad_mask)
            else:
                cap_pad_mask = torch.zeros((cap_ori_len,), dtype=torch.bool, device=cap_item.device)
                cap_padded_list.append(cap_item)
                cap_pad_masks.append(cap_pad_mask)

        cap_item_seqlens = [x.shape[0] for x in cap_padded_list]
        cap_cat = torch.cat(cap_padded_list, dim=0)

        cap_cat = weights.txt_norm.apply(cap_cat)
        cap_cat = weights.txt_in.apply(cap_cat)

        if hasattr(weights, "cap_pad_token") and hasattr(weights.cap_pad_token, "tensor"):
            cap_pad_token = weights.cap_pad_token.tensor
            cap_inner_pad_mask = torch.cat(cap_pad_masks, dim=0)
            cap_cat[cap_inner_pad_mask] = cap_pad_token.squeeze(0)

        encoder_hidden_states_list = list(cap_cat.split(cap_item_seqlens, dim=0))
        encoder_hidden_states = pad_sequence(encoder_hidden_states_list, batch_first=True, padding_value=0.0)

        device = hidden_states.device

        x_pos_ids_list = []
        cap_pos_ids_list = []

        for b in range(batch_size):
            cap_ori_len = cap_item_seqlens_ori[b]
            cap_padded_len = cap_item_seqlens[b]
            cap_pos_ids = self.scheduler.create_coordinate_grid(
                size=(cap_padded_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            cap_pos_ids_list.append(cap_pos_ids)

            x_ori_len = x_item_seqlens_ori[b]
            x_padded_len = x_item_seqlens[b]
            image_pos_ids = self.scheduler.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_padded_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)

            if x_padded_len > x_ori_len:
                padding_pos_ids = (
                    self.scheduler.create_coordinate_grid(
                        size=(1, 1, 1),
                        start=(0, 0, 0),
                        device=device,
                    )
                    .flatten(0, 2)
                    .repeat(x_padded_len - x_ori_len, 1)
                )
                image_pos_ids = torch.cat([image_pos_ids, padding_pos_ids], dim=0)

            x_pos_ids_list.append(image_pos_ids)

        x_pos_ids_cat = torch.cat(x_pos_ids_list, dim=0)
        cap_pos_ids_cat = torch.cat(cap_pos_ids_list, dim=0)

        x_freqs_cis_cat = self.scheduler.generate_freqs_cis_from_position_ids(x_pos_ids_cat, device=device)
        cap_freqs_cis_cat = self.scheduler.generate_freqs_cis_from_position_ids(cap_pos_ids_cat, device=device)

        x_freqs_cis_list = list(x_freqs_cis_cat.split(x_item_seqlens, dim=0))
        cap_freqs_cis_list = list(cap_freqs_cis_cat.split(cap_item_seqlens, dim=0))

        x_freqs_cis = pad_sequence(x_freqs_cis_list, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis_list, batch_first=True, padding_value=0.0)

        embed0 = weights.time_text_embed_timestep_embedder_linear_1.apply(self.scheduler.timesteps_proj)

        embed0 = F.silu(embed0)
        embed0 = weights.time_text_embed_timestep_embedder_linear_2.apply(embed0)
        temb_img_silu = embed0

        if self.zero_cond_t:
            temb_txt_silu = torch.zeros_like(temb_img_silu)
        else:
            pooled_text = encoder_hidden_states.mean(dim=1)
            if pooled_text.shape[-1] != temb_img_silu.shape[-1]:
                target_dim = temb_img_silu.shape[-1]
                if pooled_text.shape[-1] > target_dim:
                    pooled_text = pooled_text[..., :target_dim]
                else:
                    padding = torch.zeros(batch_size, target_dim - pooled_text.shape[-1], device=pooled_text.device, dtype=pooled_text.dtype)
                    pooled_text = torch.cat([pooled_text, padding], dim=-1)

            temb_txt_silu = F.silu(pooled_text)

        image_tokens_len = x_item_seqlens_ori[0]

        return ZPreInferModuleOutput(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_img_silu=temb_img_silu,
            temb_txt_silu=temb_txt_silu,
            x_freqs_cis=x_freqs_cis,
            cap_freqs_cis=cap_freqs_cis,
            image_tokens_len=image_tokens_len,
            x_item_seqlens=x_item_seqlens,
            cap_item_seqlens=cap_item_seqlens,
        )
