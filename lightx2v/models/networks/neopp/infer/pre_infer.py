import math

import torch
import torch.nn.functional as F

from lightx2v.models.networks.neopp.infer.module_io import NeoppPreInferModuleOutput
from lightx2v.utils.envs import *


def precompute_rope_freqs_sincos(dim: int, max_position: int, base: float = 10000.0, device=None):
    """预计算 RoPE 的 cos 和 sin 值 (1D)。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_position, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def build_abs_positions_from_grid_hw(grid_hw: torch.Tensor, device=None):
    """
    Compute patch coordinates (x, y)

    Args:
        grid_hw: (B, 2) tensor representing (H, W) per image
    """
    device = grid_hw.device
    B = grid_hw.shape[0]

    # Get the number of patches per image
    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    # Create the batch index for each patch (B x patch count)
    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)  # (N_total,)

    # Generate intra-image patch index (row-major order)
    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = patch_id_within_image - torch.cumsum(torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0)[patch_to_sample]

    # Get H/W for each patch according to its image
    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y


def apply_rotary_emb_1d(
    x: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
    positions: torch.Tensor,
):
    """对输入张量的一部分应用1D RoPE。"""
    # x: (..., seq_len, dim_part)
    # positions: (..., seq_len)
    # cos_cached: (max_pos, dim_part / 2)

    cos = cos_cached[positions]  # Shape: (positions.shape, dim_part / 2)
    sin = sin_cached[positions]  # Shape: (positions.shape, dim_part / 2)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = rotated_x1
    x_rotated[..., 1::2] = rotated_x2
    return x_rotated


def apply_2d_rotary_pos_emb(
    x: torch.Tensor, cos_cached_x: torch.Tensor, sin_cached_x: torch.Tensor, cos_cached_y: torch.Tensor, sin_cached_y: torch.Tensor, abs_positions_x: torch.Tensor, abs_positions_y: torch.Tensor
):
    """应用2D RoPE到输入张量x。"""
    dim = x.shape[-1]
    dim_half = dim // 2

    # 假设我们将embedding的前半部分用于一个方向的RoPE，后半部分用于另一个方向
    # 例如，前一半给X坐标，后一半给Y坐标 (或者反过来，但要保持一致)
    x_part_1 = x[..., :dim_half]
    x_part_2 = x[..., dim_half:]

    # 将与 abs_positions_x 相关的旋转应用于 x_part_1
    rotated_part_1 = apply_rotary_emb_1d(x_part_1, cos_cached_x, sin_cached_x, abs_positions_x)
    # 将与 abs_positions_y 相关的旋转应用于 x_part_2
    rotated_part_2 = apply_rotary_emb_1d(x_part_2, cos_cached_y, sin_cached_y, abs_positions_y)

    # 将它们重新拼接起来。确保顺序与你分割时一致。
    return torch.cat((rotated_part_1, rotated_part_2), dim=-1)


class NeoppPreInfer:
    def __init__(self, config):
        self.config = config
        self.patch_size = self.config.get("patch_size", 16)
        self.merge_size = 2
        self.embed_dim = config["vision_config"]["hidden_size"]
        self.add_noise_scale_embedding = config.get("add_noise_scale_embedding", True)
        self.noise_scale_max_value = config["noise_scale_max_value"]
        self.frequency_embedding_size = 256
        self.rope_dim_part = self.embed_dim // 2
        self.cos_cached_x, self.sin_cached_x = precompute_rope_freqs_sincos(
            self.rope_dim_part, config["vision_config"]["max_position_embeddings_vision"], base=config["vision_config"]["rope_theta_vision"], device=None
        )
        self.cos_cached_y, self.sin_cached_y = precompute_rope_freqs_sincos(
            self.rope_dim_part, config["vision_config"]["max_position_embeddings_vision"], base=config["vision_config"]["rope_theta_vision"], device=None
        )

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights):
        t = self.scheduler.timesteps[self.scheduler.step_index]

        image_prediction = self.scheduler.image_prediction

        token_h = image_prediction.shape[2] // (self.patch_size * self.merge_size)
        token_w = image_prediction.shape[3] // (self.patch_size * self.merge_size)

        z = self.patchify(image_prediction, self.patch_size * self.merge_size)
        image_input = self.patchify(image_prediction, self.patch_size, channel_first=True)

        image_input_reshaped = image_input.view(image_input.shape[0] * image_input.shape[1], -1)

        image_embeds = self.extract_feature(weights, image_input_reshaped, grid_hw=self.scheduler.grid_hw).view(1, token_h * token_w, -1)

        t_expanded = t.expand(token_h * token_w)
        timestep_embeddings = self.timestep_embedder(weights, t_expanded).view(1, token_h * token_w, -1)
        if self.add_noise_scale_embedding:
            noise_scale_tensor = torch.full_like(t_expanded, self.scheduler.noise_scale / self.noise_scale_max_value)
            noise_embeddings = self.noise_scale_embedder(weights, noise_scale_tensor).view(1, token_h * token_w, -1)
            timestep_embeddings += noise_embeddings
        image_embeds = image_embeds + timestep_embeddings

        return NeoppPreInferModuleOutput(image_embeds=image_embeds, t=t, z=z, image_token_num=token_h * token_w, timestep_embeddings=timestep_embeddings)

    def patchify(self, images, patch_size, channel_first=False):
        """
        images: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
        x = images.reshape(shape=(images.shape[0], 3, h, patch_size, w, patch_size))

        if channel_first:
            x = torch.einsum("nchpwq->nhwcpq", x)
        else:
            x = torch.einsum("nchpwq->nhwpqc", x)

        x = x.reshape(shape=(images.shape[0], h * w, patch_size**2 * 3))
        return x

    def _apply_2d_rotary_pos_emb(self, patch_embeds, grid_hw):
        """
        Apply 2D Rotary Position Embedding to the patch embeddings.
        """
        abs_pos_x, abs_pos_y = build_abs_positions_from_grid_hw(grid_hw, device=patch_embeds.device)
        embeddings = apply_2d_rotary_pos_emb(
            patch_embeds.to(torch.float32),  # RoPE calculations are often more stable in float32
            self.cos_cached_x,
            self.sin_cached_x,
            self.cos_cached_y,
            self.sin_cached_y,
            abs_pos_x,
            abs_pos_y,
        ).to(torch.bfloat16)
        return embeddings

    def extract_feature(self, weights, pixel_values: torch.FloatTensor, grid_hw=None) -> torch.Tensor:
        pixel_values = pixel_values.view(  #
            -1,
            3,
            self.patch_size,
            self.patch_size,
        )  #  [28072, 768] -> [28072, 3, 16, 16]
        patch_embeds = F.gelu(weights.vision_model_mot_gen_patch_embedding.apply(pixel_values)).view(-1, self.embed_dim)
        self.cos_cached_x = self.cos_cached_x.to(patch_embeds.device)
        self.sin_cached_x = self.sin_cached_x.to(patch_embeds.device)
        self.cos_cached_y = self.cos_cached_y.to(patch_embeds.device)
        self.sin_cached_y = self.sin_cached_y.to(patch_embeds.device)
        patch_embeds = self._apply_2d_rotary_pos_emb(patch_embeds, grid_hw)  # [28072, 1024]
        assert (grid_hw[:, 0] * grid_hw[:, 1]).sum() == patch_embeds.shape[0]

        patches_list = []
        cur_position = 0
        for i in range(grid_hw.shape[0]):
            h, w = grid_hw[i]
            patches_per_img = patch_embeds[cur_position : cur_position + h * w].view(h, w, -1).unsqueeze(0)
            patches_per_img = weights.vision_model_mot_gen_dense_embedding.apply(patches_per_img.permute(0, 3, 1, 2))
            patches_per_img = patches_per_img.permute(0, 2, 3, 1)
            patches_list.append(patches_per_img.view(-1, patches_per_img.shape[-1]))
            cur_position += h * w

        embeddings = torch.cat(patches_list, dim=0)  # (N_total // downsample_factor**2, C)

        assert cur_position == patch_embeds.shape[0]
        # assert embeddings.shape[0] == int(patch_embeds.shape[0] / self.downsample_factor**2)

        return embeddings

    def timestep_embedding(self, t: torch.Tensor, dim: int, max_period: float = 10000.0):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element. These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def timestep_embedder(self, weights, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(torch.bfloat16)
        # t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        t_emb = weights.timestep_embedder_mlp_0.apply(t_freq)
        t_emb = F.silu(t_emb)
        t_emb = weights.timestep_embedder_mlp_2.apply(t_emb)
        return t_emb

    def noise_scale_embedder(self, weights, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(torch.bfloat16)
        # t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        t_emb = weights.noise_scale_embedder_mlp_0.apply(t_freq)
        t_emb = F.silu(t_emb)
        t_emb = weights.noise_scale_embedder_mlp_2.apply(t_emb)
        return t_emb
