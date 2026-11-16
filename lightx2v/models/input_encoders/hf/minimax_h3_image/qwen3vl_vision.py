"""Image-specific Qwen3-VL interpolation matching current Transformers."""

import math

import torch
import transformers

try:
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights
except ImportError as exc:
    raise ImportError(
        "MiniMax-H3-Image requires transformers>=5.15.0 with "
        "'get_vision_interpolation_indices_and_weights'. "
        f"Installed version: {transformers.__version__}. "
        'Upgrade with: python -m pip install -U "transformers>=5.15.0"'
    ) from exc

from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_vision import MiniMaxH3Qwen3VLVisionTower, _vision_cu_seqlens, _vision_position_ids


class MiniMaxH3ImageQwen3VLVisionTower(MiniMaxH3Qwen3VLVisionTower):
    def forward(self, pixels, grid_thw):
        grid_thw = grid_thw.to(device=pixels.device)
        # Match HF's float32 interpolation; the AV tower uses a different grid.
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=int(math.sqrt(self.config["num_position_embeddings"])),
            mode="bilinear",
            align_corners=True,
            spatial_merge_size=self.spatial_merge_size,
        )
        position_ids = _vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = _vision_cu_seqlens(grid_thw)
        hidden_states = self.patch_embed(pixels)
        # HF accumulates positions in float32 before casting to the patch dtype.
        pos_embed = (self.pos_embed(indices) * weights[:, :, None]).sum(1)
        hidden_states = hidden_states + pos_embed.to(hidden_states.dtype)
        rotary = (position_ids.unsqueeze(-1) * self.rotary_inv_freq.to(position_ids.device)).flatten(1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        cos, sin = rotary.cos(), rotary.sin()
        deepstack = []
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, cos, sin)
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack.append(self.deepstack_merger_list[merger_index](hidden_states))
        return self.merger(hidden_states), deepstack
