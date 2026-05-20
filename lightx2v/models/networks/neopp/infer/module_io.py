from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NeoppPreInferModuleOutput:
    image_embeds: torch.Tensor
    t: torch.Tensor
    z: torch.Tensor
    image_token_num: int
    timestep_embeddings: torch.Tensor
    # 各 pass 专属的 image_embeds（seq_parallel 时为本 rank 的 shard，非 seq_parallel 时与 image_embeds 相同）
    # 在 _infer_t2i_i2i 中预计算，避免 _infer_pass 中反复 chunk/restore
    image_embeds_cond: Optional[torch.Tensor] = None
    image_embeds_uncond: Optional[torch.Tensor] = None
