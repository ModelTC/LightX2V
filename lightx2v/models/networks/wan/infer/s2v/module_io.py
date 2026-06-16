from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from lightx2v.models.networks.wan.infer.module_io import WanPreInferModuleOutput


@dataclass
class WanS2VPreInferModuleOutput(WanPreInferModuleOutput):
    seq_lens: Optional[torch.Tensor] = None
    freqs: Optional[torch.Tensor] = None
    original_seq_len: int = 0
    merged_audio_emb: Optional[torch.Tensor] = None
    audio_emb_global: Optional[torch.Tensor] = None
    s2v_extra: Dict[str, Any] = field(default_factory=dict)
