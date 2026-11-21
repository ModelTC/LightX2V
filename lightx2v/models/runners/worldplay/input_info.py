from dataclasses import dataclass
from typing import Any

from lightx2v.utils.input_info import ShapeState


@dataclass
class WorldPlayState(ShapeState):
    model_type: str = "ar"
    chunk_latent_frames: int = 4
    viewmats: Any = None
    Ks: Any = None
    action: Any = None
