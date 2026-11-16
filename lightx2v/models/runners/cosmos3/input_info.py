from dataclasses import dataclass, field
from typing import Any

from lightx2v.utils.input_info import ImageState


@dataclass
class Cosmos3State(ImageState):
    audio_latent_shape: list = field(default_factory=list)
    action_chunk_size: int | None = None
    vision_condition_latents: Any = field(default=None, repr=False)
    vision_condition_frame_indexes: list[int] | None = None
    action_latents: Any = field(default=None, repr=False)
    action_latent_shape: tuple[int, ...] | None = None
    action_condition_frame_indexes: list[int] | None = None
    action_domain_id: int | None = None
    raw_action_dim: int | None = None
    action_start_frame_offset: int = 1
