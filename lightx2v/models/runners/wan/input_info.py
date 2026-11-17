from dataclasses import dataclass, field
from typing import Any

from lightx2v.utils.input_info import ShapeState


@dataclass
class WanAudioState(ShapeState):
    audio_num: int = 0
    with_mask: bool = False
    stream_config: dict = field(default_factory=dict)
    overlap_frame: Any = None
    overlap_latent: Any = None
    audio_clip: Any = None
    person_mask_latens: Any = field(default=None, repr=False)
    ref_state: int = 0
    is_first: bool = False
    is_last: bool = False


@dataclass
class WanPolicyState(ShapeState):
    policy_image: Any = field(default=None, repr=False)
    policy_state: Any = field(default=None, repr=False)
