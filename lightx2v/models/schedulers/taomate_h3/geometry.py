"""Five-second TaoMate-H3 streaming geometry and absolute RoPE clocks.

Adapted from the pinned TaoMate-H3 direct inference contract. A continuation
advances 119 native frames; it does not regenerate the first request's affine
video prefix.
"""

from dataclasses import dataclass
from fractions import Fraction

_GROUP_FRAMES = 17
_GROUP_VIDEO_LATENTS = 5
_AUDIO_LATENTS_PER_SECOND = 40
_FPS = 24


def _audio_boundary(frame):
    return round(Fraction(frame * _AUDIO_LATENTS_PER_SECOND, _FPS))


def video_time(latent_index):
    full_groups, remainder = divmod(latent_index, 5)
    return Fraction(85 * full_groups + 5 * sum((1, 4, 4, 4, 4)[:remainder]), 3)


@dataclass(frozen=True)
class TaoMatePhase:
    index: int
    frame_start: int
    frame_stop: int
    video_start: int
    video_stop: int
    audio_start: int
    audio_stop: int

    @property
    def video_latents(self):
        return self.video_stop - self.video_start

    @property
    def audio_latents(self):
        return self.audio_stop - self.audio_start


@dataclass(frozen=True)
class TaoMateRequestPlan:
    request_index: int
    native_frames: int
    phases: tuple[TaoMatePhase, ...]

    @property
    def video_latents(self):
        return self.phases[-1].video_stop

    @property
    def audio_latents(self):
        return self.phases[-1].audio_stop


def request_plan(request_index):
    """Return the four native chunks for one five-second request."""

    if type(request_index) is not int or request_index < 0:
        raise ValueError("request_index must be a non-negative integer")
    first = request_index == 0
    global_start = 0 if first else 124 + (request_index - 1) * 119
    global_audio_start = _audio_boundary(global_start)
    frame = video = 0
    phases = []
    for index, groups in enumerate((2, 2, 2, 1)):
        next_frame = frame + groups * _GROUP_FRAMES + (5 if first and index == 0 else 0)
        next_video = video + groups * _GROUP_VIDEO_LATENTS + (2 if first and index == 0 else 0)
        phases.append(
            TaoMatePhase(
                index=index,
                frame_start=frame,
                frame_stop=next_frame,
                video_start=video,
                video_stop=next_video,
                audio_start=_audio_boundary(global_start + frame) - global_audio_start,
                audio_stop=_audio_boundary(global_start + next_frame) - global_audio_start,
            )
        )
        frame, video = next_frame, next_video
    return TaoMateRequestPlan(request_index, frame, tuple(phases))


__all__ = ["TaoMatePhase", "TaoMateRequestPlan", "request_plan", "video_time"]
