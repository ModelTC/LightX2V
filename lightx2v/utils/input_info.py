from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Optional

import torch


class _UnsetType:
    def __repr__(self):
        return "UNSET"


UNSET = _UnsetType()


@dataclass
class InputInfo:
    """Mutable context shared by the runner, models, and schedulers for one inference.

    Startup defaults and request values initialize the context. The inference
    pipeline then adds derived state to the same object.

    ``size`` is always ``[height, width]`` in pixels. Model tensor
    dimensions belong in ``latent_shape``.
    """

    task: str = ""
    seed: int = 0
    save_result_path: Optional[str] = None
    return_result_tensor: bool = False

    def update(self, values: Mapping[str, Any]) -> None:
        for input_field in fields(self):
            if input_field.name in values:
                setattr(self, input_field.name, values[input_field.name])


@dataclass
class T2VInputInfo(InputInfo):
    prompt: str = ""
    negative_prompt: str = ""
    # shape related
    num_frames: Optional[int] = None
    latent_shape: list = field(default_factory=list)
    size: list = field(default_factory=list)


@dataclass
class I2VInputInfo(InputInfo):
    prompt: str = ""
    negative_prompt: str = ""
    image_path: str = ""
    # shape related
    num_frames: Optional[int] = None
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    size: list = field(default_factory=list)


@dataclass
class ActionI2VInputInfo(I2VInputInfo):
    pose: Optional[str] = None
    action_path: str = ""


@dataclass
class MotusInputInfo(I2VInputInfo):
    state_path: str = ""
    save_action_path: str = ""


@dataclass
class SRInputInfo(InputInfo):
    image_path: str = ""  # Single image input
    video_path: str = ""  # Video input for SR
    sr_ratio: float = 2.0
    # shape related
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    size: list = field(default_factory=list)
    output_fps: Optional[float] = field(default=None, repr=False)


@dataclass
class Flf2vInputInfo(InputInfo):
    prompt: str = ""
    negative_prompt: str = ""
    image_path: str = ""
    last_frame_path: str = ""
    # shape related
    num_frames: Optional[int] = None
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    size: list = field(default_factory=list)


@dataclass
class VaceInputInfo(InputInfo):
    prompt: str = ""
    negative_prompt: str = ""
    ref_image_paths: Optional[str] = None
    video_path: Optional[str] = None
    mask_path: Optional[str] = None
    # shape related
    num_frames: Optional[int] = None
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    size: list = field(default_factory=list)


@dataclass
class S2VInputInfo(InputInfo):
    prompt: str = ""
    negative_prompt: str = ""
    image_path: str = ""
    video_path: str = ""
    audio_path: str = ""
    pose_video_path: str = ""
    audio_num: int = 0
    with_mask: bool = False
    stream_config: dict = field(default_factory=dict)
    # shape related
    fixed_area: str = ""
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    size: list = field(default_factory=list)
    num_frames: Optional[int] = None
    video_duration: Optional[float] = None

    # prev info
    overlap_frame: Optional[torch.Tensor] = None
    overlap_latent: Optional[torch.Tensor] = None
    # input preprocess audio
    audio_clip: Optional[torch.Tensor] = None


@dataclass
class RS2VInputInfo(InputInfo):
    prompt: str = ""
    negative_prompt: str = ""
    image_path: str = ""
    audio_path: str = ""
    audio_num: int = 0
    with_mask: bool = False
    stream_config: dict = field(default_factory=dict)
    # shape related
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    size: list = field(default_factory=list)
    num_frames: Optional[int] = None
    video_duration: Optional[float] = None

    # prev info
    overlap_frame: Optional[torch.Tensor] = None
    overlap_latent: Optional[torch.Tensor] = None
    # input preprocess audio
    audio_clip: Optional[torch.Tensor] = None
    # input reference state
    ref_state: int = 0
    # flags for first and last clip
    is_first: bool = False
    is_last: bool = False


@dataclass
class AnimateInputInfo(InputInfo):
    prompt: str = ""
    ref_video_prompt: str = "人物动作的参考视频"
    negative_prompt: str = ""
    image_path: str = ""
    pose_video_path: str = ""
    face_video_path: str = ""
    ref_image_paths: str = ""
    video_path: str = ""
    background_video_path: str = ""
    mask_path: str = ""
    # shape related
    num_frames: Optional[int] = None
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    size: list = field(default_factory=list)


@dataclass
class T2IInputInfo(InputInfo):
    prompt: str = ""
    negative_prompt: str = ""
    # shape related
    size: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    image_shapes: list = field(default_factory=list)
    txt_seq_lens: list = field(default_factory=list)  # [postive_txt_seq_len, negative_txt_seq_len]
    aspect_ratio: str = ""
    latent_image_ids: Any = field(default=None, repr=False)
    txt_ids: Optional[torch.Tensor] = field(default=None, repr=False)
    revised_prompts: Any = field(default=None, repr=False)


@dataclass
class NeoppInputInfo(InputInfo):
    seed: Optional[int] = 0
    size: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)


@dataclass
class T2TInputInfo(InputInfo):
    prompt: str = ""
    max_new_tokens: Optional[int] = None
    text_do_sample: Optional[bool] = None
    text_temperature: Optional[float] = None
    text_top_k: Optional[int] = None
    text_top_p: Optional[float] = None
    bot_task: Optional[str] = None
    system_prompt: Optional[str] = None
    stream_callback: Any = None


@dataclass
class TI2TInputInfo(T2TInputInfo):
    image_path: str = ""
    align_image_size: Optional[bool] = None


@dataclass
class I2IInputInfo(InputInfo):
    prompt: str = ""
    negative_prompt: str = ""
    image_path: str = ""
    i2i_denoise_strength: Optional[float] = None
    # shape related
    size: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    image_shapes: list = field(default_factory=list)
    txt_seq_lens: list = field(default_factory=list)  # [postive_txt_seq_len, negative_txt_seq_len]
    processed_image_size: list = field(default_factory=list)
    original_size: list = field(default_factory=list)
    aspect_ratio: str = ""
    image_encoder_output: Any = field(default=None, repr=False)
    input_image: Any = field(default=None, repr=False)
    latent_image_ids: Any = field(default=None, repr=False)
    txt_ids: Optional[torch.Tensor] = field(default=None, repr=False)


@dataclass
class Flux2I2IInputInfo(I2IInputInfo):
    inpaint_blur_size: Optional[int] = None
    inpaint_blur_sigma: Optional[float] = None


@dataclass
class HidreamI2IInputInfo(I2IInputInfo):
    keep_aspect_ratio: bool = False
    layout_bboxes: str = ""


@dataclass
class TI2IInputInfo(I2IInputInfo):
    align_image_size: Optional[bool] = None


@dataclass
class T2AVInputInfo(InputInfo):
    prompt: str = ""
    negative_prompt: str = ""
    # shape related
    video_latent_shape: list = field(default_factory=list)
    audio_latent_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    size: list = field(default_factory=list)
    num_frames: Optional[int] = None


@dataclass
class I2AVInputInfo(InputInfo):
    prompt: str = ""
    negative_prompt: str = ""
    image_path: str = ""
    image_strength: float = 1.0
    image_frame_indices: Optional[list[int]] = None
    # shape related
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    video_latent_shape: list = field(default_factory=list)
    audio_latent_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    size: list = field(default_factory=list)
    num_frames: Optional[int] = None


@dataclass
class L2AVInputInfo(T2AVInputInfo):
    last_frame_path: str = ""


@dataclass
class FL2AVInputInfo(T2AVInputInfo):
    image_path: str = ""
    last_frame_path: str = ""


@dataclass
class Ref2AVInputInfo(T2AVInputInfo):
    # Reuse the repository-wide media CLI. Comma-separated strings and Python
    # sequences are normalized by MiniMaxH3Runner.
    image_path: Any = ""
    video_path: Any = ""
    audio_path: Any = ""


@dataclass
class I2VAInputInfo(InputInfo):
    prompt: str = ""
    negative_prompt: str = ""
    image_path: str = ""
    video_path: str = ""
    action_path: str = ""
    state_path: str = ""
    action_mode: str = ""
    domain_name: str = ""
    view_point: str = ""
    save_action_path: str = ""
    # shape related
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    size: list = field(default_factory=list)
    num_frames: Optional[int] = None
    # Optional in-memory policy inputs.  Offline/CLI inference continues to use
    # image_path/state_path; long-running integrations (for example ROS) can
    # avoid writing a PNG and NPY file for every control step.
    policy_image: Any = field(default=None, repr=False)
    policy_state: Any = field(default=None, repr=False)


@dataclass
class Cosmos3InputInfo(I2VAInputInfo):
    image_shapes: list = field(default_factory=list)
    txt_seq_lens: list = field(default_factory=list)
    audio_latent_shape: list = field(default_factory=list)
    action_chunk_size: Optional[int] = None
    vision_condition_latents: Any = field(default=None, repr=False)
    vision_condition_frame_indexes: Optional[list[int]] = None
    action_latents: Any = field(default=None, repr=False)
    action_latent_shape: Optional[tuple[int, ...]] = None
    action_condition_frame_indexes: Optional[list[int]] = None
    action_domain_id: Optional[int] = None
    raw_action_dim: Optional[int] = None
    action_start_frame_offset: int = 1


@dataclass
class V2AVInputInfo(I2AVInputInfo):
    """LTX-2.3 IC-LoRA video-to-audio-video.

    Drives both motion-transfer (Union / Pose / Motion-Track-Control) and
    ICEdit-Insight editing (restoration / HD / watermark / subtitle removal).
    The reference / control video is provided pre-processed via ``video_path``.
    Optional character image conditioning is supported through the i2av-style
    ``image_path`` / ``image_strength`` / ``image_frame_indices`` fields.
    """

    # Pre-processed reference / control video (pose / canny / depth / track for
    # motion transfer, or the degraded source video for ICEdit).
    video_path: str = ""
    reference_video_strength: float = 1.0
    reference_video_frame_cap: Optional[int] = None
    # Optional: mux audio from this file after save (e.g. original driving video).
    # ``video_path`` is often a silent pose/canny/depth control clip; DefaultRunner's
    # v2av mux path is not used because LTX2Runner overrides ``process_images_after_vae_decoder``.
    mux_audio_video_path: str = ""


@dataclass
class LTX2S2VInputInfo(I2AVInputInfo):
    """LTX-2 audio-conditioned video (reference audio + optional reference images)."""

    audio_path: str = ""


@dataclass
class WorldPlayI2VInputInfo(I2VInputInfo):
    """Input info for WorldPlay model (image-to-video with action/pose conditioning)."""

    pose: str | dict | None = None
    model_type: str = "ar"  # "ar" (autoregressive) or "bi" (bidirectional)
    chunk_latent_frames: int = 4
    # Computed pose tensors (set during processing)
    viewmats: Optional[torch.Tensor] = None
    Ks: Optional[torch.Tensor] = None
    action: Optional[torch.Tensor] = None


@dataclass
class Hunyuan3DShapeInputInfo(InputInfo):
    """Input info for Hunyuan3D-2.1 image-to-3D-mesh shape generation."""

    image_path: str = ""


@dataclass
class WorldMirrorReconInputInfo(InputInfo):
    """Input info for HY-WorldMirror-2.0 3D reconstruction.

    Unlike the diffusion tasks, this task takes a directory / video / image
    and saves multi-view depth / normal / Gaussian-splat results to disk.
    """

    # Input may be a directory of images, a single image, or a video.
    input_path: str = ""
    strict_output_path: Optional[str] = None
    # Optional priors
    prior_cam_path: Optional[str] = None
    prior_depth_path: Optional[str] = None
    save_rendered: bool = False
    render_interp_per_pair: int = 15
    render_depth: bool = False


@dataclass
class WorldPlayT2VInputInfo(T2VInputInfo):
    """Input info for WorldPlay model (text-to-video with action/pose conditioning)."""

    pose: str | dict | None = None
    model_type: str = "ar"  # "ar" (autoregressive) or "bi" (bidirectional)
    chunk_latent_frames: int = 4
    # Computed pose tensors (set during processing)
    viewmats: Optional[torch.Tensor] = None
    Ks: Optional[torch.Tensor] = None
    action: Optional[torch.Tensor] = None


@dataclass
class SenseNovaVisionInputInfo(InputInfo):
    prompt: str = ""
    image_path: str = ""
    omni_vision_subtask: str = ""
    raw_output_path: str = ""
    glb_output_path: str = ""
    postprocess_predictions: Optional[bool] = None


INPUT_INFO_TYPES = {
    "t2v": T2VInputInfo,
    "i2v": I2VInputInfo,
    "sr": SRInputInfo,
    "flf2v": Flf2vInputInfo,
    "vace": VaceInputInfo,
    "s2v": S2VInputInfo,
    "rs2v": RS2VInputInfo,
    "animate": AnimateInputInfo,
    "t2t": T2TInputInfo,
    "t2i": T2IInputInfo,
    "ti2t": TI2TInputInfo,
    "ti2i": TI2IInputInfo,
    "i2i": I2IInputInfo,
    "t2av": T2AVInputInfo,
    "i2av": I2AVInputInfo,
    "l2av": L2AVInputInfo,
    "fl2av": FL2AVInputInfo,
    "ref2av": Ref2AVInputInfo,
    "i2va": I2VAInputInfo,
    "v2av": V2AVInputInfo,
    "ltx2_s2v": LTX2S2VInputInfo,
    "recon": WorldMirrorReconInputInfo,
    "i23d": Hunyuan3DShapeInputInfo,
    "omni_vision_task": SenseNovaVisionInputInfo,
}


def calculate_num_frames_from_duration(duration_seconds: float, fps: int = 16) -> int:
    """Calculate num_frames from video duration using the formula:
    num_frames = (fps * seconds + 3) // 4 * 4 + 1

    This ensures the result satisfies the VAE stride constraint: (n-1) % 4 == 0

    Args:
        duration_seconds: Video duration in seconds
        fps: Target frames per second (default 16)

    Returns:
        Frame count that satisfies VAE stride constraint

    Examples:
        1s: (16*1 + 3) // 4 * 4 + 1 = 17 frames
        3s: (16*3 + 3) // 4 * 4 + 1 = 49 frames
        5s: (16*5 + 3) // 4 * 4 + 1 = 81 frames
    """
    return align_num_frames(int(fps * duration_seconds) + 3, 4)


def align_num_frames(num_frames: int, temporal_stride: int) -> int:
    """Align a frame count so that ``num_frames - 1`` is stride-divisible."""
    return num_frames // temporal_stride * temporal_stride + 1


@dataclass
class SekoTalkInputs(InputInfo):
    num_frames: int | Any = UNSET
    seed: int | Any = UNSET
    prompt: str | Any = UNSET
    negative_prompt: str | Any = UNSET
    image_path: str | Any = UNSET
    audio_path: str | Any = UNSET
    audio_num: int | Any = UNSET
    video_duration: float | Any = UNSET
    with_mask: bool | Any = UNSET
    save_result_path: str | Any = UNSET
    return_result_tensor: bool | Any = UNSET
    stream_config: dict | Any = UNSET

    fixed_area: str | Any = UNSET
    size: list | Any = UNSET
    latent_shape: list | Any = UNSET

    # prev info
    overlap_frame: torch.Tensor | Any = UNSET
    overlap_latent: torch.Tensor | Any = UNSET
    # input preprocess audio
    audio_clip: torch.Tensor | Any = UNSET
    person_mask_latens: torch.Tensor | Any = field(default=UNSET, repr=False)

    # input reference state
    ref_state: int | Any = UNSET
    # flags for first and last clip
    is_first: bool | Any = UNSET
    is_last: bool | Any = UNSET
    # if save video by stream
    stream_save_video: bool | Any = UNSET
