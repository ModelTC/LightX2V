"""Bridge the HTTP runner contract to the existing SekoTalk RS2V Shot pipeline."""

import copy
import json
from pathlib import Path


def load_shot_settings(config_path):
    """Accept platform main -> clip files, or an existing flat inference config.

    Relative clip paths are relative to the main file; an explicit lightx2v_path
    retains the Shot CLI's repository-relative convention. Main parallel wins,
    while main video_duration supplies a request default, not a clip override.
    """
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        main = json.load(stream)
    if "clip_configs" in main:
        clips = main["clip_configs"]
        if len(clips) != 1 or clips[0].get("name") != "rs2v_clip":
            raise ValueError("RS2V HTTP requires exactly one clip named rs2v_clip")
        item = clips[0]
        if ("config" in item) == ("path" in item):
            raise ValueError("rs2v_clip requires either config or path, not both")
        if "config" in item:
            clip = copy.deepcopy(item["config"])
        else:
            base = Path(main.get("lightx2v_path", config_path.parent))
            if not base.is_absolute():
                base = config_path.parent / base
            with (base / item["path"]).open(encoding="utf-8") as stream:
                clip = json.load(stream)
        duration = main.get("video_duration", clip.get("video_duration"))
        if "parallel" in main:
            clip["parallel"] = copy.deepcopy(main["parallel"])
    else:
        clip = copy.deepcopy(main)
        duration = clip.get("video_duration")

    for old, new in (("target_fps", "fps"), ("target_video_length", "num_frames")):
        if old in clip:
            if new in clip and clip[new] != clip[old]:
                raise ValueError(f"Conflicting {old} and {new} in RS2V config")
            clip[new] = clip.pop(old)
    return clip, duration


def build_shot_rs2v_runner(args):
    # Lazy imports keep configuration inspection independent of GPU packages.
    from lightx2v.shot_runner.rs2v_infer import ShotRS2VPipeline
    from lightx2v.shot_runner.shot_base import ClipConfig
    from lightx2v.utils.set_config import build_startup_config, init_parallel

    clip, duration = load_shot_settings(args.config_json)
    # Deployment identity comes from the same CLI flags as the ordinary worker.
    clip.update(model_cls=args.model_cls, model_path=args.model_path, task=args.task)
    if getattr(args, "model_variant", None) is not None:
        clip["model_variant"] = args.model_variant
    # Shot's cached decoder assumes persistent modules; the HTTP path must
    # produce a downloadable file, not stream to the CLI livestream controller.
    for option in ("stream_save_video", "lazy_load", "unload_modules"):
        if clip.get(option):
            raise ValueError(f"RS2V Shot HTTP does not support {option}=true")
    config = build_startup_config(clip)
    if config["parallel"]:
        init_parallel(config)
    pipeline = ShotRS2VPipeline([ClipConfig(name="rs2v_clip", config_json=config)])
    return ShotRS2VRunner(pipeline, duration)


class ShotRS2VRunner:
    """Expose the HTTP worker interface without loading a second model."""

    def __init__(self, pipeline, default_duration):
        self.pipeline = pipeline
        self.clip_runner = pipeline.clip_generators["rs2v_clip"]
        self.config = self.clip_runner.config
        self.supported_tasks = self.clip_runner.supported_tasks
        self.default_duration = default_duration

    def prepare_request(self, request_data):
        request_data = {key: value for key, value in request_data.items() if value is not None}
        if "video_duration" not in request_data and self.default_duration is not None:
            request_data["video_duration"] = self.default_duration
        if request_data.get("return_result_tensor"):
            raise ValueError("RS2V Shot HTTP produces video files, not result_png tensors")
        info = self.clip_runner.prepare_request(request_data)
        if not info.image_path or not info.audio_path:
            raise ValueError("SekoTalk RS2V requires image_path and audio_path")
        # Keep runner validation and seeded defaults, but let Shot allocate its
        # own fresh per-request context and manage all segment state.
        request_data["seed"] = info.seed
        return request_data

    def set_reuse(self, reuse, reuse_prefix_segments=0):
        if reuse or reuse_prefix_segments:
            raise ValueError("SekoTalk RS2V Shot HTTP does not support reuse")

    def validate_lora(self, lora_name):
        if lora_name is not None:
            raise ValueError("SekoTalk RS2V Shot HTTP does not support dynamic LoRA")

    def _clear_request_state(self):
        runner = self.clip_runner
        # The CLI is normally one-shot. HTTP keeps the same models alive across
        # requests, including after a Python exception midway through a segment.
        runner.stop_signal = False
        for name in ("inputs", "inputs_static", "ref_img"):
            if hasattr(runner, name):
                delattr(runner, name)
        runner.input_info = None
        decoder_model = getattr(getattr(runner, "vae_decoder", None), "model", None)
        if decoder_model is not None and hasattr(decoder_model, "clear_cache"):
            decoder_model.clear_cache()
        scheduler = getattr(runner, "scheduler", None)
        if scheduler is not None:
            scheduler.clear()
            if hasattr(scheduler, "_audio_t_emb_cache"):
                scheduler._audio_t_emb_cache.clear()

    def run_request(self, request_data):
        self._clear_request_state()
        try:
            return self.pipeline.run_pipeline(request_data)
        finally:
            self._clear_request_state()
