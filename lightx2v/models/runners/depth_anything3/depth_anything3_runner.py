"""Native DA3MONO-LARGE runner for image and streaming-video depth previews."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

from lightx2v.models.networks.depth_anything3.mono import DepthAnything3Mono
from lightx2v.models.networks.depth_anything3.output import render_relative_depth_preview, save_relative_depth_preview
from lightx2v.models.networks.depth_anything3.preprocess import preprocess_image, preprocess_rgb
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.depth_anything3.checkpoint import normalize_checkpoint_weights, resolve_checkpoint
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS
from lightx2v.utils.input_info import InputInfo
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.streaming_video import StreamingVideoWriter
from lightx2v.utils.utils import mux_audio_from_video
from lightx2v_platform.base.global_var import AI_DEVICE


@dataclass
class DepthInputInfo(InputInfo):
    image_path: str = ""
    video_path: str = ""


@RUNNER_REGISTER("depth_anything3")
class DepthAnything3Runner(BaseRunner):
    """Load an official or ComfyUI-format checkpoint once and reuse it across requests."""

    input_info_cls_by_task = {"depth": DepthInputInfo}
    supported_request_fields_by_task = {"depth": (COMMON_REQUEST_FIELDS - {"return_result_tensor"}) | {"image_path", "video_path"}}

    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.device = torch.device(AI_DEVICE if AI_DEVICE != "cuda" or torch.cuda.is_available() else "cpu")

    def init_modules(self):
        variant = self.config.get("model_variant", "mono_large")
        if variant != "mono_large":
            raise ValueError(f"Depth Anything 3 supports only model_variant='mono_large'; got {variant!r}")
        mm_type = self.config.get("mm_type", "Default")
        if mm_type != "Default":
            raise ValueError(f"Depth Anything 3 does not yet support mm_type={mm_type!r}; quantized linear weights require a DA3 weight/scale loader")
        checkpoint = resolve_checkpoint(Path(self.config["model_path"]))

        with torch.device("meta"):
            model = DepthAnything3Mono(attn_type=self.config.get("attn_type", "torch_sdpa"))
        weights = load_safetensors(str(checkpoint), device="cpu")
        state = normalize_checkpoint_weights(weights)
        model.load_state_dict(state, strict=True, assign=True)
        del weights, state
        self.model = model.to(self.device).eval()
        self.config.lock()

    def run_pipeline(self, input_info):
        if bool(input_info.image_path) == bool(input_info.video_path):
            raise ValueError("Depth Anything 3 requires exactly one of image_path or video_path")
        if not input_info.save_result_path:
            raise ValueError("Depth Anything 3 requires save_result_path")
        destination = Path(input_info.save_result_path)
        if input_info.video_path:
            return self._run_video(input_info.video_path, destination)
        if destination.suffix.lower() != ".png":
            raise ValueError("Depth Anything 3 save_result_path must end in .png")
        image, original_size = preprocess_image(input_info.image_path, process_res=int(self.config.get("process_res", 504)))
        depth, sky = self._infer(image)
        return save_relative_depth_preview(depth, sky, destination, size=original_size)

    def _infer(self, image):
        with torch.inference_mode():
            depth, sky = self.model(image.to(self.device))
        return depth[0].float().cpu().numpy(), sky[0].float().cpu().numpy()

    def _run_video(self, video_path, destination):
        if destination.suffix.lower() != ".mp4":
            raise ValueError("Depth Anything 3 video save_result_path must end in .mp4")
        source = Path(video_path).expanduser().resolve()
        if source == destination.expanduser().resolve():
            raise ValueError("Depth Anything 3 video input and output paths must differ")

        import av

        with av.open(str(source)) as container:
            if not container.streams.video:
                raise ValueError(f"Input video has no video stream: {source}")
            stream = container.streams.video[0]
            fps = stream.average_rate or stream.base_rate
            if fps is None or fps <= 0:
                raise ValueError(f"Cannot determine input video frame rate: {source}")
            has_audio = bool(container.streams.audio)
            writer = StreamingVideoWriter(destination, fps=fps)
            try:
                for frame in container.decode(video=0):
                    image, original_size = preprocess_rgb(frame.to_ndarray(format="rgb24"), process_res=int(self.config.get("process_res", 504)))
                    depth, sky = self._infer(image)
                    preview = render_relative_depth_preview(depth, sky, size=original_size)
                    if preview.shape[0] % 2 or preview.shape[1] % 2:
                        preview = np.pad(preview, ((0, preview.shape[0] % 2), (0, preview.shape[1] % 2)), mode="edge")
                    rgb = np.repeat(preview[:, :, None], 3, axis=2)
                    writer.submit(torch.from_numpy(rgb[None]))
                if not writer.frames:
                    raise ValueError(f"Input video has no decodable frames: {source}")
                writer.close()
            except BaseException:
                writer.abort()
                raise
        if has_audio and mux_audio_from_video(str(source), str(destination), trim_to_shortest=False) is None:
            raise RuntimeError(f"Depth video was written but its source audio could not be muxed: {destination}")
        return destination

    def close(self):
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _maybe_freeze_gc(self):
        # Keep explicit close() able to reclaim cyclic references after the last request.
        return
