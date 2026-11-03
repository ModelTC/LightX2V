"""Native DA3MONO-LARGE runner for image and streaming-video depth previews."""

from __future__ import annotations

import gc
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors

from lightx2v.models.networks.depth_anything3.mono import DepthAnything3Mono
from lightx2v.models.networks.depth_anything3.postprocess import postprocess_mono_sky
from lightx2v.models.networks.depth_anything3.preprocess import preprocess_rgb
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.depth_anything3.checkpoint import normalize_checkpoint_weights, resolve_checkpoint
from lightx2v.models.runners.depth_anything3.output import (
    StreamingNpzWriter,
    render_relative_depth_preview,
    save_float_depth,
    save_relative_depth_preview,
    save_sky_mask,
)
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
    save_depth_path: str = ""
    save_sky_mask_path: str = ""


@RUNNER_REGISTER("depth_anything3")
class DepthAnything3Runner(BaseRunner):
    """Load an official or ComfyUI-format checkpoint once and reuse it across requests."""

    input_info_cls_by_task = {"depth": DepthInputInfo}
    supported_request_fields_by_task = {"depth": (COMMON_REQUEST_FIELDS - {"return_result_tensor"}) | {"image_path", "video_path", "save_depth_path", "save_sky_mask_path"}}

    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.device = torch.device(AI_DEVICE if AI_DEVICE != "cuda" or torch.cuda.is_available() else "cpu")
        self._cuda_graph = None
        self._graph_input = None
        self._graph_output = None
        self._graph_lock = Lock()

    def init_modules(self):
        variant = self.config.get("model_variant", "mono_large")
        if variant != "mono_large":
            raise ValueError(f"Depth Anything 3 supports only model_variant='mono_large'; got {variant!r}")
        mm_type = self.config.get("mm_type", "Default")
        if mm_type != "Default":
            raise ValueError(f"Depth Anything 3 does not yet support mm_type={mm_type!r}; quantized linear weights require a DA3 weight/scale loader")
        backbone_dtype = self.config.get("backbone_dtype", "auto")
        if backbone_dtype not in {"auto", "fp32", "bf16"}:
            raise ValueError(f"Unknown DA3 backbone_dtype: {backbone_dtype!r}")
        if backbone_dtype == "auto":
            backbone_dtype = "bf16" if self.device.type == "cuda" else "fp32"
        if backbone_dtype == "bf16" and self.device.type != "cuda":
            raise ValueError("DA3 backbone_dtype='bf16' requires a CUDA device")
        use_cuda_graph = self.config.get("use_cuda_graph", False)
        if not isinstance(use_cuda_graph, bool):
            raise ValueError("DA3 use_cuda_graph must be a boolean")
        if use_cuda_graph and self.device.type != "cuda":
            raise ValueError("DA3 use_cuda_graph requires a CUDA device")
        if self.config.get("norm_type", "torch") == "Triton" and self.device.type != "cuda":
            raise ValueError("DA3 norm_type='Triton' requires a CUDA device")
        checkpoint = resolve_checkpoint(Path(self.config["model_path"]))

        with torch.device("meta"):
            model = DepthAnything3Mono(attn_type=self.config.get("attn_type", "torch_sdpa"), norm_type=self.config.get("norm_type", "torch"))
        weights = load_safetensors(str(checkpoint), device="cpu")
        state = normalize_checkpoint_weights(weights)
        model.load_state_dict(state, strict=True, assign=True)
        del weights, state
        if backbone_dtype == "bf16":
            model.backbone.to(dtype=torch.bfloat16)
        self.model = model.to(self.device).eval()
        self.config.lock()

    def run_pipeline(self, input_info):
        if bool(input_info.image_path) == bool(input_info.video_path):
            raise ValueError("Depth Anything 3 requires exactly one of image_path or video_path")
        if not input_info.save_result_path:
            raise ValueError("Depth Anything 3 requires save_result_path")
        destination = Path(input_info.save_result_path).expanduser()
        depth_path, sky_mask_path = self._optional_output_paths(input_info, destination)
        if input_info.video_path:
            return self._run_video(input_info.video_path, destination, depth_path=depth_path, sky_mask_path=sky_mask_path)
        if destination.suffix.lower() != ".png":
            raise ValueError("Depth Anything 3 save_result_path must end in .png")
        with Image.open(input_info.image_path) as source:
            rgb = np.asarray(source.convert("RGB"))
        image, original_size = preprocess_rgb(
            rgb,
            process_res=int(self.config.get("process_res", 504)),
            process_res_method=self.config.get("process_res_method", "upper_bound_resize"),
        )
        depth, sky = self._infer(image)
        result = save_relative_depth_preview(depth, sky, destination, size=original_size)
        if depth_path is not None:
            save_float_depth(depth, depth_path)
        if sky_mask_path is not None:
            save_sky_mask(sky, sky_mask_path)
        return result

    def _optional_output_paths(self, input_info, destination):
        depth_path = Path(input_info.save_depth_path).expanduser() if input_info.save_depth_path else None
        sky_mask_path = Path(input_info.save_sky_mask_path).expanduser() if input_info.save_sky_mask_path else None
        if input_info.video_path:
            depth_path = depth_path.resolve() if depth_path is not None else None
            sky_mask_path = sky_mask_path.resolve() if sky_mask_path is not None else None
        outputs = [destination, *(path for path in (depth_path, sky_mask_path) if path is not None)]
        resolved = [path.expanduser().resolve() for path in outputs]
        source = Path(input_info.video_path or input_info.image_path).expanduser().resolve()
        if source in resolved or resolved[0] in resolved[1:] or (not input_info.video_path and len(set(resolved)) != len(resolved)):
            raise ValueError("Depth Anything 3 input and output paths must be distinct")
        if input_info.video_path:
            if destination.suffix.lower() != ".mp4":
                raise ValueError("Depth Anything 3 video save_result_path must end in .mp4")
            for path in (depth_path, sky_mask_path):
                if path is not None and (path.suffix.lower() != ".npz" or path.is_dir()):
                    raise ValueError(f"Video sidecar output must be an .npz file: {path}")
        else:
            if depth_path is not None and depth_path.suffix.lower() != ".npy":
                raise ValueError("Image save_depth_path must end in .npy")
            if sky_mask_path is not None and sky_mask_path.suffix.lower() != ".png":
                raise ValueError("Image save_sky_mask_path must end in .png")
        return depth_path, sky_mask_path

    def _capture_cuda_graph(self, image):
        self._graph_input = image.clone()
        for _ in range(3):
            self.model(self._graph_input)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                self._graph_output = self.model(self._graph_input)
        except BaseException:
            self._graph_input = None
            self._graph_output = None
            graph.reset()
            raise
        self._cuda_graph = graph

    def _infer(self, image, *, use_cuda_graph=False):
        with torch.inference_mode():
            image = image.to(self.device)
            if use_cuda_graph:
                with self._graph_lock:
                    if self._cuda_graph is None:
                        self._capture_cuda_graph(image)
                    if image.shape == self._graph_input.shape and image.dtype == self._graph_input.dtype:
                        self._graph_input.copy_(image)
                        self._cuda_graph.replay()
                        depth, sky = self._graph_output
                    else:
                        # One graph per runner bounds captured allocations for variable-size video.
                        depth, sky = self.model(image)
                    depth, sky = depth[0].float().cpu().numpy(), sky[0].float().cpu().numpy()
            else:
                depth, sky = self.model(image)
                depth, sky = depth[0].float().cpu().numpy(), sky[0].float().cpu().numpy()
            return postprocess_mono_sky(depth, sky), sky

    def _run_video(self, video_path, destination, *, depth_path=None, sky_mask_path=None):
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
            with ExitStack() as archives:
                depth_archive = archives.enter_context(StreamingNpzWriter(depth_path)) if depth_path is not None else None
                if sky_mask_path is None:
                    sky_archive = None
                elif sky_mask_path == depth_path:
                    sky_archive = depth_archive
                else:
                    sky_archive = archives.enter_context(StreamingNpzWriter(sky_mask_path))
                writer = StreamingVideoWriter(destination, fps=fps)
                try:
                    for frame_index, frame in enumerate(container.decode(video=0)):
                        image, original_size = preprocess_rgb(
                            frame.to_ndarray(format="rgb24"),
                            process_res=int(self.config.get("process_res", 504)),
                            process_res_method=self.config.get("process_res_method", "upper_bound_resize"),
                        )
                        depth, sky = self._infer(image, use_cuda_graph=self.config.get("use_cuda_graph", False))
                        preview = render_relative_depth_preview(depth, sky, size=original_size)
                        if preview.shape[0] % 2 or preview.shape[1] % 2:
                            preview = np.pad(preview, ((0, preview.shape[0] % 2), (0, preview.shape[1] % 2)), mode="edge")
                        rgb = np.repeat(preview[:, :, None], 3, axis=2)
                        writer.submit(torch.from_numpy(rgb[None]))
                        if depth_archive is not None:
                            depth_archive.write_frame(f"depth_{frame_index:06d}", depth)
                        if sky_archive is not None:
                            sky_archive.write_frame(f"sky_mask_{frame_index:06d}", (sky >= 0.5).astype(np.uint8) * 255)
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
        with self._graph_lock:
            graph = self._cuda_graph
            self._graph_output = None
            self._graph_input = None
            self._cuda_graph = None
            if graph is not None:
                torch.cuda.synchronize(self.device)
                graph.reset()
                del graph
            self.model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _maybe_freeze_gc(self):
        # Keep explicit close() able to reclaim cyclic references after the last request.
        return
