"""Native DA3MONO-LARGE runner for single-image relative depth previews."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors

from lightx2v.models.networks.depth_anything3.mono import DepthAnything3Mono
from lightx2v.models.networks.depth_anything3.output import save_relative_depth_preview
from lightx2v.models.networks.depth_anything3.preprocess import preprocess_image
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS
from lightx2v.utils.input_info import InputInfo
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@dataclass
class DepthInputInfo(InputInfo):
    image_path: str = ""


@RUNNER_REGISTER("depth_anything3")
class DepthAnything3Runner(BaseRunner):
    """Load a ComfyUI-format checkpoint once and reuse it across requests."""

    input_info_cls_by_task = {"depth": DepthInputInfo}
    supported_request_fields_by_task = {"depth": (COMMON_REQUEST_FIELDS - {"return_result_tensor"}) | {"image_path"}}

    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.device = torch.device(AI_DEVICE if AI_DEVICE != "cuda" or torch.cuda.is_available() else "cpu")

    def init_modules(self):
        variant = self.config.get("model_variant", "mono_large")
        if variant != "mono_large":
            raise ValueError(f"Depth Anything 3 supports only model_variant='mono_large'; got {variant!r}")
        model_path = Path(self.config["model_path"])
        checkpoint = model_path / "depth_anything_3_mono_large.safetensors" if model_path.is_dir() else model_path
        if not checkpoint.is_file() or checkpoint.suffix != ".safetensors":
            raise FileNotFoundError(f"DA3MONO-LARGE safetensors checkpoint not found: {checkpoint}")

        with torch.device("meta"):
            model = DepthAnything3Mono()
        weights = load_safetensors(str(checkpoint), device="cpu")
        prefix = "model."
        state = {key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in weights.items()}
        if len(state) != len(weights):
            raise ValueError("Duplicate DA3 checkpoint keys after removing the 'model.' prefix")
        model.load_state_dict(state, strict=True, assign=True)
        del weights, state
        self.model = model.to(self.device).eval()
        self.config.lock()

    def run_pipeline(self, input_info):
        if not input_info.image_path:
            raise ValueError("Depth Anything 3 requires image_path")
        if not input_info.save_result_path:
            raise ValueError("Depth Anything 3 requires save_result_path for its PNG preview")
        destination = Path(input_info.save_result_path)
        if destination.suffix.lower() != ".png":
            raise ValueError("Depth Anything 3 save_result_path must end in .png")
        image, original_size = preprocess_image(input_info.image_path, process_res=int(self.config.get("process_res", 504)))
        with torch.inference_mode():
            depth, sky = self.model(image.to(self.device))
        return save_relative_depth_preview(depth[0].float().cpu().numpy(), sky[0].float().cpu().numpy(), destination, size=original_size)

    def close(self):
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _maybe_freeze_gc(self):
        # Keep explicit close() able to reclaim cyclic references after the last request.
        return
