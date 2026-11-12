from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

from .manager import Manager
from .models import default_lightx2v_path, model_catalog, validate_selection
from .service import cache_fingerprint, server_reference


def video_class():
    from comfy_api.latest import InputImpl

    return InputImpl.VideoFromFile


def video_from_file(path: Path):
    import av

    with av.open(str(path)) as container:
        if not container.streams.video or next(container.decode(video=0), None) is None:
            raise ValueError("LightX2V output contains no decodable video frames")
    return video_class()(str(path))


def video_to_file(video, directory: Path) -> Path:
    """Use the original file only when the native video has no trim or crop."""
    import av

    cls = video_class()
    if type(video) is cls and hasattr(video, "get_active_trim_window"):
        source = video.get_stream_source()
        if isinstance(source, (str, os.PathLike)) and video.get_active_trim_window() == (0.0, 0.0):
            with av.open(str(source)) as container:
                stream = container.streams.video[0]
                if video.get_dimensions() == (stream.width, stream.height):
                    return Path(source).absolute()
    # Trimmed/cropped/tensor/composite videos must be materialized via their native API.
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid.uuid4().hex + ".mp4")
    video.save_to(str(path))
    if not path.is_file() or not path.stat().st_size:
        raise ValueError("ComfyUI did not export the input video")
    return path


def image_to_file(image, directory: Path) -> Path:
    from PIL import Image

    if len(image.shape) != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError("Expected one RGB/RGBA IMAGE; image batches are not supported")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid.uuid4().hex + ".png")
    pixels = (image[0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype("uint8")
    Image.fromarray(pixels).save(path)
    return path


def image_from_file(path: Path):
    """A single still image becomes Comfy's CPU float32 [1, H, W, 3] IMAGE."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Expected one still image, not an animation")
        pixels = np.array(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)


_manager = None
_manager_lock = threading.Lock()


def _interrupt():
    import comfy.model_management as mm

    mm.throw_exception_if_processing_interrupted()


def _unload_comfy():
    import comfy.model_management as mm

    mm.unload_all_models()
    mm.soft_empty_cache()


def manager() -> Manager:
    global _manager
    with _manager_lock:
        if _manager is None:
            import folder_paths

            root = os.environ.get("LIGHTX2V_COMFY_RUNTIME", str(Path(folder_paths.get_temp_directory()) / "lightx2v-server"))
            _manager = Manager(Path(root), before_inference=_unload_comfy)
        return _manager


class LightX2VServer:
    @classmethod
    def INPUT_TYPES(cls):
        catalog = model_catalog()
        models = catalog["models"]
        default_model = "minimax_h3" if "minimax_h3" in models else next(iter(models))
        tasks = list(dict.fromkeys(task for values in models.values() for task in values))
        return {
            "required": {
                "lightx2v_path": ("STRING", {"default": default_lightx2v_path(), "tooltip": "LightX2V source directory visible to ComfyUI. Use Browse LightX2V to choose it."}),
                "model_path": ("STRING", {"default": "", "tooltip": "Model directory visible to ComfyUI. Use Browse model to choose it."}),
                "model_cls": (list(models), {"default": default_model, "lightx2v_tasks": models, "tooltip": "Models discovered in the LightX2V source directory."}),
                "config_json": ("STRING", {"default": "", "tooltip": "Choose or edit JSON with Browse / Edit config, or enter its path. Relative paths use lightx2v_path."}),
                "task": (tasks, {"default": models[default_model][0], "tooltip": "Tasks declared by the selected model. H3 weight variant is inferred automatically."}),
            },
            "optional": {
                "CUDA_VISIBLE_DEVICES": (
                    "STRING",
                    {
                        "default": "0",
                        "tooltip": "Choose a GPU by its used/total VRAM and GPU utilization. Status updates automatically every 5 seconds while the page is visible. The saved value remains a device index or UUID.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LX2V_SERVER",)
    RETURN_NAMES = ("server",)
    FUNCTION = "connect"
    CATEGORY = "LightX2V/Server"
    DESCRIPTION = "LightX2V startup parameters. Paths refer directly to the ComfyUI filesystem. Task is defined here and used by generation/upscale requests. No model loads until inference executes."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return cache_fingerprint(kwargs)

    @classmethod
    def VALIDATE_INPUTS(cls, lightx2v_path=None, model_cls=None, task=None):
        if any(value is None for value in (lightx2v_path, model_cls, task)):
            return True  # Linked values are checked again by resolve_server during execution.
        try:
            validate_selection(model_catalog(lightx2v_path), model_cls, task)
        except (OSError, SyntaxError, ValueError) as exc:
            return str(exc)
        return True

    def connect(self, lightx2v_path, model_path, model_cls, config_json, task="t2av", CUDA_VISIBLE_DEVICES="0"):
        ref = server_reference(lightx2v_path, model_path, model_cls, config_json, task=task, CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES)
        settings = ref.resolve()
        return {"ui": {"lightx2v_service_key": [settings.service_key]}, "result": (ref,)}


class LightX2VGenerateVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LX2V_SERVER",),
                "prompt": ("STRING", {"multiline": True, "default": "A cat walks through a sunlit garden."}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 8}),
                "num_frames": ("INT", {"default": 124, "min": 1, "max": 10000}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "H3: use task=i2av for a first frame, fl2av with last_frame, or ref2av for a reference image with its matching config."}),
                "last_frame": ("IMAGE", {"tooltip": "H3: use task=l2av for a last frame, or fl2av with image as the first frame."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Blank keeps the backend default. Only submit to models supporting negative prompts."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "LightX2V/Server"

    def generate(self, server, prompt, width, height, num_frames, seed, image=None, last_frame=None, negative_prompt=""):
        settings = server.resolve()
        task = settings.settings["task"]
        if settings.settings["model_cls"] == "minimax_h3":
            required = {
                "t2av": set(),
                "i2av": {"image"},
                "l2av": {"last_frame"},
                "fl2av": {"image", "last_frame"},
                "ref2av": {"image"},
            }
            provided = {name for name, value in (("image", image), ("last_frame", last_frame)) if value is not None}
            if task in required and provided != required[task]:
                expected = ", ".join(sorted(required[task])) or "no image inputs"
                actual = ", ".join(sorted(provided)) or "no image inputs"
                suggested = {
                    frozenset(): "task=t2av for text-only generation",
                    frozenset({"image"}): "task=i2av for a first-frame image, or task=ref2av for a reference image with its matching config",
                    frozenset({"last_frame"}): "task=l2av for a last-frame image",
                    frozenset({"image", "last_frame"}): "task=fl2av for first and last frames",
                }[frozenset(provided)]
                raise ValueError(
                    f"H3 task={task} expects {expected}; received {actual}. Set {suggested} on the connected Service, or adjust the image connections. Validation stopped before loading the model."
                )
            if width % 32 or height % 32 or not (124 <= num_frames <= 362 and (num_frames - 5) % 17 == 0):
                raise ValueError("H3 requires dimensions divisible by 32 and frames = 17*n+5, in [124, 362]")
        payload = {"task": task, "prompt": prompt, "size": [height, width], "num_frames": num_frames, "seed": seed}
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        runtime = manager()
        for key, tensor in (("image_path", image), ("last_frame_path", last_frame)):
            if tensor is not None:
                payload[key] = str(image_to_file(tensor, runtime.root / "inputs"))
        result = runtime.run(settings, payload, interrupt=_interrupt, finalize=video_from_file)
        return (result,)


class LightX2VGenerateImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LX2V_SERVER",),
                "prompt": ("STRING", {"multiline": True, "default": "A cat in a sunlit garden."}),
                "width": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 8}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "image": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Blank keeps the backend default."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "LightX2V/Server"
    DESCRIPTION = "Generate one image or edit one input image through the native image task API. Select t2i or i2i on Service."

    def generate(self, server, prompt, width, height, seed, image=None, negative_prompt=""):
        settings = server.resolve()
        task = settings.settings["task"]
        if task not in {"t2i", "i2i"}:
            raise ValueError("Set task=t2i or i2i on the Service connected to Generate Image")
        if task == "i2i" and image is None:
            raise ValueError("Image editing requires an IMAGE input")
        if task == "t2i" and image is not None:
            raise ValueError("Select task=i2i to use an IMAGE input")
        if json.loads(settings.config_json).get("layered", False):
            raise ValueError("Layered image output is not supported by this single-image node")
        payload = {"task": task, "prompt": prompt, "size": [height, width], "seed": seed}
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        runtime = manager()
        if image is not None:
            payload["image_path"] = str(image_to_file(image, runtime.root / "inputs"))
        return (runtime.run(settings, payload, interrupt=_interrupt, media_type="image", finalize=image_from_file),)


class LightX2VUpscaleVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LX2V_SERVER",),
                "video": ("VIDEO",),
                "scale": ([2, 4], {"default": 2}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "upscale"
    CATEGORY = "LightX2V/Server"

    def upscale(self, server, video, scale=2, seed=42):
        settings = server.resolve()
        if settings.settings["task"] != "sr":
            raise ValueError("Set task=sr on the Service connected to Upscale Video")
        if settings.settings["model_cls"] == "relayvsr_x2" and scale != 2:
            raise ValueError("relayvsr_x2 only supports scale 2")
        runtime = manager()
        path = video_to_file(video, runtime.root / "inputs")
        payload = {"task": "sr", "video_path": str(path), "sr_ratio": scale, "seed": seed}
        result = runtime.run(settings, payload, interrupt=_interrupt, finalize=video_from_file)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "LightX2VServer": LightX2VServer,
    "LightX2VGenerateVideo": LightX2VGenerateVideo,
    "LightX2VGenerateImage": LightX2VGenerateImage,
    "LightX2VUpscaleVideo": LightX2VUpscaleVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LightX2VServer": "LightX2V Service",
    "LightX2VGenerateVideo": "LightX2V Generate Video",
    "LightX2VGenerateImage": "LightX2V Generate Image",
    "LightX2VUpscaleVideo": "LightX2V Upscale Video",
}
