from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from weakref import WeakKeyDictionary

from .manager import Manager
from .models import default_lightx2v_path, model_catalog, validate_selection
from .service import cache_fingerprint, server_reference

SEED_MAX = 0xFFFFFFFF


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


def validate_regeneration_video(path: Path, num_frames: int | None = None):
    import av

    with av.open(str(path)) as container:
        if not container.streams.video or not container.streams.audio:
            raise ValueError("H3 regeneration requires a video with audio")
        fps = float(container.streams.video[0].average_rate or 0)
        if abs(fps - 24) > 1e-4:
            raise ValueError(f"H3 regeneration requires 24 fps; received {fps}")
        frames = sum(1 for _ in container.decode(video=0))
        if num_frames is not None and frames != num_frames:
            raise ValueError(f"H3 regeneration preserves duration: source has {frames} frames, request has {num_frames}")
        return frames


def video_preview(video):
    import folder_paths

    directory = Path(folder_paths.get_temp_directory()) / "lightx2v-previews"
    directory.mkdir(parents=True, exist_ok=True)
    source = video_to_file(video, directory)
    path = directory / (uuid.uuid4().hex + ".mp4")
    try:
        os.link(source, path)
    except OSError:
        shutil.copyfile(source, path)
    return {"filename": path.name, "subfolder": directory.name, "type": "temp"}


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
                        "tooltip": "Choose one GPU or Select multiple GPUs. Usage refreshes every 5 seconds. One process starts per selected GPU; config.parallel must match the GPU count. Saved values are comma-separated indices or UUIDs.",
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
                "seed": ("INT", {"default": 42, "min": 0, "max": SEED_MAX}),
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
        return self._generate(server, prompt, width, height, num_frames, seed, image, last_frame, negative_prompt)

    def _generate(self, server, prompt, width, height, num_frames, seed, image=None, last_frame=None, negative_prompt="", draft_steps=0, resample_source_video=None, resample_start_step=-1):
        settings = server.resolve()
        task = settings.settings["task"]
        if (draft_steps or resample_source_video is not None or resample_start_step >= 0) and settings.settings["model_cls"] != "minimax_h3":
            raise ValueError("Draft and regeneration inputs require model_cls=minimax_h3")
        source = None
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
            if resample_source_video is not None:
                source = video_to_file(resample_source_video, manager().root / "inputs")
                num_frames = validate_regeneration_video(source, num_frames)
            if width % 32 or height % 32 or not (124 <= num_frames <= 362 and (num_frames - 5) % 17 == 0):
                raise ValueError("H3 requires dimensions divisible by 32 and frames = 17*n+5, in [124, 362]")
            config = json.loads(settings.config_json)
            defaults = config.get("request_defaults", {})
            steps = config.get("infer_steps")
            prefix = draft_steps or defaults.get("draft_steps")
            start = resample_start_step if resample_start_step >= 0 else defaults.get("resample_start_step", 1)
            if resample_source_video is not None or defaults.get("resample_source_video"):
                if steps is not None and prefix is not None and 0 < prefix < steps:
                    raise ValueError("H3 regeneration cannot use a draft prefix. Clear draft_steps in Batch and in the Service config defaults.")
                if steps is not None and not 0 <= start <= steps:
                    raise ValueError(f"H3 resample_start_step must be between 0 and infer_steps ({steps})")
        payload = {"task": task, "prompt": prompt, "size": [height, width], "num_frames": num_frames, "seed": seed & SEED_MAX}
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if draft_steps:
            payload["draft_steps"] = draft_steps
        if resample_start_step >= 0:
            payload["resample_start_step"] = resample_start_step
        runtime = manager()
        if source is not None:
            payload["resample_source_video"] = str(source)
        for key, tensor in (("image_path", image), ("last_frame_path", last_frame)):
            if tensor is not None:
                payload[key] = str(image_to_file(tensor, runtime.root / "inputs"))
        result = runtime.run(settings, payload, interrupt=_interrupt, finalize=video_from_file)
        return (result,)


class LightX2VRegenerateVideo(LightX2VGenerateVideo):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].pop("num_frames")
        inputs["optional"].pop("negative_prompt")
        inputs["required"]["seed"][1]["control_after_generate"] = False
        inputs["required"]["resample_source_video"] = ("VIDEO", {"tooltip": "H3 source with audio at 24 fps. Frame count is read from this video, including any applied trim."})
        inputs["required"]["resample_start_step"] = (
            "INT",
            {"default": 1, "min": -1, "max": 10000, "tooltip": "-1 uses config defaults. 0 runs all steps; infer_steps only VAE-encodes/decodes the source."},
        )
        return inputs

    FUNCTION = "regenerate"
    DESCRIPTION = "Regenerate an H3 video at the requested resolution. Source frame count is preserved. Select Video fills prompt and seed when a draft is clicked; these remain editable."

    def regenerate(self, server, prompt, width, height, seed, resample_source_video, resample_start_step=1, image=None, last_frame=None):
        return self._generate(server, prompt, width, height, None, seed, image, last_frame, resample_source_video=resample_source_video, resample_start_step=resample_start_step)


_video_candidates = WeakKeyDictionary()


class LightX2VGenerateVideoBatch(LightX2VGenerateVideo):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        negative_prompt = inputs["optional"].pop("negative_prompt")
        inputs["optional"]["draft_steps"] = (
            "INT",
            {"default": 0, "min": 0, "max": 10000, "tooltip": "H3 only. 0 uses config defaults. A positive K below infer_steps generates a K-step draft; K >= infer_steps uses the full schedule."},
        )
        inputs["optional"]["batch_size"] = (
            "INT",
            {"default": 4, "min": 1, "max": 16, "tooltip": "Generate candidates sequentially with seed, seed+1, etc. Keep seed control fixed while choosing a candidate."},
        )
        inputs["optional"]["negative_prompt"] = negative_prompt
        return inputs

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("videos",)
    OUTPUT_IS_LIST = (True,)
    DESCRIPTION = "Generate a native VIDEO list through the same service. Connect videos to Select Video to preview and choose on the canvas, or to native list/video nodes."

    def generate(self, batch_size=4, draft_steps=0, **kwargs):
        if not 1 <= batch_size <= 16:
            raise ValueError("batch_size must be between 1 and 16")
        videos = []
        for index in range(batch_size):
            seed = (kwargs["seed"] + index) & SEED_MAX
            video = self._generate(**(kwargs | {"seed": seed}), draft_steps=draft_steps)[0]
            _video_candidates[video] = {"prompt": kwargs["prompt"], "seed": seed, "preview": video_preview(video)}
            videos.append(video)
        return (videos,)


class LightX2VSelectVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "videos": ("VIDEO",),
                "index": ("INT", {"default": 0, "min": 0, "max": 16, "tooltip": "Set by clicking a draft in the canvas preview. 0 skips regeneration."}),
            },
            "optional": {"selection_batch": ("STRING", {"default": "", "tooltip": "The batch selected in the canvas preview. A new batch clears the selection."})},
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "select"
    CATEGORY = "LightX2V/Server"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    DESCRIPTION = "Preview candidates on the canvas, click one to select it, then Run to regenerate. A new batch requires a new selection."

    def select(self, videos, index, selection_batch=None):
        candidates = [_video_candidates.get(video) for video in videos]
        if not candidates or any(candidate is None for candidate in candidates):
            raise ValueError("Connect Generate Video Batch's videos output to Select Video")
        index = index[0]
        selection_batch = selection_batch[0] if selection_batch else ""
        batch_id = hashlib.sha256(json.dumps([candidate["preview"]["filename"] for candidate in candidates]).encode()).hexdigest()
        ui = {
            "lightx2v_candidates": [candidate["preview"] | {"seed": str(candidate["seed"]), "prompt": candidate["prompt"]} for candidate in candidates],
            "lightx2v_batch_id": [batch_id],
        }
        if index == 0 or selection_batch != batch_id:
            from comfy_execution.graph import ExecutionBlocker

            return {"ui": ui, "result": tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES)}
        if not 1 <= index <= len(candidates):
            raise ValueError(f"Select an index between 1 and {len(candidates)}, or 0 to preview only")
        return {"ui": ui, "result": (videos[index - 1],)}


class LightX2VGenerateImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LX2V_SERVER",),
                "prompt": ("STRING", {"multiline": True, "default": "A cat in a sunlit garden."}),
                "width": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 8}),
                "seed": ("INT", {"default": 42, "min": 0, "max": SEED_MAX}),
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
        payload = {"task": task, "prompt": prompt, "size": [height, width], "seed": seed & SEED_MAX}
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
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1, "tooltip": "Output scale, including fractional values. relayvsr_x4 supports 1–4; relayvsr_x2 supports 1–2."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": SEED_MAX}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "upscale"
    CATEGORY = "LightX2V/Server"

    def upscale(self, server, video, scale=2.0, seed=42):
        settings = server.resolve()
        if settings.settings["task"] != "sr":
            raise ValueError("Set task=sr on the Service connected to Upscale Video")
        model = settings.settings["model_cls"]
        max_scale = 2 if model == "relayvsr_x2" else 4
        if not 1 <= scale <= max_scale:
            raise ValueError(f"{model} scale must be between 1 and {max_scale}, got {scale}")
        width, height = (int(size * scale) for size in video.get_dimensions())
        if width % 2 or height % 2:
            raise ValueError(f"RelayVSR video output must have even dimensions; scale {scale} produces {width}x{height}. Choose another scale.")
        runtime = manager()
        path = video_to_file(video, runtime.root / "inputs")
        payload = {"task": "sr", "video_path": str(path), "sr_ratio": scale, "seed": seed & SEED_MAX}
        result = runtime.run(settings, payload, interrupt=_interrupt, finalize=video_from_file)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "LightX2VServer": LightX2VServer,
    "LightX2VGenerateVideo": LightX2VGenerateVideo,
    "LightX2VRegenerateVideo": LightX2VRegenerateVideo,
    "LightX2VGenerateVideoBatch": LightX2VGenerateVideoBatch,
    "LightX2VSelectVideo": LightX2VSelectVideo,
    "LightX2VGenerateImage": LightX2VGenerateImage,
    "LightX2VUpscaleVideo": LightX2VUpscaleVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LightX2VServer": "LightX2V Service",
    "LightX2VGenerateVideo": "LightX2V Generate Video",
    "LightX2VRegenerateVideo": "LightX2V Regenerate Video",
    "LightX2VGenerateVideoBatch": "LightX2V Generate Video Batch",
    "LightX2VSelectVideo": "LightX2V Select Video",
    "LightX2VGenerateImage": "LightX2V Generate Image",
    "LightX2VUpscaleVideo": "LightX2V Upscale Video",
}
