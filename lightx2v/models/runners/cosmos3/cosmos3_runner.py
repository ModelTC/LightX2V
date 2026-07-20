import gc
import json
import os
import subprocess
import tempfile
import wave
from collections import deque

import imageio
import imageio_ffmpeg as ffmpeg
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from loguru import logger
from transformers import AutoTokenizer

from lightx2v.models.audio_encoders.hf.cosmos3.sound_tokenizer import Cosmos3SoundTokenizer
from lightx2v.models.networks.cosmos3.model import Cosmos3TransformerModel
from lightx2v.models.runners.cosmos3.policy_runtime import (
    PolicySeedSequence,
    build_json_policy_prompt,
    build_official_policy_prompt,
    normalize_policy_prompt_format,
)
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.cosmos3.scheduler import Cosmos3Scheduler
from lightx2v.models.video_encoders.hf.cosmos3.vae import Cosmos3WanVAE
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import save_to_video
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)

_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."

_ACTION_VIEWPOINT_TEMPLATES = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": "This video is captured from a third-person perspective looking towards the agent from the front.",
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
}

_DROID_CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite sides, with the robot visible."
)

_EMBODIMENT_TO_DOMAIN_ID = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,
    "galbot": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "agibot_gear_gripper": 15,
    "agibot_gear_gripper_ext": 15,
    "fractal": 20,
}

_EMBODIMENT_TO_RAW_ACTION_DIM = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "galbot": 30,
    "agibotworld": 29,
    "agibot_gear_gripper": 29,
    "agibot_gear_gripper_ext": 29,
    "fractal": 10,
    "hand_pose": 57,
}


def _resize_with_zero_pad(image, height, width):
    """Match openpi_client.image_tools.resize_with_pad for one RGB image."""
    image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    current_width, current_height = image.size
    if (current_height, current_width) == (height, width):
        return np.asarray(image, dtype=np.uint8)

    ratio = max(current_width / width, current_height / height)
    resized_height = max(1, int(current_height / ratio))
    resized_width = max(1, int(current_width / ratio))
    resized = image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), 0)
    canvas.paste(resized, (max(0, (width - resized_width) // 2), max(0, (height - resized_height) // 2)))
    return np.asarray(canvas, dtype=np.uint8)


def compose_droid_policy_image(images):
    """Build the 640x540 concat view expected by Cosmos3 Policy-DROID.

    RoboLab's reference client sends a 640x360 wrist view on the top and two
    320x180 shoulder views on the bottom, ordered left then right.
    """

    required = ("wrist_cam", "over_shoulder_left_camera", "over_shoulder_right_camera")
    missing = [name for name in required if name not in images or images[name] is None]
    if missing:
        raise KeyError(f"Cosmos3 Policy-DROID requires cameras: {missing}")

    wrist = _resize_with_zero_pad(images["wrist_cam"], 360, 640)
    shoulder_views = []
    for name in ("over_shoulder_left_camera", "over_shoulder_right_camera"):
        view = _resize_with_zero_pad(images[name], 360, 640)
        # Keep this interpolation identical to RoboLab's Cosmos3Client.
        tensor = torch.from_numpy(view.copy()).permute(2, 0, 1).unsqueeze(0).float()
        tensor = torch.nn.functional.interpolate(tensor, size=(180, 320), mode="bilinear")
        shoulder_views.append(tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
    return np.ascontiguousarray(np.concatenate((wrist, np.concatenate(shoulder_views, axis=1)), axis=0))


@RUNNER_REGISTER("cosmos3")
class Cosmos3Runner(DefaultRunner):
    model_cpu_offload_seq = "transformer->vae->sound_tokenizer"
    _callback_tensor_inputs = ["latents"]

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_tokenizer = self.load_text_encoder()
        self.vae = self.load_vae()

    def load_transformer(self):
        return Cosmos3TransformerModel(
            model_path=os.path.join(self.config["model_path"], "transformer"),
            config=self.config,
            device=self.init_device,
        )

    def load_text_encoder(self):
        tokenizer_path = self.config.get("text_tokenizer_path", os.path.join(self.config["model_path"], "text_tokenizer"))
        if not os.path.exists(tokenizer_path):
            tokenizer_path = self.config["model_path"]
        logger.info(f"Loading Cosmos3 tokenizer from {tokenizer_path}")
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def load_image_encoder(self):
        return None

    def load_vae(self):
        return Cosmos3WanVAE(self.config)

    def load_sound_tokenizer(self):
        logger.info("Loading Cosmos3 sound tokenizer")
        return Cosmos3SoundTokenizer.from_pretrained(self.config["model_path"], device=AI_DEVICE, dtype=GET_DTYPE())

    def init_modules(self):
        logger.info("Initializing Cosmos3 runner modules...")
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.load_model()
        elif self.config.get("lazy_load", False):
            assert self.config.get("cpu_offload", False)
        if hasattr(self, "model") and self.model is not None:
            self.model.set_scheduler(self.scheduler)
        if self.config["task"] not in ("t2i", "t2v", "i2v", "t2av", "i2av", "i2va", "v2av"):
            raise NotImplementedError(f"Cosmos3Runner currently supports tasks t2i/t2v/i2v/t2av/i2av/i2va/v2av, got {self.config['task']}")
        if (self.config.get("enable_sound", False) or self.config["task"] in ("t2av", "i2av")) and not self.config.get("sound_gen", False):
            raise ValueError("Cosmos3 sound generation requires a checkpoint with sound_gen=True.")
        if (self.config.get("action_mode", "") or self.config["task"] in ("i2va", "v2av")) and not self.config.get("action_gen", False):
            raise ValueError("Cosmos3 action generation requires a checkpoint with action_gen=True.")
        self.run_input_encoder = self._run_input_encoder_local
        self.run_dit = self._run_dit_local
        self.config.lock()

    def init_scheduler(self):
        super().init_scheduler()
        if self.config.get("disagg_mode") == "decode":
            return
        self.scheduler = Cosmos3Scheduler(self.config)

    def _append_prompt_template(self, base: str, addition: str) -> str:
        base = (base or "").rstrip(".")
        return f"{base}. {addition}" if base else addition

    def _resolve_prompt_text(self, text: str):
        if not isinstance(text, str):
            return text
        if not os.path.isfile(text):
            return text
        if text.endswith(".json"):
            with open(text, "r") as f:
                return json.dumps(json.load(f))
        with open(text, "r") as f:
            return f.read().strip()

    def _load_action_spec(self):
        if hasattr(self, "_action_spec"):
            return self._action_spec
        action_path = getattr(self.input_info, "action_path", None) or self.config.get("action_path", None)
        self._action_spec = {}
        if action_path and os.path.isfile(action_path):
            with open(action_path, "r") as f:
                spec = json.load(f)
            if isinstance(spec, dict):
                self._action_spec = spec
            else:
                self._action_spec = {"raw_actions": spec}
        return self._action_spec

    def _get_action_mode(self):
        mode = getattr(self.input_info, "action_mode", None) or self.config.get("action_mode", "")
        if self.config["task"] in ("i2va", "v2av") and not mode:
            mode = "inverse_dynamics" if self.config["task"] == "v2av" else "forward_dynamics"
        return mode or None

    def _get_action_value(self, key, default=None):
        value = getattr(self.input_info, key, None)
        if value not in (None, ""):
            return value
        spec = self._load_action_spec()
        if key in spec and spec[key] not in (None, ""):
            return spec[key]
        return self.config.get(key, default)

    def _get_target_video_length(self):
        if self._get_action_mode():
            return int(getattr(self.input_info, "target_video_length", 0) or self.config.get("target_video_length", 1))
        input_frames = int(getattr(self.input_info, "target_video_length", 0) or 0)
        if input_frames and input_frames != 81:
            return input_frames
        return int(self.config.get("target_video_length", input_frames or 1))

    def _prepare_action_context(self):
        if hasattr(self, "_action_spec"):
            del self._action_spec
        if not self._get_action_mode():
            return
        spec = self._load_action_spec()
        if not getattr(self.input_info, "prompt", "") and spec.get("prompt"):
            self.input_info.prompt = spec["prompt"]
        chunk_size = int(self._get_action_value("action_chunk_size", self.config.get("action_chunk_size", 16)))
        self.input_info.action_chunk_size = chunk_size
        self.input_info.target_video_length = chunk_size + 1
        if spec.get("fps") and "target_fps" not in self.config:
            self.input_info.target_fps = float(spec["fps"])

    @staticmethod
    def _build_action_json_prompt(description, view_point, num_frames, fps, height, width, additional_view_description=None):
        duration_seconds = num_frames / fps if fps > 0 else 0.0
        duration = int(duration_seconds) if duration_seconds >= 0 and np.isfinite(duration_seconds) else 0
        action_end = round(duration_seconds) if duration_seconds >= 0 and np.isfinite(duration_seconds) else 0
        minutes, seconds = divmod(action_end, 60)
        desc = description.strip()
        if desc and not desc.endswith((".", "!", "?")):
            desc = f"{desc}."
        prompt = {}
        framing = _ACTION_VIEWPOINT_TEMPLATES.get(view_point)
        if framing and additional_view_description:
            framing = f"{framing} {additional_view_description}"
        if framing:
            prompt["cinematography"] = {"framing": framing}
        ratio = width / height if height > 0 else 1.0
        aspect_ratio = min(
            ("1,1", "4,3", "3,4", "16,9", "9,16"),
            key=lambda r: abs(int(r.split(",")[0]) / int(r.split(",")[1]) - ratio),
        )
        prompt["actions"] = [{"time": f"0:00-{minutes}:{seconds:02d}", "description": desc}]
        prompt["duration"] = f"{duration}s"
        prompt["fps"] = float(fps)
        prompt["resolution"] = {"H": int(height), "W": int(width)}
        prompt["aspect_ratio"] = aspect_ratio
        return json.dumps(prompt)

    def _tokenize_chat(self, text: str, is_image: bool, use_system_prompt=True):
        conversations = []
        if use_system_prompt:
            conversations.append({"role": "system", "content": _SYSTEM_PROMPT_IMAGE if is_image else _SYSTEM_PROMPT_VIDEO})
        conversations.append({"role": "user", "content": text})
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "add_vision_id": False,
            "return_dict": True,
        }
        try:
            encodings = self.text_tokenizer.apply_chat_template(conversations, **kwargs)
        except TypeError:
            kwargs.pop("add_vision_id", None)
            try:
                encodings = self.text_tokenizer.apply_chat_template(conversations, **kwargs)
            except TypeError:
                kwargs.pop("return_dict", None)
                encodings = self.text_tokenizer.apply_chat_template(conversations, **kwargs)

        if isinstance(encodings, dict):
            input_ids = encodings["input_ids"]
        elif hasattr(encodings, "input_ids"):
            input_ids = encodings.input_ids
        else:
            input_ids = encodings
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        return list(input_ids)

    def tokenize_prompt(self, prompt, negative_prompt=None):
        prompt = self._resolve_prompt_text(prompt)
        negative_prompt = self._resolve_prompt_text(negative_prompt) if negative_prompt is not None else None
        height = int(self.input_info.auto_height)
        width = int(self.input_info.auto_width)
        num_frames = self._get_target_video_length()
        fps = float(self.config.get("target_fps", 24.0))
        is_image = num_frames == 1
        negative_prompt = "" if negative_prompt is None else negative_prompt

        action_mode = self._get_action_mode()
        cond_text = prompt
        uncond_text = negative_prompt
        if action_mode:
            view_point = self._get_action_value("view_point", "ego_view")
            domain_name = self._get_action_value("domain_name", None)
            additional_view_description = None
            if action_mode == "policy" and domain_name == "droid_lerobot" and view_point == "concat_view":
                additional_view_description = _DROID_CONCAT_VIEW_DESCRIPTION
            framing = _ACTION_VIEWPOINT_TEMPLATES.get(view_point)
            if framing and additional_view_description:
                framing = f"{framing} {additional_view_description}"
            if action_mode == "policy":
                prompt_format = normalize_policy_prompt_format(self.config.get("policy_prompt_format", "official_text"))
                if prompt_format == "official_text":
                    cond_text = build_official_policy_prompt(prompt, framing, num_frames, fps, height, width)
                else:
                    cond_text = build_json_policy_prompt(prompt, framing, num_frames, fps, height, width)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    logger.info(f"Cosmos3 policy effective prompt ({prompt_format}): {cond_text}")
            else:
                cond_text = self._build_action_json_prompt(
                    prompt,
                    view_point=view_point,
                    num_frames=num_frames,
                    fps=fps,
                    height=height,
                    width=width,
                    additional_view_description=additional_view_description,
                )
        elif not is_image and self.config.get("add_duration_template", True):
            cond_text = self._append_prompt_template(cond_text, f"The video is {num_frames / fps:.1f} seconds long and is of {fps:.0f} FPS.")
            uncond_text = self._append_prompt_template(uncond_text, f"The video is not {num_frames / fps:.1f} seconds long and is not of {fps:.0f} FPS.")
        if not action_mode and self.config.get("add_resolution_template", True):
            if is_image:
                cond_text = self._append_prompt_template(cond_text, f"This image is of {height}x{width} resolution.")
                uncond_text = self._append_prompt_template(uncond_text, f"This image is not of {height}x{width} resolution.")
            else:
                cond_text = self._append_prompt_template(cond_text, f"This video is of {height}x{width} resolution.")
                uncond_text = self._append_prompt_template(uncond_text, f"This video is not of {height}x{width} resolution.")

        eos_token_id = self.text_tokenizer.eos_token_id
        vision_start_id = self.text_tokenizer.convert_tokens_to_ids("<|vision_start|>")
        if eos_token_id is None or vision_start_id is None:
            raise ValueError("Cosmos3 tokenizer must provide eos_token_id and <|vision_start|>.")

        use_system_prompt = self.config.get("use_system_prompt", not bool(action_mode))
        cond_input_ids = self._tokenize_chat(cond_text, is_image=is_image, use_system_prompt=use_system_prompt) + [eos_token_id, vision_start_id]
        uncond_input_ids = self._tokenize_chat(uncond_text, is_image=is_image, use_system_prompt=use_system_prompt) + [eos_token_id, vision_start_id]
        return cond_input_ids, uncond_input_ids

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local(self):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_tokenizer = self.load_text_encoder()
        cond_input_ids, uncond_input_ids = self.tokenize_prompt(
            self.input_info.prompt,
            negative_prompt=self.input_info.negative_prompt,
        )
        self.input_info.txt_seq_lens = [len(cond_input_ids), len(uncond_input_ids)]
        return {
            "text_encoder_output": {
                "cond_input_ids": cond_input_ids,
                "uncond_input_ids": uncond_input_ids,
            },
            "image_encoder_output": None,
        }

    def set_target_shape(self):
        if len(self.input_info.target_shape) == 2:
            height, width = self.input_info.target_shape
            height, width = int(height), int(width)
        else:
            height = int(self.config.get("target_height", 1024))
            width = int(self.config.get("target_width", 1024))

        spatial_scale = int(self.config.get("vae_scale_factor_spatial", self.config.get("vae_scale_factor", 16)))
        temporal_scale = int(self.config.get("vae_scale_factor_temporal", 4))
        if height < spatial_scale or width < spatial_scale:
            raise ValueError(f"Cosmos3 target size must be at least {spatial_scale}x{spatial_scale}, got {height}x{width}.")
        rounded_height = height // spatial_scale * spatial_scale
        rounded_width = width // spatial_scale * spatial_scale
        if rounded_height != height or rounded_width != width:
            logger.warning(f"Cosmos3 target shape rounded from {height}x{width} to {rounded_height}x{rounded_width}")
            height, width = rounded_height, rounded_width

        latent_channels = int(self.config.get("latent_channel", 48))
        pixel_frames = self._get_target_video_length()
        latent_frames = (pixel_frames - 1) // temporal_scale + 1
        self.input_info.auto_height = height
        self.input_info.auto_width = width
        self.input_info.target_shape = (1, latent_channels, latent_frames, height // spatial_scale, width // spatial_scale)
        self.input_info.image_shapes = [[(latent_frames, height // spatial_scale, width // spatial_scale)]]
        logger.info(f"Cosmos3 Runner set target shape: {width}x{height}, latent: {self.input_info.target_shape}")

    def _load_i2v_condition_frame(self):
        image_path = getattr(self.input_info, "image_path", "")
        if not image_path:
            raise ValueError("Cosmos3 i2v requires --image_path.")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Cosmos3 i2v image_path does not exist: {image_path}")
        height = int(self.input_info.auto_height)
        width = int(self.input_info.auto_width)
        resample = getattr(Image, "Resampling", Image).BILINEAR
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((width, height), resample=resample)
            frame = np.asarray(image).astype(np.float32) / 127.5 - 1.0
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        return frame.to(device=AI_DEVICE, dtype=GET_DTYPE())

    @ProfilingContext4DebugL2("Prepare i2v condition")
    def _prepare_i2v_condition_latents(self):
        if self.config["task"] not in ("i2v", "i2av"):
            return
        if hasattr(self.input_info, "vision_condition_latents") and self.input_info.vision_condition_latents is not None:
            return
        loaded_vae_here = not hasattr(self, "vae") or self.vae is None
        if loaded_vae_here:
            self.vae = self.load_vae()
        frame = self._load_i2v_condition_frame()
        num_frames = self._get_target_video_length()
        video = frame.unsqueeze(2).expand(-1, -1, num_frames, -1, -1).contiguous()
        condition_latents = self.vae.encode(video)
        self.input_info.vision_condition_latents = condition_latents
        self.input_info.vision_condition_frame_indexes = [0]
        del video, frame
        if loaded_vae_here and (self.config.get("lazy_load", False) or self.config.get("unload_modules", False)):
            del self.vae
            torch_device_module.empty_cache()
            gc.collect()

    def _frame_array_to_tensor(self, frame, height, width):
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        resample = getattr(Image, "Resampling", Image).BILINEAR
        image = Image.fromarray(frame.astype(np.uint8)).convert("RGB").resize((width, height), resample=resample)
        frame = np.asarray(image).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(frame).permute(2, 0, 1)

    def _load_video_tensor(self, video_path, num_frames, height, width, keep_first=True):
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(f"Cosmos3 action video_path does not exist: {video_path}")
        reader = imageio.get_reader(video_path)
        frames = []
        try:
            for frame in reader:
                frames.append(self._frame_array_to_tensor(np.asarray(frame), height, width))
                if len(frames) >= num_frames:
                    break
        finally:
            reader.close()
        if not frames:
            raise ValueError(f"Cosmos3 could not read frames from video_path: {video_path}")
        if keep_first:
            frames = frames[:1]
        while len(frames) < num_frames:
            frames.append(frames[-1].clone())
        return torch.stack(frames[:num_frames], dim=1).unsqueeze(0).to(device=AI_DEVICE, dtype=GET_DTYPE())

    def _load_image_tensor_by_path(self, image_path, height, width):
        if not image_path or not os.path.isfile(image_path):
            raise FileNotFoundError(f"Cosmos3 action image_path does not exist: {image_path}")
        resample = getattr(Image, "Resampling", Image).BILINEAR
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((width, height), resample=resample)
            frame = np.asarray(image).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device=AI_DEVICE, dtype=GET_DTYPE())

    def _load_policy_image_tensor_by_path(self, image_path, height, width):
        """Load a policy observation with the aspect-preserving bottom/right padding used for DROID."""
        if not image_path or not os.path.isfile(image_path):
            raise FileNotFoundError(f"Cosmos3 policy image_path does not exist: {image_path}")
        with Image.open(image_path) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return self._load_policy_image_tensor(frame, height, width)

    @staticmethod
    def _load_policy_image_tensor(frame, height, width):
        """Convert an in-memory RGB Policy-DROID observation to a model tensor."""
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Cosmos3 policy image must be HxWx3 RGB, got {frame.shape}.")
        source_height, source_width = frame.shape[:2]
        scale = min(width / source_width, height / source_height, 1.0)
        resized_width = max(1, int(scale * source_width + 0.5))
        resized_height = max(1, int(scale * source_height + 0.5))
        if (resized_width, resized_height) != (source_width, source_height):
            resample = getattr(Image, "Resampling", Image).BICUBIC
            frame = np.asarray(Image.fromarray(frame, mode="RGB").resize((resized_width, resized_height), resample=resample), dtype=np.uint8)

        padding_right = width - resized_width
        padding_bottom = height - resized_height
        if padding_right < 0 or padding_bottom < 0:
            raise ValueError(f"Cosmos3 policy resize produced {resized_height}x{resized_width} for target {height}x{width}.")
        if padding_right or padding_bottom:
            pad_mode = "edge" if padding_right >= resized_width or padding_bottom >= resized_height else "reflect"
            frame = np.pad(frame, ((0, padding_bottom), (0, padding_right), (0, 0)), mode=pad_mode)
        frame = frame.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device=AI_DEVICE, dtype=GET_DTYPE())

    def _load_policy_state(self, raw_action_dim):
        payload = getattr(self.input_info, "policy_state", None)
        if payload is None:
            state_path = getattr(self.input_info, "state_path", None) or self.config.get("state_path", "")
            if not state_path or not os.path.isfile(state_path):
                raise FileNotFoundError(f"Cosmos3 policy state_path does not exist: {state_path}")

            suffix = os.path.splitext(state_path)[1].lower()
            if suffix == ".json":
                with open(state_path, "r") as f:
                    payload = json.load(f)
            elif suffix in (".npy", ".npz"):
                payload = np.load(state_path, allow_pickle=True)
                if isinstance(payload, np.lib.npyio.NpzFile):
                    payload = {key: payload[key] for key in payload.files}
                elif isinstance(payload, np.ndarray) and payload.shape == () and isinstance(payload.item(), dict):
                    payload = payload.item()
            else:
                payload = np.loadtxt(state_path, delimiter=",", dtype=np.float32)

        if isinstance(payload, dict):
            if "state" in payload:
                payload = payload["state"]
            elif "joint_position" in payload and "gripper_position" in payload:
                payload = np.concatenate(
                    [
                        np.asarray(payload["joint_position"], dtype=np.float32).reshape(-1),
                        np.asarray(payload["gripper_position"], dtype=np.float32).reshape(-1),
                    ]
                )
            else:
                raise ValueError("Cosmos3 policy state JSON/NPZ must contain `state` or joint/gripper fields.")

        state = np.asarray(payload, dtype=np.float32).reshape(-1)
        if state.size != raw_action_dim:
            raise ValueError(f"Cosmos3 policy state must contain {raw_action_dim} floats, got {state.size}.")
        if self.config.get("policy_flip_gripper", True):
            state = state.copy()
            state[-1] = 1.0 - state[-1]
        return torch.as_tensor(state, device=AI_DEVICE, dtype=GET_DTYPE())

    def _get_action_chunk_index(self):
        return int(getattr(self, "_action_chunk_index", self.config.get("action_chunk_index", 0)))

    def _get_action_num_chunks(self):
        spec = self._load_action_spec()
        default_num_chunks = int(spec.get("num_chunks", 1) or 1)
        if "action_chunks" in spec:
            default_num_chunks = len(spec["action_chunks"])
        num_chunks = int(self.config.get("action_num_chunks", default_num_chunks) or default_num_chunks)
        if "action_chunks" in spec:
            num_chunks = min(num_chunks, len(spec["action_chunks"]))
        return max(num_chunks, 1)

    def _is_action_forward_multichunk(self):
        return self._get_action_mode() == "forward_dynamics" and bool(self.config.get("action_multichunk", False)) and self._get_action_num_chunks() > 1

    def _load_action_raw_actions(self, chunk_size, raw_action_dim):
        spec = self._load_action_spec()
        actions = spec.get("raw_actions")
        actions_from_chunks = False
        if actions is None and "action_chunks" in spec:
            chunk_index = self._get_action_chunk_index()
            actions = spec["action_chunks"][chunk_index]
            actions_from_chunks = True
        if actions is None and "raw_actions" not in spec:
            action_path = getattr(self.input_info, "action_path", None) or self.config.get("action_path", None)
            if action_path and os.path.isfile(action_path):
                with open(action_path, "r") as f:
                    actions = json.load(f)
        if actions is None:
            raise ValueError("Cosmos3 forward_dynamics requires raw actions in --action_path or config action_path.")
        actions = torch.as_tensor(actions, dtype=GET_DTYPE(), device=AI_DEVICE)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions.squeeze(0)
        if actions.ndim != 2:
            raise ValueError(f"Cosmos3 raw actions must have shape [T, D], got {tuple(actions.shape)}")
        if actions.shape[1] != raw_action_dim:
            raise ValueError(f"Cosmos3 raw action dim mismatch for domain: expected {raw_action_dim}, got {actions.shape[1]}")
        if not actions_from_chunks and actions.shape[0] > chunk_size:
            start = self._get_action_chunk_index() * chunk_size
            if start >= actions.shape[0]:
                raise ValueError(f"Cosmos3 action_chunk_index={self._get_action_chunk_index()} is out of range for raw_actions length {actions.shape[0]}.")
            actions = actions[start : start + chunk_size]
        if actions.shape[0] < chunk_size:
            actions = torch.cat([actions, actions[-1:].expand(chunk_size - actions.shape[0], -1)], dim=0)
        return actions[:chunk_size]

    @ProfilingContext4DebugL2("Prepare action condition")
    def _prepare_action_condition_latents(self):
        action_mode = self._get_action_mode()
        if not action_mode:
            return
        if hasattr(self.input_info, "action_latents") or hasattr(self.input_info, "action_latent_shape"):
            return
        chunk_size = int(getattr(self.input_info, "action_chunk_size", 0) or self._get_action_value("action_chunk_size", 16))
        action_dim = int(self.config.get("action_dim", self.config.get("max_action_dim", 64)))
        domain_name = self._get_action_value("domain_name", None)
        if not domain_name:
            raise ValueError("Cosmos3 action generation requires domain_name in config or action JSON.")
        if domain_name not in _EMBODIMENT_TO_DOMAIN_ID or domain_name not in _EMBODIMENT_TO_RAW_ACTION_DIM:
            raise ValueError(f"Unsupported Cosmos3 action domain_name={domain_name!r}")
        raw_action_dim = int(self._get_action_value("raw_action_dim", _EMBODIMENT_TO_RAW_ACTION_DIM[domain_name]))
        if raw_action_dim > action_dim:
            raise ValueError(f"Cosmos3 raw_action_dim={raw_action_dim} exceeds model action_dim={action_dim}")

        height = int(self.input_info.auto_height)
        width = int(self.input_info.auto_width)
        num_frames = chunk_size + 1
        image_path = getattr(self.input_info, "image_path", None) or self.config.get("image_path", "")
        video_path = getattr(self.input_info, "video_path", None) or self.config.get("video_path", "")
        policy_image = getattr(self.input_info, "policy_image", None)

        loaded_vae_here = not hasattr(self, "vae") or self.vae is None
        if loaded_vae_here:
            self.vae = self.load_vae()
        if action_mode == "inverse_dynamics":
            video = self._load_video_tensor(video_path, num_frames, height, width, keep_first=False)
        elif action_mode == "policy" and policy_image is not None:
            frame = self._load_policy_image_tensor(policy_image, height, width)
            video = torch.full(
                (frame.shape[0], frame.shape[1], num_frames, frame.shape[2], frame.shape[3]),
                -1.0,
                device=frame.device,
                dtype=frame.dtype,
            )
            video[:, :, 0] = frame
        elif image_path:
            if action_mode == "policy":
                frame = self._load_policy_image_tensor_by_path(image_path, height, width)
                video = torch.full(
                    (frame.shape[0], frame.shape[1], num_frames, frame.shape[2], frame.shape[3]),
                    -1.0,
                    device=frame.device,
                    dtype=frame.dtype,
                )
                video[:, :, 0] = frame
            else:
                frame = self._load_image_tensor_by_path(image_path, height, width)
                video = frame.unsqueeze(2).expand(-1, -1, num_frames, -1, -1).contiguous()
        else:
            video = self._load_video_tensor(video_path, num_frames, height, width, keep_first=True)

        condition_latents = self.vae.encode(video)
        self.input_info.vision_condition_latents = condition_latents
        if action_mode == "inverse_dynamics":
            self.input_info.vision_condition_frame_indexes = list(range(condition_latents.shape[2]))
        else:
            self.input_info.vision_condition_frame_indexes = [0]

        self.input_info.action_domain_id = int(_EMBODIMENT_TO_DOMAIN_ID[domain_name])
        self.input_info.raw_action_dim = raw_action_dim
        if action_mode == "forward_dynamics":
            raw_actions = self._load_action_raw_actions(chunk_size, raw_action_dim)
            if raw_action_dim < action_dim:
                padding = torch.zeros((chunk_size, action_dim - raw_action_dim), device=AI_DEVICE, dtype=raw_actions.dtype)
                raw_actions = torch.cat([raw_actions, padding], dim=-1)
            self.input_info.action_latents = raw_actions
            self.input_info.action_condition_frame_indexes = list(range(chunk_size))
        elif action_mode == "inverse_dynamics":
            self.input_info.action_latent_shape = (chunk_size, action_dim)
            self.input_info.action_condition_frame_indexes = []
            self.input_info.action_start_frame_offset = 1
        elif action_mode == "policy":
            if not self.config.get("policy_use_state", True):
                raise ValueError("Cosmos3-Nano-Policy-DROID requires policy_use_state=True.")
            state = self._load_policy_state(raw_action_dim)
            action_latents = torch.zeros((chunk_size + 1, action_dim), device=AI_DEVICE, dtype=GET_DTYPE())
            action_latents[0, :raw_action_dim] = state
            self.input_info.action_latents = action_latents
            self.input_info.action_condition_frame_indexes = [0]
            self.input_info.action_start_frame_offset = 0
        else:
            raise ValueError(f"Unsupported Cosmos3 action_mode={action_mode!r}")

        del video
        if loaded_vae_here and (self.config.get("lazy_load", False) or self.config.get("unload_modules", False)):
            del self.vae
            torch_device_module.empty_cache()
            gc.collect()

    def _clear_action_condition_state(self):
        for name in (
            "vision_condition_latents",
            "vision_condition_frame_indexes",
            "action_latents",
            "action_latent_shape",
            "action_condition_frame_indexes",
            "action_domain_id",
            "raw_action_dim",
            "action_start_frame_offset",
        ):
            if hasattr(self.input_info, name):
                delattr(self.input_info, name)

    @ProfilingContext4DebugL2("Run DiT")
    def _run_dit_local(self, total_steps=None):
        if (self.config.get("lazy_load", False) or self.config.get("unload_modules", False)) and (not hasattr(self, "model") or self.model is None):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
        self._prepare_i2v_condition_latents()
        self._prepare_action_condition_latents()
        self.model.scheduler.prepare(self.input_info)
        if hasattr(self.input_info, "vision_condition_latents"):
            self.input_info.vision_condition_latents = None
        return self.run(total_steps)

    @ProfilingContext4DebugL1(
        "Run VAE Decoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_labels=["Cosmos3Runner"],
    )
    def run_vae_decoder(self, latents):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae = self.load_vae()
        images = self.vae.decode(latents, self.input_info)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae
            torch_device_module.empty_cache()
            gc.collect()
        return images

    @ProfilingContext4DebugL1("Run Sound Decoder")
    def run_sound_decoder(self, sound_latents):
        if sound_latents is None:
            return None
        if not hasattr(self, "sound_tokenizer") or self.sound_tokenizer is None:
            self.sound_tokenizer = self.load_sound_tokenizer()
        decoder_dtype = next(self.sound_tokenizer.parameters()).dtype
        sound = self.sound_tokenizer.decode(sound_latents.to(device=AI_DEVICE, dtype=decoder_dtype)).detach().cpu()
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.sound_tokenizer
            torch_device_module.empty_cache()
            gc.collect()
        return sound

    def _write_wav(self, path, audio, sample_rate):
        audio = audio.detach().float().cpu().clamp(-1.0, 1.0)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        pcm = (audio.transpose(0, 1).numpy() * 32767.0).round().astype(np.int16)
        with wave.open(path, "wb") as f:
            f.setnchannels(int(audio.shape[0]))
            f.setsampwidth(2)
            f.setframerate(int(sample_rate))
            f.writeframes(pcm.tobytes())

    def _mux_generated_audio(self, video_path, audio):
        if audio is None:
            return
        sample_rate = int(self.config.get("sound_sampling_rate", 48000))
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        target_dir = os.path.dirname(video_path) or "."
        os.makedirs(target_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".cosmos3_mux_", dir=target_dir) as tmp_dir:
            wav_path = os.path.join(tmp_dir, "cosmos3_sound.wav")
            tmp_video_path = os.path.join(tmp_dir, "cosmos3_muxed.mp4")
            self._write_wav(wav_path, audio, sample_rate)
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                video_path,
                "-i",
                wav_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-f",
                "mp4",
                tmp_video_path,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                stderr = result.stderr.decode(errors="ignore") if result.stderr else "Unknown error"
                logger.warning(f"Cosmos3 generated audio mux failed, keep silent video. Error: {stderr}")
                return
            os.replace(tmp_video_path, video_path)

    def _collect_action_output(self, action_latents):
        if action_latents is None:
            return None
        action_mode = self._get_action_mode()
        if action_mode not in ("inverse_dynamics", "policy"):
            return None
        raw_action_dim = getattr(self.input_info, "raw_action_dim", None)
        # NumPy does not support torch.bfloat16. Policy actions are also more
        # convenient for downstream robot code when persisted as float32.
        action = action_latents.detach().float().cpu()
        if raw_action_dim is not None:
            action = action[:, : int(raw_action_dim)]
        if action_mode == "policy":
            history_length = int(self.config.get("policy_history_length", 1))
            action = action[history_length:]
            if self.config.get("policy_flip_gripper", True) and action.shape[-1] > 0:
                action[:, -1] = 1.0 - action[:, -1]
        return action

    def _save_action_output(self, action):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if action is None:
            return
        save_action_path = getattr(self.input_info, "save_action_path", None)
        if not save_action_path:
            save_result_path = getattr(self.input_info, "save_result_path", None)
            if not save_result_path:
                return
            root, _ = os.path.splitext(save_result_path)
            save_action_path = f"{root}_action.json"
        save_dir = os.path.dirname(save_action_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if save_action_path.endswith(".npy"):
            np.save(save_action_path, action.numpy())
        else:
            with open(save_action_path, "w") as f:
                json.dump(action.tolist(), f)
        logger.info(f"Action saved: {save_action_path}")

    def run(self, total_steps=None):
        if total_steps is None:
            total_steps = self.model.scheduler.infer_steps
        for step_index in range(total_steps):
            logger.info(f"==> step_index: {step_index + 1} / {total_steps}")
            with ProfilingContext4DebugL1("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)
            with ProfilingContext4DebugL1("infer_main"):
                self.model.infer(self.inputs)
            with ProfilingContext4DebugL1("step_post"):
                self.model.scheduler.step_post()
            if self.progress_callback:
                self.progress_callback(((step_index + 1) / total_steps) * 100, 100)
        return self.model.scheduler.latents, self.model.scheduler.generator

    def _images_to_video_tensor(self, images):
        if isinstance(images, torch.Tensor):
            if images.dim() == 5:
                video = images[0].permute(1, 2, 3, 0).contiguous()
            elif images.dim() == 4:
                video = images
            else:
                raise ValueError(f"Cosmos3 video tensor output must be 4D or 5D, got {tuple(images.shape)}")
            return video.detach().float().cpu().clamp(0, 1)
        frames = []
        for image in images:
            if isinstance(image, torch.Tensor):
                frame = image.detach().float().cpu()
            else:
                frame = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0)
            frames.append(frame)
        if not frames:
            raise ValueError("Cosmos3 video output is empty.")
        return torch.stack(frames, dim=0).float().clamp(0, 1)

    def _save_frame_tensor(self, frame, path):
        frame = frame.detach().float().cpu().clamp(0, 1)
        frame = (frame.numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(frame).save(path)

    def _save_images(self, images, input_info, log_prefix="Image saved", sound=None):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if input_info.return_result_tensor or not input_info.save_result_path:
            return
        save_path = input_info.save_result_path
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if self._is_video_output():
            video = self._images_to_video_tensor(images)
            save_to_video(
                video.clamp(0, 1),
                save_path,
                fps=float(self.config.get("target_fps", 24.0)),
                method=self.config.get("save_video_method", "ffmpeg"),
            )
            self._mux_generated_audio(save_path, sound)
            logger.info(f"Video saved: {save_path}")
            return
        image_prefix, image_suffix = os.path.splitext(save_path)
        image_suffix = image_suffix.lstrip(".") or "png"
        if isinstance(images, list) and len(images) > 1:
            for idx, image in enumerate(images):
                path = f"{image_prefix}_{idx:05d}.{image_suffix}"
                image.save(path)
                logger.info(f"{log_prefix}: {path}")
            return
        image = images[0] if isinstance(images, list) else images
        image.save(f"{image_prefix}.{image_suffix}")
        logger.info(f"{log_prefix}: {image_prefix}.{image_suffix}")

    def _finalize_pipeline_outputs(self, input_info, images, latents=None, generator=None, sound=None, action=None):
        if latents is not None:
            del latents
        if generator is not None:
            del generator
        torch_device_module.empty_cache()
        gc.collect()
        output_key = "video" if self._is_video_output() else "images"
        if input_info.return_result_tensor:
            outputs = {output_key: images}
            if sound is not None:
                outputs["sound"] = sound
            if action is not None:
                outputs["action"] = action
            return outputs
        if input_info.save_result_path is not None:
            outputs = {output_key: None}
            if action is not None:
                outputs["action"] = None
            return outputs
        outputs = {output_key: images}
        if sound is not None:
            outputs["sound"] = sound
        if action is not None:
            outputs["action"] = action
        return outputs

    def _is_video_output(self):
        return int(self.config.get("target_video_length", 1)) > 1

    def end_run(self):
        if hasattr(self, "model") and self.model is not None:
            self.model.scheduler.clear()
        elif hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.clear()
        if hasattr(self, "inputs"):
            del self.inputs
        self.input_info = None
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "text_tokenizer"):
                del self.text_tokenizer
            torch_device_module.empty_cache()
            gc.collect()

    @ProfilingContext4DebugL1("RUN action multichunk pipeline")
    def _run_action_forward_multichunk_pipeline(self, input_info):
        num_chunks = self._get_action_num_chunks()
        original_image_path = getattr(input_info, "image_path", "")
        save_result_path = getattr(input_info, "save_result_path", "") or ""
        temp_root = os.path.dirname(save_result_path) or "."
        os.makedirs(temp_root, exist_ok=True)

        stitched_videos = []
        generator = None
        current_image_path = original_image_path
        try:
            with tempfile.TemporaryDirectory(prefix=".cosmos3_rollout_", dir=temp_root) as tmp_dir:
                for chunk_index in range(num_chunks):
                    logger.info(f"Cosmos3 action forward rollout chunk: {chunk_index + 1} / {num_chunks}")
                    self._action_chunk_index = chunk_index
                    input_info.image_path = current_image_path
                    self._clear_action_condition_state()

                    latents, generator = self.run_dit()
                    images = self.run_vae_decoder(latents)
                    video = self._images_to_video_tensor(images)
                    if video.shape[0] <= 1:
                        raise ValueError(f"Cosmos3 action chunk output must contain condition + generated frames, got {video.shape[0]} frame.")

                    chunk_video = video if chunk_index == 0 else video[1:]
                    stitched_videos.append(chunk_video.cpu())
                    if chunk_index + 1 < num_chunks:
                        next_frame_path = os.path.join(tmp_dir, f"chunk_{chunk_index + 1:05d}_last.png")
                        self._save_frame_tensor(video[-1], next_frame_path)
                        current_image_path = next_frame_path

                    if hasattr(self, "model") and self.model is not None:
                        self.model.scheduler.clear()
                    del latents, images, video
                    torch_device_module.empty_cache()
                    gc.collect()
        finally:
            if hasattr(self, "_action_chunk_index"):
                del self._action_chunk_index
            input_info.image_path = original_image_path
            self._clear_action_condition_state()

        video = torch.cat(stitched_videos, dim=0)
        self._save_images(video, input_info, log_prefix="Video saved", sound=None)
        self.end_run()
        return self._finalize_pipeline_outputs(input_info, video, generator=generator)

    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info
        self._prepare_action_context()
        self.set_target_shape()
        self.inputs = self.run_input_encoder()
        logger.info(f"input_info: {self.input_info}")
        if self._is_action_forward_multichunk():
            return self._run_action_forward_multichunk_pipeline(input_info)
        latents, generator = self.run_dit()
        sound_latents = getattr(self.model.scheduler, "sound_latents", None) if hasattr(self, "model") else None
        action_latents = getattr(self.model.scheduler, "action_latents", None) if hasattr(self, "model") else None
        sound = self.run_sound_decoder(sound_latents)
        action = self._collect_action_output(action_latents)
        if self._get_action_mode() == "policy" and not self.config.get("decode_video", False):
            self._save_action_output(action)
            del latents, generator
            self.end_run()
            torch_device_module.empty_cache()
            gc.collect()
            if input_info.return_result_tensor or not getattr(input_info, "save_action_path", None):
                return {"action": action}
            return {"action": None}
        images = self.run_vae_decoder(latents)
        self._save_images(images, input_info, log_prefix="Image saved", sound=sound)
        self._save_action_output(action)
        self.end_run()
        return self._finalize_pipeline_outputs(input_info, images, latents=latents, generator=generator, sound=sound, action=action)


class Cosmos3Policy:
    """Long-running Policy-DROID wrapper used by interactive control loops.

    The normal ``lightx2v.infer`` entry point is intentionally single-shot.
    This wrapper keeps model weights resident, accepts in-memory observations,
    and exposes one action at a time from the generated action chunk.
    """

    def __init__(self, config, *, actions_per_plan=None, binarize_gripper=True):
        from lightx2v.utils.input_info import init_empty_input_info

        if str(config.get("action_mode", "")).strip().lower() != "policy":
            raise ValueError("Cosmos3Policy requires action_mode='policy'.")
        if str(config.get("domain_name", "")).strip().lower() != "droid_lerobot":
            raise ValueError("Cosmos3Policy requires domain_name='droid_lerobot'.")

        self.config = config
        self._input_info_factory = lambda: init_empty_input_info("i2va")
        self.action_dim = int(config.get("raw_action_dim", 8))
        self.action_chunk_size = int(config.get("action_chunk_size", 32))
        requested = self.action_chunk_size if actions_per_plan is None else int(actions_per_plan)
        self.actions_per_plan = max(1, min(requested, self.action_chunk_size))
        self.binarize_gripper = bool(binarize_gripper)
        self.prompt_format = normalize_policy_prompt_format(config.get("policy_prompt_format", "official_text"))
        self._seed_sequence = PolicySeedSequence(config.get("seed", 0))
        self.pending_actions = deque()

        self.runner = Cosmos3Runner(config)
        self.runner.init_modules()

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(config, **kwargs)

    def _plan(self, images, state, task_description):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size != self.action_dim:
            raise ValueError(f"Cosmos3 Policy-DROID state length {state.size} != {self.action_dim}")

        input_info = self._input_info_factory()
        input_info.seed = self._seed_sequence.next_seed()
        input_info.prompt = str(task_description)
        input_info.negative_prompt = ""
        input_info.action_mode = "policy"
        input_info.domain_name = "droid_lerobot"
        input_info.view_point = str(self.config.get("view_point", "concat_view"))
        input_info.return_result_tensor = True
        input_info.policy_image = compose_droid_policy_image(images)
        input_info.policy_state = state

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f"Cosmos3 policy plan: seed={input_info.seed}, prompt_format={self.prompt_format}")

        result = self.runner.run_pipeline(input_info)
        chunk = result.get("action") if isinstance(result, dict) else None
        if chunk is None:
            raise RuntimeError("Cosmos3 Policy-DROID inference returned no action chunk")
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.detach().float().cpu().numpy()
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != self.action_dim:
            raise ValueError(f"Cosmos3 Policy-DROID action chunk must be Nx{self.action_dim}, got {chunk.shape}")
        if self.binarize_gripper:
            chunk = chunk.copy()
            chunk[:, -1] = (chunk[:, -1] > 0.5).astype(np.float32)
        self.pending_actions.extend(chunk[: self.actions_per_plan])

    def next_action(self, *, images, state, task_description):
        if not self.pending_actions:
            self._plan(images, state, task_description)
        return np.asarray(self.pending_actions.popleft(), dtype=np.float32)

    def reset(self):
        self.pending_actions.clear()

    def close(self):
        self.pending_actions.clear()
