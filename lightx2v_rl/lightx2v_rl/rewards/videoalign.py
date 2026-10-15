"""VideoAlign inference preserving FewStepRL's RGB TA / grayscale MQ recipe.

Model imports are lazy. Run this backend in the separate reward environment:
the checkpoint's original Qwen2-VL model layout requires Transformers 4.51.3.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ._vendor.videoalign.prompt_template import build_prompt

IMAGE_FACTOR = 28
FRAME_FACTOR = 2
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
SPECIAL_TOKENS = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]


def grayscale_video(video: torch.Tensor) -> torch.Tensor:
    """Match the old NumPy float64 dot + uint8 truncation exactly."""
    frames = video.detach().cpu().permute(0, 2, 3, 1).numpy()
    gray = np.dot(frames[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return torch.from_numpy(np.repeat(gray[..., None], 3, axis=-1)).permute(0, 3, 1, 2)


def sample_frame_indices(
    total_frames: int,
    source_fps: float,
    data_config: Mapping,
    sample_fps: float | None = None,
) -> torch.Tensor:
    """Port VideoAlign smart_nframes and FewStepRL's in-memory index selection."""
    if total_frames < 2 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("VideoAlign sampling needs at least two frames and positive finite source_fps")
    sample_type = data_config.get("sample_type", "uniform")
    if sample_type == "uniform":
        nframes = data_config.get("num_frames")
        if sample_fps is None and nframes is not None:
            nframes = round(nframes / FRAME_FACTOR) * FRAME_FACTOR
        else:
            fps = data_config.get("fps", 2.0) if sample_fps is None else sample_fps
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("VideoAlign sample fps must be positive and finite")
            max_frames = math.floor(min(768, total_frames) / FRAME_FACTOR) * FRAME_FACTOR
            nframes = min(max(total_frames / source_fps * fps, 4), max_frames)
            nframes = round(nframes / FRAME_FACTOR) * FRAME_FACTOR
        nframes = min(nframes, total_frames)
        if not 2 <= nframes <= total_frames:
            raise ValueError(f"VideoAlign frame count {nframes} is outside [2, {total_frames}]")
        return torch.linspace(0, total_frames - 1, int(nframes)).round().long()
    if sample_type == "multi_pts":
        # Original recipe: four six-frame windows, sampled from an 8-fps stream.
        nframes = int(total_frames * 8 // source_fps)
        if nframes < 7:
            raise ValueError("VideoAlign multi_pts requires at least seven frames after 8-fps sampling")
        frame_indices = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        points = torch.linspace(3, nframes - 4, 4).round().long().tolist()
        return torch.tensor([index for point in points for index in frame_indices[point - 3 : point + 3]])
    raise ValueError(f"Unsupported VideoAlign sample_type: {sample_type}")


def resized_frame_shape(height: int, width: int, max_pixels: float) -> tuple[int, int]:
    """VideoAlign smart_resize with its factor-28 video minimum pixel count."""
    if height <= 0 or width <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError("VideoAlign frame dimensions must be positive with aspect ratio <= 200")
    if not math.isfinite(max_pixels) or max_pixels < IMAGE_FACTOR**2:
        raise ValueError("VideoAlign max_frame_pixels must be positive and at least 784")
    h_bar = max(IMAGE_FACTOR, round(height / IMAGE_FACTOR) * IMAGE_FACTOR)
    w_bar = max(IMAGE_FACTOR, round(width / IMAGE_FACTOR) * IMAGE_FACTOR)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = math.floor(height / beta / IMAGE_FACTOR) * IMAGE_FACTOR
        w_bar = math.floor(width / beta / IMAGE_FACTOR) * IMAGE_FACTOR
    elif h_bar * w_bar < VIDEO_MIN_PIXELS:
        beta = math.sqrt(VIDEO_MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / IMAGE_FACTOR) * IMAGE_FACTOR
        w_bar = math.ceil(width * beta / IMAGE_FACTOR) * IMAGE_FACTOR
    if min(h_bar, w_bar) <= 0:
        raise ValueError("VideoAlign resize produced an empty dimension")
    return h_bar, w_bar


def prepare_video(
    video: torch.Tensor,
    source_fps: float,
    data_config: Mapping,
    sample_fps: float | None = None,
) -> torch.Tensor:
    """Return sampled float TCHW pixels in [0, 255], as in the old wrapper."""
    if video.dtype != torch.uint8 or video.ndim != 4 or video.shape[1] != 3 or not video.shape[0]:
        raise ValueError("VideoAlign expects a nonempty uint8 video with shape [T, 3, H, W]")
    video = video.detach().cpu()
    if video.shape[0] < FRAME_FACTOR:
        video = torch.cat([video, video[-1:].repeat(FRAME_FACTOR - video.shape[0], 1, 1, 1)])
    video = video[sample_frame_indices(video.shape[0], source_fps, data_config, sample_fps)]
    nframes, _, height, width = video.shape
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR),
        int(VIDEO_MIN_PIXELS * 1.05),
    )
    max_pixels = data_config.get("max_frame_pixels") or max_pixels
    target = resized_frame_shape(height, width, max_pixels)
    if target == (height, width):
        return video.float()
    # torchvision's uint8 tensor bicubic resize: cast, interpolate, clamp, round,
    # restore uint8. Keep that quantization before the processor's rescale.
    resized = F.interpolate(video.float(), size=target, mode="bicubic", align_corners=False, antialias=True)
    return resized.clamp(0, 255).round().to(torch.uint8).float()


def normalize_scores(scores: Mapping[str, float], inference_config: Mapping | None) -> dict[str, float]:
    result = {name: float(value) for name, value in scores.items()}
    if inference_config is not None:
        for name, value in result.items():
            mean, std = float(inference_config[f"{name}_mean"]), float(inference_config[f"{name}_std"])
            if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
                raise ValueError(f"Invalid VideoAlign normalization statistics for {name}")
            result[name] = (value - mean) / std
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("VideoAlign returned a non-finite reward")
    return result


def _checkpoint_path(checkpoint_dir: Path) -> Path:
    if (checkpoint_dir / "model.pth").is_file():
        return checkpoint_dir
    candidates = [
        path for path in checkpoint_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(f"No VideoAlign checkpoint-* directory under {checkpoint_dir}")
    return max(candidates, key=lambda path: int(path.name.removeprefix("checkpoint-")))


def _load_model(checkpoint_dir: Path, base_model_path: Path, config: dict, dtype: torch.dtype, attention: str):
    import transformers

    if transformers.__version__.split("+")[0] != "4.51.3":
        raise RuntimeError(
            "VideoAlign's migrated checkpoint requires transformers==4.51.3. "
            "Run the reward server in its separate reward environment; do not downgrade the verl environment."
        )
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    from ._vendor.videoalign.model import Qwen2VLRewardModelBT

    model_config, lora = config["model_config"], config["peft_lora_config"]
    if model_config.get("load_in_4bit") or model_config.get("load_in_8bit"):
        raise ValueError("The migrated VideoAlign backend expects an unquantized checkpoint")
    processor = AutoProcessor.from_pretrained(str(base_model_path), padding_side="right", local_files_only=True)
    special_token_ids = None
    if model_config.get("use_special_tokens"):
        processor.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    model = Qwen2VLRewardModelBT.from_pretrained(
        str(base_model_path),
        output_dim=model_config["output_dim"],
        reward_token=model_config["reward_token"],
        special_token_ids=special_token_ids,
        torch_dtype=dtype,
        attn_implementation=attention,
        use_cache=False,
        local_files_only=True,
    )
    if special_token_ids is not None:
        model.resize_token_embeddings(len(processor.tokenizer))
    # Match VideoAlign's creation order: cast the base before PEFT. PEFT keeps
    # LoRA parameters in fp32 by default, including when checkpoint tensors are
    # bf16; a whole-model dtype cast after loading would undo that behavior.
    model.to(dtype=dtype)
    if lora.get("lora_enable"):
        excluded = lora.get("lora_namespan_exclude") or []
        if isinstance(excluded, str):
            excluded = ast.literal_eval(excluded)
        targets = [
            name for name, module in model.named_modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
            and not any(part in name for part in excluded)
        ]
        if lora.get("num_lora_modules", -1) > 0:
            targets = targets[-lora["num_lora_modules"] :]
        model = get_peft_model(model, LoraConfig(
            target_modules=targets,
            r=lora["lora_r"],
            lora_alpha=lora["lora_alpha"],
            lora_dropout=lora["lora_dropout"],
            task_type=lora["lora_task_type"],
            use_rslora=lora.get("use_rslora", False),
            bias="none",
            modules_to_save=lora.get("lora_modules_to_save"),
        ))
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    checkpoint = _checkpoint_path(checkpoint_dir)
    if (checkpoint / "model.pth").is_file():
        state = torch.load(checkpoint / "model.pth", map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(state, strict=True)
    else:
        from safetensors.torch import load_file

        adapter = load_file(str(checkpoint / "adapter_model.safetensors"))
        adapter = {
            key.replace(".lora_A.", ".lora_A.default.").replace(".lora_B.", ".lora_B.default."): value
            for key, value in adapter.items()
        }
        state = torch.load(checkpoint / "non_lora_state_dict.pth", map_location="cpu", weights_only=True)
        combined = model.state_dict()
        combined.update(state)
        combined.update(adapter)
        model.load_state_dict(combined, strict=True)
    return model.eval().requires_grad_(False), processor


class VideoAlignReward:
    """Lazy local VideoAlign scorer. MQ uses grayscale; VQ and TA use RGB.

    ``checkpoint_dir`` contains model_config.json and checkpoint-11352/model.pth;
    ``base_model_path`` is the separately migrated Qwen2-VL-2B-Instruct directory.
    Scores use checkpoint z-score normalization by default, never [0, 1] scaling.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        base_model_path: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        dimensions: Sequence[str] = ("MQ", "TA"),
        sample_fps: float | None = None,
        normalize: bool = True,
        attention: str = "sdpa",
    ):
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.base_model_path = Path(base_model_path).expanduser()
        self.device = torch.device(device)
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported VideoAlign dtype: {dtype}")
        self.dtype = getattr(torch, dtype)
        self.dimensions = tuple(dict.fromkeys(name.removeprefix("videoalign_").upper() for name in dimensions))
        if not self.dimensions or any(name not in {"VQ", "MQ", "TA"} for name in self.dimensions):
            raise ValueError("VideoAlign dimensions must be a nonempty subset of VQ, MQ, TA")
        if sample_fps is not None and (not math.isfinite(sample_fps) or sample_fps <= 0):
            raise ValueError("VideoAlign sample_fps must be positive and finite")
        if attention not in {"sdpa", "flash_attention_2", "eager"}:
            raise ValueError(f"Unsupported VideoAlign attention backend: {attention}")
        self.sample_fps, self.normalize, self.attention = sample_fps, normalize, attention
        self.model = self.processor = self.config = None

    def load(self) -> VideoAlignReward:
        """Load local assets once, for service startup checks before listening."""
        self._ensure_loaded()
        return self

    def _ensure_loaded(self):
        if self.model is not None:
            return
        with (self.checkpoint_dir / "model_config.json").open() as handle:
            config = json.load(handle)
        if not (self.base_model_path / "config.json").is_file():
            raise FileNotFoundError(f"Missing local Qwen2-VL base model: {self.base_model_path}")
        model, processor = _load_model(
            self.checkpoint_dir, self.base_model_path, config, self.dtype, self.attention
        )
        self.model, self.processor, self.config = model.to(self.device), processor, config

    def _predict(self, video: torch.Tensor, prompt: str, source_fps: float) -> dict[str, float]:
        data = self.config["data_config"]
        tensor = prepare_video(video, source_fps, data, self.sample_fps)
        content = {
            "type": "video", "video": "file://in_memory.mp4",
            "max_pixels": data.get("max_frame_pixels"), "sample_type": data.get("sample_type", "uniform"),
        }
        if self.sample_fps is not None or data.get("num_frames") is None:
            content["fps"] = self.sample_fps if self.sample_fps is not None else data.get("fps", 2.0)
        else:
            content["nframes"] = data["num_frames"]
        chat = [[{"role": "user", "content": [content, {
            "type": "text", "text": build_prompt(prompt, data["eval_dim"], data["prompt_template_type"]),
        }]}]]
        batch = self.processor(
            text=self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True),
            images=None, videos=[tensor], padding=True, return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            rewards = self.model(return_dict=True, **batch)["logits"]
        if rewards.shape != (1, 3):
            raise ValueError(f"VideoAlign must return [1, 3] VQ/MQ/TA logits, got {tuple(rewards.shape)}")
        scores = {name: rewards[0, index].item() for index, name in enumerate(("VQ", "MQ", "TA"))}
        return normalize_scores(scores, self.config.get("inference_config") if self.normalize else None)

    @torch.inference_mode()
    def score(
        self,
        video: torch.Tensor,
        prompt: str,
        source_fps: float = 16.0,
        *,
        dimensions: Sequence[str] | None = None,
    ) -> dict[str, float]:
        if video.dtype != torch.uint8 or video.ndim != 4 or video.shape[1] != 3 or not video.shape[0]:
            raise ValueError("VideoAlign expects a nonempty uint8 video with shape [T, 3, H, W]")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("VideoAlign requires a nonempty text prompt")
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("VideoAlign source_fps must be positive and finite")
        requested = self.dimensions if dimensions is None else tuple(
            dict.fromkeys(name.removeprefix("videoalign_").upper() for name in dimensions)
        )
        if not requested or any(name not in {"VQ", "MQ", "TA"} for name in requested):
            raise ValueError("VideoAlign dimensions must be a nonempty subset of VQ, MQ, TA")
        self._ensure_loaded()
        result = {}
        if any(name in requested for name in ("VQ", "TA")):
            rgb_scores = self._predict(video, prompt, source_fps)
            result.update({f"videoalign_{name.lower()}": rgb_scores[name] for name in requested if name != "MQ"})
        if "MQ" in requested:
            result["videoalign_mq"] = self._predict(grayscale_video(video), prompt, source_fps)["MQ"]
        return result
