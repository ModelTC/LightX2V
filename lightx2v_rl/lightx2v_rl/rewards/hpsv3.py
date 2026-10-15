"""Framewise HPSv3 rewards migrated from FewStepRL.

Copyright 2025 Jintao Zhang, Haoxu Wang, Jianfei Chen (Apache-2.0).
HPSv3 inference portions: Copyright (c) 2024 HPSv3 Team (MIT); see
``_vendor/hpsv3/LICENSE``. This module never imports FewStepRL at runtime.

The model is loaded only on first use. Run it in the separate reward environment
with Transformers 4.51.3; verl-omni's Transformers 5 environment is incompatible
with the original Qwen2-VL reward checkpoint architecture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

GENERAL_PROMPT = "A high-quality image"
VARIANTS = frozenset({"hpsv3_standard", "hpsv3_general", "hpsv3_percentile"})
IMAGE_PIXELS = 256 * 28 * 28


def aggregate_frame_scores(scores: Sequence[float], *, percentile: bool = False) -> float:
    """Match FewStepRL: arithmetic mean, or mean of top floor(30% * T)."""
    if not scores:
        raise ValueError("HPSv3 requires at least one frame score")
    values = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("HPSv3 returned a non-finite frame score")
    if percentile:
        values = sorted(values, reverse=True)[: max(1, int(len(values) * 0.3))]
    # Legacy reward tensors are float32, even though NumPy averages in float64.
    return float(np.float32(np.mean(values)))


def _resize_frame(image: Image.Image) -> Image.Image:
    """The PIL path of HPSv3's Qwen smart_resize, with its fixed pixel budget."""
    width, height = image.size
    if min(height, width) <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError("HPSv3 frame dimensions must be nonzero with aspect ratio <= 200")
    factor = 28
    target_height = max(factor, round(height / factor) * factor)
    target_width = max(factor, round(width / factor) * factor)
    if target_height * target_width > IMAGE_PIXELS:
        beta = math.sqrt((height * width) / IMAGE_PIXELS)
        target_height = math.floor(height / beta / factor) * factor
        target_width = math.floor(width / beta / factor) * factor
    elif target_height * target_width < IMAGE_PIXELS:
        beta = math.sqrt(IMAGE_PIXELS / (height * width))
        target_height = math.ceil(height * beta / factor) * factor
        target_width = math.ceil(width * beta / factor) * factor
    return image.convert("RGB").resize((target_width, target_height), Image.Resampling.BICUBIC)


class HPSv3Scorer:
    """Score all frames of one uint8 TCHW video with a shared HPSv3 model.

    No temporal subsampling is applied. ``frame_chunk_size=None`` reproduces the
    legacy unchunked inference; a positive size bounds the frame inference batch.
    ``standard`` and ``percentile`` reuse the same prompt-conditioned logits.
    """

    def __init__(
        self,
        base_model_path: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        frame_chunk_size: int | None = None,
        attn_implementation: str = "sdpa",
    ) -> None:
        self.base_model_path = Path(base_model_path).expanduser()
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.device = torch.device(device)
        if frame_chunk_size is not None and (type(frame_chunk_size) is not int or frame_chunk_size < 1):
            raise ValueError("HPSv3 frame_chunk_size must be a positive integer or null")
        if attn_implementation not in {"sdpa", "flash_attention_2", "eager"}:
            raise ValueError("HPSv3 attn_implementation must be sdpa, flash_attention_2, or eager")
        self.frame_chunk_size = frame_chunk_size
        self.attn_implementation = attn_implementation
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load local weights once; useful before advertising service readiness."""
        self._load()

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.base_model_path.is_dir():
            raise FileNotFoundError(f"HPSv3 local Qwen2-VL base model is missing: {self.base_model_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"HPSv3 checkpoint is missing: {self.checkpoint_path}")
        if self.checkpoint_path.suffix != ".safetensors":
            raise ValueError("HPSv3 checkpoint must be the migrated HPSv3.safetensors file")

        import transformers

        if transformers.__version__ != "4.51.3":
            raise RuntimeError(
                "HPSv3 requires transformers==4.51.3 in the isolated reward environment; "
                f"found {transformers.__version__}. Start the reward server with .venv-reward."
            )
        from safetensors.torch import load_file
        from transformers import AutoProcessor

        from ._vendor.hpsv3.model import Qwen2VLRewardModelBT

        processor = AutoProcessor.from_pretrained(
            str(self.base_model_path), padding_side="right", local_files_only=True, trust_remote_code=False
        )
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<|Reward|>"]})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(["<|Reward|>"])
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        model = Qwen2VLRewardModelBT.from_pretrained(
            str(self.base_model_path),
            output_dim=2,
            reward_token="special",
            special_token_ids=special_token_ids,
            torch_dtype=dtype,
            attn_implementation=self.attn_implementation,
            rm_head_type="ranknet",
            use_cache=False,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        model.resize_token_embeddings(len(processor.tokenizer))
        model.to(dtype=dtype)
        model.rm_head.to(torch.float32)
        model.config.tokenizer_padding_side = processor.tokenizer.padding_side
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.load_state_dict(load_file(str(self.checkpoint_path), device="cpu"), strict=True)
        model.requires_grad_(False).eval().to(self.device)
        self._processor = processor
        self._model = model

    @torch.inference_mode()
    def _score_frames(self, frames: list[Image.Image], prompt: str) -> list[float]:
        from ._vendor.hpsv3.prompt import INSTRUCTION, PROMPT_WITH_SPECIAL_TOKEN

        self._load()
        chunk_size = self.frame_chunk_size or len(frames)
        scores = []
        for start in range(0, len(frames), chunk_size):
            chunk = frames[start : start + chunk_size]
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame, "min_pixels": IMAGE_PIXELS, "max_pixels": IMAGE_PIXELS},
                            {"type": "text", "text": INSTRUCTION.format(text_prompt=prompt) + PROMPT_WITH_SPECIAL_TOKEN},
                        ],
                    }
                ]
                for frame in chunk
            ]
            batch = self._processor(
                text=self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                images=[_resize_frame(frame) for frame in chunk],
                padding=True,
                return_tensors="pt",
                videos_kwargs={"do_rescale": True},
            )
            batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logits = self._model(return_dict=True, **batch)["logits"]
            if logits.shape != (len(chunk), 2):
                raise ValueError(f"HPSv3 expected one (mu, sigma) pair per frame; got {tuple(logits.shape)}")
            scores.extend(logits[:, 0].float().cpu().tolist())
        return scores

    def score(
        self,
        video: torch.Tensor,
        prompt: str,
        *,
        variants: Sequence[str] = ("hpsv3_general", "hpsv3_percentile"),
    ) -> dict[str, float]:
        unknown = set(variants) - VARIANTS
        if unknown:
            raise ValueError(f"Unknown HPSv3 reward variants: {sorted(unknown)}")
        if not isinstance(video, torch.Tensor) or video.dtype != torch.uint8 or video.ndim != 4 or video.shape[1] != 3:
            raise ValueError("HPSv3 video must be a uint8 tensor in TCHW format with 3 channels")
        if min(video.shape) <= 0:
            raise ValueError("HPSv3 video must contain at least one nonempty frame")
        if not isinstance(prompt, str):
            raise TypeError("HPSv3 prompt must be a string")
        if not variants:
            return {}
        arrays = video.detach().permute(0, 2, 3, 1).contiguous().cpu().numpy()
        frames = [Image.fromarray(frame) for frame in arrays]
        result = {}
        if "hpsv3_general" in variants:
            result["hpsv3_general"] = aggregate_frame_scores(self._score_frames(frames, GENERAL_PROMPT))
        if {"hpsv3_standard", "hpsv3_percentile"}.intersection(variants):
            scores = self._score_frames(frames, prompt)
            if "hpsv3_standard" in variants:
                result["hpsv3_standard"] = aggregate_frame_scores(scores)
            if "hpsv3_percentile" in variants:
                result["hpsv3_percentile"] = aggregate_frame_scores(scores, percentile=True)
        return result
