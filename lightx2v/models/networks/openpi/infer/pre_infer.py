"""LIBERO input construction for the native PyTorch OpenPI backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import sentencepiece
import torch
from PIL import Image

from ..observation import Observation

LOGGER = logging.getLogger(__name__)
IMAGE_SIZE = 224
LIBERO_STATE_DIM = 8


def load_norm_stats(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)["norm_stats"]
    if "state" not in payload or "actions" not in payload:
        raise ValueError(f"Invalid LIBERO norm_stats file: {path}")
    # Match upstream by preserving the JSON quantiles as float64.
    return {key: {stat: np.asarray(value) for stat, value in stats.items()} for key, stats in payload.items()}


def normalize_quantile(value: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    q01 = stats["q01"][: value.shape[-1]]
    q99 = stats["q99"][: value.shape[-1]]
    return (value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def _require_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an HWC uint8 RGB image, got shape={array.shape}, dtype={array.dtype}")
    return np.ascontiguousarray(array)


def _resize_with_pad(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    height, width = image.shape[:2]
    if (height, width) == (size, size):
        return np.array(image, copy=True)
    ratio = max(width / size, height / size)
    resized_height = int(height / ratio)
    resized_width = int(width / ratio)
    resized = Image.fromarray(image, mode="RGB").resize((resized_width, resized_height), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), 0)
    canvas.paste(resized, ((size - resized_width) // 2, (size - resized_height) // 2))
    return np.asarray(canvas, dtype=np.uint8).copy()


class PaligemmaTokenizer:
    """PaliGemma SentencePiece tokenizer backed by a local model file."""

    def __init__(self, model_path: str | Path, max_len: int = 200):
        self.max_len = max_len
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"PaliGemma tokenizer not found: {model_path}")
        self.processor = sentencepiece.SentencePieceProcessor(model_proto=model_path.read_bytes())

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self.processor.encode(cleaned, add_bos=True) + self.processor.encode("\n")
        if len(tokens) > self.max_len:
            LOGGER.warning("Prompt uses %d tokens; truncating to %d", len(tokens), self.max_len)
        tokens = tokens[: self.max_len]
        mask = [True] * len(tokens)
        padding = self.max_len - len(tokens)
        tokens += [0] * padding
        mask += [False] * padding
        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=np.bool_)


class OpenPIPreInfer:
    """Convert LIBERO observations to padded model tensors."""

    def __init__(
        self,
        norm_stats_path: str | Path,
        tokenizer_path: str | Path,
        device: torch.device | str,
        action_dim: int = 32,
        max_token_len: int = 200,
    ):
        self.norm_stats = load_norm_stats(norm_stats_path)
        self.tokenizer = PaligemmaTokenizer(tokenizer_path, max_len=max_token_len)
        self.device = torch.device(device)
        self.action_dim = action_dim

    def infer(self, images: dict[str, np.ndarray], state: np.ndarray, task_description: str) -> Observation:
        base = _resize_with_pad(_require_rgb(images["agentview"]))
        left_wrist = _resize_with_pad(_require_rgb(images["wrist"]))
        right_wrist = np.zeros_like(base)

        # Quantile normalization runs at simulator input precision upstream.
        raw_state = np.asarray(state).reshape(-1)
        if raw_state.shape != (LIBERO_STATE_DIM,):
            raise ValueError(f"pi05_libero expects an 8-D state, got {raw_state.shape}")
        normalized_state = normalize_quantile(raw_state, self.norm_stats["state"])
        padded_state = np.pad(normalized_state, (0, self.action_dim - LIBERO_STATE_DIM))

        tokens, token_mask = self.tokenizer.tokenize(task_description)

        image_arrays = {
            "base_0_rgb": base,
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        }
        images = {key: torch.from_numpy(value).unsqueeze(0).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0 for key, value in image_arrays.items()}
        return Observation(
            images=images,
            image_masks={
                "base_0_rgb": torch.ones(1, dtype=torch.bool, device=self.device),
                "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device=self.device),
                "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=self.device),
            },
            state=torch.from_numpy(padded_state).unsqueeze(0).to(self.device),
            tokenized_prompt=torch.from_numpy(tokens).unsqueeze(0).to(self.device),
            tokenized_prompt_mask=torch.from_numpy(token_mask).unsqueeze(0).to(self.device),
        )
