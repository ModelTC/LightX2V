"""LIBERO input construction for the native PyTorch OpenPI backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
        payload = json.load(handle)
    payload = payload.get("norm_stats", payload)
    if "state" not in payload or "actions" not in payload:
        raise ValueError(f"Invalid LIBERO norm_stats file: {path}")
    # Keep JSON's float64 precision here. Upstream OpenPI constructs NormStats
    # with ``np.asarray`` as well; casting the quantiles early changes the
    # normalized state by roughly 1e-7.
    return {key: {stat: np.asarray(value) for stat, value in stats.items()} for key, stats in payload.items()}


def normalize_quantile(value: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    q01 = stats["q01"][: value.shape[-1]]
    q99 = stats["q99"][: value.shape[-1]]
    return (value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def _parse_rgb(image: Any) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as pil:
            array = np.asarray(pil.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D RGB image, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        # This matches LiberoInputs: floating environment frames are [0, 1].
        array = np.clip(array, 0.0, 1.0)
        array = np.rint(array * 255.0).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
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
    """Local-file PaliGemma SentencePiece tokenizer matching OpenPI."""

    def __init__(self, model_path: str | Path, max_len: int = 200):
        self.max_len = int(max_len)
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"PaliGemma tokenizer not found: {model_path}")
        self.processor = sentencepiece.SentencePieceProcessor(model_proto=model_path.read_bytes())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned = str(prompt).strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            bins = np.linspace(-1, 1, 257)[:-1]
            discrete = np.digitize(state, bins=bins) - 1
            state_text = " ".join(map(str, discrete))
            text = f"Task: {cleaned}, State: {state_text};\nAction: "
            tokens = self.processor.encode(text, add_bos=True)
        else:
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
    """Map logical LIBERO observations to the model's padded torch tensors."""

    def __init__(
        self,
        norm_stats_path: str | Path,
        tokenizer_path: str | Path,
        device: torch.device | str,
        action_dim: int = 32,
        max_token_len: int = 200,
        discrete_state_input: bool = False,
    ):
        self.norm_stats = load_norm_stats(norm_stats_path)
        self.tokenizer = PaligemmaTokenizer(tokenizer_path, max_len=max_token_len)
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.discrete_state_input = bool(discrete_state_input)

    def infer(self, images: dict, state: Any, task_description: str) -> Observation:
        try:
            agentview = images["agentview"]
            wrist = images["wrist"]
        except KeyError as exc:
            raise KeyError("OpenPI expects images with logical keys 'agentview' and 'wrist'") from exc

        base = _resize_with_pad(_parse_rgb(agentview))
        left_wrist = _resize_with_pad(_parse_rgb(wrist))
        right_wrist = np.zeros_like(base)

        raw_state = np.asarray(state, dtype=np.float32).reshape(-1)
        if raw_state.shape != (LIBERO_STATE_DIM,):
            raise ValueError(f"pi05_libero expects an 8-D state, got {raw_state.shape}")
        normalized_state = normalize_quantile(raw_state, self.norm_stats["state"])
        padded_state = np.pad(normalized_state, (0, self.action_dim - LIBERO_STATE_DIM))

        tokenizer_state = normalized_state if self.discrete_state_input else None
        tokens, token_mask = self.tokenizer.tokenize(task_description, tokenizer_state)

        image_arrays = {
            "base_0_rgb": base,
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        }
        data = {
            "image": {key: torch.from_numpy(value).unsqueeze(0).to(self.device) for key, value in image_arrays.items()},
            "image_mask": {
                "base_0_rgb": torch.ones(1, dtype=torch.bool, device=self.device),
                "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device=self.device),
                "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=self.device),
            },
            "state": torch.from_numpy(padded_state).unsqueeze(0).to(self.device),
            "tokenized_prompt": torch.from_numpy(tokens).unsqueeze(0).to(self.device),
            "tokenized_prompt_mask": torch.from_numpy(token_mask).unsqueeze(0).to(self.device),
        }
        return Observation.from_dict(data)
