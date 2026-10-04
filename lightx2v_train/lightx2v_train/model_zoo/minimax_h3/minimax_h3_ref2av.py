"""LightX2V-Train wrapper for MiniMax-H3's Ref2AV transformer partition."""

from collections import Counter
from math import prod

import torch

from lightx2v_train.utils.registry import MODEL_REGISTER

from ..native.minimax_h3 import (
    MiniMaxH3ReferenceGeometry,
    video_latent_num_frames,
)
from .minimax_h3_t2av import MiniMaxH3T2AVModel


def _single_value(value, name):
    """Undo batch-size-one collation for scalar cache metadata."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"MiniMax-H3 Ref2AV {name} must contain one value, got {tuple(value.shape)}.")
        return value.detach().reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"MiniMax-H3 Ref2AV {name} must contain one value, got {value!r}.")
        return _single_value(value[0], name)
    return value


def _cached_rows(value, name, width):
    if not torch.is_tensor(value):
        raise TypeError(f"MiniMax-H3 Ref2AV {name} must be a tensor.")
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(f"MiniMax-H3 Ref2AV currently requires batch_size=1, got {name}={tuple(value.shape)}.")
        value = value[0]
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"MiniMax-H3 Ref2AV {name} must have shape [rows, {width}], got {tuple(value.shape)}.")
    return value


@MODEL_REGISTER("minimax_h3_ref2av")
class MiniMaxH3Ref2AVModel(MiniMaxH3T2AVModel):
    """Train Ref2AV from pre-encoded prompt and ordered reference conditions.

    A ``condition_path`` payload uses this schema (before batch collation)::

        conditioning.positive = {
            "task": "ref2av",
            "prompt_embeds": Tensor[1, L, 5120],
            "text_token_tags": LongTensor[L],
            "target_height": int,
            "target_width": int,
            "target_num_frames": int,  # 17*n+5, requested 4-15s
            "references": [
                {
                    "kind": "image" | "video" | "audio",
                    "normalized": True,
                    # image/video only: clean normalized, patchified rows
                    "video_latents": Tensor[Rv, 96],
                    "num_latent_frames": int,
                    "latent_height": int,
                    "latent_width": int,
                    # audio, or an optional video soundtrack: clean rows
                    "audio_latents": Tensor[2*A, 32],
                    "num_audio_latents": int,
                },
                ...
            ],
        }

    The list order is semantic and is preserved. Target video/audio latents are
    deliberately absent: pure DMD samples its generated rows online.
    """

    transformer_component = "transformer_ref"

    def prepare_text_condition(self, condition):
        if not isinstance(condition, dict):
            raise TypeError(f"MiniMax-H3 Ref2AV cached condition must be a dict, got {type(condition)!r}.")
        if "prompt_embeds" not in condition or "text_token_tags" not in condition:
            raise KeyError("MiniMax-H3 Ref2AV condition requires prompt_embeds and text_token_tags.")

        prompt_embeds = condition["prompt_embeds"]
        text_token_tags = condition["text_token_tags"]
        if not torch.is_tensor(prompt_embeds) or not torch.is_tensor(text_token_tags):
            raise TypeError("MiniMax-H3 Ref2AV prompt embeddings and tags must be tensors.")
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
            raise ValueError(f"MiniMax-H3 Ref2AV prompt_embeds must have shape [1, tokens, dim], got {tuple(prompt_embeds.shape)}.")
        expected_text_dim = int(getattr(self, "text_dim", 5120))
        if prompt_embeds.shape[2] != expected_text_dim:
            raise ValueError(f"MiniMax-H3 Ref2AV prompt_embeds dim must be {expected_text_dim}, got {prompt_embeds.shape[2]}.")
        if text_token_tags.ndim == 2:
            if text_token_tags.shape[0] != 1:
                raise ValueError("MiniMax-H3 Ref2AV currently requires batch_size=1.")
            text_token_tags = text_token_tags[0]
        if text_token_tags.ndim != 1 or text_token_tags.shape[0] != prompt_embeds.shape[1]:
            raise ValueError(f"MiniMax-H3 Ref2AV text_token_tags must contain one tag per prompt row; got {tuple(text_token_tags.shape)} for {prompt_embeds.shape[1]} rows.")
        valid_tags = torch.tensor([0, 1], device=text_token_tags.device)
        if not bool(torch.isin(text_token_tags, valid_tags).all()):
            raise ValueError("MiniMax-H3 Ref2AV text_token_tags must be video=0 or text=1.")
        if not bool((text_token_tags == 0).any()):
            raise ValueError("MiniMax-H3 Ref2AV prompt cache has no reference vision rows; encode the Ref2VA multimodal presentation, not a T2AV prompt.")

        task = str(_single_value(condition.get("task", "ref2av"), "task")).lower()
        if task == "ref2va":
            task = "ref2av"
        if task != "ref2av":
            raise ValueError(f"MiniMax-H3 Ref2AV condition requires task='ref2av', got {task!r}.")

        target_height = int(_single_value(condition.get("target_height"), "target_height"))
        target_width = int(_single_value(condition.get("target_width"), "target_width"))
        target_num_frames = int(_single_value(condition.get("target_num_frames"), "target_num_frames"))
        if min(target_height, target_width) <= 0 or target_height % 32 or target_width % 32:
            raise ValueError(f"MiniMax-H3 Ref2AV target_height/target_width must be positive multiples of 32, got {target_height}x{target_width}.")
        video_latent_num_frames(target_num_frames)

        references = condition.get("references")
        if not isinstance(references, (list, tuple)) or not references:
            raise ValueError("MiniMax-H3 Ref2AV requires a non-empty ordered references list.")
        if len(references) > 12:
            raise ValueError(f"MiniMax-H3 Ref2AV accepts at most 12 references, got {len(references)}.")

        expected_video_width = self.video_latent_channels * prod(self.patch_size)
        geometry = []
        clean_video_rows = []
        clean_audio_rows = []
        kinds = []
        for index, entry in enumerate(references):
            if not isinstance(entry, dict):
                raise TypeError(f"MiniMax-H3 Ref2AV references[{index}] must be a dict, got {type(entry)!r}.")
            kind = str(_single_value(entry.get("kind"), f"references[{index}].kind")).lower()
            if kind not in {"image", "video", "audio"}:
                raise ValueError(f"MiniMax-H3 Ref2AV references[{index}].kind must be image/video/audio, got {kind!r}.")
            kinds.append(kind)
            if not bool(
                _single_value(
                    entry.get("normalized", False),
                    f"references[{index}].normalized",
                )
            ):
                raise ValueError(f"MiniMax-H3 Ref2AV references[{index}] must be marked normalized=true.")

            video_rows = entry.get("video_latents")
            audio_rows = entry.get("audio_latents")
            num_latent_frames = 0
            latent_height = 0
            latent_width = 0
            num_audio_latents = 0
            if kind != "audio":
                video_rows = _cached_rows(
                    video_rows,
                    f"references[{index}].video_latents",
                    expected_video_width,
                )
                num_latent_frames = int(
                    _single_value(
                        entry.get("num_latent_frames"),
                        f"references[{index}].num_latent_frames",
                    )
                )
                latent_height = int(
                    _single_value(
                        entry.get("latent_height"),
                        f"references[{index}].latent_height",
                    )
                )
                latent_width = int(
                    _single_value(
                        entry.get("latent_width"),
                        f"references[{index}].latent_width",
                    )
                )
                if num_latent_frames <= 0 or latent_height <= 0 or latent_width <= 0 or latent_height % self.patch_size[1] or latent_width % self.patch_size[2]:
                    raise ValueError(f"MiniMax-H3 Ref2AV references[{index}] has invalid visual geometry.")
                if kind == "image" and num_latent_frames != 1:
                    raise ValueError(f"MiniMax-H3 Ref2AV image references[{index}] must have one latent frame.")
                expected_rows = num_latent_frames * (latent_height // self.patch_size[1]) * (latent_width // self.patch_size[2])
                if video_rows.shape[0] != expected_rows:
                    raise ValueError(f"MiniMax-H3 Ref2AV references[{index}] visual geometry requires {expected_rows} rows, got {video_rows.shape[0]}.")
                clean_video_rows.append(video_rows)
            elif video_rows is not None:
                raise ValueError(f"MiniMax-H3 Ref2AV audio references[{index}] cannot contain video_latents.")

            if audio_rows is not None:
                if kind == "image":
                    raise ValueError(f"MiniMax-H3 Ref2AV image references[{index}] cannot carry audio; use a following ordered audio reference.")
                audio_rows = _cached_rows(
                    audio_rows,
                    f"references[{index}].audio_latents",
                    self.audio_latent_channels,
                )
                num_audio_latents = int(
                    _single_value(
                        entry.get("num_audio_latents"),
                        f"references[{index}].num_audio_latents",
                    )
                )
                expected_rows = num_audio_latents * 2
                if num_audio_latents <= 0 or audio_rows.shape[0] != expected_rows:
                    raise ValueError(f"MiniMax-H3 Ref2AV references[{index}] audio geometry requires {expected_rows} rows, got {audio_rows.shape[0]}.")
                clean_audio_rows.append(audio_rows)
            elif kind == "audio":
                raise KeyError(f"MiniMax-H3 Ref2AV audio references[{index}] requires audio_latents.")

            geometry.append(
                MiniMaxH3ReferenceGeometry(
                    kind=kind,
                    num_latent_frames=num_latent_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    num_audio_latents=num_audio_latents,
                )
            )

        counts = Counter(kinds)
        for kind, limit in (("image", 9), ("video", 3), ("audio", 3)):
            if counts[kind] > limit:
                raise ValueError(f"MiniMax-H3 Ref2AV accepts at most {limit} {kind} references, got {counts[kind]}.")
        if counts["image"] + counts["video"] == 0:
            raise ValueError("MiniMax-H3 Ref2AV audio references cannot be used without an image or video reference.")

        condition_video_latents = torch.cat(clean_video_rows).to(
            self.device,
            dtype=torch.float32,
        )
        condition_audio_latents = torch.cat(clean_audio_rows).to(self.device, dtype=torch.float32) if clean_audio_rows else None
        return {
            "prompt_embeds": prompt_embeds.to(
                self.device,
                dtype=self.running_dtype,
            ),
            "text_token_tags": text_token_tags.to(self.device, dtype=torch.long),
            "task": task,
            "references": tuple(geometry),
            "condition_video_latents": condition_video_latents,
            "condition_audio_latents": condition_audio_latents,
            "target_height": target_height,
            "target_width": target_width,
            "target_num_frames": target_num_frames,
        }
