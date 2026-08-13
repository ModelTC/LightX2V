"""Native LTX-2.5 duration prediction head."""

from __future__ import annotations

import json

import safetensors
import torch
from torch import nn


class AttentionPooler(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_queries: int = 1, num_heads: int = 4):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        queries = self.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        return pooled


class DurationHead(nn.Module):
    """Predict duration in seconds from video and/or audio connector tokens."""

    def __init__(
        self,
        video_cross_attention_dim: int = 4096,
        audio_cross_attention_dim: int = 2048,
        pooler_hidden_dim: int = 256,
        num_queries: int = 1,
        num_pooler_heads: int = 4,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.video_input_proj = nn.Linear(video_cross_attention_dim, pooler_hidden_dim)
        self.video_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)
        self.audio_input_proj = nn.Linear(audio_cross_attention_dim, pooler_hidden_dim)
        self.audio_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)
        self.attention_pooler = AttentionPooler(pooler_hidden_dim, num_queries, num_pooler_heads)
        self.mlp_hidden = nn.Linear(pooler_hidden_dim * num_queries, mlp_hidden)
        self.mlp_out = nn.Linear(mlp_hidden, 1)

    def forward(
        self,
        video_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if video_tokens is None and audio_tokens is None:
            raise ValueError("DurationHead requires video_tokens and/or audio_tokens")
        groups = []
        if video_tokens is not None:
            groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)
        pooled = self.attention_pooler(torch.cat(groups, dim=1)).flatten(1)
        hidden = torch.nn.functional.gelu(self.mlp_hidden(pooled), approximate="tanh")
        return self.mlp_out(hidden).squeeze(-1).exp()


class LTX25DurationPredictor:
    """Load the 15-tensor LTX-2.5 duration head and predict an ``8k+1`` frame count.

    ``device`` is always the compute device.  With ``cpu_offload=True`` the
    3.8 MB head rests on CPU between calls and is moved to ``device`` only for
    prediction.  Lazy/disk streaming is intentionally unnecessary for this
    small component.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        cpu_offload: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.cpu_offload = cpu_offload
        self.head = self._load()

    def _load(self) -> DurationHead:
        load_device = torch.device("cpu") if self.cpu_offload else self.device
        with safetensors.safe_open(self.checkpoint_path, framework="pt", device=str(load_device)) as handle:
            metadata = handle.metadata() or {}
            config = json.loads(metadata.get("config", "{}"))
            transformer_config = config.get("transformer", {})
            head_config = config.get("duration_head", {})
            with torch.device("meta"):
                head = DurationHead(
                    video_cross_attention_dim=transformer_config.get("cross_attention_dim", 4096),
                    audio_cross_attention_dim=transformer_config.get("audio_cross_attention_dim", 2048),
                    pooler_hidden_dim=head_config.get("pooler_hidden_dim", 256),
                    num_queries=head_config.get("num_queries", 1),
                    num_pooler_heads=head_config.get("num_pooler_heads", 4),
                    mlp_hidden=head_config.get("mlp_hidden", 256),
                )
            state = {key.removeprefix("duration_head."): handle.get_tensor(key).to(dtype=self.dtype) for key in handle.keys() if key.startswith("duration_head.")}

        incompatible = head.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"Invalid LTX-2.5 duration checkpoint: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
        return head.to(load_device).eval()

    @torch.inference_mode()
    def predict_seconds(
        self,
        video_context: torch.Tensor | None,
        audio_context: torch.Tensor | None,
    ) -> float:
        if video_context is None and audio_context is None:
            raise ValueError("Duration prediction requires video_context and/or audio_context")
        run_device = self.device
        if self.cpu_offload:
            try:
                self.head = self.head.to(run_device)
            except BaseException:
                self.head = self.head.to("cpu")
                raise
        try:
            video = video_context.to(device=run_device, dtype=self.dtype) if video_context is not None else None
            audio = audio_context.to(device=run_device, dtype=self.dtype) if audio_context is not None else None
            seconds = self.head(video, audio)
            if seconds.shape != (1,):
                raise ValueError(f"Duration prediction only supports batch size 1, got {tuple(seconds.shape)}")
            return float(seconds.item())
        finally:
            if self.cpu_offload:
                self.head = self.head.to("cpu")

    def predict(
        self,
        video_context: torch.Tensor | None,
        audio_context: torch.Tensor | None,
        frame_rate: float,
        min_seconds: float = 1.0,
        max_seconds: float = 20.0,
    ) -> int:
        """Return the clamped frame count on the causal VAE's ``8k+1`` grid."""
        if frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive, got {frame_rate}")
        if min_seconds <= 0 or max_seconds < min_seconds:
            raise ValueError(f"Invalid duration bounds [{min_seconds}, {max_seconds}]")

        seconds = self.predict_seconds(video_context, audio_context)
        min_frames = round(min_seconds * frame_rate)
        max_frames = round(max_seconds * frame_rate)
        raw_frames = max(min_frames, min(round(seconds * frame_rate), max_frames))
        frames = ((raw_frames - 1) // 8) * 8 + 1
        if frames < min_frames:
            frames = min(-(-(min_frames - 1) // 8) * 8 + 1, max_frames)
        return frames

    __call__ = predict


__all__ = ["AttentionPooler", "DurationHead", "LTX25DurationPredictor"]
