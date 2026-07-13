from typing import Any

import torch
from einops import rearrange

from ..modules.model2_2 import WanModel
from .layers import expert_block_forward, timestep_embedding


class FastWAMVideoDiT(WanModel):
    """Wan2.2 video expert with FastWAM's split pre/post interface."""

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode: str = "causal",
        video_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
    ):
        del has_image_pos_emb, in_dim_control_adapter, action_dim, action_group_causal_mask_mode
        if has_image_input or has_ref_conv or add_control_adapter:
            raise ValueError("FastWAM training only supports the Wan2.2 latent-input video expert.")
        if require_vae_embedding or require_clip_embedding or not fuse_vae_embedding_in_latents:
            raise ValueError("FastWAM requires the first-frame VAE latent to be fused into the video input.")
        if action_conditioned:
            raise ValueError("FastWAM conditions actions through the MoT action expert, not the video expert.")
        if hidden_dim != num_heads * attn_head_dim:
            raise ValueError(
                f"hidden_dim must equal num_heads * attn_head_dim, got {hidden_dim} and "
                f"{num_heads} * {attn_head_dim}"
            )

        super().__init__(
            model_type="ti2v",
            patch_size=tuple(patch_size),
            text_len=512,
            in_dim=in_dim,
            dim=hidden_dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=eps,
        )
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    def _validate_inputs(self, x, timestep, context, context_mask):
        if x.ndim != 5:
            raise ValueError(f"latents must be [B,C,T,H,W], got {tuple(x.shape)}")
        if context.ndim != 3:
            raise ValueError(f"context must be [B,L,D], got {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"timestep must be [B] or [1], got {tuple(timestep.shape)}")
        batch_size = x.shape[0]
        if context.shape[0] != batch_size:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}")
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"timestep length must be 1 or {batch_size}, got {timestep.shape[0]}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("Training timestep length must match batch size.")
            timestep = timestep.expand(batch_size)
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
        elif context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"context_mask must have shape {tuple(context.shape[:2])}, got {tuple(context_mask.shape)}"
            )
        return x, timestep, context_mask

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        if video_seq_len % video_tokens_per_frame != 0:
            raise ValueError("Video token count must be divisible by tokens per frame.")
        if self.video_attention_mask_mode == "per_frame_causal":
            frame_count = video_seq_len // video_tokens_per_frame
            mask = torch.tril(torch.ones((frame_count, frame_count), dtype=torch.bool, device=device))
            return mask.repeat_interleave(video_tokens_per_frame, 0).repeat_interleave(
                video_tokens_per_frame, 1
            )
        if self.video_attention_mask_mode == "first_frame_causal":
            mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            mask[:video_tokens_per_frame, video_tokens_per_frame:] = False
            return mask
        raise ValueError(f"Unsupported video_attention_mask_mode: {self.video_attention_mask_mode}")

    def _video_frequencies(self, grid_size, device):
        frames, height, width = grid_size
        half_dim = self.attn_head_dim // 2
        split_sizes = [half_dim - 2 * (half_dim // 3), half_dim // 3, half_dim // 3]
        time_freqs, height_freqs, width_freqs = self.freqs.split(split_sizes, dim=1)
        frequencies = torch.cat(
            [
                time_freqs[:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                height_freqs[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                width_freqs[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ],
            dim=-1,
        )
        return frequencies.reshape(frames * height * width, 1, -1).to(device)

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del action, control_camera_latents_input
        x, timestep, context_mask = self._validate_inputs(x, timestep, context, context_mask)
        if not self.seperated_timestep or not fuse_vae_embedding_in_latents:
            raise ValueError("FastWAM video expert requires separated timestep with fused first-frame latents.")

        batch_size = x.shape[0]
        patch_h, patch_w = self.patch_size[1:]
        if x.shape[3] % patch_h or x.shape[4] % patch_w:
            raise ValueError("Latent spatial dimensions must be divisible by the Wan patch size.")
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)
        token_timesteps = timestep.view(batch_size, 1, 1).expand(
            batch_size, x.shape[2], tokens_per_frame
        ).clone()
        token_timesteps[:, 0] = 0
        token_timesteps = token_timesteps.reshape(batch_size, -1)
        time_embedding = self.time_embedding(
            timestep_embedding(self.freq_dim, token_timesteps.reshape(-1))
        ).reshape(batch_size, -1, self.hidden_dim)
        time_modulation = self.time_projection(time_embedding).unflatten(2, (6, self.hidden_dim))

        patches = self.patch_embedding(x)
        frames, height, width = patches.shape[2:]
        tokens = rearrange(patches, "b c f h w -> b (f h w) c").contiguous()
        context = self.text_embedding(context)
        context_mask = context_mask.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        return {
            "tokens": tokens,
            "freqs": self._video_frequencies((frames, height, width), tokens.device),
            "t": time_embedding,
            "t_mod": time_modulation,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_size": (frames, height, width),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        time_embedding = pre_state["t"]
        shift, scale = (
            self.head.modulation.unsqueeze(0).to(time_embedding) + time_embedding.unsqueeze(2)
        ).chunk(2, dim=2)
        tokens = self.head.head(
            self.head.norm(tokens) * (1 + scale.squeeze(2)) + shift.squeeze(2)
        )
        frames, height, width = pre_state["meta"]["grid_size"]
        return rearrange(
            tokens,
            "b (f h w) (pf ph pw c) -> b c (f pf) (h ph) (w pw)",
            f=frames,
            h=height,
            w=width,
            pf=self.patch_size[0],
            ph=self.patch_size[1],
            pw=self.patch_size[2],
            c=self.out_dim,
        )

    def forward(self, x, timestep, context, context_mask=None, action=None, fuse_vae_embedding_in_latents=False):
        state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        tokens = state["tokens"]
        self_mask = self.build_video_to_video_mask(
            tokens.shape[1], state["meta"]["tokens_per_frame"], tokens.device
        )
        for block in self.blocks:
            def run(value, current_block=block):
                return expert_block_forward(
                    current_block,
                    value,
                    state["context"],
                    state["t_mod"],
                    state["freqs"],
                    state["context_mask"],
                    self_mask,
                )

            if self.use_gradient_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(run, tokens, use_reentrant=False)
            else:
                tokens = run(tokens)
        return self.post_dit(tokens, state)
