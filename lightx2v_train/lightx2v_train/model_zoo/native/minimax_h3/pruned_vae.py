"""Depth-pruned H3 decoder and recoverability-aware, one-block-per-group search.

Search adapts TinyFusion's Gumbel gates and temporary LoRA to reconstruction:
https://github.com/VainF/TinyFusion
The decoder geometry follows Diffusers' AutoencoderKLMiniMaxH3 implementation:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py
"""

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoViTDecoder3d
from torch import nn
from torch.utils.checkpoint import checkpoint

from .vae_protocol import MiniMaxH3VideoVAE
from .vae_geometry import SPATIAL_COMPRESSION_RATIO, TEMPORAL_COMPRESSION_RATIO
from .video_vae import PIXEL_MEAN, PIXEL_STD


class SearchLoRALinear(nn.Linear):
    """Temporary search update; original checkpoint weight names stay unchanged."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        nn.Module.__init__(self)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(linear.weight.new_empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(linear.weight.new_zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = F.linear(hidden_states, self.weight, self.bias)
        update = F.linear(F.linear(hidden_states, self.lora_A), self.lora_B)
        return output + update * self.scale


def _add_search_lora(module: nn.Module, rank: int, alpha: float) -> None:
    for name, child in tuple(module.named_children()):
        if isinstance(child, SearchLoRALinear):
            child.lora_A.requires_grad_(True)
            child.lora_B.requires_grad_(True)
        elif isinstance(child, nn.Linear):
            setattr(module, name, SearchLoRALinear(child, rank, alpha))
        else:
            _add_search_lora(child, rank, alpha)


def _decoder_config(teacher_config: Mapping) -> dict:
    return {
        "in_channels": teacher_config["latent_channels"],
        "out_channels": teacher_config["out_channels"],
        "patch_size": math.prod(teacher_config["spatial_downsample_factors"]),
        "patch_size_t": math.prod(teacher_config["temporal_downsample_factors"]),
        "num_layers": teacher_config["decoder_num_layers"],
        "num_attention_heads": teacher_config["decoder_num_attention_heads"],
        "attention_head_dim": teacher_config["decoder_attention_head_dim"],
        "num_register_tokens": teacher_config["decoder_num_register_tokens"],
        "ffn_mult": teacher_config["decoder_ffn_mult"],
        "rope_theta": teacher_config["decoder_rope_theta"],
        "rope_dim_ratio": teacher_config["decoder_rope_dim_ratio"],
        "norm_eps": teacher_config["decoder_norm_eps"],
    }


def _decoder_rotary_emb(decoder, latent_shape, batch, device):
    grids = [
        2.0 * (torch.arange(0.5, size, device=device, dtype=torch.float32) / size) - 1.0
        for size in latent_shape
    ]
    position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
    position_ids = position_ids.unsqueeze(0).expand(batch, -1, -1)
    suffix_ids = position_ids.new_zeros(batch, decoder.num_register_tokens + 1, 3)
    return decoder.rope(torch.cat([position_ids, suffix_ids], dim=1))


def _project_decoder_tokens(decoder, hidden_states, latent_shape):
    batch = hidden_states.shape[0]
    frames, height, width = latent_shape
    hidden_states = decoder.proj_out(decoder.norm_out(hidden_states))[:, : frames * height * width]
    hidden_states = hidden_states.reshape(
        batch, frames, height, width, decoder.out_channels, decoder.patch_size_t, decoder.patch_size, decoder.patch_size
    )
    return hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().reshape(
        batch, decoder.out_channels, frames * decoder.patch_size_t, height * decoder.patch_size, width * decoder.patch_size
    )


def decode_teacher_suffix(decoder, full_tokens, latent_shape, teacher_feature_index, *, gradient_checkpointing=True):
    """Decode post-block tokens; frozen teacher weights still transmit input gradients."""
    if not 0 <= teacher_feature_index < len(decoder.transformer_blocks):
        raise ValueError("Teacher suffix anchor is outside the decoder.")
    expected_tokens = math.prod(latent_shape) + decoder.num_register_tokens + 1
    if full_tokens.shape[1] != expected_tokens:
        raise ValueError("Teacher suffix requires the full video, register, and auxiliary token sequence.")
    rotary_emb = _decoder_rotary_emb(decoder, latent_shape, full_tokens.shape[0], full_tokens.device)
    hidden_states = full_tokens
    for block in decoder.transformer_blocks[teacher_feature_index + 1 :]:
        if torch.is_grad_enabled() and gradient_checkpointing:
            hidden_states = checkpoint(block, hidden_states, rotary_emb, use_reentrant=False)
        else:
            hidden_states = block(hidden_states, rotary_emb)
    return _project_decoder_tokens(decoder, hidden_states, latent_shape)


class MiniMaxH3PrunedDecoder(MiniMaxH3VideoViTDecoder3d):
    """The original decoder blocks and geometry, with optional differentiable gates."""

    def __init__(self, *, decoder_config: dict, search: dict | None = None):
        super().__init__(**decoder_config)
        self.search_config = search
        self.feature_indices = tuple(range(len(self.transformer_blocks)))
        self.feature_projections = nn.ModuleList()
        self._search_noise = None
        if search is not None:
            self.group_size = int(search.get("group_size", 3))
            depth = len(self.transformer_blocks)
            if self.group_size < 1 or depth % self.group_size:
                raise ValueError("Search group_size must evenly divide the teacher decoder depth.")
            self.temperature = float(search.get("temperature", 4.0))
            self.gate_scale = float(search.get("gate_scale", 100.0))
            if self.temperature <= 0 or self.gate_scale <= 0:
                raise ValueError("Search temperature and gate_scale must be positive.")
            self.gate_logits = nn.Parameter(torch.zeros(depth // self.group_size, self.group_size))

    def configure_search_trainable(self) -> None:
        self.requires_grad_(False)
        rank = int(self.search_config.get("lora_rank", 8))
        if rank < 1:
            raise ValueError("Search lora_rank must be positive.")
        alpha = float(self.search_config.get("lora_alpha", 2 * rank))
        for block in self.transformer_blocks:
            _add_search_lora(block, rank, alpha)
        self.gate_logits.requires_grad_(True)

    def prepare_search_step(self, batch_size: int) -> None:
        # Reuse randomness, not an autograd graph, across windows of the same examples.
        if self.search_config is not None:
            shape = (batch_size, *self.gate_logits.shape)
            self._search_noise = -torch.empty(shape, device=self.gate_logits.device).exponential_().log()

    def clear_search_step(self) -> None:
        self._search_noise = None

    def selected_layers(self, logits: torch.Tensor | None = None) -> list[int]:
        choices = (self.gate_logits if logits is None else logits).detach().argmax(dim=-1)
        offsets = torch.arange(len(choices), device=choices.device) * self.group_size
        return (offsets + choices).cpu().tolist()

    def _search_mask(self, batch_size: int) -> torch.Tensor:
        logits = self.gate_logits.float().unsqueeze(0).expand(batch_size, -1, -1) * self.gate_scale
        if not self.training:
            return F.one_hot(logits.argmax(dim=-1), self.group_size).to(logits.dtype).flatten(1)
        noise = self._search_noise
        if noise is None:
            noise = -torch.empty_like(logits).exponential_().log()
        elif noise.shape[0] != batch_size:
            raise ValueError("prepare_search_step batch size differs from the decoded tile batch.")
        soft = ((logits + noise) / self.temperature).softmax(dim=-1)
        hard = F.one_hot(soft.argmax(dim=-1), self.group_size).to(soft.dtype)
        return (hard - soft.detach() + soft).flatten(1)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def forward(self, hidden_states: torch.Tensor, *, return_features: bool = False, auxiliary_feature_index: int | None = None):
        if auxiliary_feature_index is not None and not 0 <= auxiliary_feature_index < len(self.transformer_blocks):
            raise ValueError("Auxiliary feature index is outside the student decoder.")
        batch, channels, frames, height, width = hidden_states.shape
        latent_shape = (frames, height, width)
        patches = frames * height * width
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch, patches, channels)
        hidden_states = self.proj_in(hidden_states)
        register_tokens = self.register_tokens.expand(batch, -1, -1)
        hidden_states = torch.cat(
            [hidden_states, register_tokens, torch.zeros_like(hidden_states[:, :1])], dim=1
        )

        rotary_emb = _decoder_rotary_emb(self, latent_shape, batch, hidden_states.device)

        mask = self._search_mask(batch) if self.search_config is not None else None
        features = []
        auxiliary_tokens = None
        for index, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                output = checkpoint(block, hidden_states, rotary_emb, use_reentrant=False)
            else:
                output = block(hidden_states, rotary_emb)
            if mask is None:
                hidden_states = output
            else:
                # All candidate blocks run: identity bypass must retain the gate's gradient.
                gate = mask[:, index, None, None].to(output.dtype)
                hidden_states = gate * output + (1 - gate) * hidden_states
            if return_features and index in self.feature_indices:
                feature = hidden_states[:, :patches].reshape(batch, frames, height, width, -1)
                features.append(feature.permute(0, 4, 1, 2, 3))
            if index == auxiliary_feature_index:
                auxiliary_tokens = hidden_states

        hidden_states = _project_decoder_tokens(self, hidden_states, latent_shape)
        if auxiliary_feature_index is not None:
            return hidden_states, tuple(features), auxiliary_tokens
        return (hidden_states, tuple(features)) if return_features else hidden_states


class MiniMaxH3PrunedVideoVAE(MiniMaxH3VideoVAE):
    """H3 latent/stitching protocol backed by a searched or fixed-depth ViT decoder."""

    config_name = "pruned_vae_config.json"
    weights_name = "minimax_h3_pruned_vae.safetensors"

    def __init__(
        self,
        teacher_config: dict,
        *,
        kept_layers: list[int] | None = None,
        search: dict | None = None,
        use_tiling: bool = True,
        tile_sample_min_height: int = 256,
        tile_sample_min_width: int = 256,
        tile_sample_min_overlap_height: int = 64,
        tile_sample_min_overlap_width: int = 64,
    ):
        nn.Module.__init__(self)
        decoder_config = _decoder_config(teacher_config)
        if (
            decoder_config["patch_size"] != SPATIAL_COMPRESSION_RATIO
            or decoder_config["patch_size_t"] != TEMPORAL_COMPRESSION_RATIO
            or teacher_config["clip_length"] != 17
            or teacher_config["token_drop"] != 3
        ):
            raise ValueError("The H3 stitching protocol requires compression (4,16,16), clip_length=17, token_drop=3.")
        depth = decoder_config["num_layers"]
        if search is not None and kept_layers is not None:
            raise ValueError("Search starts from the full teacher; kept_layers is only for the fixed student.")
        self.kept_layers = list(range(depth)) if kept_layers is None else list(kept_layers)
        if not self.kept_layers or self.kept_layers != sorted(set(self.kept_layers)):
            raise ValueError("kept_layers must be nonempty, unique, and in teacher order.")
        if self.kept_layers[0] < 0 or self.kept_layers[-1] >= depth:
            raise ValueError("kept_layers contains an index outside the teacher decoder.")
        decoder_config["num_layers"] = len(self.kept_layers)
        self.decoder = MiniMaxH3PrunedDecoder(decoder_config=decoder_config, search=search)
        latent_channels = teacher_config["latent_channels"]
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1)
        self.register_buffer("latents_mean", torch.tensor(teacher_config["latents_mean"], dtype=torch.float32))
        self.register_buffer("latents_std", torch.tensor(teacher_config["latents_std"], dtype=torch.float32))
        self.register_buffer("pixel_mean", torch.tensor(PIXEL_MEAN, dtype=torch.float32), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(PIXEL_STD, dtype=torch.float32), persistent=False)
        self.use_tiling = use_tiling
        self.tile_sample_min_height = tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width
        self.tile_sample_min_overlap_height = tile_sample_min_overlap_height
        self.tile_sample_min_overlap_width = tile_sample_min_overlap_width
        self.architecture_config = {
            "teacher_config": dict(teacher_config),
            "kept_layers": self.kept_layers if search is None else None,
            "search": dict(search) if search is not None else None,
            "tile_sample_min_height": tile_sample_min_height,
            "tile_sample_min_width": tile_sample_min_width,
            "tile_sample_min_overlap_height": tile_sample_min_overlap_height,
            "tile_sample_min_overlap_width": tile_sample_min_overlap_width,
        }
        if search is not None:
            self.configure_search_trainable()

    def configure_search_trainable(self) -> None:
        self.requires_grad_(False)
        self.decoder.configure_search_trainable()

    def forward(self, normalized_latents: torch.Tensor, *, return_features: bool = False, auxiliary_feature_index: int | None = None):
        vae_latents = self.denormalize_latents(normalized_latents)
        return self.decoder(
            self.post_quant_conv(vae_latents),
            return_features=return_features,
            auxiliary_feature_index=auxiliary_feature_index,
        )

    def initialize_from_teacher_state(self, teacher_state: Mapping[str, torch.Tensor]) -> None:
        """Copy only original decoder weights, never search LoRA updates or gates."""
        target = self.state_dict()
        copied = {}
        for name in target:
            if name in ("latents_mean", "latents_std") or name.endswith(("lora_A", "lora_B", "gate_logits")):
                continue
            source_name = name
            prefix = "decoder.transformer_blocks."
            if name.startswith(prefix):
                index, suffix = name[len(prefix):].split(".", 1)
                source_name = f"{prefix}{self.kept_layers[int(index)]}.{suffix}"
            copied[name] = teacher_state[source_name]
        self.load_state_dict({**target, **copied}, strict=True)

    def initialize_from_teacher(self, teacher_vae) -> None:
        self.initialize_from_teacher_state(teacher_vae.state_dict())

    def prepare_search_step(self, batch_size: int) -> None:
        self.decoder.prepare_search_step(batch_size)

    def clear_search_step(self) -> None:
        self.decoder.clear_search_step()

    def selected_layers(self, logits: torch.Tensor | None = None) -> list[int]:
        return self.decoder.selected_layers(logits)

    def fsdp_modules(self) -> tuple[nn.Module, ...]:
        return tuple(self.decoder.transformer_blocks)
