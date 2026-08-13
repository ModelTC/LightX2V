from pathlib import Path

import safetensors
import torch

from lightx2v.models.video_encoders.hf.ltx2.video_vae.diffusion_video_decoder import DiffusionVideoDecoder
from lightx2v.models.video_encoders.hf.ltx2.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
from lightx2v.models.video_encoders.hf.ltx2.video_vae.video_vae import VideoDecoder, VideoEncoder
from lightx2v.utils.ltx2_utils import KeyValueOperationResult, ModelConfigurator, SDOps

_CONV_VAE_CLASS_NAME = "CausalVideoAutoencoder"


def is_diffusion_video_vae(config: dict) -> bool:
    """Return whether safetensors metadata describes the LTX-2.5 DiffVAE."""
    return config.get("vae", {}).get("_class_name", _CONV_VAE_CLASS_NAME) != _CONV_VAE_CLASS_NAME


def _prepare_video_encoder_kwargs(config: dict) -> dict:
    """Normalize the flat LTX-2.x and nested LTX-2.5 encoder configs."""
    if "encoder" in config:
        encoder_config = config["encoder"]
        out_channels = encoder_config.get("out_channels", config.get("latent_channels", 128))
        encoder_blocks = encoder_config.get("blocks", encoder_config.get("encoder_blocks", []))
    else:
        encoder_config = config
        out_channels = config.get("latent_channels", 128)
        encoder_blocks = config.get("encoder_blocks", [])

    return {
        "convolution_dimensions": encoder_config.get("dims", config.get("dims", 3)),
        "in_channels": encoder_config.get("in_channels", 3),
        "out_channels": out_channels,
        "encoder_blocks": encoder_blocks,
        "patch_size": encoder_config.get("patch_size", 4),
        "norm_layer": NormLayerType(encoder_config.get("norm_layer", "pixel_norm")),
        "latent_log_var": LogVarianceType(encoder_config.get("latent_log_var", "uniform")),
        "encoder_spatial_padding_mode": PaddingModeType(encoder_config.get("spatial_padding_mode", config.get("encoder_spatial_padding_mode", "zeros"))),
    }


class VideoEncoderConfigurator(ModelConfigurator[VideoEncoder]):
    """Configurator for creating a video VAE Encoder from a configuration dictionary."""

    @classmethod
    def from_config(cls: type[VideoEncoder], config: dict) -> VideoEncoder:
        config = config.get("vae", {})
        return VideoEncoder(**_prepare_video_encoder_kwargs(config))


class VideoDecoderConfigurator(ModelConfigurator[VideoDecoder | DiffusionVideoDecoder]):
    """Configurator for creating a video VAE Decoder from a configuration dictionary."""

    @classmethod
    def from_config(cls: type[VideoDecoder], config: dict) -> VideoDecoder | DiffusionVideoDecoder:
        vae_config = config.get("vae", {})
        if vae_config.get("_class_name", _CONV_VAE_CLASS_NAME) == _CONV_VAE_CLASS_NAME:
            return VideoDecoder(
                convolution_dimensions=vae_config.get("dims", 3),
                in_channels=vae_config.get("latent_channels", 128),
                out_channels=vae_config.get("out_channels", 3),
                decoder_blocks=vae_config.get("decoder_blocks", []),
                patch_size=vae_config.get("patch_size", 4),
                norm_layer=NormLayerType(vae_config.get("norm_layer", "pixel_norm")),
                causal=vae_config.get("causal_decoder", False),
                timestep_conditioning=vae_config.get("timestep_conditioning", True),
                decoder_spatial_padding_mode=PaddingModeType(vae_config.get("spatial_padding_mode", "reflect")),
                base_channels=vae_config.get("decoder_base_channels", 128),
            )

        decoder_config = vae_config.get("decoder", vae_config)
        architecture: dict = {}
        for name in ("stage_channels", "stage_depths"):
            if name in decoder_config:
                architecture[name] = tuple(decoder_config[name])
        for name in ("stage_kernels",):
            if name in decoder_config:
                architecture[name] = tuple(tuple(item) for item in decoder_config[name])
        if "upsamples" in decoder_config:
            architecture["upsamples"] = tuple((tuple(stride), reduction) for stride, reduction in decoder_config["upsamples"])
        if "stage5_kernel" in decoder_config:
            architecture["stage5_kernel"] = tuple(decoder_config["stage5_kernel"])
        if "stage5_channels" in decoder_config:
            architecture["stage5_channels"] = decoder_config["stage5_channels"]

        return DiffusionVideoDecoder(
            in_channels=decoder_config.get("in_channels", vae_config.get("latent_channels", 128)),
            out_channels=decoder_config.get("out_channels", 3),
            patch_size=decoder_config.get("patch_size", 4),
            head_dim=decoder_config.get("head_dim", decoder_config.get("na_head_dim", 64)),
            t_emb_dim=decoder_config.get("t_emb_dim", 384),
            default_num_inference_steps=decoder_config.get("default_num_inference_steps", 2),
            timestep_scale_multiplier=decoder_config.get("timestep_scale_multiplier", 1.0),
            model_output_type=vae_config.get("model_output_type", "v"),
            **architecture,
        )


VAE_DECODER_COMFY_KEYS_FILTER = (
    SDOps("VAE_DECODER_COMFY_KEYS_FILTER")
    .with_matching(prefix="vae.decoder.")
    .with_matching(prefix="vae.per_channel_statistics.")
    .with_matching(prefix="decoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("vae.decoder.", "")
    .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
    .with_replacement("decoder.", "")
)

VAE_ENCODER_COMFY_KEYS_FILTER = (
    SDOps("VAE_ENCODER_COMFY_KEYS_FILTER")
    .with_matching(prefix="vae.encoder.")
    .with_matching(prefix="vae.per_channel_statistics.")
    .with_matching(prefix="encoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("vae.encoder.", "")
    .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
    .with_replacement("encoder.", "")
)


def _split_fused_qkv_param(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    if value.shape[0] % 3:
        raise ValueError(f"Fused QKV parameter {key!r} has indivisible leading dim {value.shape[0]}")
    size = value.shape[0] // 3
    leaf = "weight" if key.endswith(".weight") else "bias"
    prefix = key[: -len(leaf)]
    return [
        KeyValueOperationResult(f"{prefix}to_q.{leaf}", value[:size].detach().clone()),
        KeyValueOperationResult(f"{prefix}to_k.{leaf}", value[size : 2 * size].detach().clone()),
        KeyValueOperationResult(f"{prefix}to_v.{leaf}", value[2 * size :].detach().clone()),
    ]


_GATE_FOLD_TARGETS = (
    (".attn.proj.weight", ".gate_msa"),
    (".attn.proj.bias", ".gate_msa"),
    (".mlp.w_down.weight", ".gate_mlp"),
    (".mlp.w_down.bias", ".gate_mlp"),
    (".context_proj.weight", ".gate_ctx"),
    (".context_proj.bias", ".gate_ctx"),
)


def _strip_decoder_prefix(key: str) -> str | None:
    for prefix in ("vae.decoder.", "decoder."):
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    return None


def _read_diffusion_vae_gates(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    gates: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118 - SafeOpen is not a Mapping
            stripped = _strip_decoder_prefix(key)
            if stripped is not None and stripped.endswith((".gate_msa", ".gate_mlp", ".gate_ctx")):
                gates[stripped] = handle.get_tensor(key)
    return gates


def _gate_key(param_key: str) -> str | None:
    for leaf, gate_suffix in _GATE_FOLD_TARGETS:
        if param_key.endswith(leaf):
            return param_key[: -len(leaf)] + gate_suffix
    return None


def _build_diffusion_decoder_sd_ops(gates: dict[str, torch.Tensor]) -> SDOps:
    def drop(_key: str, _value: torch.Tensor) -> list[KeyValueOperationResult]:
        return []

    def fold(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        gate = gates.get(_gate_key(key) or "")
        if gate is None:
            return [KeyValueOperationResult(key, value)]
        gate = gate.to(device=value.device, dtype=torch.float32)
        folded = value.float() * (gate[:, None] if value.ndim == 2 else gate)
        return [KeyValueOperationResult(key, folded.to(value.dtype))]

    return (
        SDOps("DIFFUSION_VAE_DECODER_KEYS_FILTER")
        .with_matching(prefix="vae.decoder.")
        .with_matching(prefix="decoder.")
        .with_matching(prefix="vae.per_channel_statistics.")
        .with_matching(prefix="per_channel_statistics.")
        .with_replacement("vae.decoder.", "")
        .with_replacement("decoder.", "")
        .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
        .with_replacement("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.")
        .with_replacement("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
        .with_kv_operation(drop, key_prefix="coarse_")
        .with_kv_operation(drop, key_suffix=".gate_msa")
        .with_kv_operation(drop, key_suffix=".gate_mlp")
        .with_kv_operation(drop, key_suffix=".gate_ctx")
        .with_kv_operation(fold, key_suffix=".attn.proj.weight")
        .with_kv_operation(fold, key_suffix=".attn.proj.bias")
        .with_kv_operation(fold, key_suffix=".mlp.w_down.weight")
        .with_kv_operation(fold, key_suffix=".mlp.w_down.bias")
        .with_kv_operation(fold, key_suffix=".context_proj.weight")
        .with_kv_operation(fold, key_suffix=".context_proj.bias")
        .with_kv_operation(_split_fused_qkv_param, key_suffix=".qkv.weight")
        .with_kv_operation(_split_fused_qkv_param, key_suffix=".qkv.bias")
    )


DIFFUSION_VAE_DECODER_KEYS_FILTER = _build_diffusion_decoder_sd_ops({})


def video_decoder_sd_ops_for_checkpoint(checkpoint_path: str, config: dict | None = None) -> SDOps:
    """Select ConvVAE or DiffVAE key transforms, including legacy gate folding."""
    if config is None:
        from lightx2v.utils.ltx2_utils import SafetensorsModelStateDictLoader

        config = SafetensorsModelStateDictLoader().metadata(checkpoint_path)
    if not is_diffusion_video_vae(config):
        return VAE_DECODER_COMFY_KEYS_FILTER
    return _build_diffusion_decoder_sd_ops(_read_diffusion_vae_gates(checkpoint_path))
