"""Load independently exported LightVAE encoder/decoder without training imports."""

import json

import torch
from loguru import logger
from safetensors import safe_open
from torch import nn

from .weights import load_safetensors_subset


class ResidualShortcut(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.conv_shortcut = block.conv_shortcut

    def forward(self, x):
        return x if self.conv_shortcut is None else self.conv_shortcut(x)


def read_architecture(path, kind, config):
    if path is None:
        return None
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        expected = "minimax_h3_pruned_encoder" if kind == "encoder" else "minimax_h3_pruned_vae"
        if meta.get("model_type") != expected:
            raise ValueError(f"Wrong LightVAE {kind} checkpoint type: {path}")
        arch = json.loads(meta["architecture"])
        if arch.get("search") is not None:
            raise ValueError("LightVAE requires an exported fixed architecture, not a search checkpoint")
        for key, value in arch["teacher_config"].items():
            if not key.startswith("_") and config.get(key) != value:
                raise ValueError(f"LightVAE teacher config mismatch for {key}: {path}")
        for key in ("latents_mean", "latents_std"):
            if not torch.allclose(f.get_tensor(key).float(), torch.tensor(config[key]).float(), atol=1e-6, rtol=1e-6):
                raise ValueError(f"LightVAE latent normalization mismatch: {key}")
    field = "kept_layers" if kind == "decoder" else "kept_residual_indices"
    kept = arch[field]
    depth = config["decoder_num_layers"] if kind == "decoder" else len(config["block_out_channels"]) * config["layers_per_block"]
    if not kept or any(type(i) is not int for i in kept) or kept != sorted(set(kept)) or kept[0] < 0 or kept[-1] >= depth:
        raise ValueError(f"Invalid LightVAE {field}: {kept}")
    return arch


def load_components(model, original_path, encoder_path, decoder_path, encoder_arch, decoder_arch):
    if decoder_arch is not None:
        # Exported decoder blocks are already densely reindexed 0..K-1.
        model.decoder.transformer_blocks = nn.ModuleList(list(model.decoder.transformer_blocks)[: len(decoder_arch["kept_layers"])])
    if encoder_arch is not None:
        kept = set(encoder_arch["kept_residual_indices"])
        index = 0
        for stage in model.encoder.down_blocks:
            for i, block in enumerate(stage.resnets):
                if index not in kept:
                    stage.resnets[i] = ResidualShortcut(block)
                index += 1
    reports = {}
    for kind, names, path in (
        ("encoder", ("encoder", "quant_conv"), encoder_path),
        ("decoder", ("decoder", "post_quant_conv"), decoder_path),
    ):
        view = nn.Module()
        for name in names:
            view.add_module(name, getattr(model, name))
        reports[kind] = load_safetensors_subset(view, path or original_path)
        logger.info("H3 {} weights: {} (LightVAE={})", kind, path or original_path, path is not None)
    model.lightvae_architectures = {"encoder": encoder_arch, "decoder": decoder_arch}
    return reports
