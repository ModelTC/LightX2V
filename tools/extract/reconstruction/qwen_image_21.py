"""Differentiable, one-unit-at-a-time Qwen-Image 2.1 backend.

Use the Diffusers reference operators; LightX2V inference kernels with out=
buffers cannot be used to train LoRA factors. No training-framework dependency.
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..export_student import torch_dtype

TEMPORAL_LAYERS = (
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "modulation.1",
    "norm_out.linear",
)


def reference():
    try:
        from diffusers.models.transformers import transformer_qwenimage21
    except ImportError as exc:
        raise ImportError("Reconstruction requires a Diffusers build containing models/transformers/transformer_qwenimage21.py, as used by this repository's Qwen-Image 2.1 training backend.") from exc
    return transformer_qwenimage21


def read_architecture(weights, explicit=None):
    root = weights.source.path
    candidates = [Path(explicit)] if explicit else [root / "config.json", root.parent / "config.json"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise ValueError("Cannot find the base transformer's config.json; set model_config explicitly")
    config = json.loads(path.read_text())
    if config.get("_class_name") != "QwenImage21Transformer2DModel":
        raise ValueError("Only QwenImage21Transformer2DModel is supported")
    defaults = {
        "patch_size": 1,
        "in_channels": 64,
        "out_channels": 64,
        "num_layers": 32,
        "attention_head_dim": 128,
        "num_attention_heads": 32,
        "context_in_dim": 4096,
        "mlp_ratio": 3,
        "axes_dims_rope": [16, 56, 56],
        "eps": 1e-6,
        "causal_condition": True,
    }
    defaults.update({key: config[key] for key in defaults if key in config})
    if defaults["patch_size"] != 1 or not defaults["causal_condition"]:
        raise ValueError("This backend requires patch_size=1 and causal_condition=true")
    return defaults


class LoRALinear(nn.Module):
    def __init__(self, linear, down, up, alpha, trainable):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.down = nn.Parameter(down.to(device=linear.weight.device, dtype=torch.float32).clone(), requires_grad=trainable)
        self.up = nn.Parameter(up.to(device=linear.weight.device, dtype=torch.float32).clone(), requires_grad=trainable)
        self.scale = float(alpha) / down.shape[0]

    def forward(self, x):
        # Explicit casts keep frozen operators in inference precision while
        # retaining FP32 optimizer/master parameters and differentiable casts.
        return F.linear(x, self.weight, self.bias) + F.linear(F.linear(x, self.down.to(x.dtype)), self.up.to(x.dtype)) * self.scale


class InputDtypeLayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x).to(x.dtype)


class Unit(nn.Module):
    def __init__(self, kind, architecture, block_index=None):
        super().__init__()
        ref = reference()
        self.kind, self.block_index = kind, block_index
        self.checkpointing = False
        c = architecture
        dim = c["num_attention_heads"] * c["attention_head_dim"]
        if kind == "temporal":
            self.time_text_embed = ref.QwenImage21TimestepProjEmbeddings(dim)
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=False))
            self.norm_out = ref.QwenImage21AdaLayerNormContinuous(dim, dim, eps=c["eps"])
        elif kind == "input":
            self.img_in = nn.Linear(c["in_channels"], dim, bias=False)
            self.txt_in = ref.QwenImage21TextProjection(c["context_in_dim"], dim, eps=c["eps"])
        elif kind == "block":
            self.transformer_blocks = nn.ModuleDict({str(block_index): ref.QwenImage21TransformerBlock(dim, c["num_attention_heads"], c["attention_head_dim"], c["mlp_ratio"], c["eps"])})
        elif kind == "output":
            # The time projection was already reconstructed. Its cached output
            # feeds the parameterless normalization here.
            self.proj_out = nn.Linear(dim, c["out_channels"] or c["in_channels"], bias=False)
            self.output_norm = InputDtypeLayerNorm(dim, elementwise_affine=False, eps=c["eps"])
        else:
            raise ValueError(kind)
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                module.__class__ = InputDtypeLayerNorm

    def forward(self, record):
        if self.kind == "temporal":
            proj = self.time_text_embed.time_proj(record["timestep"]).to(record["timestep"].dtype)
            embedder = self.time_text_embed.timestep_embedder
            first = embedder.linear_1(proj)
            embedding = embedder.linear_2(embedder.act(first))
            return {"first": first, "embedding": embedding, "modulation": self.modulation(embedding), "scale": self.norm_out.linear(self.norm_out.silu(embedding))}
        if self.kind == "input":
            image = self.img_in(record["hidden_states"])
            text = self.txt_in(record["encoder_hidden_states"])
            zeros = text.new_zeros(1, record["target_tokens"] // 4, text.shape[-1])
            x = torch.cat((text, zeros), dim=1).repeat_interleave(record["repeats"], dim=1)
            x[:, record["image_pad_mask"]] = image
            return x
        if self.kind == "block":
            block = self.transformer_blocks[str(self.block_index)]

            def forward(x):
                return block(
                    hidden_states=x,
                    modulation=record["modulation"],
                    rotary_emb=record["rotary_emb"],
                    target_token_mask=record["target_token_mask"],
                    segments=record["segments"],
                    key_valid=record["key_valid"],
                )

            return checkpoint(forward, record["x"], use_reentrant=False) if self.checkpointing and torch.is_grad_enabled() else forward(record["x"])
        # Only denoising-target tokens contribute to the output objective.
        x = record["x"][:, record["target_token_mask"]]
        return self.proj_out(self.output_norm(x) * (1 + record["scale"][:1, None]))


def build_unit(kind, architecture, weights, device, dtype, factors=None, active=(), block_index=None, checkpointing=False):
    with torch.device("meta"):
        unit = Unit(kind, architecture, block_index)
    state = {}
    for key, tensor in unit.state_dict().items():
        if key not in weights or tuple(tensor.shape) != weights.shapes[key]:
            raise ValueError(f"Model configuration/weight mismatch for {key}: expected {tuple(tensor.shape)}, found {weights.shapes.get(key)}")
        state[key] = weights[key].to(device=device, dtype=dtype)
    unit.load_state_dict(state, strict=True, assign=True)
    if kind == "temporal":
        # Nonpersistent frequency buffers constructed on meta have no checkpoint
        # entries. Recreate them on CPU in FP32, then move without dtype casting.
        unit.time_text_embed.time_proj = reference().QwenImage21TemporalTimesteps(256).to(device)
    unit.requires_grad_(False)
    for name, module in list(unit.named_modules()):
        if not isinstance(module, nn.Linear) or not factors or f"transformer.{name}.lora_down.weight" not in factors:
            continue
        prefix = f"transformer.{name}"
        replacement = LoRALinear(module, factors[f"{prefix}.lora_down.weight"], factors[f"{prefix}.lora_up.weight"], factors[f"{prefix}.alpha"], name in active)
        parent, _, child = name.rpartition(".")
        setattr(unit.get_submodule(parent), child, replacement)
    unit.checkpointing = checkpointing
    return unit.eval()


def update_factors(unit, factors):
    for name, module in unit.named_modules():
        if isinstance(module, LoRALinear):
            factors[f"transformer.{name}.lora_down.weight"] = module.down.detach().float().cpu().clone()
            factors[f"transformer.{name}.lora_up.weight"] = module.up.detach().float().cpu().clone()


def move(record, device, dtype):
    return {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype) if isinstance(value, torch.Tensor) else value for key, value in record.items()}


def prepare_record(raw, architecture, rope):
    """Validate and expand a single-sample Diffusers input on CPU."""
    ref = reference()
    raw = dict(raw)
    image, text, mask = raw["hidden_states"], raw["encoder_hidden_states"], raw["img_mask"].bool()
    if image.ndim != 3 or text.ndim != 3 or image.shape[0] != 1 or text.shape[0] != 1 or mask.shape[0] != 1:
        raise ValueError("Each calibration record must contain exactly one sample")
    if image.shape[-1] != architecture["in_channels"] or text.shape[-1] != architecture["context_in_dim"]:
        raise ValueError("Calibration embedding dimensions do not match the model")
    shapes = [tuple(shape) for shape in raw["img_shapes"]]
    if not shapes or any(len(s) != 3 or s[0] != 1 or s[1] <= 0 or s[2] <= 0 for s in shapes):
        raise ValueError("img_shapes must contain positive (1, height, width) entries")
    target_tokens = shapes[-1][1] * shapes[-1][2]
    if target_tokens % 4 or mask.shape[1] != text.shape[1] + target_tokens // 4 or not mask[0, text.shape[1] :].all():
        raise ValueError("img_mask must include target image slots (one slot per four tokens)")
    repeats = torch.where(mask[0], 4, 1)
    expanded = mask[0].repeat_interleave(repeats)
    if int(expanded.sum()) != image.shape[1]:
        raise ValueError("Image slots and image latent token counts disagree")
    ids, target_mask = ref.QwenImage21Transformer2DModel.build_token_metadata(expanded, shapes)
    valid = torch.ones(1, expanded.numel(), dtype=torch.bool)
    text_valid = raw.get("encoder_hidden_states_mask")
    text_valid = torch.ones(text.shape[:2], dtype=torch.bool) if text_valid is None else text_valid.bool()
    if text_valid.shape != text.shape[:2]:
        raise ValueError("encoder_hidden_states_mask shape differs from text embeddings")
    valid[:, ~expanded] = text_valid[:, ~mask[0, : text.shape[1]]]
    timestep = raw["timestep"].reshape(-1)
    if timestep.numel() != 1 or not torch.isfinite(timestep).all() or not ((timestep >= 0) & (timestep <= 1)).all():
        raise ValueError("Calibration timestep must be a single normalized value in [0, 1]")
    return {
        "hidden_states": image,
        "encoder_hidden_states": text,
        "timestep": timestep,
        "repeats": repeats,
        "image_pad_mask": expanded,
        "target_tokens": target_tokens,
        "target_token_mask": target_mask,
        "key_valid": valid,
        "rotary_emb": rope(shapes, expanded, device=torch.device("cpu")),
        "segments": ref._qwenimage21_prefix_segments(ids, int((~target_mask).sum())),
    }


def trace_records(trace, step_indices, compute_dtype):
    """Convert existing native validation traces, including reference images."""
    if trace.get("format_version") != 1:
        raise ValueError("Expected native validation trace format_version=1")
    condition = trace["condition"]
    indices = step_indices if step_indices is not None else list(range(len(trace["steps"])))
    if not indices or max(indices) >= len(trace["steps"]):
        raise ValueError("step_indices outside the captured trace")
    for index in indices:
        step = trace["steps"][index]
        latents = step["latents"]
        source = condition.get("source_latents")
        shapes = list(condition["source_img_shapes"]) + [(1, trace["height"] // 16, trace["width"] // 16)]
        if latents.shape[1] != shapes[-1][1] * shapes[-1][2]:
            raise ValueError("Trace canvas and latent shape disagree")
        if latents.shape[1] % 4:
            raise ValueError("Target latent token count must be divisible by four")
        yield {
            "hidden_states": torch.cat((source, latents), dim=1) if source is not None else latents,
            "encoder_hidden_states": condition["prompt_embed"],
            "encoder_hidden_states_mask": condition.get("prompt_embed_mask"),
            "img_mask": torch.cat((condition["image_pad_mask"], torch.ones(1, latents.shape[1] // 4, dtype=torch.bool)), dim=1),
            "img_shapes": shapes,
            # Match native inference's low-precision timestep cast before /1000.
            "timestep": (trace["timesteps"][step["step_index"]].to(torch_dtype(compute_dtype)) / 1000).reshape(1),
        }
