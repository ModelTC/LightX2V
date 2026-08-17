"""LTX-2.5 Gemma 4 single-file assets and checkpoint key mappings.

The released text encoder is a self-contained safetensors file: Gemma weights,
the LTX readout projections, Hugging Face config, tokenizer and processor assets
all live in that one file.  This module reconstructs those objects directly in
memory; it intentionally has no dependency on the upstream ``ltx_core`` package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import safetensors
import torch
import transformers
from tokenizers import Tokenizer
from transformers import (
    AutoModelForImageTextToText,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    ProcessorMixin,
)
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.video_processing_utils import BaseVideoProcessor

from lightx2v.models.input_encoders.hf.ltx2.utils import KeyValueOperationResult, SDOps

GEMMA_CONFIG_METADATA_KEY = "gemma_config"
TOKENIZER_JSON_TENSOR_KEY = "tokenizer_json"
HF_ASSET_TENSOR_PREFIX = "hf_asset__"
TOKENIZER_MAX_LENGTH = 1024

_REQUIRED_SIDECARS = ("tokenizer_config.json", "processor_config.json")
_TOKENIZER_CONFIG_SKIP = frozenset(
    {
        "tokenizer_class",
        "auto_map",
        "model_max_length",
        "backend",
        "is_local",
        "local_files_only",
        "processor_class",
        "added_tokens_decoder",
    }
)
_SUBPROCESSOR_BASES: dict[str, type] = {
    "image_processor": BaseImageProcessor,
    "feature_extractor": FeatureExtractionMixin,
    "video_processor": BaseVideoProcessor,
}


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().cpu().numpy()
    if str(array.dtype) == "uint8":
        return array.tobytes()
    return array.astype("uint8").tobytes()


def _resolve_transformers_class(name: str, base: type) -> type:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"Invalid transformers class name {name!r}")
    cls = getattr(transformers, name, None)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, base):
        raise ValueError(f"transformers.{name} is not a {base.__name__}")
    return cls


def _resolve_subprocessor_class(attribute: str, type_name: str) -> type:
    base = _SUBPROCESSOR_BASES[attribute]
    # The upstream pipeline explicitly selects the PIL image implementation;
    # its preprocessing values differ slightly from the torchvision backend.
    if attribute == "image_processor" and not type_name.endswith("Pil"):
        pil_cls = getattr(transformers, f"{type_name}Pil", None)
        if isinstance(pil_cls, type) and issubclass(pil_cls, base):
            return pil_cls
    return _resolve_transformers_class(type_name, base)


@dataclass(frozen=True, slots=True)
class LTX25GemmaAssets:
    source: str
    config_dict: Mapping[str, Any]
    tokenizer_json: bytes
    sidecars: Mapping[str, bytes]

    @classmethod
    def load(cls, path: str | Path) -> "LTX25GemmaAssets":
        path = Path(path)
        if not path.is_file() or path.suffix != ".safetensors":
            raise FileNotFoundError(f"LTX-2.5 text encoder must be a .safetensors file, got {path}")

        with safetensors.safe_open(str(path), framework="pt") as handle:
            metadata = handle.metadata() or {}
            raw_config = metadata.get(GEMMA_CONFIG_METADATA_KEY)
            if raw_config is None:
                raise ValueError(f"{path} is missing safetensors metadata {GEMMA_CONFIG_METADATA_KEY!r}")

            keys = set(handle.keys())
            if TOKENIZER_JSON_TENSOR_KEY not in keys:
                raise ValueError(f"{path} is missing embedded tensor {TOKENIZER_JSON_TENSOR_KEY!r}")
            tokenizer_json = _tensor_to_bytes(handle.get_tensor(TOKENIZER_JSON_TENSOR_KEY))
            sidecars = {key.removeprefix(HF_ASSET_TENSOR_PREFIX): _tensor_to_bytes(handle.get_tensor(key)) for key in keys if key.startswith(HF_ASSET_TENSOR_PREFIX)}

        missing = [name for name in _REQUIRED_SIDECARS if name not in sidecars]
        if missing:
            raise ValueError(f"{path} is missing embedded Hugging Face assets: {', '.join(missing)}")
        return cls(
            source=str(path),
            config_dict=json.loads(raw_config),
            tokenizer_json=tokenizer_json,
            sidecars=sidecars,
        )

    def sidecar_json(self, name: str) -> dict[str, Any]:
        try:
            return json.loads(self.sidecars[name])
        except KeyError as exc:
            raise KeyError(f"Embedded Gemma assets in {self.source} do not contain {name!r}") from exc

    def build_config(self) -> PretrainedConfig:
        model_type = self.config_dict.get("model_type")
        if model_type != "gemma4_unified":
            raise ValueError(f"Expected Gemma 4 unified assets, got model_type={model_type!r}")
        return CONFIG_MAPPING[model_type].from_dict(dict(self.config_dict))

    def build_tokenizer(self, max_length: int = TOKENIZER_MAX_LENGTH) -> PreTrainedTokenizerFast:
        tokenizer_config = self.sidecar_json("tokenizer_config.json")
        kwargs = {key: value for key, value in tokenizer_config.items() if key not in _TOKENIZER_CONFIG_SKIP}
        if "chat_template.jinja" in self.sidecars:
            kwargs.setdefault("chat_template", self.sidecars["chat_template.jinja"].decode())
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_buffer(self.tokenizer_json),
            model_max_length=max_length,
            **kwargs,
        )
        tokenizer.padding_side = "left"
        return tokenizer

    def build_processor(self, tokenizer: PreTrainedTokenizerBase) -> ProcessorMixin:
        processor_config = self.sidecar_json("processor_config.json")
        processor_name = processor_config.get("processor_class")
        processor_cls = _resolve_transformers_class(processor_name, ProcessorMixin)

        components: dict[str, Any] = {"tokenizer": tokenizer}
        for name in processor_cls.get_attributes():
            if name in components:
                continue
            if name not in _SUBPROCESSOR_BASES:
                raise ValueError(f"Unsupported {processor_name} component {name!r}")
            subconfig = processor_config.get(name)
            if not isinstance(subconfig, dict):
                raise ValueError(f"Embedded processor config is missing {name!r}")
            type_name = subconfig.get(f"{name}_type")
            if not type_name:
                raise ValueError(f"Embedded {name!r} config has no {name}_type")
            components[name] = _resolve_subprocessor_class(name, type_name).from_dict(dict(subconfig))

        extra = {key: value for key, value in processor_config.items() if key not in processor_cls.get_attributes() and key != "processor_class"}
        return processor_cls(**components, **extra)

    def build_model(self, *, attn_implementation: str | dict | None = None) -> torch.nn.Module:
        config = self.build_config()
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        with torch.device("meta"):
            return AutoModelForImageTextToText.from_config(config)


def _with_tied_lm_head(ops: SDOps) -> SDOps:
    return ops.with_kv_operation(
        operation=lambda key, value: [
            KeyValueOperationResult(key, value),
            KeyValueOperationResult("model.lm_head.weight", value),
        ],
        key_prefix="model.model.language_model.embed_tokens.weight",
    )


# Released LTX-2.5 TE files use Comfy's flattened Gemma 4 layout.
LTX25_GEMMA_KEY_OPS = _with_tied_lm_head(
    SDOps("LTX25_GEMMA_KEY_OPS")
    .with_matching(prefix="model.")
    .with_matching(prefix="vision_model.")
    .with_matching(prefix="multi_modal_projector.")
    .with_matching(prefix="audio_projector.")
    .with_replacement("model.layers.", "model.model.language_model.layers.")
    .with_replacement("model.embed_tokens.", "model.model.language_model.embed_tokens.")
    .with_replacement("model.norm.", "model.model.language_model.norm.")
    .with_replacement("vision_model.", "model.model.embed_vision.")
    .with_replacement(
        "multi_modal_projector.embedding_projection.",
        "model.model.embed_vision.multimodal_embedder.embedding_projection.",
    )
    .with_replacement("audio_projector.", "model.model.embed_audio.")
)


LTX25_EMBEDDINGS_KEY_OPS = (
    SDOps("LTX25_EMBEDDINGS_KEY_OPS")
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement(
        "text_embedding_projection.video_aggregate_embed.",
        "feature_extractor.video_aggregate_embed.",
    )
    .with_matching(prefix="text_embedding_projection.audio_aggregate_embed.")
    .with_replacement(
        "text_embedding_projection.audio_aggregate_embed.",
        "feature_extractor.audio_aggregate_embed.",
    )
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement(
        "model.diffusion_model.video_embeddings_connector.",
        "embeddings_processor.video_connector.",
    )
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_replacement(
        "model.diffusion_model.audio_embeddings_connector.",
        "embeddings_processor.audio_connector.",
    )
)


def populate_gemma4_buffers(model: torch.nn.Module) -> None:
    """Materialize non-persistent Gemma 4 buffers after meta construction."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    language_model = model.model.language_model
    config = model.config.text_config
    rotary = language_model.rotary_emb
    # Transformers 5.15 moved Gemma 4's local/global attention dimensions into
    # heterogeneous per-layer configs. Reading ``config.head_dim`` globally is
    # rejected there, and opting out of the guard would silently give full
    # attention the wrong RoPE width. Older releases expose no heterogeneous
    # view, so retain their original global-config initialization.
    is_heterogeneous = bool(getattr(config, "is_heterogeneous", False))
    for layer_type in dict.fromkeys(config.layer_types):
        rope_config = config
        if is_heterogeneous:
            per_layer_configs = config.per_layer_config
            try:
                # Transformers >= 5.15 exposes a layer-type keyed view.
                rope_config = per_layer_configs[layer_type]
            except (KeyError, TypeError, ValueError):
                # Compatibility with the older index-based view.
                rope_config = per_layer_configs[config.layer_types.index(layer_type)]

        rope_parameters = rope_config.rope_parameters[layer_type]
        if rope_parameters is None:
            continue
        rope_type = rope_parameters["rope_type"]
        if rope_type == "default":
            inv_freq, attention_scaling = rotary.compute_default_rope_parameters(
                rope_config,
                layer_type=layer_type,
            )
        else:
            kwargs = {"layer_type": layer_type}
            if rope_config is config and layer_type == "full_attention" and rope_type == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](rope_config, **kwargs)
        rotary.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
        rotary.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
        setattr(rotary, f"{layer_type}_attention_scaling", attention_scaling)

    language_model.embed_tokens.register_buffer(
        "embed_scale",
        torch.tensor(config.hidden_size**0.5, device="cpu"),
        persistent=False,
    )


def read_safetensors_metadata(path: str | Path) -> dict[str, Any]:
    """Read and JSON-decode every safetensors metadata entry."""
    with safetensors.safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    return result


__all__ = [
    "LTX25GemmaAssets",
    "LTX25_EMBEDDINGS_KEY_OPS",
    "LTX25_GEMMA_KEY_OPS",
    "populate_gemma4_buffers",
    "read_safetensors_metadata",
]
