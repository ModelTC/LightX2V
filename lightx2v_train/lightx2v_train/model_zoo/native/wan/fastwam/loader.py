import inspect
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from ..modules.t5 import T5EncoderModel
from ..modules.vae2_2 import Wan2_2_VAE
from .video_dit import FastWAMVideoDiT

logger = logging.getLogger(__name__)
@dataclass
class Wan22LoadedComponents:
    dit: FastWAMVideoDiT
    vae: Wan2_2_VAE
    text_encoder: T5EncoderModel | None
    tokenizer: Any | None


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"video_dit_config must be a dict, got {type(dit_config)}")
    signature = inspect.signature(FastWAMVideoDiT.__init__)
    allowed = {name for name in signature.parameters if name != "self"}
    required = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is inspect.Signature.empty
    }
    unknown = sorted(set(dit_config) - allowed)
    missing = sorted(required - set(dit_config))
    if unknown:
        raise ValueError(f"Unknown video_dit_config keys: {unknown}")
    if missing:
        raise ValueError(f"Missing video_dit_config keys: {missing}")
    return dict(dit_config)


def _require_path(path: Path, description: str, directory: bool = False) -> str:
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"Wan model {description} {kind} does not exist: {path}")
    return str(path)


def _resolve_model_paths(model_path: str):
    root = Path(model_path).expanduser().resolve()
    _require_path(root, "root", directory=True)
    single_dit = root / "diffusion_pytorch_model.safetensors"
    if single_dit.is_file():
        dit_path: str | list[str] = str(single_dit)
    else:
        shards = sorted(root.glob("diffusion_pytorch_model-*-of-*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"Wan diffusion weights do not exist under: {root}")
        dit_path = [str(path) for path in shards]
    return (
        dit_path,
        _require_path(root / "models_t5_umt5-xxl-enc-bf16.pth", "text encoder"),
        _require_path(root / "Wan2.2_VAE.pth", "VAE"),
        _require_path(root / "google" / "umt5-xxl", "tokenizer", directory=True),
    )


def _load_video_expert(path, config, torch_dtype, device):
    model = FastWAMVideoDiT(**config)
    paths = path if isinstance(path, list) else [path]
    state_dict = {}
    for shard_path in paths:
        shard = load_file(shard_path, device="cpu")
        state_dict.update({key: value.to(torch_dtype) for key, value in shard.items()})
    model.load_state_dict(state_dict, strict=True)
    return model.to(device=device, dtype=torch_dtype)


def load_wan22_ti2v_5b_components(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    tokenizer_max_len: int = 512,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
):
    start = time.time()
    if dit_config is None:
        raise ValueError("video_dit_config is required for Wan2.2-TI2V-5B loading.")
    config = _validate_dit_config(dit_config)
    dit_path, text_encoder_path, vae_path, tokenizer_path = _resolve_model_paths(model_path)

    if skip_dit_load_from_pretrain:
        dit = FastWAMVideoDiT(**config).to(device=device, dtype=torch_dtype)
    else:
        dit = _load_video_expert(dit_path, config, torch_dtype, device)

    text_encoder = None
    tokenizer = None
    if load_text_encoder:
        text_encoder = T5EncoderModel(
            text_len=int(tokenizer_max_len),
            dtype=torch_dtype,
            device=torch.device(device),
            checkpoint_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        tokenizer = text_encoder.tokenizer

    vae = Wan2_2_VAE(
        vae_pth=vae_path,
        dtype=torch_dtype,
        device=device,
    )
    vae.model.to(device=device, dtype=torch_dtype)
    vae.temporal_downsample_factor = 4
    vae.upsampling_factor = 16
    logger.info("Loaded Wan2.2 FastWAM components in %.2f seconds", time.time() - start)
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )
