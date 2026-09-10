# Copyright 2026 The LightX2V Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from loguru import logger
from safetensors import safe_open
from safetensors import torch as st
from tqdm import tqdm

from lightx2v.models.video_encoders.hf.minimax_h3.fp8_encoder_conv_policy import (
    FP8_ENCODER_CONV_MODES,
    FP8_ENCODER_CONV_PROFILE_METADATA_KEY,
    FP8_ENCODER_CONV_UNQUANTIZED_MODULE_NAMES,
    FP8_ENCODER_CONV_WEIGHT_QMAX_METADATA_KEY,
    get_fp8_encoder_conv_policy,
)

_EXPECTED_ENCODER_CONV_WEIGHT_COUNT = 33
_UNQUANTIZED_DECODER_LINEAR_WEIGHT_NAMES = {"decoder.proj_in.weight"}
_UNQUANTIZED_ENCODER_CONV_WEIGHT_NAMES = {f"encoder.{name}.weight" for name in FP8_ENCODER_CONV_UNQUANTIZED_MODULE_NAMES}


def _validate_fp8_decoder_checkpoint(checkpoint, checkpoint_keys: set[str]) -> None:
    """Validate the existing Decoder tensors; this converter leaves them unchanged."""
    decoder_linear_weight_names = {key for key in checkpoint_keys if key.startswith("decoder.") and key.endswith(".weight") and len(checkpoint.get_slice(key).get_shape()) == 2}
    fp8_weight_names = decoder_linear_weight_names - _UNQUANTIZED_DECODER_LINEAR_WEIGHT_NAMES
    if not fp8_weight_names:
        raise ValueError("h3_video_vae_encoder conversion requires an existing FP8 Video VAE decoder checkpoint")

    for weight_name in sorted(fp8_weight_names):
        scale_name = f"{weight_name}_scale"
        bias_name = f"{weight_name.removesuffix('.weight')}.bias"
        if scale_name not in checkpoint_keys or bias_name not in checkpoint_keys:
            raise ValueError(f"Incomplete FP8 decoder tensors for {weight_name!r}")
        if str(checkpoint.get_slice(weight_name).get_dtype()) != "F8_E4M3":
            raise TypeError(f"Expected FP8 decoder weight {weight_name!r}")
        if str(checkpoint.get_slice(scale_name).get_dtype()) != "F32":
            raise TypeError(f"Expected FP32 decoder scale {scale_name!r}")
        if str(checkpoint.get_slice(bias_name).get_dtype()) != "F16":
            raise TypeError(f"Expected FP16 decoder bias {bias_name!r}; use a runtime-compatible FP8 decoder checkpoint")


def convert_h3_video_vae_encoder_fp8(args) -> None:
    """Quantize the validated MiniMax-H3 Video VAE Encoder Conv3D weights."""
    if not args.quantized or args.linear_type != "fp8":
        raise ValueError("h3_video_vae_encoder conversion requires --quantized --linear_type fp8")
    if args.output_ext != ".safetensors" or not args.single_file:
        raise ValueError("h3_video_vae_encoder conversion requires --output_ext .safetensors --single_file")
    if args.direction is not None or args.lora_path is not None:
        raise ValueError("h3_video_vae_encoder FP8 conversion does not support key conversion or LoRA merging")
    if args.vae_encoder_conv_mode not in FP8_ENCODER_CONV_MODES:
        raise ValueError(f"h3_video_vae_encoder conversion requires --vae_encoder_conv_mode in {FP8_ENCODER_CONV_MODES}")
    conv_policy = get_fp8_encoder_conv_policy(args.vae_encoder_conv_mode)

    source_path = Path(args.source)
    if not source_path.is_file() or source_path.suffix != ".safetensors":
        raise ValueError("h3_video_vae_encoder conversion requires one source safetensors file")
    output_root = Path(args.output)
    output_path = output_root / f"{args.output_name}{args.output_ext}"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output: {output_path}")

    with safe_open(source_path, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = tuple(checkpoint.keys())
        checkpoint_key_set = set(checkpoint_keys)
        _validate_fp8_decoder_checkpoint(checkpoint, checkpoint_key_set)

        encoder_conv_weight_names = {key for key in checkpoint_keys if key.startswith("encoder.") and key.endswith(".weight") and len(checkpoint.get_slice(key).get_shape()) == 5}
        if len(encoder_conv_weight_names) != _EXPECTED_ENCODER_CONV_WEIGHT_COUNT:
            raise ValueError(f"Expected {_EXPECTED_ENCODER_CONV_WEIGHT_COUNT} H3 Encoder Conv3D weights, found {len(encoder_conv_weight_names)}")
        missing_unquantized_weight_names = _UNQUANTIZED_ENCODER_CONV_WEIGHT_NAMES - encoder_conv_weight_names
        if missing_unquantized_weight_names:
            raise KeyError(f"H3 Encoder checkpoint is missing unquantized Conv3D weights: {sorted(missing_unquantized_weight_names)}")
        quantized_weight_names = encoder_conv_weight_names - _UNQUANTIZED_ENCODER_CONV_WEIGHT_NAMES
        existing_scale_names = {f"{key}_scale" for key in encoder_conv_weight_names} & checkpoint_key_set
        if existing_scale_names:
            raise ValueError(f"H3 Encoder checkpoint already contains FP8 Conv3D scales: {sorted(existing_scale_names)[:8]}")

        output_root.mkdir(parents=True, exist_ok=True)
        output_tensors: dict[str, torch.Tensor] = {}
        device = torch.device(args.device)
        for key in tqdm(checkpoint_keys, desc="Quantizing H3 Video VAE Encoder"):
            tensor = checkpoint.get_tensor(key)
            if key not in quantized_weight_names:
                output_tensors[key] = tensor.clone()
                continue
            if tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
                raise TypeError(f"Expected floating-point Conv3D weight for {key}, got {tensor.dtype}")
            weight = tensor.to(device=device, dtype=torch.float32)
            weight_scale = (weight.abs().max() / conv_policy.weight_qmax).clamp_min(torch.finfo(torch.float32).tiny)
            weight_fp8 = (weight / weight_scale).clamp(-conv_policy.weight_qmax, conv_policy.weight_qmax).to(torch.float8_e4m3fn)
            output_tensors[key] = weight_fp8.cpu().contiguous()
            output_tensors[f"{key}_scale"] = weight_scale.cpu().contiguous()

        metadata = dict(checkpoint.metadata() or {})

    metadata.pop(FP8_ENCODER_CONV_PROFILE_METADATA_KEY, None)
    metadata.pop(FP8_ENCODER_CONV_WEIGHT_QMAX_METADATA_KEY, None)
    if conv_policy.required_checkpoint_profile is not None:
        metadata[FP8_ENCODER_CONV_PROFILE_METADATA_KEY] = conv_policy.required_checkpoint_profile
        metadata[FP8_ENCODER_CONV_WEIGHT_QMAX_METADATA_KEY] = str(conv_policy.weight_qmax)
    total_size_bytes = sum(tensor.numel() * tensor.element_size() for tensor in output_tensors.values())
    logger.info(
        "Saving MiniMax-H3 Video VAE with {} FP8 Encoder Conv3D weights ({:.2f} GiB): {}",
        len(quantized_weight_names),
        total_size_bytes / 1024**3,
        output_path,
    )
    with TemporaryDirectory(prefix=f".{args.output_name}.", dir=output_root) as temporary_dir:
        temporary_path = Path(temporary_dir) / output_path.name
        st.save_file(output_tensors, temporary_path, metadata=metadata)
        os.replace(temporary_path, output_path)
