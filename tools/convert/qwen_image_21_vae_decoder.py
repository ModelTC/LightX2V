r"""Convert the VAE decoder's supported convolutions to FP8 with FP16 accumulation.

Example (other tensors retain their original dtype)::

    python tools/convert/converter.py \
        --model_type qwen_image_21_vae_decoder \
        --source /path/to/Qwen-Image-2.1/vae/diffusion_pytorch_model.safetensors \
        --output /path/to/Qwen-Image-2.1-vae-fp8 \
        --output_name qwen_image_21_vae_decoder_fp8 \
        --quantized --linear_type fp8 --single_file --device cpu

Set vae_decoder_conv_mode="cutlass_fp8_f16_accum" and vae_quantized_ckpt to
that output file. This mixed qmax21 profile retains high-resolution convolutions
in their original precision to fit the CUTLASS per-buffer size limit at 2K.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lightx2v.models.video_encoders.hf.qwen_image_21.fp8_conv import FP8_DECODER_PROFILE, FP8_DECODER_PROFILE_KEY, FP8_DECODER_QMAX, is_fp8_decoder_conv


def convert_qwen_image_21_vae_decoder_fp8(args):
    if not args.quantized or args.linear_type != "fp8" or not args.single_file or args.output_ext != ".safetensors":
        raise ValueError("qwen_image_21_vae_decoder requires --quantized --linear_type fp8 --single_file --output_ext .safetensors")
    if args.direction is not None or args.lora_path is not None:
        raise ValueError("VAE decoder conversion does not support key conversion or LoRA merging")
    source = Path(args.source)
    if not source.is_file() or source.suffix != ".safetensors":
        raise ValueError("Expected one original Qwen-Image-2.1 VAE safetensors file")
    output = Path(args.output) / f"{args.output_name}{args.output_ext}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        names = [key for key in checkpoint.keys() if key.endswith(".weight") and is_fp8_decoder_conv(key.removesuffix(".weight"), checkpoint.get_slice(key).get_shape())]
        if len(names) != 32 or any(key.endswith("weight_scale") for key in checkpoint.keys()):
            raise ValueError("Expected the original VAE checkpoint with 32 supported decoder convolutions")
        tensors = {}
        for key in checkpoint.keys():
            tensor = checkpoint.get_tensor(key)
            if tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                raise ValueError(f"Expected an unquantized floating-point tensor: {key}")
            if key not in names:
                tensors[key] = tensor
                continue
            weight = tensor.to(device=args.device, dtype=torch.float32)
            if not torch.isfinite(weight).all():
                raise ValueError(f"Non-finite VAE weight: {key}")
            scale = (weight.abs().amax() / FP8_DECODER_QMAX).clamp_min(torch.finfo(torch.float32).tiny)
            tensors[key] = (weight / scale).clamp(-FP8_DECODER_QMAX, FP8_DECODER_QMAX).to(torch.float8_e4m3fn).cpu().contiguous()
            tensors[f"{key}_scale"] = scale.cpu().contiguous()
        metadata = dict(checkpoint.metadata() or {})
    metadata[FP8_DECODER_PROFILE_KEY] = FP8_DECODER_PROFILE
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.stem}.", dir=output.parent) as temporary_dir:
        temporary = Path(temporary_dir) / output.name
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, output)
    print(f"Saved {len(names)} FP8 decoder convolutions to {output}")
