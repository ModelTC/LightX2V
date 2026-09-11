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

"""Policies and checkpoint contracts for MiniMax-H3 VAE Encoder FP8 Conv3D."""

from dataclasses import dataclass

import torch

FP8_ENCODER_CONV_F16_ACCUM_PROFILE = "h3-vae-encoder-fp8-f16-accum"
FP8_ENCODER_CONV_PROFILE_METADATA_KEY = "h3_encoder_conv_profile"
FP8_ENCODER_CONV_WEIGHT_QMAX_METADATA_KEY = "h3_encoder_conv_weight_qmax"
FP8_ENCODER_CONV_UNQUANTIZED_MODULE_NAMES = {
    "conv_in",
    "conv_out",
    "down_blocks.1.resnets.0.conv_shortcut",
    "down_blocks.3.resnets.0.conv1",
    "down_blocks.3.resnets.0.conv_shortcut",
    "down_blocks.5.resnets.0.conv_shortcut",
}


@dataclass(frozen=True)
class Fp8EncoderConvPolicy:
    mode: str
    weight_qmax: float
    activation_qmax: float
    accumulator_dtype: torch.dtype
    required_checkpoint_profile: str | None = None


_FP8_ENCODER_CONV_POLICIES = {
    "cutlass_fp8_f32_accum": Fp8EncoderConvPolicy(
        mode="cutlass_fp8_f32_accum",
        weight_qmax=torch.finfo(torch.float8_e4m3fn).max,
        activation_qmax=torch.finfo(torch.float8_e4m3fn).max,
        accumulator_dtype=torch.float32,
    ),
    "cutlass_fp8_f16_accum": Fp8EncoderConvPolicy(
        mode="cutlass_fp8_f16_accum",
        weight_qmax=21.0,
        activation_qmax=21.0,
        accumulator_dtype=torch.float16,
        required_checkpoint_profile=FP8_ENCODER_CONV_F16_ACCUM_PROFILE,
    ),
}
FP8_ENCODER_CONV_MODES = tuple(_FP8_ENCODER_CONV_POLICIES)


def get_fp8_encoder_conv_policy(mode: str) -> Fp8EncoderConvPolicy:
    if mode not in _FP8_ENCODER_CONV_POLICIES:
        raise ValueError(f"Unsupported H3 VAE Encoder FP8 Conv3D mode: {mode!r}")
    return _FP8_ENCODER_CONV_POLICIES[mode]


def validate_fp8_encoder_conv_checkpoint_metadata(
    policy: Fp8EncoderConvPolicy,
    checkpoint_profile: str | None,
    weight_qmax: float | None,
) -> None:
    if policy.required_checkpoint_profile is None:
        if checkpoint_profile is not None:
            raise ValueError(f"MiniMax-H3 Encoder Conv3D mode {policy.mode!r} uses standard FP8 weights, but the checkpoint declares profile {checkpoint_profile!r}")
        return
    if checkpoint_profile != policy.required_checkpoint_profile:
        raise ValueError(f"MiniMax-H3 Encoder Conv3D mode {policy.mode!r} requires checkpoint profile {policy.required_checkpoint_profile!r}, got {checkpoint_profile!r}")
    if weight_qmax != policy.weight_qmax:
        raise ValueError(f"MiniMax-H3 Encoder Conv3D profile {policy.required_checkpoint_profile!r} requires weight_qmax={policy.weight_qmax}, got {weight_qmax!r}")
