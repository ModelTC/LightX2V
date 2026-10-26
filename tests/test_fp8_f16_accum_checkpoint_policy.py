"""Each model profile checks every FP8 checkpoint shard."""

import pytest
import torch
from safetensors.torch import save_file

from lightx2v.utils.fp8_f16_accum_checkpoint import (
    H3_QUANTIZATION_PROFILE,
    QWEN_IMAGE_21_QUANTIZATION_PROFILE,
    T5_QUANTIZATION_PROFILE,
    WAN_QUANTIZATION_PROFILE,
    validate_checkpoint,
)


@pytest.mark.parametrize(
    "profile,extra",
    [
        (WAN_QUANTIZATION_PROFILE, {}),
        (T5_QUANTIZATION_PROFILE, {"activation_qmax": "7"}),
        (QWEN_IMAGE_21_QUANTIZATION_PROFILE, {}),
        (H3_QUANTIZATION_PROFILE, {}),
    ],
)
def test_fp8_checkpoint_policy_validates_every_shard(tmp_path, profile, extra):
    checkpoint = tmp_path / "shards"
    checkpoint.mkdir()
    metadata = {"quantization_profile": profile, "weight_qmax": "14", **extra}
    save_file({"weight": torch.ones(1)}, checkpoint / "00001.safetensors", metadata=metadata)
    save_file({"weight": torch.ones(1)}, checkpoint / "00002.safetensors", metadata={**metadata, "weight_qmax": "448"})

    with pytest.raises(ValueError, match="weight_qmax"):
        validate_checkpoint(checkpoint, profile=profile)

    save_file({"weight": torch.ones(1)}, checkpoint / "00002.safetensors", metadata=metadata)
    validate_checkpoint(checkpoint, profile=profile)
