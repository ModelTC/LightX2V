"""Each model policy checks every shard after the shared FP8 metadata refactor."""

import pytest
import torch
from safetensors.torch import save_file

from lightx2v.models.input_encoders.hf.wan.t5 import fp8_f16_accum_policy as t5
from lightx2v.models.networks.minimax_h3 import fp8_f16_accum_policy as h3
from lightx2v.models.networks.qwen_image_21 import fp8_f16_accum_policy as qwen
from lightx2v.models.networks.wan import fp8_f16_accum_policy as wan


@pytest.mark.parametrize(
    "validate,profile,extra",
    [
        (wan.validate_checkpoint, wan.QUANTIZATION_PROFILE, {}),
        (t5.validate_checkpoint, t5.QUANTIZATION_PROFILE, {"activation_qmax": "7"}),
        (qwen.validate_checkpoint, qwen.QUANTIZATION_PROFILE, {}),
        (h3.validate_fp8_f16_accum_checkpoint, h3.FP8_F16_ACCUM_QUANTIZATION_PROFILE, {}),
    ],
)
def test_fp8_checkpoint_policy_validates_every_shard(tmp_path, validate, profile, extra):
    checkpoint = tmp_path / "shards"
    checkpoint.mkdir()
    metadata = {"quantization_profile": profile, "weight_qmax": "14", **extra}
    save_file({"weight": torch.ones(1)}, checkpoint / "00001.safetensors", metadata=metadata)
    save_file({"weight": torch.ones(1)}, checkpoint / "00002.safetensors", metadata={**metadata, "weight_qmax": "448"})

    with pytest.raises(ValueError, match="weight_qmax"):
        validate(checkpoint)

    save_file({"weight": torch.ones(1)}, checkpoint / "00002.safetensors", metadata=metadata)
    validate(checkpoint)
