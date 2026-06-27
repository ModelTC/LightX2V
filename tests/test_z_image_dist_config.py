import json
from pathlib import Path


def test_z_image_nvidia_dist_uses_head_divisible_seq_parallel():
    config = json.loads(Path("configs/platforms/nvidia/dist_8/z_image_turbo_t2i.json").read_text(encoding="utf-8"))
    script = Path("scripts/platforms/nvidia/dist_8/z_image_turbo_t2i.sh").read_text(encoding="utf-8")

    assert config["enable_cfg"] is False
    assert config["parallel"] == {
        "seq_p_size": 2,
        "seq_p_attn_type": "ulysses",
    }
    assert "gpus=${GPUS:-2}" in script
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"' in script
