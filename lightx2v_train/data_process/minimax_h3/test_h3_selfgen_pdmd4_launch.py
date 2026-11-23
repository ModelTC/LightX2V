"""Exercise the real launcher without starting distributed training."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from test_h3_action_pdmd_selfgen_gan_stages import weights
from test_h3_action_pdmd_selfgen_views import make_selfgen

ROOT = Path(__file__).resolve().parents[2]


def test_dcm_selfgen_four_step_launch(tmp_path, monkeypatch):
    cache = make_selfgen(tmp_path)
    model = tmp_path / "model"
    for component in ("transformer", "text_encoder", "vae"):
        (model / component).mkdir(parents=True)
        (model / component / "config.json").write_text("{}")
    checkpoint = tmp_path / "temporary_export"
    weights(checkpoint, 1700)
    dcm = tmp_path / "dcm_ema"
    dcm.mkdir()
    shutil.copyfile(checkpoint / "pytorch_lora_weights.safetensors", dcm / "pytorch_lora_weights.safetensors")
    output = tmp_path / "training"
    environment = {**os.environ, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29717",
                   "H3_CODE_ROOT": str(ROOT.parent), "H3_PYTHON": sys.executable,
                   "H3_MODEL_PATH": str(model), "H3_STUDENT_INIT": str(dcm),
                   "H3_SELFGEN_CACHE": str(cache / "metadata.jsonl"), "H3_PDMD_OUTPUT": str(output)}
    result = subprocess.run(["bash", str(ROOT / "scripts/run_minimax_h3_selfgen_pdmd4_gan_32gpu.sh"), "--dry-run"],
                            env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"selfgen_rows": 64' in result.stdout
    assert '"optimizer_updates": 3000' in result.stdout
    assert "--nnodes=4 --nproc_per_node=8" in result.stdout
    assert result.stdout.count("-m torch.distributed.run") == 1
    assert not output.exists()
    for key, value in environment.items():
        if key.startswith("H3_"):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("H3_SELFGEN_VIEW", str(output / "data/selfgen"))
    config = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/train/dmd/minimax_h3_selfgen_pdmd4_gan_shift6_12_fsdp32.yaml"), resolve=True)
    from lightx2v_train.trainers.dmd.official_pdmd_training import validate_official_pdmd_config
    validate_official_pdmd_config(config)
    assert config["training"]["student"]["checkpoint_path"] == str(dcm)
    assert "checkpoint_path" not in config["training"]["fake"]
    assert config["training"]["dmd"]["num_inference_steps"] == 4
    assert config["data"]["train"]["tasks"] == ["t2av"]
    assert config["training"]["dmd"]["gan"]["real_cache"] == str(output / "data/selfgen/t2av/landscape/metadata.jsonl")
