"""PDD launchers preserve preprocessing and keep the old PDMD defaults."""

import os
import subprocess
from unittest.mock import patch

import pytest
from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.pdd.config import PDDConfig

from lightx2v_train.trainers.dmd.tests import test_h3_image_launcher as existing

REPO, stages = existing.REPO, existing.stages


def test_method_defaults_and_explicit_override(monkeypatch):
    for name in ("H3_IMAGE_TRAINING_METHOD", "H3_IMAGE_CONFIG", "H3_IMAGE_MAX_ITERS", "H3_PDD_MAX_ITERS"):
        monkeypatch.delenv(name, raising=False)
    previous = stages.arguments([])
    pdd = stages.arguments(["--method", "pdd"])
    assert previous.method == "pdmd" and previous.max_iters == 15000
    assert "configs/train/dmd/" in str(previous.config)
    assert pdd.max_iters == 30000 and "configs/train/pdd/" in str(pdd.config)
    assert stages.commands(previous)["pe"] == stages.commands(pdd)["pe"]
    assert stages.commands(previous)["cache"] == stages.commands(pdd)["cache"]


def test_pdd_requires_original_single_head_shard_before_preprocessing(tmp_path):
    helper = existing.H3ImageLauncherTests()
    args = helper.fixture(tmp_path)
    args.method = "pdd"
    with patch.object(stages.subprocess, "run") as run:
        with pytest.raises(FileNotFoundError, match="transformer/model.safetensors.index.json"):
            stages.validate_models(args)
        run.assert_not_called()


def test_pdd_preprocessing_handoff_sets_output_and_iteration_environment(tmp_path):
    helper = existing.H3ImageLauncherTests()
    args = helper.fixture(tmp_path)
    args.method = "pdd"
    with patch.object(stages, "validate_environment"), patch.object(stages, "validate_models"), patch.object(stages.subprocess, "run") as run:
        run.side_effect = lambda command, **kwargs: helper.preprocessing_result(args, command, **kwargs)
        stages.run(args)
    assert len(run.call_args_list) == 3
    env = run.call_args.kwargs["env"]
    assert env["H3_PDD_OUTPUT"] == str(args.output_dir)
    assert env["H3_PDD_MAX_ITERS"] == "12"
    assert "train.py" in str(run.call_args.args[0])


@pytest.mark.parametrize("task,gpus,nfe", [("image", 8, 2), ("fl2av", 32, 8), ("ref2av", 32, 8)])
def test_shipped_configs_follow_the_pdd_schema(monkeypatch, task, gpus, nfe):
    for name in ("H3_MODEL_PATH", "H3_IMAGE_MODEL_PATH", "H3_PDD_CACHE", "H3_IMAGE_CACHE", "H3_PDD_OUTPUT"):
        monkeypatch.setenv(name, "/example/" + name)
    monkeypatch.delenv("H3_PDD_MAX_ITERS", raising=False)
    name = "minimax_h3_image_pdd2_2048_fsdp8.yaml" if task == "image" else f"minimax_h3_{task}_pdd8_fsdp32.yaml"
    config = load_config(REPO / "lightx2v_train/configs/train/pdd" / name)
    parsed = PDDConfig.from_config(config)
    assert parsed.num_inference_steps == nfe
    assert config["distributed"]["fsdp2"]["size"] == gpus
    training = config["training"]
    assert training["method"] == "pdd" and training["train_type"] == "lora"
    assert not {"dmd", "fake", "student"} & training.keys()
    assert training["teacher"]["guidance_scale"] == 1.0
    assert training["max_train_iters"] == "30000"
    assert training["save_every_iters"] == 100 and training["save_total_limit"] == 2
    assert training["optimizer"]["learning_rate"] == 5e-6
    assert training["lora"]["rank_pattern"]["norm_out.linear"] == 16
    assert training["output_dir"].endswith("/train")
    if task != "image":
        matching = config["model"]["capabilities"]["distribution_matching"]
        assert "video_loss_weight" not in matching and "audio_loss_weight" not in matching
    if task == "fl2av":
        assert config["data"]["train"]["require_all_routes_per_optimizer_step"] is False


@pytest.mark.parametrize("task", ["fl2av", "ref2av"])
def test_video_wrapper_dry_run_does_not_require_cluster_or_data(tmp_path, task):
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("H3_", "MASTER_"))}
    environment.update(H3_CODE_ROOT=str(REPO), H3_PDD_OUTPUT=str(tmp_path / "not-created"))
    script = REPO / f"lightx2v_train/scripts/run_minimax_h3_{task}_pdd8_fsdp32_32gpu_acp.sh"
    result = subprocess.run(["bash", str(script), "--dry-run"], env=environment, capture_output=True, text=True, check=True)
    assert "--nnodes=4" in result.stdout and "--nproc_per_node=8" in result.stdout
    assert f"minimax_h3_{task}_pdd8_fsdp32.yaml" in result.stdout
    assert "ACP_MASTER_ADDR:ACP_MASTER_PORT" in result.stdout
    assert not (tmp_path / "not-created").exists()
