import ast
import builtins
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightx2v.models.runners.bagel.bagel_runner import BagelRunner
from lightx2v.models.runners.bagel.sensenova_vision_runner import SenseNovaVisionRunner
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.utils import set_config


@pytest.mark.parametrize("model_cls", ["cosmos3", "lingbot_va", "fastwam"])
def test_ros_policy_uses_current_startup_api(tmp_path, model_cls):
    root = Path(__file__).resolve().parents[1]
    path = root / "lightx2v_ros/src/inference/inference" / f"{model_cls}_node/main.py"
    tree = ast.parse(path.read_text())
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "lightx2v.utils.set_config"]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"build_policy_config", "_build_policy_config"})
    namespace = {}
    exec(compile(ast.Module(body=[*imports, method], type_ignores=[]), str(path), "exec"), namespace)
    deployment = tmp_path / "deployment.json"
    deployment.write_text('{"target_video_length": 81, "raw_action_dim": 8, "policy_prompt_format": "json"}')
    values = {"config_json": str(deployment), "model_path": str(tmp_path), "prompt_format": "official_text", "seed": 7}
    node = SimpleNamespace(
        get_parameter=lambda key: SimpleNamespace(value=values[key]),
        seed=7,
        contract=SimpleNamespace(action_dim=8, state_dim=8, name="libero"),
        get_logger=Mock(),
    )

    config = namespace[method.name](node)

    assert config["model_cls"] == model_cls
    assert config["task"] == "i2va"
    assert config["target_video_length"] == 81
    assert config["warmup"] is False
    if model_cls == "cosmos3":
        assert config["policy_prompt_format"] == "official_text"


@pytest.mark.parametrize(
    ("preferred", "fallback"),
    [("", "low_noise_model"), ("low_noise_model", "distill_models/low_noise_model"), ("distill_models/low_noise_model", "original"), ("original", "transformer")],
)
def test_checkpoint_directory_priority(tmp_path, preferred, fallback):
    for subfolder, width in ((preferred, 128), (fallback, 256)):
        folder = tmp_path / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps({"hidden_size": width}))
    deployment = tmp_path / "deployment.json"
    deployment.write_text('{"hidden_size": 512, "target_video_length": 81, "infer_steps": 4}')

    config = set_config.build_startup_config({"model_cls": "wan2.1", "model_path": str(tmp_path), "task": "t2v", "config_json": str(deployment)})

    assert config["hidden_size"] == 128
    assert config["target_video_length"] == 81
    assert config["infer_steps"] == 4


@pytest.mark.parametrize("model_cls", ["hunyuan_video_1.5", "worldplay_ar", "worldplay_bi", "worldplay_distill"])
def test_named_transformer_directory(tmp_path, model_cls):
    transformer_path = tmp_path / "transformer" / "480p_i2v"
    transformer_path.mkdir(parents=True)
    (transformer_path / "config.json").write_text('{"hidden_size": 128}')
    (tmp_path / "config.json").write_text('{"hidden_size": 256}')

    config = set_config.build_startup_config({"model_cls": model_cls, "model_path": str(tmp_path), "task": "i2v", "transformer_model_name": "480p_i2v"})

    assert config["hidden_size"] == 128
    assert config["transformer_model_path"] == str(transformer_path)


def test_longcat_merges_both_model_configs(tmp_path):
    (tmp_path / "config.json").write_text('{"root_setting": 1, "hidden_size": 128}')
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text('{"transformer_setting": 2, "hidden_size": 256}')

    config = set_config.build_startup_config({"model_cls": "longcat_image", "model_path": str(tmp_path), "task": "t2i"})

    assert config["root_setting"] == 1
    assert config["transformer_setting"] == 2
    assert config["hidden_size"] == 256


@pytest.mark.parametrize("subfolder", ["", "transformer"])
def test_ltx_preserves_runtime_rope_selector(tmp_path, subfolder):
    folder = tmp_path / subfolder
    folder.mkdir(exist_ok=True)
    (folder / "config.json").write_text('{"rope_type": "split", "hidden_size": 128}')

    config = set_config.build_startup_config({"model_cls": "ltx2", "model_path": str(tmp_path), "task": "t2av", "rope_type": "torch"})

    assert config["rope_type"] == "torch"
    assert config["hidden_size"] == 128


def test_cosmos_reads_vae_once_and_uses_its_scale_factors(tmp_path, monkeypatch):
    vae_path = tmp_path / "vae" / "config.json"
    vae_path.parent.mkdir()
    vae_path.write_text('{"block_out_channels": [1, 2, 3], "scale_factor_spatial": 12, "scale_factor_temporal": 2}')
    real_open = builtins.open
    vae_reads = []

    def record_open(path, *args, **kwargs):
        if str(path) == str(vae_path):
            vae_reads.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", record_open)

    config = set_config.build_startup_config({"model_cls": "cosmos3", "model_path": str(tmp_path), "task": "i2va"})

    assert len(vae_reads) == 1
    assert config["vae_scale_factor"] == 12
    assert config["vae_scale_factor_spatial"] == 12
    assert config["vae_scale_factor_temporal"] == 2


@pytest.mark.parametrize(("task", "subfolder"), [("t2av", "transformer"), ("ref2av", "transformer_ref")])
def test_minimax_selects_task_weights_and_keeps_spatial_scale(tmp_path, task, subfolder):
    folder = tmp_path / subfolder
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "config.json").write_text('{"block_out_channels": [1, 2, 3, 4, 5, 6]}')

    config = set_config.build_startup_config({"model_cls": "minimax_h3", "model_path": str(tmp_path), "task": task})

    assert config["dit_original_ckpt"] == str(folder)
    assert config["vae_scale_factor"] == 16
    assert config["vae_spatial_scale_factor"] == 16
    assert config["enable_cfg"] is False


@pytest.mark.parametrize(
    ("task", "subtask", "json_subtask", "expected", "error"),
    [
        ("omni_vision_task", None, None, None, "Unsupported omni-vision subtask"),
        ("omni_vision_task", "", "recon3d", None, "Unsupported omni-vision subtask"),
        ("omni_vision_task", "invalid", None, None, "Unsupported omni-vision subtask"),
        ("omni_vision_task", "recon3d", None, "recon3d", None),
        ("omni_vision_task", None, "recon3d", "recon3d", None),
        ("omni_vision_task", "depth", "recon3d", "depth", None),
        ("t2i", "recon3d", None, None, "does not support request fields: omni_vision_subtask"),
        ("t2i", "", None, None, "does not support request fields: omni_vision_subtask"),
    ],
)
def test_cli_leaves_subtask_validation_to_runner(tmp_path, task, subtask, json_subtask, expected, error):
    for filename in ("llm_config.json", "vit_config.json"):
        (tmp_path / filename).write_text("{}")
    deployment = tmp_path / "deployment.json"
    deployment.write_text(json.dumps({"omni_vision_subtask": json_subtask} if json_subtask else {}))
    args = SimpleNamespace(
        model_cls="sensenova_vision" if task == "omni_vision_task" else "bagel",
        model_path=str(tmp_path),
        config_json=str(deployment),
        task=task,
        omni_vision_subtask=subtask,
    )

    config, request = set_config.build_cli_inputs(args)
    runner_cls = SenseNovaVisionRunner if task == "omni_vision_task" else BagelRunner
    runner = object.__new__(runner_cls)
    BaseRunner.__init__(runner, config)

    if error:
        with pytest.raises(ValueError, match=error):
            runner.prepare_request(request)
    else:
        input_info = runner.prepare_request(request)
        assert input_info.task == task
        assert input_info.omni_vision_subtask == expected


def test_sensenova_normalizes_each_request_without_changing_defaults():
    config = {"task": "omni_vision_task", "omni_vision_subtask": "recon3d"}
    runner = object.__new__(SenseNovaVisionRunner)
    BaseRunner.__init__(runner, config)
    request = {"omni_vision_subtask": " RAW-QUERY "}

    explicit = runner.prepare_request(request)
    default = runner.prepare_request({})

    assert explicit.omni_vision_subtask == "understanding"
    assert default.omni_vision_subtask == "recon3d"
    assert config["omni_vision_subtask"] == "recon3d"
    assert request == {"omni_vision_subtask": " RAW-QUERY "}
    assert runner.input_info is None


@pytest.mark.parametrize(
    ("tp", "cfg", "sp", "enable_cfg", "shape", "names"),
    [
        (1, 1, 1, False, (1, 1), ("cfg_p", "seq_p")),
        (1, 2, 2, True, (2, 2), ("cfg_p", "seq_p")),
        (2, 1, 1, False, (2,), ("tensor_p",)),
        (2, 2, 4, True, (2, 4, 2), ("cfg_p", "seq_p", "tensor_p")),
        (2, 2, 1, False, (2, 2), ("cfg_p", "tensor_p")),
    ],
)
def test_parallel_topology_and_flags(monkeypatch, tp, cfg, sp, enable_cfg, shape, names):
    mesh = Mock()
    reduce = Mock()
    monkeypatch.setattr(set_config, "AI_DEVICE", "cpu")
    monkeypatch.setattr(set_config, "init_device_mesh", mesh)
    monkeypatch.setattr(set_config.dist, "get_world_size", lambda: tp * cfg * sp)
    monkeypatch.setattr(set_config.dist, "all_reduce", reduce)
    config = {"parallel": {"tensor_p_size": tp, "cfg_p_size": cfg, "seq_p_size": sp}, "enable_cfg": enable_cfg}

    set_config.init_parallel(config)

    mesh.assert_called_once_with("cpu", shape, mesh_dim_names=names)
    assert config["tensor_parallel"] == (tp > 1)
    assert config["seq_parallel"] == (sp > 1)
    assert config["cfg_parallel"] == (enable_cfg and cfg > 1)
    reduce.assert_called_once()


def test_parallel_rejects_mismatched_world_size(monkeypatch):
    mesh = Mock()
    monkeypatch.setattr(set_config, "init_device_mesh", mesh)
    monkeypatch.setattr(set_config.dist, "get_world_size", lambda: 2)

    with pytest.raises(ValueError, match="Parallel sizes must match"):
        set_config.init_parallel({"parallel": {"tensor_p_size": 4}})

    mesh.assert_not_called()


def test_phase_parallel_keeps_model_specific_flags(monkeypatch):
    from lightx2v.models.networks.hunyuan_image3 import parallel

    def initialize(config):
        config.update({"tensor_parallel": True, "seq_parallel": True, "cfg_parallel": False, "device_mesh": "phase_mesh"})

    monkeypatch.setattr(parallel, "initialize_hunyuan_image3_parallel_runtime", initialize)
    monkeypatch.setattr(set_config, "AI_DEVICE", "cpu")
    monkeypatch.setattr(set_config.dist, "get_world_size", lambda: 4)
    reduce = Mock()
    monkeypatch.setattr(set_config.dist, "all_reduce", reduce)
    config = {"model_cls": "hunyuan_image3", "parallel": {"phase_aware": True, "cfg_p_size": 4}, "enable_cfg": True}

    set_config.init_parallel(config)

    assert config["device_mesh"] == "phase_mesh"
    assert config["tensor_parallel"] is True
    assert config["seq_parallel"] is True
    assert config["cfg_parallel"] is False
    reduce.assert_called_once()
