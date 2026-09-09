import argparse
import ast
import gc
import importlib.util
import json
import os
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from loguru import logger

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.cosmos3 import cosmos3_runner
from lightx2v.models.runners.wan import wan_runner
from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner
from lightx2v.utils.lockable_dict import LockableDict
from lightx2v.utils.set_config import get_default_config

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = (
    "post_async_t2i_and_wait",
    "post_sync_t2i_base64",
    "post_sync_t2i_presigned",
    "post_sync_i2i_base64",
    "post_sync_i2i_presigned",
)


def load_client(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/server" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CapturedPayload(Exception):
    def __init__(self, payload):
        self.payload = payload


def capture_post(url, *, json, timeout):
    raise CapturedPayload(json)


@pytest.mark.parametrize("name", CLIENTS)
@pytest.mark.parametrize("explicit", [False, True])
def test_image_client_main_only_sends_explicit_request_fields(monkeypatch, name, explicit):
    client = load_client(name)
    argv = [name, "--prompt", "a cat"]
    if "i2i" in name:
        argv += ["--image_base64", "aW1hZ2U="]
    if "presigned" in name:
        argv += ["--presigned_url", "https://example.invalid/upload"]
    if explicit:
        argv += ["--negative_prompt", "", "--seed", "0", "--aspect_ratio", "1:1", "--save_result_path", ""]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(client.requests, "post", capture_post)

    with pytest.raises(CapturedPayload) as captured:
        client.main()

    payload = captured.value.payload
    assert payload["prompt"] == "a cat"
    expected = {"negative_prompt": "", "seed": 0, "aspect_ratio": "1:1", "save_result_path": ""}
    assert {key: payload[key] for key in expected if key in payload} == (expected if explicit else {})
    assert "target_shape" not in payload
    if "i2i" in name:
        assert payload["image_path"] == "aW1hZ2U="


@pytest.mark.parametrize("output_path", [None, "result.png"])
def test_async_client_downloads_only_saved_results(monkeypatch, output_path):
    client = load_client("post_async_t2i_and_wait")
    monkeypatch.setattr("sys.argv", ["post_async_t2i_and_wait", "--prompt", "image"])
    monkeypatch.setattr(client, "submit_t2i_task", lambda **kwargs: "task1")
    monkeypatch.setattr(client, "wait_task_done", lambda **kwargs: {"status": "completed", "save_result_path": output_path})
    downloads = []
    monkeypatch.setattr(client, "download_result", lambda *args: downloads.append(args) or Path(args[2]))

    client.main()

    assert len(downloads) == (0 if output_path is None else 1)


@pytest.mark.parametrize("explicit", [False, True])
def test_benchmark_preserves_request_omission(monkeypatch, explicit):
    client = load_client("benchmark_sync_s3_latency")
    argv = ["benchmark", "--prompt", "a cat"]
    if explicit:
        argv += ["--negative_prompt", "", "--seed", "0", "--aspect_ratio", "1:1", "--save_result_path", ""]
    monkeypatch.setattr("sys.argv", argv)
    parse_args = argparse.ArgumentParser.parse_args

    def capture_args(parser):
        args = parse_args(parser)
        raise CapturedPayload(client.build_sync_payload(args))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_args)
    with pytest.raises(CapturedPayload) as captured:
        client.main()

    expected = {"prompt": "a cat"}
    if explicit:
        expected.update(negative_prompt="", seed=0, aspect_ratio="1:1", save_result_path="")
    assert captured.value.payload == expected


@pytest.mark.parametrize("explicit", [False, True])
def test_seko_client_preserves_request_omission(monkeypatch, explicit):
    client = load_client("post_seko_talk_ar")
    argv = ["post_seko_talk_ar"]
    if explicit:
        argv += ["--negative_prompt", "", "--seed", "0", "--save_result_path", ""]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(client.requests, "post", capture_post)

    with pytest.raises(CapturedPayload) as captured:
        client.main()

    payload = captured.value.payload
    expected = {"negative_prompt": "", "seed": 0, "save_result_path": ""}
    assert {key: payload[key] for key in expected if key in payload} == (expected if explicit else {})


class RecordingCosmosRunner(cosmos3_runner.Cosmos3Runner):
    def __init__(self, config):
        BaseRunner.__init__(self, config)
        self.scheduler = SimpleNamespace(sample_guide_scale=float(config.get("sample_guide_scale", 4.0)))
        self.requests = []

    def init_modules(self):
        pass

    def run_request(self, input_info):
        self.requests.append(input_info)
        return {"action": np.zeros((2, 8), dtype=np.float32)}


@pytest.mark.parametrize(
    ("cfg_config", "accepts_negative_prompt"),
    [
        ({}, False),
        ({"enable_cfg": False}, False),
        ({"enable_cfg": False, "sample_guide_scale": 1.0}, False),
        ({"enable_cfg": True, "sample_guide_scale": 0.5}, True),
        ({"enable_cfg": True, "sample_guide_scale": 6.0}, True),
    ],
    ids=["defaults", "disabled", "unit_scale", "fractional_scale", "enabled"],
)
def test_cosmos_policy_uses_runner_contract_without_injecting_negative_prompt(monkeypatch, cfg_config, accepts_negative_prompt):
    monkeypatch.setattr(cosmos3_runner, "Cosmos3Runner", RecordingCosmosRunner)
    config = {"task": "i2va", "action_mode": "policy", "domain_name": "droid_lerobot", **cfg_config}
    policy = cosmos3_runner.Cosmos3Policy(config)
    images = {name: np.zeros((8, 8, 3), dtype=np.uint8) for name in ("wrist_cam", "over_shoulder_left_camera", "over_shoulder_right_camera")}

    action = policy.next_action(images=images, state=np.zeros(8), task_description="pick up the cup")

    assert action.shape == (8,)
    assert policy.runner.requests[0].negative_prompt == ""
    assert policy.runner.requests[0].policy_image.shape == (540, 640, 3)
    for negative_prompt in ("", "blur"):
        if accepts_negative_prompt:
            assert policy.runner.prepare_request({"task": policy.runner.config["task"], "negative_prompt": negative_prompt}).negative_prompt == negative_prompt
        else:
            with pytest.raises(ValueError, match="negative_prompt"):
                policy.runner.prepare_request({"task": policy.runner.config["task"], "negative_prompt": negative_prompt})

    policy.reset()
    with pytest.raises(ValueError, match="state length"):
        policy.next_action(images=images, state=np.zeros(7), task_description="pick up the cup")


@pytest.mark.parametrize(("field", "value"), [("action_mode", "other"), ("domain_name", "other")])
def test_cosmos_policy_keeps_domain_boundary(field, value):
    config = {"action_mode": "policy", "domain_name": "droid_lerobot", field: value}
    with pytest.raises(ValueError, match=field):
        cosmos3_runner.Cosmos3Policy(config)


@pytest.fixture
def gradio_harness():
    # Load only this function to avoid starting the Gradio UI and its file logger.
    path = ROOT / "app/gradio_demo.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_inference")
    requests = []
    configs = []

    def build_runner(config):
        assert isinstance(config, LockableDict)
        configs.append(config)
        runner_cls = Wan22MoeRunner if config["model_cls"] == "wan2.2_moe" else WanRunner
        runner = object.__new__(runner_cls)
        BaseRunner.__init__(runner, config)
        runner.run_request = lambda input_info: requests.append(input_info)
        runner.switch_lora = Mock(return_value=True)
        return runner

    auto_fields = (
        "rope_chunk",
        "rope_chunk_size",
        "cpu_offload",
        "offload_granularity",
        "lazy_load",
        "t5_cpu_offload",
        "clip_cpu_offload",
        "vae_cpu_offload",
        "unload_modules",
        "attention_type",
        "quant_op",
        "use_tiling_vae",
        "clean_cuda_cache",
    )
    namespace = {
        "argparse": argparse,
        "json": json,
        "os": os,
        "gc": gc,
        "logger": logger,
        "cleanup_memory": lambda: None,
        "get_auto_config_dict": lambda **kwargs: {f"{name}_val": False for name in auto_fields},
        "extract_op_name": lambda value: value,
        "generate_unique_filename": lambda *args, **kwargs: "result.mp4",
        "get_default_config": get_default_config,
        "build_runner": build_runner,
        "output_dir": ".",
        "global_runner": None,
        "current_startup_config": None,
        "current_lora_configs": [],
    }
    model_path = ROOT / "app/utils/model_utils.py"
    model_tree = ast.parse(model_path.read_text())
    model_function = next(node for node in model_tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_model_configs")
    namespace["build_wan21"] = lambda model_path, dit_path, *args: {"model_cls": "wan2.1", "model_path": model_path, "dit_original_ckpt": dit_path}
    namespace["build_wan22"] = lambda model_path, high_path, low_path, *args: {
        "model_cls": "wan2.2_moe",
        "model_path": model_path,
        "high_noise_original_ckpt": high_path,
        "low_noise_original_ckpt": low_path,
    }
    exec(compile(ast.Module(body=[model_function], type_ignores=[]), str(model_path), "exec"), namespace)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    yield namespace, configs, requests
    gc.unfreeze()


@pytest.mark.parametrize("cfg_scale", [1, 5])
def test_gradio_config_reaches_runner_with_real_lockable_dict(gradio_harness, cfg_scale):
    namespace, configs, requests = gradio_harness

    result = namespace["run_inference"](model_type_input="Wan2.1", model_path_input="/path/to/model", prompt="a cat", image_path="input.png", cfg_scale=cfg_scale, seed=0)

    assert result == "result.mp4"
    assert configs[0]["model_cls"] == "wan2.1"
    assert configs[0]["model_path"] == "/path/to/model"
    assert configs[0]["enable_cfg"] == (cfg_scale != 1)
    assert requests[0].seed == 0
    assert requests[0].prompt == "a cat"


def test_gradio_releases_frozen_runner_before_loading_replacement(gradio_harness):
    namespace, configs, _ = gradio_harness
    namespace["cleanup_memory"] = gc.collect
    infer = namespace["run_inference"]
    kwargs = {"model_type_input": "Wan2.1", "model_path_input": "/models/wan", "prompt": "a cat", "image_path": "input.png"}
    infer(**kwargs)
    old_runner = weakref.ref(namespace["global_runner"])
    namespace["global_runner"].cycle = namespace["global_runner"]
    namespace["global_runner"]._maybe_freeze_gc()
    build_runner = namespace["build_runner"]

    def create_replacement(config):
        assert namespace["global_runner"] is None
        assert old_runner() is None
        return build_runner(config)

    namespace["build_runner"] = create_replacement
    infer(**kwargs, infer_steps=8)

    assert len(configs) == 2


@pytest.mark.parametrize("model_type", ["Wan2.1", "Wan2.2"])
def test_gradio_updates_strength_at_same_lora_path(gradio_harness, model_type):
    namespace, configs, requests = gradio_harness
    infer = namespace["run_inference"]
    kwargs = {"model_type_input": model_type, "model_path_input": "/models/wan", "prompt": "a cat", "image_path": "input.png", "use_lora": True, "lora_path": "style.safetensors"}
    infer(**kwargs, lora_strength=1.0)
    runner = namespace["global_runner"]

    infer(**kwargs, lora_strength=0.5)

    path = "/models/wan/loras/style.safetensors"
    if model_type == "Wan2.2":
        runner.switch_lora.assert_called_once_with(high_lora_path=path, high_lora_strength=0.5, low_lora_path=path, low_lora_strength=0.5)
    else:
        runner.switch_lora.assert_called_once_with(path, 0.5)
    infer(**{**kwargs, "prompt": "a dog", "image_path": "another.png", "seed": 7}, lora_strength=0.5, num_frames=49, aspect_ratio="16:9")
    assert len(configs) == 1
    assert len(requests) == 3
    assert runner.switch_lora.call_count == 1


@pytest.mark.parametrize(
    ("model_type", "path_field"),
    [("Wan2.1", "dit_path_input"), ("Wan2.2", "high_noise_path_input"), ("Wan2.2", "low_noise_path_input")],
)
def test_gradio_rebuilds_when_checkpoint_config_changes(gradio_harness, model_type, path_field):
    namespace, configs, _ = gradio_harness
    infer = namespace["run_inference"]
    kwargs = {"model_type_input": model_type, "model_path_input": "/models/wan", "prompt": "a cat", "image_path": "input.png"}
    infer(**kwargs, **{path_field: "original.safetensors"})
    original_runner = namespace["global_runner"]

    infer(**kwargs, **{path_field: "replacement.safetensors"})
    replacement_runner = namespace["global_runner"]
    infer(**kwargs, **{path_field: "replacement.safetensors"})

    assert len(configs) == 2
    assert replacement_runner is not original_runner
    assert namespace["global_runner"] is replacement_runner


@pytest.mark.parametrize("model_type", ["Wan2.1", "Wan2.2"])
def test_gradio_rebuilds_when_enabling_or_disabling_lora(gradio_harness, model_type):
    namespace, configs, _ = gradio_harness
    infer = namespace["run_inference"]
    kwargs = {"model_type_input": model_type, "model_path_input": "/models/wan", "prompt": "a cat", "image_path": "input.png", "lora_path": "style.safetensors"}
    infer(**kwargs, use_lora=False)
    base_runner = namespace["global_runner"]
    infer(**kwargs, use_lora=True)
    lora_runner = namespace["global_runner"]
    infer(**kwargs, use_lora=False)

    assert len(configs) == 3
    assert lora_runner is not base_runner
    assert namespace["global_runner"] is not lora_runner
    assert configs[0].get("lora_configs") is None
    assert configs[1]["lora_configs"]
    assert configs[2].get("lora_configs") is None


@pytest.mark.parametrize("branch", ["high", "low"])
def test_gradio_updates_individual_moe_lora_strength(gradio_harness, branch):
    namespace, configs, _ = gradio_harness
    infer = namespace["run_inference"]
    kwargs = {
        "model_type_input": "Wan2.2",
        "model_path_input": "/models/wan",
        "prompt": "a cat",
        "image_path": "input.png",
        "use_lora": True,
        "high_noise_lora_path": "high.safetensors",
        "low_noise_lora_path": "low.safetensors",
    }
    infer(**kwargs)
    runner = namespace["global_runner"]

    infer(**kwargs, **{f"{branch}_noise_lora_strength": 0.0})

    runner.switch_lora.assert_called_once_with(
        high_lora_path="/models/wan/loras/high.safetensors",
        high_lora_strength=0.0 if branch == "high" else 1.0,
        low_lora_path="/models/wan/loras/low.safetensors",
        low_lora_strength=0.0 if branch == "low" else 1.0,
    )
    assert len(configs) == 1


@pytest.mark.parametrize("remaining_branch", ["high", "low"])
@pytest.mark.parametrize("initial_both", [False, True])
def test_gradio_loads_single_moe_lora_branch(gradio_harness, monkeypatch, remaining_branch, initial_both):
    namespace, configs, _ = gradio_harness
    namespace["cleanup_memory"] = gc.collect
    monkeypatch.setattr(wan_runner, "WanModel", lambda **kwargs: kwargs)
    build_runner = namespace["build_runner"]

    def load_models(config):
        runner = build_runner(config)
        runner.high_noise_model_path = "/models/wan/high_noise_model"
        runner.low_noise_model_path = "/models/wan/low_noise_model"
        runner.init_device = "cpu"
        runner.distill_method = None
        config["boundary"] = 0.9
        runner.model = runner.load_transformer()
        return runner

    namespace["build_runner"] = load_models
    infer = namespace["run_inference"]
    kwargs = {"model_type_input": "Wan2.2", "model_path_input": "/models/wan", "prompt": "a cat", "image_path": "input.png", "use_lora": True}
    if initial_both:
        infer(**kwargs, high_noise_lora_path="high.safetensors", low_noise_lora_path="low.safetensors")
    infer(**kwargs, **{f"{remaining_branch}_noise_lora_path": f"{remaining_branch}.safetensors"})

    assert len(configs) == (2 if initial_both else 1)
    models = namespace["global_runner"].model.model
    assert len(models) == 2
    for model, branch in zip(models, ("high", "low")):
        assert model["model_path"] == f"/models/wan/{branch}_noise_model"
        if branch == remaining_branch:
            assert model["lora_path"] == f"/models/wan/loras/{branch}.safetensors"
            assert model["lora_strength"] == 1.0
        else:
            assert "lora_path" not in model
            assert "lora_strength" not in model


@pytest.mark.parametrize("model_type", ["wan2.1", "high_noise_model", "low_noise_model"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_wan_lora_loading_preserves_branch_weights(monkeypatch, model_type, dynamic):
    loras = [{"name": name, "path": f"{name}.safetensors", "strength": 0.0} for name in ("high_noise_model", "low_noise_model")]
    expected = loras if model_type == "wan2.1" else [item for item in loras if item["name"] == model_type]
    adapter = Mock()
    adapter_factory = Mock(return_value=adapter)
    monkeypatch.setattr(wan_runner, "LoraAdapter", adapter_factory)

    model = wan_runner.build_wan_model_with_lora(lambda **kwargs: kwargs, {"lora_dynamic_apply": dynamic}, {"model_path": "base"}, loras, model_type)

    if dynamic:
        assert model == {"model_path": "base", "lora_path": expected[0]["path"], "lora_strength": 0.0}
        adapter_factory.assert_not_called()
    else:
        assert model == {"model_path": "base"}
        adapter.apply_lora.assert_called_once_with(expected, model_type=model_type)


@pytest.mark.parametrize("dynamic", [False, True])
def test_wan_missing_branch_loads_base_model_without_lora(monkeypatch, dynamic):
    adapter_factory = Mock()
    monkeypatch.setattr(wan_runner, "LoraAdapter", adapter_factory)
    loras = [{"name": "high_noise_model", "path": "high.safetensors", "strength": 1.0}]

    model = wan_runner.build_wan_model_with_lora(lambda **kwargs: kwargs, {"lora_dynamic_apply": dynamic}, {"model_path": "low"}, loras, "low_noise_model")

    assert model == {"model_path": "low"}
    adapter_factory.assert_not_called()


@pytest.mark.parametrize("lora_path", [None, "adapter.safetensors"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_weight_loading_registers_only_current_model_lora(lora_path, dynamic):
    model = SimpleNamespace(
        config={"lora_dynamic_apply": dynamic, "lora_configs": [{"name": "high_noise_model", "path": "adapter.safetensors", "strength": 0.0}]},
        lora_path=lora_path,
        lora_strength=0.0,
        original_weight_dict={"weight": "base"},
        pre_weight=Mock(),
        transformer_weights=Mock(),
        _register_lora=Mock(),
    )

    BaseTransformerModel._apply_weights(model)

    model.pre_weight.load.assert_called_once_with({"weight": "base"})
    model.transformer_weights.load.assert_called_once_with({"weight": "base"})
    if dynamic and lora_path is not None:
        model._register_lora.assert_called_once_with(lora_path, 0.0)
    else:
        model._register_lora.assert_not_called()


def test_gradio_does_not_generate_or_cache_failed_lora_switch(gradio_harness):
    namespace, configs, requests = gradio_harness
    infer = namespace["run_inference"]
    kwargs = {"model_type_input": "Wan2.1", "model_path_input": "/models/wan", "prompt": "a cat", "image_path": "input.png", "use_lora": True, "lora_path": "style.safetensors"}
    infer(**kwargs)
    runner = namespace["global_runner"]
    runner.switch_lora.return_value = False

    with pytest.raises(RuntimeError, match="LoRA"):
        infer(**kwargs, lora_strength=0.5)

    assert len(requests) == 1
    runner.switch_lora.return_value = True
    infer(**kwargs, lora_strength=0.5)
    assert runner.switch_lora.call_count == 2
    assert len(configs) == 1
    assert len(requests) == 2
