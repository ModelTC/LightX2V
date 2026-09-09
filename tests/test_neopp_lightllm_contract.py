import base64
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lightx2v.models.runners.neopp.neopp_runner import NeoppRunner
from lightx2v.pipeline import LightX2VPipeline


@pytest.fixture
def create_pipeline(tmp_path, monkeypatch):
    """Exercise the LightLLM adapter contract with real CPU request/scheduler methods."""
    monkeypatch.setattr("lightx2v.models.runners.neopp.neopp_runner.AI_DEVICE", "cpu")
    monkeypatch.setattr("lightx2v.models.schedulers.neopp.scheduler.AI_DEVICE", "cpu")
    monkeypatch.setattr("lightx2v.models.runners.base_runner.seed_all", torch.manual_seed)

    def load_model(runner):
        infer = SimpleNamespace(
            kv_cache={},
            fi_moe_autotune=SimpleNamespace(cache_rebuild_needed=lambda: False, session=lambda **kwargs: nullcontext()),
        )
        model = SimpleNamespace(transformer_infer=infer, consumed_inputs=[])

        def infer_step(inputs):
            model.consumed_inputs.append(inputs)
            infer.kv_cache["current"] = inputs["past_key_values_cond"]

        model.infer = infer_step
        model.set_scheduler = lambda scheduler: setattr(model, "scheduler", scheduler)
        runner.model = model

    monkeypatch.setattr(NeoppRunner, "load_model", load_model)
    encode_base64 = NeoppRunner.process_images_after_vae_decoder
    # LightLLM replaces this output hook to return encoded image bytes.
    monkeypatch.setattr(NeoppRunner, "process_images_after_vae_decoder", lambda runner: base64.b64decode(encode_base64(runner)))

    config_path = tmp_path / "neopp.json"
    config_path.write_text(
        json.dumps(
            {
                "cpu_offload": True,
                "infer_steps": 1,
                "patch_size": 16,
                "enable_cfg": True,
                "seed": 37,
                "save_result_for_debug": True,
                "llm_config": {"head_dim": 32, "rope_theta": 10000, "rope_theta_hw": 10000},
            }
        ),
        encoding="utf-8",
    )

    def create(task=None):
        task_args = {} if task is None else {"task": task}
        pipeline = LightX2VPipeline(model_path=str(tmp_path), model_cls="neopp", support_tasks=["t2i", "i2i"], **task_args)
        pipeline.create_generator(config_json=str(config_path))
        pipeline.runner.config.lock()
        return pipeline

    with torch.random.fork_rng(devices=[]):
        yield create


def generate_from_kv(pipeline, *, task, **request):
    runner = pipeline.runner
    runner.set_inference_params(index_offset_cond=9, index_offset_uncond=3, cfg_scale=3.5, timestep_shift=2.0, output_format="png")
    runner.set_kvcache(torch.ones(1, 2, 3, 4), torch.zeros(1, 2, 3, 4))
    return pipeline.generate(task=task, save_result_path="", target_shape=[32, 32], **request)


@pytest.mark.parametrize(("task", "expected_task"), [(None, "t2i"), ("i2i", "i2i")])
def test_support_tasks_selects_startup_task_without_setting_request_default(create_pipeline, task, expected_task):
    pipeline = create_pipeline(task)

    assert pipeline.task == (task or "")
    assert pipeline.runner.config["task"] == expected_task
    assert "support_tasks" not in pipeline.runner.config
    if task is None:
        with pytest.raises(ValueError, match="task is required"):
            pipeline.generate()


def test_modify_config_preserves_lightllm_bytes_output_and_kv_injection(create_pipeline):
    pipeline = create_pipeline()
    pipeline.modify_config({"load_kv_cache_in_pipeline_for_debug": False, "save_result_for_debug": False})

    result = generate_from_kv(pipeline, task="t2i", seed=0)
    runner = pipeline.runner

    assert isinstance(result, bytes)
    assert result.startswith(b"\x89PNG\r\n\x1a\n")
    assert runner.config.locked
    assert runner.input_info.save_result_path == ""
    inputs = runner.model.consumed_inputs[-1]
    assert torch.equal(inputs["past_key_values_cond"], torch.ones(1, 2, 3, 4))
    assert torch.equal(inputs["past_key_values_uncond"], torch.zeros(1, 2, 3, 4))
    assert runner.model.cfg_scale == 3.5
    assert runner.scheduler.timestep_shift == 2.0
    assert runner.past_key_values_cond is None
    assert runner.past_key_values_uncond is None
    assert runner.model.transformer_infer.kv_cache == {}


def test_same_runner_accepts_t2i_and_i2i_encoded_kv_requests(create_pipeline):
    pipeline = create_pipeline()
    pipeline.modify_config({"save_result_for_debug": False})
    runner = pipeline.runner

    generate_from_kv(pipeline, task="t2i", seed=0)
    first_input = runner.input_info
    result = generate_from_kv(pipeline, task="i2i", seed=0)

    assert isinstance(result, bytes)
    assert pipeline.runner is runner
    assert runner.input_info is not first_input
    assert (first_input.task, runner.input_info.task) == ("t2i", "i2i")
    assert runner.config["task"] == "t2i"

    generate_from_kv(pipeline, task="t2i", seed=0)
    assert runner.input_info.task == "t2i"


def test_omitted_seed_restores_code_default_after_explicit_zero(create_pipeline):
    pipeline = create_pipeline()
    pipeline.modify_config({"save_result_for_debug": False})

    generate_from_kv(pipeline, task="t2i")
    default_noise = pipeline.runner.scheduler.image_prediction.clone()
    assert pipeline.runner.input_info.seed == 42

    generate_from_kv(pipeline, task="t2i", seed=0)
    assert pipeline.runner.input_info.seed == 0
    assert not torch.equal(pipeline.runner.scheduler.image_prediction, default_noise)

    generate_from_kv(pipeline, task="t2i")
    assert pipeline.runner.input_info.seed == 42
    assert torch.equal(pipeline.runner.scheduler.image_prediction, default_noise)
    assert pipeline.runner.config["seed"] == 37


def test_explicit_none_continues_restored_session_rng(create_pipeline):
    pipeline = create_pipeline()
    pipeline.modify_config({"save_result_for_debug": False})

    generate_from_kv(pipeline, task="t2i", seed=321)
    session_state = torch.get_rng_state()
    scheduler = pipeline.runner.scheduler
    expected_next_noise = scheduler.noise_scale * torch.randn_like(scheduler.image_prediction)

    generate_from_kv(pipeline, task="t2i", seed=999)
    torch.set_rng_state(session_state)
    generate_from_kv(pipeline, task="t2i", seed=None)

    assert pipeline.runner.input_info.seed is None
    assert torch.equal(scheduler.image_prediction, expected_next_noise)

    generate_from_kv(pipeline, task="t2i")
    assert pipeline.runner.input_info.seed == 42
