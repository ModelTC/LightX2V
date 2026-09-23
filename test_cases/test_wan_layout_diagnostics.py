"""Regression checks for bounded diagnostics and CPU UniPC coefficients."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from lightx2v.common.offload.timing import InferenceTimer
from lightx2v.models.schedulers.wan import scheduler as wan_scheduler
from lightx2v_platform.base.global_var import AI_DEVICE
from scripts.wan.layout.diagnostics import instrument, run_prefix


def test_prefix_preserves_schedule_and_stops_before_decode():
    calls = []
    scheduler = SimpleNamespace(infer_steps=40, timesteps=tuple(range(40)), latents=torch.tensor(0))

    def pre(step_index):
        calls.append(step_index)

    def post():
        scheduler.latents += 1

    scheduler.step_pre, scheduler.step_post = pre, post
    runner = SimpleNamespace(
        model=SimpleNamespace(scheduler=scheduler, infer=lambda inputs: None),
        prepare_request=lambda request: SimpleNamespace(seed=request["seed"]),
        run_input_encoder=lambda: {"encoded": True},
        init_run=lambda: None,
        init_run_segment=lambda index: None,
        video_segment_num=1,
    )
    result = run_prefix(runner, {"seed": 42}, 4)
    assert result.item() == 4
    assert calls == [0, 1, 2, 3]
    assert scheduler.infer_steps == 40 and len(scheduler.timesteps) == 40
    assert not hasattr(runner, "inputs")
    with pytest.raises(ValueError, match="scheduler length"):
        run_prefix(runner, {"seed": 42}, 41)

    @contextmanager
    def interrupted(index):
        if index == 1:
            raise RuntimeError("interrupted")
        yield

    with pytest.raises(RuntimeError, match="interrupted"):
        run_prefix(runner, {"seed": 42}, 4, interrupted)
    assert not hasattr(runner, "inputs")


device_module = getattr(torch, AI_DEVICE)
requires_device = pytest.mark.skipif(AI_DEVICE not in ("cuda", "npu") or not device_module.is_available(), reason="requires CUDA or Ascend")


@requires_device
def test_diagnostic_spans_and_hooks_restore_on_error():
    timer = InferenceTimer(device_module, capacity=8)
    timer.enabled = True
    value = torch.ones((32, 32), device=AI_DEVICE)
    stream = device_module.Stream()
    stream.wait_stream(device_module.current_stream())
    with device_module.stream(stream), timer.measure("block_compute"):
        result = value @ value
    with timer.measure("wait_compute", device_timing=False):
        stream.synchronize()
    rows = timer.collect()
    assert rows[0]["device_span_ms"] > 0
    assert rows[1]["device_span_ms"] is None
    assert all(row["host_ms"] >= 0 for row in rows)
    assert result[0, 0].item() == 32
    assert timer.collect() == []
    timer.enabled = False
    with timer.measure("disabled"):
        pass
    assert timer.collect() == []

    def operation(*args, **kwargs):
        return value

    manager = SimpleNamespace(transfer_timer=None, diagnostic_timer=None)
    infer = SimpleNamespace(offload_manager=manager, run_block=operation, infer_self_attn=operation, infer_cross_attn=operation, infer_ffn=operation)
    model = SimpleNamespace(
        transformer_infer=infer,
        _infer_cond_uncond=operation,
        scheduler=SimpleNamespace(step_pre=operation, step_post=operation),
        pre_weight=SimpleNamespace(to_cuda=operation, to_cpu=operation),
        transformer_weights=SimpleNamespace(non_block_weights_to_cuda=operation, non_block_weights_to_cpu=operation),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        with instrument(SimpleNamespace(model=model), timer, SimpleNamespace(context={})):
            assert infer.run_block is not operation
            raise RuntimeError("interrupted")
    assert infer.run_block is operation
    assert model._infer_cond_uncond is operation
    assert manager.transfer_timer is None and manager.diagnostic_timer is None


@requires_device
@pytest.mark.parametrize("method", ["multistep_uni_p_bh_update", "multistep_uni_c_bh_update"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_unipc_host_coefficients_match_device_solution(monkeypatch, method, order):
    # Compare against the existing device solve on CUDA, or an independent CPU run on Ascend.
    reference_device = AI_DEVICE if AI_DEVICE == "cuda" else "cpu"
    source = torch.linspace(-0.5, 0.5, 24).reshape(1, 2, 3, 4)

    def update(device):
        scheduler = object.__new__(wan_scheduler.WanScheduler)
        scheduler.sigmas = torch.tensor([0.95, 0.82, 0.68, 0.53, 0.37, 0.2])
        scheduler.step_index = 3
        scheduler.model_outputs = [(source + shift).to(device) for shift in (0.1, 0.2, 0.3)]
        scheduler.timestep_list = [900, 800, 700]
        sample = source.to(device)
        if method == "multistep_uni_p_bh_update":
            return scheduler.multistep_uni_p_bh_update(sample, sample=sample, order=order)
        return scheduler.multistep_uni_c_bh_update(sample, last_sample=sample - 0.1, this_sample=sample, order=order)

    monkeypatch.setattr(wan_scheduler, "AI_DEVICE", reference_device)
    reference = update(reference_device).cpu()
    solve = torch.linalg.solve
    solved_on = []

    def cpu_solve(a, b):
        solved_on.append(a.device.type)
        assert a.device.type == b.device.type == "cpu"
        return solve(a, b)

    monkeypatch.setattr(wan_scheduler, "AI_DEVICE", "npu")
    monkeypatch.setattr(torch.linalg, "solve", cpu_solve)
    actual = update(AI_DEVICE)
    assert actual.device.type == AI_DEVICE
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-5, atol=1e-6)
    if order == 3 or (method == "multistep_uni_c_bh_update" and order == 2):
        assert solved_on == ["cpu"]
