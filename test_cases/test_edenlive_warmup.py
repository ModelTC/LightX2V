from types import SimpleNamespace

import pytest
import torch

from lightx2v.models.networks.wan.infer.edenlive.transformer_infer import EdenLiveARTransformerInfer, EdenLiveTransformerInfer
from lightx2v.models.runners.wan import edenlive_runner as module
from lightx2v.models.schedulers.wan.edenlive.scheduler import EdenLiveARScheduler


def make_runner(monkeypatch, mode="none", fail=False):
    events = []
    monkeypatch.setattr(module, "AI_DEVICE", "cpu")
    monkeypatch.setattr(torch.cpu, "synchronize", lambda: events.append("sync"), raising=False)
    scheduler = SimpleNamespace(generator=object(), latents=None, condition_frame=None)

    def clear():
        scheduler.generator = None
        scheduler.latents = None
        scheduler.condition_frame = None
        events.append("clear")

    def prepare(seed, shape, condition):
        scheduler.generator = object()
        scheduler.latents = torch.zeros(shape)
        events.append("prepare")

    scheduler.clear = clear
    scheduler.prepare = prepare
    runner = SimpleNamespace(
        config={"out_dim": 48, "vae_stride": [4, 16, 16], "num_frames": 9, "cpu_offload": mode != "none", "offload_granularity": mode},
        scheduler=scheduler,
        model=SimpleNamespace(transformer_infer=SimpleNamespace(clear_request_state=lambda: events.append("infer_clear")), to_cpu=lambda: events.append("cpu")),
        _WARMUP_RESOLUTIONS=((32, 32), (32, 64)),
    )

    def encode(image, frames):
        assert scheduler.generator is None
        events.append(("encode", tuple(image.shape), frames))
        if fail:
            raise RuntimeError("encoder failed")
        return {"image_encoder_output": {"clean_first_frame": torch.zeros(48, 1, 2, 2)}}

    def diffusion(step_indices):
        assert step_indices == (0,)
        assert scheduler.latents is not None
        events.extend(["step_pre", "infer", "step_post", "decode"])
        return torch.zeros(1, 3, 9, 32, 32)

    runner._encode_inputs = encode
    runner._run_diffusion = diffusion
    runner.clear_warmup_state = lambda: module.EdenLiveRunner.clear_warmup_state(runner)
    return runner, events


@pytest.mark.parametrize("mode", ["none", "block", "model"])
def test_warmup_two_shapes_and_request_state(monkeypatch, mode):
    runner, events = make_runner(monkeypatch, mode)
    module.EdenLiveRunner._run_warmup(runner)
    assert [event for event in events if isinstance(event, tuple)] == [("encode", (3, 32, 32), 9), ("encode", (3, 32, 64), 9)]
    assert events.count("infer") == events.count("step_post") == events.count("decode") == 2
    assert events.count("cpu") == (2 if mode == "model" else 0)
    assert events.count("clear") == events.count("infer_clear") == 2
    assert runner.input_info is None and not hasattr(runner, "inputs")
    assert runner.scheduler.generator is None and runner.scheduler.latents is None
    assert runner.proxy_pixels is None and runner.prompt_embeddings == {} and runner.prompts == []


def test_warmup_exception_cleanup(monkeypatch):
    runner, events = make_runner(monkeypatch, "model", fail=True)
    with pytest.raises(RuntimeError, match="encoder failed"):
        module.EdenLiveRunner._run_warmup(runner)
    assert events[-4:] == ["sync", "cpu", "clear", "infer_clear"]
    assert runner.input_info is None and runner.scheduler.generator is None
    assert not hasattr(runner, "inputs")


def test_specialized_subclass_must_opt_in():
    class OtherRunner(module.Wan21EdenLiveRunner):
        pass

    runner = OtherRunner.__new__(OtherRunner)
    runner.config = {"task": "i2v"}
    with pytest.raises(NotImplementedError, match="warmup is not supported"):
        runner.run_warmup()


def test_scheduler_clear_drops_request_tensors():
    scheduler = EdenLiveARScheduler(dict(model_cls="wan22_edenlive_ar", infer_steps=4, sample_shift=5, seq_parallel=False, dim=256, num_heads=2))
    scheduler.generator = object()
    scheduler.latents = scheduler.latents_input = scheduler.condition_frame = torch.ones(1)
    scheduler.cache_update = True
    scheduler.chunk_start = 12
    scheduler.clear()
    assert scheduler.generator is None and scheduler.latents is None
    assert scheduler.latents_input is None and scheduler.condition_frame is None
    assert not scheduler.cache_update and scheduler.chunk_start == 0


@pytest.mark.parametrize("infer_class,expected", [(EdenLiveTransformerInfer, 2), (EdenLiveARTransformerInfer, 4)])
def test_compile_cache_uses_staging_or_layer_identity(monkeypatch, infer_class, expected):
    compiled = []
    monkeypatch.setattr(torch, "compile", lambda fn, dynamic: compiled.append(fn) or fn)
    infer = infer_class.__new__(infer_class)
    infer.init_compile({"use_compile": True})
    infer.block_offload = True
    infer.infer_block = lambda block, value: value + 1
    slots = [object(), object()]
    for _ in range(2):
        for layer in range(4):
            assert infer.run_block(layer, slots[layer % 2], 0) == 1
    assert len(compiled) == len(infer.compiled_blocks) == expected


def test_eager_does_not_compile(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: pytest.fail("eager compiled"))
    infer = EdenLiveTransformerInfer.__new__(EdenLiveTransformerInfer)
    infer.init_compile({"use_compile": False})
    infer.infer_block = lambda block, value: value + 1
    assert infer.run_block(0, object(), 2) == 3
    assert not infer.compiled_blocks
