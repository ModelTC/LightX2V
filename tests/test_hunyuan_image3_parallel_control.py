from types import SimpleNamespace

import pytest
import torch

import lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner as runner_module
from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner


def _make_runner(*, seq_parallel, cfg_parallel):
    cfg_p_group = object() if cfg_parallel else None
    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {
        "seq_parallel": seq_parallel,
        "cfg_parallel": cfg_parallel,
        "enable_cfg": cfg_parallel,
        "device_mesh": SimpleNamespace(get_group=lambda mesh_dim: cfg_p_group),
    }
    runner.model = SimpleNamespace(seq_p_group=object() if seq_parallel else None)
    runner._test_cfg_p_group = cfg_p_group
    return runner


def _mock_initialized_dist(monkeypatch):
    world_group = object()
    monkeypatch.setattr(runner_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runner_module.dist, "group", SimpleNamespace(WORLD=world_group))
    return world_group


@pytest.mark.parametrize(
    ("seq_parallel", "cfg_parallel", "expected_group"),
    [
        (True, False, "seq"),
        (False, True, "cfg"),
        (True, True, "world"),
        (False, False, None),
    ],
)
def test_hunyuan_parallel_control_group_covers_cfg_dimension(
    monkeypatch,
    seq_parallel,
    cfg_parallel,
    expected_group,
):
    world_group = _mock_initialized_dist(monkeypatch)
    runner = _make_runner(seq_parallel=seq_parallel, cfg_parallel=cfg_parallel)

    group = runner._parallel_control_group()

    if expected_group == "seq":
        assert group is runner.model.seq_p_group
    elif expected_group == "cfg":
        assert group is runner._test_cfg_p_group
    elif expected_group == "world":
        assert group is world_group
    else:
        assert group is None


@pytest.mark.parametrize(
    ("seq_parallel", "cfg_parallel", "expected_group"),
    [
        (True, False, "seq"),
        (False, True, "cfg"),
        (True, True, "world"),
    ],
)
def test_hunyuan_parallel_tensor_broadcast_and_barrier_use_control_group(
    monkeypatch,
    seq_parallel,
    cfg_parallel,
    expected_group,
):
    world_group = _mock_initialized_dist(monkeypatch)
    runner = _make_runner(seq_parallel=seq_parallel, cfg_parallel=cfg_parallel)
    if expected_group == "seq":
        expected = runner.model.seq_p_group
    elif expected_group == "cfg":
        expected = runner._test_cfg_p_group
    else:
        expected = world_group
    events = []

    monkeypatch.setattr(runner_module.dist, "get_global_rank", lambda group, rank: 13)
    monkeypatch.setattr(
        runner_module.dist,
        "broadcast",
        lambda tensor, src, group: events.append(("broadcast", tensor, src, group)),
    )
    monkeypatch.setattr(runner_module.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(runner_module.dist, "barrier", lambda group: events.append(("barrier", group)))

    tensor = torch.tensor([3])
    assert runner._broadcast_parallel_tensor(tensor) is tensor
    runner._parallel_barrier()

    assert events == [
        ("broadcast", tensor, 13, expected),
        ("barrier", expected),
    ]


def test_hunyuan_local_runner_control_is_a_noop(monkeypatch):
    monkeypatch.setattr(runner_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(runner_module.dist, "is_initialized", lambda: False)
    runner = _make_runner(seq_parallel=True, cfg_parallel=True)
    tensor = torch.tensor([5])

    assert runner._broadcast_parallel_tensor(tensor) is tensor
    assert runner._parallel_barrier() is None


def test_hunyuan_nccl_parallel_barrier_uses_pipeline_latent_device(monkeypatch):
    _mock_initialized_dist(monkeypatch)
    runner = _make_runner(seq_parallel=False, cfg_parallel=True)
    runner.model.pipeline_devices = ["cuda:2", "cuda:3"]
    events = []

    monkeypatch.setattr(runner_module.dist, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(runner_module.torch.cuda, "set_device", lambda device: events.append(("set_device", device)))
    monkeypatch.setattr(
        runner_module.dist,
        "barrier",
        lambda group, device_ids: events.append(("barrier", group, device_ids)),
    )

    runner._parallel_barrier()

    assert events == [
        ("set_device", torch.device("cuda:2")),
        ("barrier", runner._test_cfg_p_group, [2]),
    ]
