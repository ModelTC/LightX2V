import argparse
import json
import multiprocessing as mp
import queue
import time
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.utils.set_config import set_config

_WORLD_SIZE = 2
_PROCESS_TIMEOUT_SECONDS = 45


def _write_tiny_hunyuan_configs(tmp_path, *, q_heads=32, kv_heads=8):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "hunyuan_image_3_moe",
                "num_hidden_layers": 2,
                "num_attention_heads": q_heads,
                "num_key_value_heads": kv_heads,
                "hidden_size": q_heads * 4,
            }
        )
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps({"enable_cfg": True, "hunyuan_cfg_mode": "serial"}))
    return model_path, runtime_path


def _sp_args(model_path, runtime_path, *, size, attn_type):
    return argparse.Namespace(
        seed=123,
        model_cls="hunyuan_image3",
        task="t2i",
        support_tasks=[],
        model_path=str(model_path),
        config_json=str(runtime_path),
        hunyuan_sp_size=size,
        hunyuan_sp_attn_type=attn_type,
        hunyuan_image3_pipeline_layout="interleaved",
    )


def test_hunyuan_sequence_parallel_cli_builds_normalized_parallel_config(tmp_path):
    model_path, runtime_path = _write_tiny_hunyuan_configs(tmp_path)

    config = set_config(
        _sp_args(
            model_path,
            runtime_path,
            size=2,
            attn_type="kv-all-gather",
        )
    )

    assert config["parallel"] == {
        "seq_p_size": 2,
        "cfg_p_size": 1,
        "seq_p_attn_type": "kv_all_gather",
    }
    assert config["hunyuan_cfg_mode"] == "serial"
    assert config["hunyuan_image3_pipeline_layout"] == "interleaved"


def test_hunyuan_sequence_parallel_size_one_disables_parallel(tmp_path):
    model_path, runtime_path = _write_tiny_hunyuan_configs(tmp_path)

    config = set_config(_sp_args(model_path, runtime_path, size=1, attn_type="ulysses"))

    assert config["parallel"] is False


def test_hunyuan_sequence_parallel_size_one_preserves_cfg_parallel(tmp_path):
    model_path, runtime_path = _write_tiny_hunyuan_configs(tmp_path)
    runtime_path.write_text(
        json.dumps(
            {
                "enable_cfg": True,
                "hunyuan_cfg_mode": "parallel",
                "parallel": {
                    "cfg_p_size": 2,
                    "seq_p_size": 2,
                    "seq_p_attn_type": "ulysses",
                },
            }
        )
    )

    config = set_config(_sp_args(model_path, runtime_path, size=1, attn_type="ulysses"))

    assert config["parallel"] == {"cfg_p_size": 2, "seq_p_size": 1}
    assert config["hunyuan_cfg_mode"] == "parallel"


@pytest.mark.parametrize("attn_type", ["kv_all_gather", "ulysses"])
def test_hunyuan_hybrid_cfg_sequence_parallel_config_is_supported(tmp_path, attn_type):
    model_path, runtime_path = _write_tiny_hunyuan_configs(tmp_path)
    runtime_path.write_text(
        json.dumps(
            {
                "enable_cfg": True,
                "hunyuan_cfg_mode": "parallel",
                "parallel": {"cfg_p_size": 2, "seq_p_size": 2},
            }
        )
    )

    config = set_config(_sp_args(model_path, runtime_path, size=2, attn_type=attn_type))

    assert config["parallel"] == {
        "cfg_p_size": 2,
        "seq_p_size": 2,
        "seq_p_attn_type": attn_type,
    }
    assert config["hunyuan_cfg_mode"] == "parallel"


def test_hunyuan_cfg_parallel_requires_cfg_and_parallel_mode(tmp_path):
    model_path, runtime_path = _write_tiny_hunyuan_configs(tmp_path)
    runtime_path.write_text(
        json.dumps(
            {
                "enable_cfg": False,
                "hunyuan_cfg_mode": "parallel",
                "parallel": {"cfg_p_size": 2, "seq_p_size": 2},
            }
        )
    )
    with pytest.raises(ValueError, match="cfg_p_size=2 requires enable_cfg=true"):
        set_config(_sp_args(model_path, runtime_path, size=2, attn_type="kv_all_gather"))

    runtime_path.write_text(
        json.dumps(
            {
                "enable_cfg": True,
                "hunyuan_cfg_mode": "serial",
                "parallel": {"cfg_p_size": 2, "seq_p_size": 2},
            }
        )
    )
    with pytest.raises(ValueError, match="cfg_p_size=2 requires hunyuan_cfg_mode='parallel'"):
        set_config(_sp_args(model_path, runtime_path, size=2, attn_type="kv_all_gather"))


def test_hunyuan_cfg_parallel_size_is_limited_to_cond_uncond_pair(tmp_path):
    model_path, runtime_path = _write_tiny_hunyuan_configs(tmp_path)
    runtime_path.write_text(
        json.dumps(
            {
                "enable_cfg": True,
                "hunyuan_cfg_mode": "parallel",
                "parallel": {"cfg_p_size": 3, "seq_p_size": 2},
            }
        )
    )

    with pytest.raises(ValueError, match="cfg_p_size must be 1 or 2"):
        set_config(_sp_args(model_path, runtime_path, size=2, attn_type="kv_all_gather"))


def test_hunyuan_hybrid_parallel_config_builds_two_dimensional_mesh(monkeypatch):
    import lightx2v.utils.set_config as config_module

    mesh = object()
    init_calls = []
    monkeypatch.setattr(config_module, "AI_DEVICE", "cpu")
    monkeypatch.setattr(config_module.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(config_module.dist, "all_reduce", lambda tensor: tensor)
    monkeypatch.setattr(
        config_module,
        "init_device_mesh",
        lambda device, shape, mesh_dim_names: init_calls.append((device, shape, mesh_dim_names)) or mesh,
    )
    config = {
        "parallel": {"cfg_p_size": 2, "seq_p_size": 2, "seq_p_attn_type": "ulysses"},
        "enable_cfg": True,
        "seq_parallel": False,
        "cfg_parallel": False,
    }

    config_module.set_parallel_config(config)

    assert init_calls == [("cpu", (2, 2), ("cfg_p", "seq_p"))]
    assert config["device_mesh"] is mesh
    assert config["seq_parallel"] is True
    assert config["cfg_parallel"] is True
    assert config["tensor_parallel"] is False


def test_hunyuan_ulysses_rejects_world_size_that_does_not_divide_gqa_heads(tmp_path):
    model_path, runtime_path = _write_tiny_hunyuan_configs(tmp_path, q_heads=32, kv_heads=8)

    with pytest.raises(ValueError, match=r"Ulysses.*Q=32, KV=8, seq_p_size=3"):
        set_config(_sp_args(model_path, runtime_path, size=3, attn_type="ulysses"))


def test_hunyuan_sequence_parallel_rejects_nonserial_cfg(tmp_path):
    model_path, runtime_path = _write_tiny_hunyuan_configs(tmp_path)
    runtime_path.write_text(json.dumps({"enable_cfg": True, "hunyuan_cfg_mode": "batch"}))

    with pytest.raises(ValueError, match="hunyuan_cfg_mode='serial'"):
        set_config(_sp_args(model_path, runtime_path, size=2, attn_type="kv_all_gather"))


@pytest.mark.parametrize(
    ("seq_rank", "expected"),
    [
        (0, ["cuda:0", "cuda:2", "cuda:4"]),
        (1, ["cuda:1", "cuda:3", "cuda:5"]),
    ],
)
def test_hunyuan_sp_model_selects_wan_style_interleaved_pipeline_lane(monkeypatch, seq_rank, expected):
    import lightx2v.models.networks.hunyuan_image3.model as model_module

    group = object()
    config = {
        "pipeline_parallel": True,
        "seq_parallel": True,
        "hunyuan_image3_pipeline_layout": "interleaved",
        "device_mesh": SimpleNamespace(get_group=lambda mesh_dim: group),
    }
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_module.torch.cuda, "device_count", lambda: 6)
    monkeypatch.setattr(model_module.torch.cuda, "current_device", lambda: seq_rank)
    monkeypatch.setattr(model_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(model_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_module.dist, "get_world_size", lambda process_group=None: 2)
    monkeypatch.setattr(model_module.dist, "get_rank", lambda process_group=None: seq_rank)

    assert model_module.resolve_pipeline_devices(config, "cuda") == expected


def test_hunyuan_sp_model_rejects_nondivisible_pipeline_device_count(monkeypatch):
    import lightx2v.models.networks.hunyuan_image3.model as model_module

    config = {
        "pipeline_parallel": True,
        "seq_parallel": True,
        "device_mesh": SimpleNamespace(get_group=lambda mesh_dim: object()),
    }
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_module.torch.cuda, "device_count", lambda: 5)
    monkeypatch.setattr(model_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(model_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_module.dist, "get_world_size", lambda process_group=None: 2)
    monkeypatch.setattr(model_module.dist, "get_rank", lambda process_group=None: 0)

    with pytest.raises(ValueError, match=r"pipeline device count .* divisible by cfg_p_size \* seq_p_size"):
        model_module.resolve_pipeline_devices(config, "cuda")


def test_hunyuan_sp_model_splits_explicit_pipeline_device_pool(monkeypatch):
    import lightx2v.models.networks.hunyuan_image3.model as model_module

    group = object()
    config = {
        "pipeline_parallel_devices": "cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5",
        "seq_parallel": True,
        "device_mesh": SimpleNamespace(get_group=lambda mesh_dim: group),
    }
    monkeypatch.setattr(model_module.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(model_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(model_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_module.dist, "get_world_size", lambda process_group=None: 2)
    monkeypatch.setattr(model_module.dist, "get_rank", lambda process_group=None: 1)

    assert model_module.resolve_pipeline_devices(config, "cuda") == ["cuda:1", "cuda:3", "cuda:5"]


@pytest.mark.parametrize(
    ("global_rank", "expected"),
    [
        (0, ["cuda:0", "cuda:4"]),
        (1, ["cuda:1", "cuda:5"]),
        (2, ["cuda:2", "cuda:6"]),
        (3, ["cuda:3", "cuda:7"]),
    ],
)
def test_hunyuan_hybrid_model_uses_disjoint_global_rank_pipeline_lanes(monkeypatch, global_rank, expected):
    import lightx2v.models.networks.hunyuan_image3.model as model_module

    config = {
        "pipeline_parallel": True,
        "seq_parallel": True,
        "cfg_parallel": True,
        "hunyuan_image3_pipeline_layout": "interleaved",
    }
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_module.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(model_module.torch.cuda, "current_device", lambda: global_rank)
    monkeypatch.setattr(model_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(model_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_module.dist, "get_world_size", lambda process_group=None: 4)
    monkeypatch.setattr(model_module.dist, "get_rank", lambda process_group=None: global_rank)

    assert model_module.resolve_pipeline_devices(config, "cuda") == expected


def test_hunyuan_pure_cfg_model_keeps_contiguous_pipeline_lane(monkeypatch):
    import lightx2v.models.networks.hunyuan_image3.model as model_module

    cfg_group = object()
    config = {
        "pipeline_parallel": True,
        "seq_parallel": False,
        "cfg_parallel": True,
        "parallel": {"cfg_p_size": 2, "seq_p_size": 1},
        "device_mesh": SimpleNamespace(get_group=lambda mesh_dim: cfg_group),
    }
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_module.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(model_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(model_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_module.dist, "get_rank", lambda process_group=None: 1)

    assert model_module.resolve_pipeline_devices(config, "cuda") == ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]


def test_hunyuan_model_accepts_parallel_cfg_branch_with_sequence_parallel(monkeypatch):
    import lightx2v.models.networks.hunyuan_image3.model as model_module

    model = model_module.HunyuanImage3Model.__new__(model_module.HunyuanImage3Model)
    model.config = {
        "seq_parallel": True,
        "cfg_parallel": True,
        "enable_cfg": True,
        "hunyuan_cfg_mode": "parallel",
        "use_taylor_cache": False,
        "enable_kv_cache": True,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }
    model.seq_p_group = object()
    model.sequence_parallel_attn_type = "ulysses"
    monkeypatch.setattr(model_module.dist, "get_world_size", lambda process_group=None: 2)

    model._validate_sequence_parallel_config()


def test_nvidia_parallel_init_remains_original_rank_based_implementation():
    source = Path("lightx2v_platform/base/nvidia.py").read_text()

    assert "pg_options.is_high_priority_stream = True" in source
    assert "dist.init_process_group(backend=\"nccl\", pg_options=pg_options)" in source
    assert "torch.cuda.set_device(dist.get_rank())" in source
    assert source.index("dist.init_process_group") < source.index("torch.cuda.set_device(dist.get_rank())")
    assert "LIGHTX2V_CUDA_DEVICE_INDEX" not in source


def test_hunyuan_dist_sp_scripts_and_configs_expose_both_backends_flashinfer_and_cache():
    script = Path("scripts/dist_infer/run_hunyuan_image3_t2i_dist_sp.sh").read_text()
    kv_wrapper = Path("scripts/dist_infer/run_hunyuan_image3_t2i_dist_kv_all_gather.sh").read_text()
    ulysses_wrapper = Path("scripts/dist_infer/run_hunyuan_image3_t2i_dist_ulysses.sh").read_text()

    assert "SP_ATTN_TYPE=kv_all_gather" in kv_wrapper
    assert "SP_ATTN_TYPE=ulysses" in ulysses_wrapper
    assert "-m lightx2v.infer" in script
    assert "hunyuan_image3_sp_entry.py" not in script
    assert '--hunyuan_sp_size "$SP_SIZE"' in script
    assert '--hunyuan_sp_attn_type "$SP_ATTN_TYPE"' in script
    assert '--moe_impl "${moe_impl:-flashinfer}"' in script
    assert '--flashinfer_autotune_mode "$resolved_autotune_mode"' in script
    assert '--enable_kv_cache "${enable_kv_cache:-true}"' in script
    assert "--hunyuan_cfg_mode serial" in script

    expected_attention_impl = {"kv_all_gather": "flash_attention_2", "ulysses": "sdpa"}
    for backend in ("kv_all_gather", "ulysses"):
        config = json.loads(Path(f"configs/dist_infer/hunyuan_image3_t2i_dist_{backend}.json").read_text())
        assert config["hunyuan_cfg_mode"] == "serial"
        assert config["moe_impl"] == "flashinfer"
        assert config["attn_impl"] == expected_attention_impl[backend]
        assert config["enable_kv_cache"] is True
        assert config["enable_text_kv_cache"] is True
        assert config["parallel"] == {
            "cfg_p_size": 1,
            "seq_p_size": 2,
            "seq_p_attn_type": backend,
        }


def test_hunyuan_hybrid_cfg_sp_scripts_and_configs_contract():
    script = Path("scripts/dist_infer/run_hunyuan_image3_t2i_dist_cfg_sp.sh").read_text()
    wrappers = {
        "kv_all_gather": Path("scripts/dist_infer/run_hunyuan_image3_t2i_dist_cfg_kv_all_gather.sh").read_text(),
        "ulysses": Path("scripts/dist_infer/run_hunyuan_image3_t2i_dist_cfg_ulysses.sh").read_text(),
    }

    assert "world_size=$((CFG_SIZE * SP_SIZE))" in script
    assert "visible_gpu_count % world_size" in script
    assert '--nproc_per_node="$world_size"' in script
    assert "--hunyuan_cfg_mode parallel" in script
    assert '--hunyuan_sp_size "$SP_SIZE"' in script
    assert '--hunyuan_sp_attn_type "$SP_ATTN_TYPE"' in script
    assert '--moe_impl "${moe_impl:-flashinfer}"' in script
    assert '--flashinfer_autotune_mode "$resolved_autotune_mode"' in script
    assert '--enable_kv_cache "${enable_kv_cache:-true}"' in script
    assert '--enable_text_kv_cache "${enable_text_kv_cache:-${enable_kv_cache:-true}}"' in script
    assert "MIN_PIPELINE_GPUS_PER_LANE" not in script
    assert "MIN_FREE_GPU_MEMORY_MIB" not in script
    assert "--query-compute-apps" not in script

    expected_attention_impl = {"kv_all_gather": "flash_attention_2", "ulysses": "sdpa"}
    for backend, wrapper in wrappers.items():
        assert f"SP_ATTN_TYPE={backend}" in wrapper
        assert "run_hunyuan_image3_t2i_dist_cfg_sp.sh" in wrapper
        config = json.loads(Path(f"configs/dist_infer/hunyuan_image3_t2i_dist_cfg_{backend}.json").read_text())
        assert config["enable_cfg"] is True
        assert config["hunyuan_cfg_mode"] == "parallel"
        assert config["moe_impl"] == "flashinfer"
        assert config["attn_impl"] == expected_attention_impl[backend]
        assert config["flashinfer_autotune_mode"] == "tune"
        assert config["enable_kv_cache"] is True
        assert config["enable_text_kv_cache"] is True
        assert config["use_taylor_cache"] is False
        assert config["parallel"] == {
            "cfg_p_size": 2,
            "seq_p_size": 2,
            "seq_p_attn_type": backend,
        }


def test_hunyuan_registered_flash_attention_uses_local_ulysses_head_count():
    from lightx2v.models.networks.hunyuan_image3.infer.transformer_infer import HunyuanImage3TransformerInfer

    class IdentityFlattenedKernel:
        def apply(self, q, k, v, **kwargs):
            del k, v, kwargs
            return q.reshape(q.shape[0] * q.shape[1], -1)

    infer = HunyuanImage3TransformerInfer.__new__(HunyuanImage3TransformerInfer)
    infer.attn_impl = "flash_attn2"
    infer.attn_kernel = IdentityFlattenedKernel()
    infer.num_heads = 4
    infer.head_dim = 2
    infer._attn_cu_seqlens_cache = {}
    query = torch.randn(1, 2, 3, 2)

    output = infer._apply_registered_attention_kernel(
        query,
        torch.randn_like(query),
        torch.randn_like(query),
        causal=False,
    )

    assert output.shape == query.shape
    torch.testing.assert_close(output, query)


@pytest.mark.parametrize("output_rank", [True, False])
def test_hunyuan_distributed_flashinfer_autotune_uses_single_cache_writer(monkeypatch, tmp_path, output_rank):
    from contextlib import contextmanager

    import lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner as runner_module
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    events = []

    class FakeTuner:
        def load_configs(self, path):
            events.append(("load", path))
            return True

        def save_configs(self, path):
            events.append(("save", path))

    fake_tuner = FakeTuner()

    class FakeAutoTuner:
        @staticmethod
        def get():
            return fake_tuner

    @contextmanager
    def fake_autotune(**kwargs):
        events.append(("enter", kwargs))
        yield
        events.append(("exit", kwargs))

    monkeypatch.setattr(runner_module, "FlashInferAutoTuner", FakeAutoTuner)
    monkeypatch.setattr(runner_module, "flashinfer_autotune", fake_autotune)
    monkeypatch.setattr(runner_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(runner_module.dist, "is_initialized", lambda: True)
    world_group = object()
    monkeypatch.setattr(runner_module.dist, "group", SimpleNamespace(WORLD=world_group))
    monkeypatch.setattr(runner_module.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(runner_module.dist, "get_global_rank", lambda group, rank: rank)
    monkeypatch.setattr(runner_module.dist, "all_reduce", lambda tensor, op, group: events.append(("all_reduce", int(tensor.item()), group)))
    monkeypatch.setattr(runner_module.dist, "broadcast", lambda tensor, src, group: events.append(("broadcast", src, group)))

    cache_path = tmp_path / "autotune.json"
    cache_path.write_text("{}")
    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {"seq_parallel": True, "cfg_parallel": True, "enable_cfg": True}
    runner.model = SimpleNamespace(seq_p_group=object())
    runner._is_output_rank = lambda: output_rank

    with runner._distributed_flashinfer_autotune_context(str(cache_path), (128, 256), True):
        events.append(("body", None))

    assert ("load", str(cache_path)) in events
    assert any(event[0] == "enter" and event[1]["cache"] is None for event in events)
    assert any(event[0] == "all_reduce" and event[2] is world_group for event in events)
    assert any(event[0] == "broadcast" and event[2] is world_group for event in events)
    assert (("save", str(cache_path)) in events) is output_rank


def test_hunyuan_flash_attention_does_not_infer_segments_from_opaque_custom_mask():
    from lightx2v.models.networks.hunyuan_image3.infer.module_io import HunyuanImage3PreInferOutput
    from lightx2v.models.networks.hunyuan_image3.infer.transformer_infer import HunyuanImage3TransformerInfer

    infer = HunyuanImage3TransformerInfer.__new__(HunyuanImage3TransformerInfer)
    infer.attn_impl = "flash_attn2"
    infer.attn_kernel = object()
    infer._attn_fallback_warnings = set()
    pre_infer_out = HunyuanImage3PreInferOutput(
        hidden_states=torch.zeros(1, 4, 8),
        attention_mask=torch.eye(4, dtype=torch.bool).reshape(1, 1, 4, 4),
        position_ids=torch.arange(4).reshape(1, 4),
        full_attn_slices=None,
    )

    assert infer._prepare_attention_segment_specs(pre_infer_out) is None
    torch.manual_seed(2040)
    query = torch.randn(1, 2, 4, 3)
    key = torch.randn(1, 2, 4, 3)
    value = torch.randn(1, 2, 4, 3)
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=pre_infer_out.attention_mask, dropout_p=0.0)
    actual = infer._registered_attention(
        query,
        key,
        value,
        pre_infer_out.attention_mask,
        position_ids=pre_infer_out.position_ids,
        full_attn_slices=None,
    )
    torch.testing.assert_close(actual, expected)


def _hybrid_attention_mask(seq_len):
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    mask[1:4, 1:4] = True
    return mask.unsqueeze(0).unsqueeze(0)


def _pad_sequence(tensor, padded_seq_len, sequence_dim):
    padding_size = padded_seq_len - tensor.shape[sequence_dim]
    if not padding_size:
        return tensor
    shape = list(tensor.shape)
    shape[sequence_dim] = padding_size
    return torch.cat((tensor, tensor.new_zeros(shape)), dim=sequence_dim)


def _local_sequence_shard(tensor, rank, local_seq_len, sequence_dim):
    return tensor.narrow(sequence_dim, rank * local_seq_len, local_seq_len).contiguous()


def _run_worker(worker, rendezvous_path, result_queue, rank):
    error = None
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{rendezvous_path}",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=20),
        )
        worker(rank)
    except BaseException:
        error = traceback.format_exc()
    finally:
        was_initialized = dist.is_initialized()
        if was_initialized:
            dist.destroy_process_group()
        result_queue.put(
            {
                "rank": rank,
                "error": error,
                "destroyed": was_initialized and not dist.is_initialized(),
            }
        )
    if error is not None:
        raise AssertionError(error)


def _spawn_and_wait(worker, tmp_path):
    rendezvous_path = tmp_path / f"gloo_rendezvous_{worker.__name__}"
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_run_worker,
            args=(worker, str(rendezvous_path), result_queue, rank),
        )
        for rank in range(_WORLD_SIZE)
    ]

    for process in processes:
        process.start()

    deadline = time.monotonic() + _PROCESS_TIMEOUT_SECONDS
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))

    timed_out = [process for process in processes if process.is_alive()]
    for process in timed_out:
        process.terminate()
    for process in timed_out:
        process.join(5)

    results = []
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            break

    assert not timed_out, f"Distributed worker timed out: {[process.pid for process in timed_out]}"
    assert len(results) == _WORLD_SIZE, f"Expected {_WORLD_SIZE} worker results, got {results}"
    errors = [result for result in results if result["error"]]
    assert not errors, "\n".join(result["error"] for result in errors)
    assert all(result["destroyed"] for result in results)
    assert all(process.exitcode == 0 for process in processes)


def _kv_all_gather_worker(rank):
    from lightx2v.models.networks.hunyuan_image3.infer.kv_cache import HunyuanImage3StaticKVCache
    from lightx2v.models.networks.hunyuan_image3.infer.module_io import HunyuanImage3SequenceParallelState
    from lightx2v.models.networks.hunyuan_image3.infer.transformer_infer import HunyuanImage3TransformerInfer
    from lightx2v.models.networks.hunyuan_image3.infer.utils import repeat_kv
    from lightx2v.models.networks.hunyuan_image3.model import HunyuanImage3Model

    seq_len = 5
    padded_seq_len = 6
    local_seq_len = padded_seq_len // _WORLD_SIZE
    valid_local_seq_len = min(local_seq_len, max(0, seq_len - rank * local_seq_len))
    q_heads, kv_heads, head_dim = 4, 2, 3

    torch.manual_seed(2026)
    global_query = torch.randn(1, q_heads, seq_len, head_dim)
    global_key = torch.randn(1, kv_heads, seq_len, head_dim)
    global_value = torch.randn(1, kv_heads, seq_len, head_dim)
    padded_query = _pad_sequence(global_query, padded_seq_len, sequence_dim=2)
    padded_key = _pad_sequence(global_key, padded_seq_len, sequence_dim=2)
    padded_value = _pad_sequence(global_value, padded_seq_len, sequence_dim=2)
    local_query = _local_sequence_shard(padded_query, rank, local_seq_len, sequence_dim=2)
    local_key = _local_sequence_shard(padded_key, rank, local_seq_len, sequence_dim=2)
    local_value = _local_sequence_shard(padded_value, rank, local_seq_len, sequence_dim=2)

    state = HunyuanImage3SequenceParallelState(
        attn_type="kv_all_gather",
        original_seq_len=seq_len,
        padded_seq_len=padded_seq_len,
        local_seq_len=local_seq_len,
        local_start=rank * local_seq_len,
        valid_local_seq_len=valid_local_seq_len,
        global_position_ids=torch.arange(seq_len).unsqueeze(0),
    )
    infer = HunyuanImage3TransformerInfer.__new__(HunyuanImage3TransformerInfer)
    infer.seq_p_group = dist.group.WORLD
    infer._sp_gather_buffers = {}

    gathered_key, gathered_value = infer._sequence_parallel_gather_kv(local_key, local_value, state)
    torch.testing.assert_close(gathered_key, global_key)
    torch.testing.assert_close(gathered_value, global_value)

    full_mask = _hybrid_attention_mask(seq_len)
    local_mask = full_mask[:, :, rank * local_seq_len : min((rank + 1) * local_seq_len, seq_len)]
    repeated_key = repeat_kv(gathered_key, q_heads // kv_heads)
    repeated_value = repeat_kv(gathered_value, q_heads // kv_heads)
    local_output = torch.zeros_like(local_query)
    if valid_local_seq_len:
        local_output[:, :, :valid_local_seq_len] = F.scaled_dot_product_attention(
            local_query[:, :, :valid_local_seq_len],
            repeated_key,
            repeated_value,
            attn_mask=local_mask,
            dropout_p=0.0,
        )

    model = HunyuanImage3Model.__new__(HunyuanImage3Model)
    model.seq_p_group = dist.group.WORLD
    model._sp_gather_buffers = {}
    gathered_output = model._seq_parallel_post_process(
        local_output.transpose(1, 2).reshape(1, local_seq_len, q_heads * head_dim),
        state,
    ).reshape(1, seq_len, q_heads, head_dim).transpose(1, 2)
    expected = F.scaled_dot_product_attention(
        global_query,
        repeat_kv(global_key, q_heads // kv_heads),
        repeat_kv(global_value, q_heads // kv_heads),
        attn_mask=full_mask,
        dropout_p=0.0,
    )
    torch.testing.assert_close(gathered_output, expected, atol=1e-6, rtol=1e-6)

    cache = HunyuanImage3StaticKVCache(num_layers=1, max_cache_len=seq_len)
    cached_key, cached_value = cache.update(
        gathered_key,
        gathered_value,
        layer_idx=0,
        cache_position=torch.arange(seq_len),
    )
    assert cached_key.shape == (1, kv_heads, seq_len, head_dim)
    torch.testing.assert_close(cached_key, global_key)
    torch.testing.assert_close(cached_value, global_value)
    checksums = [torch.empty(2) for _ in range(_WORLD_SIZE)]
    dist.all_gather(checksums, torch.stack((cached_key.sum(), cached_value.sum())))
    torch.testing.assert_close(checksums[0], checksums[1])

    update_positions = torch.tensor([[1, 3]])
    update_key = torch.full((1, kv_heads, 2, head_dim), 17.0)
    update_value = torch.full((1, kv_heads, 2, head_dim), 23.0)
    update_state = HunyuanImage3SequenceParallelState(
        attn_type="kv_all_gather",
        original_seq_len=2,
        padded_seq_len=2,
        local_seq_len=1,
        local_start=rank,
        valid_local_seq_len=1,
        global_position_ids=update_positions,
    )
    gathered_update_key, gathered_update_value = infer._sequence_parallel_gather_kv(
        update_key[:, :, rank : rank + 1],
        update_value[:, :, rank : rank + 1],
        update_state,
    )
    cached_key, cached_value = cache.update(
        gathered_update_key,
        gathered_update_value,
        layer_idx=0,
        cache_position=update_positions,
    )
    expected_key = global_key.clone()
    expected_value = global_value.clone()
    expected_key[:, :, update_positions[0]] = update_key
    expected_value[:, :, update_positions[0]] = update_value
    torch.testing.assert_close(cached_key, expected_key)
    torch.testing.assert_close(cached_value, expected_value)


def _ulysses_worker(rank):
    from lightx2v.models.networks.hunyuan_image3.infer.kv_cache import HunyuanImage3StaticKVCache
    from lightx2v.models.networks.hunyuan_image3.infer.module_io import HunyuanImage3SequenceParallelState
    from lightx2v.models.networks.hunyuan_image3.infer.transformer_infer import HunyuanImage3TransformerInfer
    from lightx2v.models.networks.hunyuan_image3.infer.utils import repeat_kv

    seq_len = 5
    padded_seq_len = 6
    local_seq_len = padded_seq_len // _WORLD_SIZE
    valid_local_seq_len = min(local_seq_len, max(0, seq_len - rank * local_seq_len))
    q_heads, kv_heads, head_dim = 4, 2, 3

    torch.manual_seed(2027)
    global_query = torch.randn(1, q_heads, seq_len, head_dim)
    global_key = torch.randn(1, kv_heads, seq_len, head_dim)
    global_value = torch.randn(1, kv_heads, seq_len, head_dim)
    padded_query = _pad_sequence(global_query, padded_seq_len, sequence_dim=2)
    padded_key = _pad_sequence(global_key, padded_seq_len, sequence_dim=2)
    padded_value = _pad_sequence(global_value, padded_seq_len, sequence_dim=2)
    local_query = _local_sequence_shard(padded_query, rank, local_seq_len, sequence_dim=2)
    local_key = _local_sequence_shard(padded_key, rank, local_seq_len, sequence_dim=2)
    local_value = _local_sequence_shard(padded_value, rank, local_seq_len, sequence_dim=2)
    full_mask = _hybrid_attention_mask(seq_len)

    state = HunyuanImage3SequenceParallelState(
        attn_type="ulysses",
        original_seq_len=seq_len,
        padded_seq_len=padded_seq_len,
        local_seq_len=local_seq_len,
        local_start=rank * local_seq_len,
        valid_local_seq_len=valid_local_seq_len,
        global_position_ids=torch.arange(seq_len).unsqueeze(0),
        global_attention_mask=full_mask,
    )
    infer = HunyuanImage3TransformerInfer.__new__(HunyuanImage3TransformerInfer)
    infer.seq_p_group = dist.group.WORLD

    head_query, head_key, head_value = infer._sequence_parallel_ulysses_seq_to_head(
        local_query,
        local_key,
        local_value,
        state,
    )
    assert head_query.shape == (1, q_heads // _WORLD_SIZE, seq_len, head_dim)
    assert head_key.shape == (1, kv_heads // _WORLD_SIZE, seq_len, head_dim)
    expected_head_key = global_key[:, rank : rank + 1]
    expected_head_value = global_value[:, rank : rank + 1]
    torch.testing.assert_close(head_key, expected_head_key)
    torch.testing.assert_close(head_value, expected_head_value)

    head_output = F.scaled_dot_product_attention(
        head_query,
        repeat_kv(head_key, (q_heads // _WORLD_SIZE) // (kv_heads // _WORLD_SIZE)),
        repeat_kv(head_value, (q_heads // _WORLD_SIZE) // (kv_heads // _WORLD_SIZE)),
        attn_mask=full_mask,
        dropout_p=0.0,
    )
    local_output = infer._sequence_parallel_ulysses_head_to_seq(head_output, state)
    assert local_output.shape == (1, q_heads, local_seq_len, head_dim)

    expected_global = F.scaled_dot_product_attention(
        global_query,
        repeat_kv(global_key, q_heads // kv_heads),
        repeat_kv(global_value, q_heads // kv_heads),
        attn_mask=full_mask,
        dropout_p=0.0,
    )
    expected_local = _local_sequence_shard(
        _pad_sequence(expected_global, padded_seq_len, sequence_dim=2),
        rank,
        local_seq_len,
        sequence_dim=2,
    )
    torch.testing.assert_close(local_output, expected_local, atol=1e-6, rtol=1e-6)
    if valid_local_seq_len < local_seq_len:
        assert torch.count_nonzero(local_output[:, :, valid_local_seq_len:]) == 0

    cache = HunyuanImage3StaticKVCache(num_layers=1, max_cache_len=seq_len)
    cache.update(
        head_key,
        head_value,
        layer_idx=0,
        cache_position=torch.arange(seq_len),
    )
    assert cache.layers[0].key.shape == (1, kv_heads // _WORLD_SIZE, seq_len, head_dim)
    gathered_key_heads = [torch.empty_like(cache.layers[0].key) for _ in range(_WORLD_SIZE)]
    gathered_value_heads = [torch.empty_like(cache.layers[0].value) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered_key_heads, cache.layers[0].key)
    dist.all_gather(gathered_value_heads, cache.layers[0].value)
    torch.testing.assert_close(torch.cat(gathered_key_heads, dim=1), global_key)
    torch.testing.assert_close(torch.cat(gathered_value_heads, dim=1), global_value)

    update_positions = torch.tensor([[1, 3]])
    update_query = torch.full((1, q_heads, 2, head_dim), 11.0)
    update_key = torch.full((1, kv_heads, 2, head_dim), 17.0)
    update_value = torch.full((1, kv_heads, 2, head_dim), 23.0)
    update_state = HunyuanImage3SequenceParallelState(
        attn_type="ulysses",
        original_seq_len=2,
        padded_seq_len=2,
        local_seq_len=1,
        local_start=rank,
        valid_local_seq_len=1,
        global_position_ids=update_positions,
    )
    _, update_key_heads, update_value_heads = infer._sequence_parallel_ulysses_seq_to_head(
        update_query[:, :, rank : rank + 1],
        update_key[:, :, rank : rank + 1],
        update_value[:, :, rank : rank + 1],
        update_state,
    )
    cache.update(
        update_key_heads,
        update_value_heads,
        layer_idx=0,
        cache_position=update_positions,
    )
    gathered_key_heads = [torch.empty_like(cache.layers[0].key) for _ in range(_WORLD_SIZE)]
    gathered_value_heads = [torch.empty_like(cache.layers[0].value) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered_key_heads, cache.layers[0].key)
    dist.all_gather(gathered_value_heads, cache.layers[0].value)
    expected_key = global_key.clone()
    expected_value = global_value.clone()
    expected_key[:, :, update_positions[0]] = update_key
    expected_value[:, :, update_positions[0]] = update_value
    torch.testing.assert_close(torch.cat(gathered_key_heads, dim=1), expected_key)
    torch.testing.assert_close(torch.cat(gathered_value_heads, dim=1), expected_value)


class _FakeLinear:
    def __init__(self, weight):
        self.weight = weight
        self.bias = None

    def apply(self, hidden_states):
        return hidden_states @ self.weight


def _cache_attention_mask(position_ids, kv_len=7):
    mask = torch.zeros(len(position_ids), kv_len, dtype=torch.bool)
    for row, position in enumerate(position_ids):
        visible_stop = 6 if 2 <= position < 6 else position + 1
        mask[row, :visible_stop] = True
    return mask.reshape(1, 1, len(position_ids), kv_len)


def _full_infer_attention_cache_worker(rank):
    from lightx2v.models.networks.hunyuan_image3.infer.kv_cache import HunyuanImage3StaticKVCache
    from lightx2v.models.networks.hunyuan_image3.infer.module_io import HunyuanImage3PreInferOutput
    from lightx2v.models.networks.hunyuan_image3.infer.transformer_infer import HunyuanImage3TransformerInfer
    from lightx2v.models.networks.hunyuan_image3.model import HunyuanImage3Model

    config = {
        "num_layers": 1,
        "hidden_size": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "attention_head_dim": 2,
        "hidden_act": "silu",
        "attn_impl": "torch_sdpa",
        "seq_parallel": False,
    }
    torch.manual_seed(2030)
    phase = SimpleNamespace(
        qkv_proj=_FakeLinear(torch.randn(8, 16) * 0.2),
        o_proj=_FakeLinear(torch.eye(8)),
        query_layernorm=None,
        key_layernorm=None,
    )
    steps = [
        (torch.randn(1, 7, 8), list(range(7))),
        (torch.randn(1, 5, 8), [1, 2, 3, 4, 5]),
    ]
    full_attn_slices = [[(2, 6)]]

    baseline_infer = HunyuanImage3TransformerInfer(config)
    baseline_cache = HunyuanImage3StaticKVCache(num_layers=1, max_cache_len=7)
    expected_outputs = []
    for hidden_states, positions in steps:
        position_ids = torch.tensor(positions).reshape(1, -1)
        expected_outputs.append(
            baseline_infer.infer_attention(
                0,
                phase,
                hidden_states,
                _cache_attention_mask(positions),
                position_ids,
                None,
                full_attn_slices,
                baseline_cache,
            )
        )

    for backend in ("kv_all_gather", "ulysses"):
        model = HunyuanImage3Model.__new__(HunyuanImage3Model)
        model.config = {"seq_parallel": True}
        model.seq_p_group = dist.group.WORLD
        model.sequence_parallel_attn_type = backend
        model._sp_gather_buffers = {}
        infer = HunyuanImage3TransformerInfer(config)
        infer.seq_p_group = dist.group.WORLD
        cache = HunyuanImage3StaticKVCache(num_layers=1, max_cache_len=7)

        for step_index, (hidden_states, positions) in enumerate(steps):
            pre_infer_out = HunyuanImage3PreInferOutput(
                hidden_states=hidden_states.clone(),
                attention_mask=_cache_attention_mask(positions),
                position_ids=torch.tensor(positions).reshape(1, -1),
                full_attn_slices=full_attn_slices,
            )
            pre_infer_out = model._seq_parallel_pre_process(pre_infer_out)
            local_output = infer.infer_attention(
                0,
                phase,
                pre_infer_out.hidden_states,
                pre_infer_out.attention_mask,
                pre_infer_out.position_ids,
                None,
                pre_infer_out.full_attn_slices,
                cache,
                pre_infer_out.sequence_parallel_state,
            )
            output = model._seq_parallel_post_process(
                local_output,
                pre_infer_out.sequence_parallel_state,
            )
            torch.testing.assert_close(output, expected_outputs[step_index], atol=1e-5, rtol=1e-5)

        if backend == "kv_all_gather":
            torch.testing.assert_close(cache.layers[0].key, baseline_cache.layers[0].key)
        else:
            gathered_heads = [torch.empty_like(cache.layers[0].key) for _ in range(_WORLD_SIZE)]
            dist.all_gather(gathered_heads, cache.layers[0].key)
            torch.testing.assert_close(torch.cat(gathered_heads, dim=1), baseline_cache.layers[0].key)


def _sequence_parallel_collectives_worker(rank):
    _kv_all_gather_worker(rank)
    dist.barrier()
    _ulysses_worker(rank)
    dist.barrier()
    _full_infer_attention_cache_worker(rank)


def test_hunyuan_two_rank_gloo_kv_all_gather_and_ulysses_with_odd_trim_hybrid_attention_and_cache(tmp_path):
    _spawn_and_wait(_sequence_parallel_collectives_worker, tmp_path)
