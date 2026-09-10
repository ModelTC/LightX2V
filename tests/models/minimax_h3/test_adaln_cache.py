"""Exercise real cache math and IO with small official/native checkpoints."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_checkpoint_adapter import C, write_raw
from test_scheduler_layout import modules as scheduler_modules

ROOT = Path(__file__).parents[3]


@pytest.fixture
def cache_modules(monkeypatch):
    scheduler_modules.__wrapped__(monkeypatch)
    envs = types.ModuleType("lightx2v.utils.envs")
    envs.GET_DTYPE = lambda: torch.bfloat16
    monkeypatch.setitem(sys.modules, envs.__name__, envs)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    monkeypatch.setitem(sys.modules, "lightx2v.models.networks.minimax_h3.checkpoint", C)
    load("lightx2v.models.networks.minimax_h3.infer.module_io", "lightx2v/models/networks/minimax_h3/infer/module_io.py")
    load("lightx2v.models.networks.minimax_h3.infer.pre_infer", "lightx2v/models/networks/minimax_h3/infer/pre_infer.py")
    load("lightx2v.models.networks.minimax_h3.adaln_cache_guide", "lightx2v/models/networks/minimax_h3/adaln_cache_guide.py")
    cache = load("lightx2v.models.networks.minimax_h3.adaln_cache", "lightx2v/models/networks/minimax_h3/adaln_cache.py")
    builder = load("h3_adaln_builder_under_test", "tools/cache_minimax_h3_adaln/builder.py")
    return cache, builder


@pytest.fixture
def checkpoint(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    config, tensors, weight_map = write_raw(raw)
    for name in tensors:
        if name != "rope.inv_freq":
            tensors[name] = tensors[name] * 0.001
    for shard in set(weight_map.values()):
        save_file({k: v for k, v in tensors.items() if weight_map[k] == shard}, raw / shard)
    config = C._native_config(config)
    config.update(dit_original_ckpt=str(raw), use_adaln_cache=True, adaln_cache_dir=str(tmp_path / "cache"), task="fl2av", infer_steps=4)
    return config, tensors, weight_map


def native_projections(tensors):
    # Independent expected mapping for the cache's four projection families.
    pairs = [("time_embedder.linear_1", "time_embedder.proj_in"), ("time_embedder.linear_2", "time_embedder.proj_out"), ("norm_out.linear", "final_layer.adaln_proj.linear")]
    pairs.extend((f"transformer_blocks.{i}.adaln_proj.linear", f"blocks.{i}.adaln_proj.linear") for i in range(2))
    return {target + suffix: tensors[source + suffix] for target, source in pairs for suffix in (".weight", ".bias")}


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_raw_generation_matches_native_and_loads(cache_modules, checkpoint, tmp_path, monkeypatch, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    cache, builder = cache_modules
    monkeypatch.setattr(builder, "AI_DEVICE", device)
    monkeypatch.setattr(builder, "torch_device_module", getattr(torch, device))
    config, tensors, _ = checkpoint
    expected = native_projections(tensors)
    reader = builder._CheckpointTensors(builder._checkpoint_files(config), config=config)
    for name, tensor in expected.items():
        actual = reader.get(name)
        assert actual.dtype == tensor.dtype
        assert torch.equal(actual, tensor)
    with pytest.raises(KeyError, match="missing"):
        reader.get("missing.weight")
    raw_path = builder.build_persistent_adaln_cache(config)
    raw_tables = load_file(raw_path / "adaln_cache.safetensors")
    spec = cache._build_spec(config)
    assert cache._validate_cache(raw_path, spec)
    tables, norm = cache.load_persistent_adaln_cache(config, device)
    for entry in spec["entries"]:
        key = tuple(cache._timesteps_from_bits(entry["timestep_bits"]).tolist())
        for i in range(spec["num_layers"]):
            tensor = tables[key][i]
            assert tensor.device.type == device and tensor.dtype == torch.bfloat16
            assert tuple(tensor.shape) == cache._expected_table_shape(spec, entry)
            assert torch.equal(tensor.cpu(), raw_tables[cache._block_key(i, entry)])
        assert norm[key].dtype == torch.bfloat16 and norm[key].device.type == device
        assert tuple(norm[key].shape) == cache._expected_norm_out_shape(spec, entry)
        assert torch.equal(norm[key].cpu(), raw_tables[cache._norm_out_key(entry)])
    native = tmp_path / "native"
    native.mkdir()
    save_file(expected, native / "model.safetensors")
    for indexed in (False, True):
        if indexed:
            (native / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in expected}}))
        native_config = dict(config, dit_original_ckpt=str(native), adaln_cache_dir=str(tmp_path / f"native-cache-{indexed}"))
        native_path = builder.build_persistent_adaln_cache(native_config)
        native_tables = load_file(native_path / "adaln_cache.safetensors")
        assert raw_tables.keys() == native_tables.keys()
        assert all(torch.equal(value, native_tables[name]) for name, value in raw_tables.items())
    with pytest.raises(FileExistsError):
        builder.build_persistent_adaln_cache(config)
    # Loader still rejects corrupted cache tensors.
    key = next(iter(raw_tables))
    raw_tables[key] = raw_tables[key].float()
    save_file(raw_tables, raw_path / "adaln_cache.safetensors")
    with pytest.raises(FileNotFoundError, match="AdaLN cache not found"):
        cache.load_persistent_adaln_cache(config, device)


@pytest.mark.parametrize("damage", ["shape", "dtype", "missing_key", "missing_shard", "runtime_config", "rope"])
def test_invalid_raw_checkpoint_rejected(cache_modules, checkpoint, damage):
    _, builder = cache_modules
    config, tensors, weight_map = checkpoint
    directory = Path(config["dit_original_ckpt"])
    name = "time_embedder.proj_in.weight"
    if damage == "runtime_config":
        config["hidden_size"] += 1
    elif damage == "missing_shard":
        (directory / weight_map[name]).unlink()
    elif damage == "missing_key":
        del weight_map[name]
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        if damage == "shape":
            tensors[name] = tensors[name][:1]
        elif damage == "dtype":
            tensors[name] = tensors[name].to(torch.bfloat16)
        else:
            name = "rope.inv_freq"
            tensors[name] = torch.zeros_like(tensors[name])
        shard = weight_map[name]
        save_file({k: v for k, v in tensors.items() if weight_map[k] == shard}, directory / shard)
    with pytest.raises((ValueError, FileNotFoundError)):
        builder.build_persistent_adaln_cache(config)
    assert not any(Path(config["adaln_cache_dir"]).rglob("manifest.json"))
    assert not any(Path(config["adaln_cache_dir"]).rglob(".building-*"))


@pytest.mark.parametrize("task", ["t2av", "i2av", "l2av", "fl2av"])
def test_base_profiles_share_raw_cache(cache_modules, checkpoint, task):
    cache, builder = cache_modules
    config, _, _ = checkpoint
    path = builder.build_persistent_adaln_cache(config)
    requested = dict(config, task=task)
    assert cache._cache_path(requested) == path
    assert cache._validate_cache(path, cache._build_spec(requested))
    assert cache.load_persistent_adaln_cache(requested, "cpu")
    ref_config = dict(config, task="ref2av")
    assert cache._cache_path(ref_config) != path
    assert not cache._validate_cache(path, cache._build_spec(ref_config))
    with pytest.raises(FileNotFoundError):
        cache.load_persistent_adaln_cache(ref_config, "cpu")


@pytest.mark.parametrize("damage", ["manifest", "profile", "shape", "missing_tensor"])
def test_cache_compatibility_validation(cache_modules, checkpoint, damage):
    cache, builder = cache_modules
    config, _, _ = checkpoint
    path = builder.build_persistent_adaln_cache(config)
    manifest_path = path / "manifest.json"
    if damage in {"manifest", "profile"}:
        manifest = json.loads(manifest_path.read_text())
        if damage == "manifest":
            manifest["video_flow_shift"] += 1
        else:
            manifest["entries"][0]["name"] = "ref2av_video_step_000"
        manifest_path.write_text(json.dumps(manifest))
    else:
        tensor_path = path / "adaln_cache.safetensors"
        tensors = load_file(tensor_path)
        name = next(iter(tensors))
        if damage == "shape":
            tensors[name] = tensors[name][:1].contiguous()
        else:
            del tensors[name]
        save_file(tensors, tensor_path)
    assert not cache._validate_cache(path, cache._build_spec(config))
    with pytest.raises(FileNotFoundError, match="AdaLN cache not found"):
        cache.load_persistent_adaln_cache(config, "cpu")


@pytest.mark.parametrize(
    "update, error",
    [
        ({"use_adaln_cache": False}, ValueError),
        ({"adaln_cache_dir": ""}, ValueError),
        ({"dummy_model": True}, NotImplementedError),
        ({"task": "unsupported"}, ValueError),
    ],
)
def test_invalid_cache_config(cache_modules, checkpoint, update, error):
    cache, builder = cache_modules
    config, _, _ = checkpoint
    config.update(update)
    with pytest.raises(error):
        builder.build_persistent_adaln_cache(config)
    with pytest.raises(error):
        cache.load_persistent_adaln_cache(config, "cpu")
