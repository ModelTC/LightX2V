import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).parents[3]


class _FakeWeightModule:
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def add_module(self, name, module):
        self._modules[name] = module
        setattr(self, name, module)

    def load(self, weight_dict):
        for module in self._modules.values():
            if hasattr(module, "load"):
                module.load(weight_dict)

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        for module in self._modules.values():
            if hasattr(module, "load_state_dict"):
                module.load_state_dict(destination, block_index, adapter_block_index)

    def to_cuda(self):
        pass

    def to_cpu(self):
        pass


class _FakeWeightModuleList(_FakeWeightModule):
    def __init__(self, modules=None):
        super().__init__()
        self._list = []
        if modules is not None:
            for module in modules:
                self.append(module)

    def append(self, module):
        self._list.append(module)
        self.add_module(str(len(self._list) - 1), module)

    def __getitem__(self, index):
        return self._list[index]

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)


def _resolve_block_name(name, block_index):
    parts = name.split(".", 2)
    if len(parts) == 3 and parts[0] == "transformer_blocks" and parts[1].isdigit():
        return f"transformer_blocks.{int(block_index)}.{parts[2]}"
    return name


class _FakeLinear:
    def __init__(self, weight_name, bias_name=None, create_cuda_buffer=False, **_kwargs):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.create_cuda_buffer = create_cuda_buffer
        self.base_attrs = [(weight_name, "weight", True)]
        if bias_name is not None:
            self.base_attrs.append((bias_name, "bias", False))

    def load(self, weight_dict):
        for name, attr_name, transpose in self.base_attrs:
            tensor = weight_dict[name]
            if transpose:
                tensor = tensor.t()
            if self.create_cuda_buffer:
                setattr(self, f"{attr_name}_cuda_buffer", tensor.clone())
            else:
                setattr(self, attr_name, tensor.clone())
                if tensor.device.type == "cpu":
                    setattr(self, f"pin_{attr_name}", tensor.clone())
                    del weight_dict[name]

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        for name, attr_name, _transpose in self.base_attrs:
            actual_name = _resolve_block_name(name, block_index)
            if actual_name in destination:
                buffer = getattr(self, f"{attr_name}_cuda_buffer")
                setattr(self, attr_name, buffer.copy_(destination[actual_name]))


class _FakeRMS:
    def __init__(self, weight_name, create_cuda_buffer=False, **_kwargs):
        self.weight_name = weight_name
        self.create_cuda_buffer = create_cuda_buffer
        self.base_attrs = [(weight_name, "weight", False)]

    def load(self, weight_dict):
        tensor = weight_dict[self.weight_name]
        if self.create_cuda_buffer:
            self.weight_cuda_buffer = tensor.clone()
        else:
            self.weight = tensor.clone()
            self.pin_weight = tensor.clone()
            if tensor.device.type == "cpu":
                del weight_dict[self.weight_name]

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        actual_name = _resolve_block_name(self.weight_name, block_index)
        if actual_name in destination:
            self.weight = self.weight_cuda_buffer.copy_(destination[actual_name])


class _FakeLeaf:
    base_attrs = ()

    def __init__(self, *_args, **_kwargs):
        pass

    def set_config(self, *_args, **_kwargs):
        pass


class _FakeBaseTransformerModel:
    load_ckpt_called = False
    init_weights_called = False

    def __init__(self, model_path, config, device, model_type=None, lora_path=None, lora_strength=1.0):
        self.device = torch.device(device)
        self.model_path = model_path
        self.config = config
        self.lora_path = lora_path
        self.lora_strength = lora_strength
        self.model_type = model_type
        self.cpu_offload = config.get("cpu_offload", False)
        self.offload_granularity = config.get("offload_granularity", "block")
        self.lazy_load = config.get("lazy_load", False)
        self.dit_quantized = config.get("dit_quantized", False)
        self.use_tp = config.get("tensor_parallel", False)
        self.tp_size = 1
        self.tp_rank = 0
        self.seq_p_group = None
        self.sensitive_layer = {}

    def _init_weights(self, weight_dict=None):
        _FakeBaseTransformerModel.init_weights_called = True
        if weight_dict is None:
            self._load_ckpt(False, {})

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        _FakeBaseTransformerModel.load_ckpt_called = True
        raise AssertionError("full checkpoint loading must not run in disk streaming")

    def _apply_weights(self, weight_dict=None):
        pass

    def _init_offload_manager(self):
        raise AssertionError("WeightAsyncStreamManager/offload manager must not initialize in disk streaming")


def _load_module(module_name, relative_path):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def h3_model_modules(monkeypatch):
    for package_name in [
        "lightx2v",
        "lightx2v.common",
        "lightx2v.common.modules",
        "lightx2v.models",
        "lightx2v.models.networks",
        "lightx2v.models.networks.minimax_h3",
        "lightx2v.models.networks.minimax_h3.infer",
        "lightx2v.models.networks.minimax_h3.weights",
        "lightx2v.utils",
    ]:
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    guide = _load_module("h3_adaln_cache_guide_under_test", "lightx2v/models/networks/minimax_h3/adaln_cache_guide.py")
    monkeypatch.setitem(sys.modules, "lightx2v.models.networks.minimax_h3.adaln_cache_guide", guide)

    # Cache IO is outside these isolated model/streaming tests.
    cache = types.ModuleType("lightx2v.models.networks.minimax_h3.adaln_cache")
    cache.validate_adaln_cache_config = lambda config: None
    cache.load_persistent_adaln_cache = lambda config, device: ({}, {})
    monkeypatch.setitem(sys.modules, cache.__name__, cache)

    weight_module = types.ModuleType("lightx2v.common.modules.weight_module")
    weight_module.WeightModule = _FakeWeightModule
    weight_module.WeightModuleList = _FakeWeightModuleList
    monkeypatch.setitem(sys.modules, "lightx2v.common.modules.weight_module", weight_module)

    registry = types.ModuleType("lightx2v.utils.registry_factory")
    registry.MM_WEIGHT_REGISTER = {"Default": _FakeLinear, "Default-ForceFp32": _FakeLinear}
    registry.RMS_WEIGHT_REGISTER = {"torch_native": _FakeRMS}
    registry.ROPE_REGISTER = {"torch_real_rope": _FakeLeaf}
    registry.ATTN_WEIGHT_REGISTER = {"flash_attn3": _FakeLeaf}
    monkeypatch.setitem(sys.modules, "lightx2v.utils.registry_factory", registry)

    envs = types.ModuleType("lightx2v.utils.envs")
    envs.GET_DTYPE = lambda: torch.bfloat16
    monkeypatch.setitem(sys.modules, "lightx2v.utils.envs", envs)

    base_model = types.ModuleType("lightx2v.models.networks.base_model")
    base_model.BaseTransformerModel = _FakeBaseTransformerModel
    monkeypatch.setitem(sys.modules, "lightx2v.models.networks.base_model", base_model)

    triton_ops = types.ModuleType("lightx2v.models.networks.minimax_h3.infer.triton_ops")
    triton_ops.MiniMaxH3TritonRope = _FakeLeaf
    monkeypatch.setitem(sys.modules, "lightx2v.models.networks.minimax_h3.infer.triton_ops", triton_ops)

    infer_module = types.ModuleType("lightx2v.models.networks.minimax_h3.infer.transformer_infer")

    class MiniMaxH3TransformerInfer:
        def __init__(self, config):
            self.config = config

    infer_module.MiniMaxH3TransformerInfer = MiniMaxH3TransformerInfer
    monkeypatch.setitem(sys.modules, "lightx2v.models.networks.minimax_h3.infer.transformer_infer", infer_module)

    offload_module = types.ModuleType("lightx2v.models.networks.minimax_h3.infer.offload")

    class MiniMaxH3OffloadTransformerInfer:
        pass

    offload_module.MiniMaxH3OffloadTransformerInfer = MiniMaxH3OffloadTransformerInfer
    monkeypatch.setitem(sys.modules, "lightx2v.models.networks.minimax_h3.infer.offload", offload_module)

    for module_name, class_name in [
        ("module_io", "MiniMaxH3SequenceParallelState"),
        ("post_infer", "MiniMaxH3PostInfer"),
        ("pre_infer", "MiniMaxH3PreInfer"),
    ]:
        module = types.ModuleType(f"lightx2v.models.networks.minimax_h3.infer.{module_name}")
        setattr(module, class_name, _FakeLeaf)
        monkeypatch.setitem(sys.modules, f"lightx2v.models.networks.minimax_h3.infer.{module_name}", module)

    tensor_parallel = types.ModuleType("lightx2v.models.networks.minimax_h3.weights.tensor_parallel")
    tensor_parallel.unwrap_tp_linear = lambda obj: obj
    monkeypatch.setitem(sys.modules, "lightx2v.models.networks.minimax_h3.weights.tensor_parallel", tensor_parallel)

    checkpoint_module = _load_module(
        "lightx2v.models.networks.minimax_h3.checkpoint",
        "lightx2v/models/networks/minimax_h3/checkpoint.py",
    )
    pre_module = _load_module(
        "minimax_h3_pre_weights_for_model_test",
        "lightx2v/models/networks/minimax_h3/weights/pre_weights.py",
    )
    post_module = _load_module(
        "minimax_h3_post_weights_for_model_test",
        "lightx2v/models/networks/minimax_h3/weights/post_weights.py",
    )
    transformer_module = _load_module(
        "minimax_h3_transformer_weights_for_model_test",
        "lightx2v/models/networks/minimax_h3/weights/transformer_weights.py",
    )

    weights_package = sys.modules["lightx2v.models.networks.minimax_h3.weights"]
    weights_package.MiniMaxH3PreWeights = pre_module.MiniMaxH3PreWeights
    weights_package.MiniMaxH3PostWeights = post_module.MiniMaxH3PostWeights
    weights_package.MiniMaxH3TransformerWeights = transformer_module.MiniMaxH3TransformerWeights

    model_module = _load_module(
        "minimax_h3_model_under_test",
        "lightx2v/models/networks/minimax_h3/model.py",
    )
    return checkpoint_module, pre_module, post_module, transformer_module, infer_module, offload_module, model_module


def _iter_base_attrs(module):
    if hasattr(module, "base_attrs"):
        yield from module.base_attrs
    for child in getattr(module, "_modules", {}).values():
        yield from _iter_base_attrs(child)


def _tensors_from_roots(roots, block_index=None, value=1):
    tensors = {}
    for root in roots:
        for name, _attr_name, transpose in _iter_base_attrs(root):
            actual_name = _resolve_block_name(name, block_index) if block_index is not None else name
            if transpose:
                tensor = torch.full((2, 3), value, dtype=torch.bfloat16)
            elif actual_name.endswith(".bias"):
                tensor = torch.full((2,), value, dtype=torch.bfloat16)
            else:
                tensor = torch.full((3,), value, dtype=torch.bfloat16)
            tensors[actual_name] = tensor
    return tensors


def _write_fake_checkpoint(tmp_path, pre_module, post_module, transformer_module, num_layers=2):
    config = {"num_layers": num_layers, "num_refiner_layers": 1}
    tensors = {}
    tensors.update(_tensors_from_roots([pre_module.MiniMaxH3PreWeights(config)], value=3))
    tensors.update(_tensors_from_roots([post_module.MiniMaxH3PostWeights(config)], value=4))
    block_template = transformer_module.MiniMaxH3TransformerBlockWeights(0, config)
    for block_index in range(num_layers):
        tensors.update(_tensors_from_roots([block_template], block_index=block_index, value=block_index + 1))

    names = sorted(tensors)
    shard_1_names = set(names[::2])
    shard_1 = {name: tensors[name] for name in names if name in shard_1_names}
    shard_2 = {name: tensors[name] for name in names if name not in shard_1_names}
    save_file(shard_1, tmp_path / "model-00001-of-00002.safetensors")
    save_file(shard_2, tmp_path / "model-00002-of-00002.safetensors")
    weight_map = {
        **{name: "model-00001-of-00002.safetensors" for name in shard_1},
        **{name: "model-00002-of-00002.safetensors" for name in shard_2},
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return tensors


def _config(tmp_path, **overrides):
    config = {
        "seq_parallel": False,
        "cfg_parallel": False,
        "enable_cfg": False,
        "cpu_offload": True,
        "use_adaln_cache": True,
        "adaln_cache_dir": str(tmp_path / "adaln-cache"),
        "offload_granularity": "block",
        "dit_disk_streaming": True,
        "dit_original_ckpt": str(tmp_path),
        "lazy_load": False,
        "dit_quantized": False,
        "dit_quant_scheme": "Default",
        "tensor_parallel": False,
        "num_layers": 2,
        "num_refiner_layers": 1,
    }
    config.update(overrides)
    return config


def test_non_disk_streaming_uses_base_init_weights(tmp_path, h3_model_modules):
    _checkpoint_module, _pre_module, _post_module, _transformer_module, _infer_module, _offload_module, model_module = h3_model_modules
    _FakeBaseTransformerModel.init_weights_called = False
    _FakeBaseTransformerModel.load_ckpt_called = False

    with pytest.raises(AssertionError, match="full checkpoint loading"):
        model_module.MiniMaxH3Model(str(tmp_path), _config(tmp_path, dit_disk_streaming=False), torch.device("cpu"))

    assert _FakeBaseTransformerModel.init_weights_called is True
    assert _FakeBaseTransformerModel.load_ckpt_called is True


def test_disk_streaming_model_init_skips_full_checkpoint_load(tmp_path, monkeypatch, h3_model_modules):
    checkpoint_module, pre_module, post_module, transformer_module, infer_module, offload_module, model_module = h3_model_modules
    _write_fake_checkpoint(tmp_path, pre_module, post_module, transformer_module, num_layers=2)
    calls = []

    class SpyCheckpoint(checkpoint_module.MiniMaxH3ShardCheckpoint):
        def load_tensors(self, names, device="cpu"):
            calls.append(tuple(names))
            return super().load_tensors(names, device=device)

    monkeypatch.setattr(transformer_module, "MiniMaxH3ShardCheckpoint", SpyCheckpoint)
    _FakeBaseTransformerModel.init_weights_called = False
    _FakeBaseTransformerModel.load_ckpt_called = False

    model = model_module.MiniMaxH3Model(str(tmp_path), _config(tmp_path), torch.device("cpu"))

    assert _FakeBaseTransformerModel.load_ckpt_called is False
    assert model.transformer_weights.disk_streaming is True
    assert len(model.transformer_weights.blocks) == 0
    assert model.transformer_weights.streaming_block is not None
    assert model.pre_weight.proj_in.pin_weight is not None
    assert model.post_weight.proj_out.pin_weight is not None
    assert model.transformer_infer_class is infer_module.MiniMaxH3TransformerInfer
    assert model.transformer_infer_class is not offload_module.MiniMaxH3OffloadTransformerInfer
    assert not hasattr(model.transformer_infer, "offload_manager")

    block0_names = model.transformer_weights.checkpoint.tensor_names_for_block(0)
    block1_names = model.transformer_weights.checkpoint.tensor_names_for_block(1)
    prepost_names = model_module._collect_declared_base_tensor_names(model.pre_weight, model.post_weight)
    assert calls == [block0_names, prepost_names]
    assert not any(set(call) == set(block1_names) for call in calls)


def test_disk_streaming_rejects_cpu_offload_false(tmp_path, h3_model_modules):
    _checkpoint_module, _pre_module, _post_module, _transformer_module, _infer_module, _offload_module, model_module = h3_model_modules

    with pytest.raises(ValueError, match="requires cpu_offload=true"):
        model_module.MiniMaxH3Model(str(tmp_path), _config(tmp_path, cpu_offload=False), torch.device("cpu"))


def test_disk_streaming_rejects_non_block_offload(tmp_path, h3_model_modules):
    _checkpoint_module, _pre_module, _post_module, _transformer_module, _infer_module, _offload_module, model_module = h3_model_modules

    with pytest.raises(ValueError, match="requires offload_granularity='block'"):
        model_module.MiniMaxH3Model(str(tmp_path), _config(tmp_path, offload_granularity="model"), torch.device("cpu"))


def test_cpu_offload_requires_persistent_cache(tmp_path, h3_model_modules):
    model_module = h3_model_modules[-1]
    with pytest.raises(ValueError, match="cpu_offload=true requires use_adaln_cache=true"):
        model_module.MiniMaxH3Model(str(tmp_path), _config(tmp_path, use_adaln_cache=False), torch.device("cpu"))
