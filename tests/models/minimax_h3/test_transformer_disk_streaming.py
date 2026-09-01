import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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


def _load_module(module_name, relative_path):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def h3_modules(monkeypatch):
    for package_name in [
        "lightx2v",
        "lightx2v.common",
        "lightx2v.common.modules",
        "lightx2v.common.transformer_infer",
        "lightx2v.models",
        "lightx2v.models.networks",
        "lightx2v.models.networks.minimax_h3",
        "lightx2v.models.networks.minimax_h3.infer",
        "lightx2v.utils",
    ]:
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    weight_module = types.ModuleType("lightx2v.common.modules.weight_module")
    weight_module.WeightModule = _FakeWeightModule
    weight_module.WeightModuleList = _FakeWeightModuleList
    monkeypatch.setitem(sys.modules, "lightx2v.common.modules.weight_module", weight_module)

    registry = types.ModuleType("lightx2v.utils.registry_factory")
    registry.MM_WEIGHT_REGISTER = {"Default": _FakeLinear}
    registry.RMS_WEIGHT_REGISTER = {"torch_native": _FakeRMS}
    registry.ROPE_REGISTER = {"torch_real_rope": _FakeLeaf}
    registry.ATTN_WEIGHT_REGISTER = {"flash_attn3": _FakeLeaf}
    monkeypatch.setitem(sys.modules, "lightx2v.utils.registry_factory", registry)

    triton_ops = types.ModuleType("lightx2v.models.networks.minimax_h3.infer.triton_ops")
    triton_ops.MiniMaxH3TritonRope = _FakeLeaf
    monkeypatch.setitem(sys.modules, "lightx2v.models.networks.minimax_h3.infer.triton_ops", triton_ops)

    checkpoint_module = _load_module(
        "lightx2v.models.networks.minimax_h3.checkpoint",
        "lightx2v/models/networks/minimax_h3/checkpoint.py",
    )
    weights_module = _load_module(
        "minimax_h3_transformer_weights_under_test",
        "lightx2v/models/networks/minimax_h3/weights/transformer_weights.py",
    )

    base_infer = types.ModuleType("lightx2v.common.transformer_infer.transformer_infer")

    class _BaseTransformerInfer:
        def init_compile(self, config):
            self.use_compile = config.get("use_compile", False)

        def run_block(self, block_idx, block, *args):
            return self.infer_block(block, *args)

    base_infer.BaseTransformerInfer = _BaseTransformerInfer
    monkeypatch.setitem(sys.modules, "lightx2v.common.transformer_infer.transformer_infer", base_infer)

    envs = types.ModuleType("lightx2v.utils.envs")
    envs.GET_DTYPE = lambda: torch.float32
    monkeypatch.setitem(sys.modules, "lightx2v.utils.envs", envs)
    infer_module = _load_module(
        "minimax_h3_transformer_infer_under_test",
        "lightx2v/models/networks/minimax_h3/infer/transformer_infer.py",
    )

    return checkpoint_module, weights_module, infer_module


def _iter_base_attrs(module):
    if hasattr(module, "base_attrs"):
        yield from module.base_attrs
    for child in getattr(module, "_modules", {}).values():
        yield from _iter_base_attrs(child)


def _block_tensors_from_template(block, block_index, value):
    tensors = {}
    for name, _attr_name, transpose in _iter_base_attrs(block):
        actual_name = _resolve_block_name(name, block_index)
        if transpose:
            tensor = torch.full((2, 3), value, dtype=torch.float32)
        elif actual_name.endswith(".bias"):
            tensor = torch.full((2,), value, dtype=torch.float32)
        else:
            tensor = torch.full((3,), value, dtype=torch.float32)
        tensors[actual_name] = tensor
    return tensors


def _write_fake_official_checkpoint(tmp_path, weights_module, num_layers=2):
    template = weights_module.MiniMaxH3TransformerBlockWeights(0, {"num_layers": num_layers})
    block_tensors = {}
    for block_index in range(num_layers):
        block_tensors.update(_block_tensors_from_template(template, block_index, block_index + 1))

    names = sorted(block_tensors)
    shard_1_names = set(names[::2])
    shard_1 = {name: block_tensors[name] for name in names if name in shard_1_names}
    shard_2 = {name: block_tensors[name] for name in names if name not in shard_1_names}
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
    return block_tensors


def test_transformer_weights_stream_official_shards_one_block_at_a_time(tmp_path, monkeypatch, h3_modules):
    checkpoint_module, weights_module, _infer_module = h3_modules
    _write_fake_official_checkpoint(tmp_path, weights_module, num_layers=2)
    requested_names = []

    class SpyCheckpoint(checkpoint_module.MiniMaxH3ShardCheckpoint):
        def load_tensors(self, names, device="cpu"):
            requested_names.append(tuple(names))
            return super().load_tensors(names, device=device)

    monkeypatch.setattr(weights_module, "MiniMaxH3ShardCheckpoint", SpyCheckpoint)

    weights = weights_module.MiniMaxH3TransformerWeights(
        {
            "dit_disk_streaming": True,
            "dit_original_ckpt": str(tmp_path),
            "num_layers": 2,
            "dit_quantized": False,
            "tensor_parallel": False,
        }
    )

    assert weights.disk_streaming is True
    assert len(weights.blocks) == 0
    assert weights.checkpoint.block_indices == (0, 1)
    assert weights.streaming_block_indices == (0, 1)
    assert weights.streaming_block.attn.to_q.weight_cuda_buffer.device.type == "cpu"

    block0_names = weights.checkpoint.tensor_names_for_block(0)
    block1_names = weights.checkpoint.tensor_names_for_block(1)
    assert requested_names == [block0_names]

    block0 = weights.load_streaming_block(0)
    block_id = id(block0)
    buffer = block0.attn.to_q.weight_cuda_buffer
    buffer_id = id(buffer)
    assert torch.all(block0.attn.to_q.weight == 1)
    assert requested_names[-1] == block0_names

    block1 = weights.load_streaming_block(1)
    assert id(block1) == block_id
    assert id(block1.attn.to_q.weight_cuda_buffer) == buffer_id
    assert torch.all(block1.attn.to_q.weight == 2)
    assert block1.attn.to_q.weight.shape == (3, 2)
    assert requested_names[-1] == block1_names
    assert all(set(names) in [set(block0_names), set(block1_names)] for names in requested_names)


def test_transformer_disk_streaming_rejects_missing_block(tmp_path, h3_modules):
    _checkpoint_module, weights_module, _infer_module = h3_modules
    _write_fake_official_checkpoint(tmp_path, weights_module, num_layers=1)

    with pytest.raises(ValueError, match="checkpoint block indices mismatch"):
        weights_module.MiniMaxH3TransformerWeights(
            {
                "dit_disk_streaming": True,
                "dit_original_ckpt": str(tmp_path),
                "num_layers": 2,
                "dit_quantized": False,
                "tensor_parallel": False,
            }
        )


def test_transformer_infer_dispatches_to_disk_streaming(h3_modules):
    _checkpoint_module, _weights_module, infer_module = h3_modules
    infer = infer_module.MiniMaxH3TransformerInfer({"num_attention_heads": 1, "use_adaln_cache": False})
    loaded = []
    ran = []

    class FakeBlockWeights:
        disk_streaming = True
        checkpoint = SimpleNamespace(block_indices=(0, 1))

        def load_streaming_block(self, block_index):
            loaded.append(block_index)
            return f"block-{block_index}"

    def run_block(block_index, block, hidden_states, pre_infer_out):
        ran.append((block_index, block, hidden_states))
        return hidden_states + block_index + 1

    infer.run_block = run_block
    pre_infer_out = SimpleNamespace(hidden_states=0)

    assert infer.infer(FakeBlockWeights(), pre_infer_out) == 3
    assert loaded == [0, 1]
    assert ran == [(0, "block-0", 0), (1, "block-1", 1)]
