import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
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


def _resolve_qwen_layer_name(name, layer_index):
    prefix = "model.language_model.layers."
    if not name.startswith(prefix):
        return name
    parts = name.split(".", 4)
    if len(parts) == 5 and parts[3].isdigit():
        return f"{prefix}{int(layer_index)}.{parts[4]}"
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
            actual_name = _resolve_qwen_layer_name(name, block_index)
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
        actual_name = _resolve_qwen_layer_name(self.weight_name, block_index)
        if actual_name in destination:
            self.weight = self.weight_cuda_buffer.copy_(destination[actual_name])


class _FakeEmbedding:
    def __init__(self, weight_name, *_args, **_kwargs):
        self.weight_name = weight_name
        self.weight = None

    def apply(self, input_indices):
        return F.embedding(input_indices, self.weight)


class _FakeAttention:
    def __init__(self, *_args, **_kwargs):
        pass

    def apply(self, q, *_args, **_kwargs):
        return q.reshape(q.shape[0], -1)


class _FakeLeaf:
    def __init__(self, *_args, **_kwargs):
        pass


class _FakeAttnWeightTemplate:
    def __init__(self, *_args, **_kwargs):
        pass


def _load_qwen_module(monkeypatch):
    for package_name in [
        "lightx2v",
        "lightx2v.common",
        "lightx2v.common.modules",
        "lightx2v.common.offload",
        "lightx2v.common.ops",
        "lightx2v.common.ops.attn",
        "lightx2v.common.ops.embedding",
        "lightx2v.common.ops.mm",
        "lightx2v.common.ops.norm",
        "lightx2v.models",
        "lightx2v.models.input_encoders",
        "lightx2v.models.input_encoders.hf",
        "lightx2v.models.input_encoders.hf.minimax_h3",
        "lightx2v.models.networks",
        "lightx2v.models.networks.minimax_h3",
        "lightx2v.models.networks.minimax_h3.weights",
        "lightx2v.utils",
        "lightx2v_platform",
        "lightx2v_platform.base",
    ]:
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    weight_module = types.ModuleType("lightx2v.common.modules.weight_module")
    weight_module.WeightModule = _FakeWeightModule
    weight_module.WeightModuleList = _FakeWeightModuleList
    monkeypatch.setitem(sys.modules, "lightx2v.common.modules.weight_module", weight_module)

    event_manager = types.ModuleType("lightx2v.common.offload.event_manager")

    class RaisingEventSlotWeightAsyncStreamManager:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("EventSlotWeightAsyncStreamManager must not be instantiated")

    event_manager.EventSlotWeightAsyncStreamManager = RaisingEventSlotWeightAsyncStreamManager
    monkeypatch.setitem(sys.modules, "lightx2v.common.offload.event_manager", event_manager)

    attn_template = types.ModuleType("lightx2v.common.ops.attn.template")
    attn_template.AttnWeightTemplate = _FakeAttnWeightTemplate
    monkeypatch.setitem(sys.modules, "lightx2v.common.ops.attn.template", attn_template)
    for module_name, class_name in [
        ("lightx2v.common.ops.attn.torch_sdpa", "TorchSDPAWeight"),
        ("lightx2v.common.ops.embedding.embedding_weight", "EmbeddingWeight"),
        ("lightx2v.common.ops.mm.mm_weight", "MMWeight"),
        ("lightx2v.common.ops.norm.rms_norm_weight", "RMSWeightFP32Qwen"),
    ]:
        module = types.ModuleType(module_name)
        setattr(module, class_name, _FakeLeaf)
        monkeypatch.setitem(sys.modules, module_name, module)

    vision = types.ModuleType("lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_vision")
    vision.MiniMaxH3Qwen3VLVisionTower = _FakeLeaf
    monkeypatch.setitem(sys.modules, "lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_vision", vision)

    packing = types.ModuleType("lightx2v.models.networks.minimax_h3.packing")
    packing.VIDEO_TAG = 2
    monkeypatch.setitem(sys.modules, "lightx2v.models.networks.minimax_h3.packing", packing)

    packing_ref = types.ModuleType("lightx2v.models.networks.minimax_h3.packing_ref2av")
    packing_ref.build_ref2av_presentation = lambda *_args, **_kwargs: None
    packing_ref.sample_reference_video_frames = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "lightx2v.models.networks.minimax_h3.packing_ref2av", packing_ref)

    tp = types.ModuleType("lightx2v.models.networks.minimax_h3.weights.tensor_parallel")
    tp.MiniMaxH3TensorParallelLinear = _FakeLinear
    tp.unwrap_tp_linear = lambda obj: obj
    monkeypatch.setitem(sys.modules, "lightx2v.models.networks.minimax_h3.weights.tensor_parallel", tp)

    envs = types.ModuleType("lightx2v.utils.envs")
    envs.GET_DTYPE = lambda: torch.bfloat16
    monkeypatch.setitem(sys.modules, "lightx2v.utils.envs", envs)

    registry = types.ModuleType("lightx2v.utils.registry_factory")
    registry.ATTN_WEIGHT_REGISTER = {"torch_sdpa": _FakeAttention}
    registry.EMBEDDING_WEIGHT_REGISTER = {"Default": _FakeEmbedding}
    registry.MM_WEIGHT_REGISTER = {"Default": _FakeLinear}
    registry.RMS_WEIGHT_REGISTER = {"fp32_variance_qwen": _FakeRMS}
    monkeypatch.setitem(sys.modules, "lightx2v.utils.registry_factory", registry)

    global_var = types.ModuleType("lightx2v_platform.base.global_var")
    global_var.AI_DEVICE = "cpu"
    monkeypatch.setitem(sys.modules, "lightx2v_platform.base.global_var", global_var)

    spec = importlib.util.spec_from_file_location(
        "qwen3vl_disk_streaming_under_test",
        REPO_ROOT / "lightx2v/models/input_encoders/hf/minimax_h3/qwen3vl.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.AI_DEVICE = "cpu"
    module.torch_device_module = SimpleNamespace(synchronize=lambda: None)
    return module


@pytest.fixture()
def qwen_module(monkeypatch):
    return _load_qwen_module(monkeypatch)


def _tiny_text_config():
    return {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }


def _make_backbone(qwen_module):
    return qwen_module._Qwen3VLTextBackboneWeights(
        {"text_encoder_disk_streaming": True},
        _tiny_text_config(),
        num_layers=2,
        attn_type="torch_sdpa",
        block_offload=False,
        disk_streaming=True,
    )


def _tensors_for_names(names, value):
    tensors = {}
    for name in names:
        if name.endswith("embed_tokens.weight"):
            tensors[name] = torch.full((32, 8), value, dtype=torch.bfloat16)
        elif name.endswith(".weight") and any(proj in name for proj in (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.", ".gate_proj.", ".up_proj.", ".down_proj.")):
            tensors[name] = torch.full((2, 3), value, dtype=torch.bfloat16)
        else:
            tensors[name] = torch.full((3,), value, dtype=torch.bfloat16)
    return tensors


def _write_fake_checkpoint(tmp_path, backbone, qwen_module):
    template_layer = backbone.layers[0]
    embedding_name = backbone.embed_tokens.weight_name
    layer0_names = qwen_module._Qwen3VLTextBackboneWeights._layer_tensor_names(template_layer, 0)
    layer1_names = qwen_module._Qwen3VLTextBackboneWeights._layer_tensor_names(template_layer, 1)
    tensors = {}
    tensors.update(_tensors_for_names((embedding_name,), 5))
    tensors.update(_tensors_for_names(layer0_names, 1))
    tensors.update(_tensors_for_names(layer1_names, 2))

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
    return embedding_name, layer0_names, layer1_names, weight_map


def test_qwen3vl_disk_streaming_reuses_one_layer_buffer(tmp_path, monkeypatch, qwen_module):
    backbone = _make_backbone(qwen_module)
    embedding_name, layer0_names, layer1_names, weight_map = _write_fake_checkpoint(tmp_path, backbone, qwen_module)
    calls = []
    original_loader = qwen_module._load_selected_checkpoint_tensors

    def spy_loader(text_encoder_path, weight_map_arg, names):
        calls.append(tuple(names))
        return original_loader(text_encoder_path, weight_map_arg, names)

    monkeypatch.setattr(qwen_module, "_load_selected_checkpoint_tensors", spy_loader)

    backbone.init_disk_streaming(tmp_path, weight_map)
    assert calls == [layer0_names]
    assert backbone.offload_manager is None
    layer0 = backbone.load_streaming_layer(0)
    layer_id = id(layer0)
    buffer = layer0.self_attn.q_proj.weight_cuda_buffer
    buffer_id = id(buffer)
    assert torch.all(layer0.self_attn.q_proj.weight == 1)

    layer1 = backbone.load_streaming_layer(1)
    assert id(layer1) == layer_id
    assert id(layer1.self_attn.q_proj.weight_cuda_buffer) == buffer_id
    assert torch.all(layer1.self_attn.q_proj.weight == 2)

    hidden_states = backbone._forward_streaming_embedding(torch.tensor([0, 1], dtype=torch.long))
    assert hidden_states.dtype == torch.bfloat16
    assert tuple(hidden_states.shape) == (2, 8)
    assert getattr(backbone.embed_tokens, "weight", None) is None
    assert getattr(backbone.embed_tokens, "pin_weight", None) is None

    assert calls == [layer0_names, layer0_names, layer1_names, (embedding_name,)]
    assert all(set(call) in [set(layer0_names), set(layer1_names), {embedding_name}] for call in calls)


def test_qwen3vl_disk_streaming_rejects_vision_inputs(tmp_path, qwen_module):
    backbone = _make_backbone(qwen_module)
    _embedding_name, _layer0_names, _layer1_names, weight_map = _write_fake_checkpoint(tmp_path, backbone, qwen_module)
    backbone.init_disk_streaming(tmp_path, weight_map)

    with pytest.raises(NotImplementedError, match="text-only t2av"):
        backbone.forward(
            torch.tensor([0], dtype=torch.long),
            vision_mask=torch.tensor([True]),
            vision_embeds=torch.zeros((1, 8), dtype=torch.bfloat16),
        )


def test_qwen3vl_release_disk_streaming_buffer_clears_device_refs(tmp_path, qwen_module):
    backbone = _make_backbone(qwen_module)
    _embedding_name, _layer0_names, _layer1_names, weight_map = _write_fake_checkpoint(tmp_path, backbone, qwen_module)
    backbone.init_disk_streaming(tmp_path, weight_map)
    old_layer = backbone.streaming_layer
    assert old_layer.self_attn.q_proj.weight_cuda_buffer is not None

    backbone.release_disk_streaming_buffer()

    assert backbone.streaming_layer is None
    assert old_layer.self_attn.q_proj.weight is None
    assert old_layer.self_attn.q_proj.weight_cuda_buffer is None
    assert old_layer.input_layernorm.weight is None
    assert old_layer.input_layernorm.weight_cuda_buffer is None


@pytest.mark.parametrize("mode", ["disk", "resident", "block"])
@pytest.mark.parametrize("release_buffers", [True, False])
@pytest.mark.parametrize("fail_forward", [False, True])
def test_public_infer_offload_lifecycle(tmp_path, monkeypatch, qwen_module, mode, release_buffers, fail_forward):
    from unittest.mock import Mock

    # Exercise the real constructor's mode flags without needing an accelerator.
    monkeypatch.setattr(qwen_module, "AI_DEVICE", "mps")
    encoder = qwen_module.MiniMaxH3Qwen3VLTextEncoder(
        {
            "task": "t2av",
            "text_encoder_cpu_offload": True,
            "text_encoder_offload_granularity": "model" if mode == "resident" else "block",
            "text_encoder_disk_streaming": mode == "disk",
            "text_encoder_release_block_offload_buffers": release_buffers,
            "text_encoder_load_on_init": False,
        }
    )
    monkeypatch.setattr(qwen_module, "AI_DEVICE", "cpu")
    monkeypatch.setattr(qwen_module, "MINIMAX_H3_TEXT_HIDDEN_SIZE", 8)
    backbone = qwen_module._Qwen3VLTextBackboneWeights(
        encoder.config,
        _tiny_text_config(),
        num_layers=2,
        block_offload=encoder.block_offload,
        disk_streaming=encoder.disk_streaming,
    )
    encoder.text_encoder = backbone
    encoder.tokenizer = Mock(return_value={"input_ids": [0, 1, 2]})
    assert (encoder.cpu_offload, encoder.block_offload, encoder.disk_streaming) == (True, mode == "block", mode == "disk")
    events = []

    def layer_forward(self, hidden_states, position_embeddings):
        events.append("layer")
        if fail_forward and events.count("layer") == 2:
            raise RuntimeError("injected forward failure")
        return hidden_states + 1

    monkeypatch.setattr(qwen_module._Qwen3VLDecoderLayerWeights, "forward", layer_forward)
    if mode == "disk":
        _, _, _, weight_map = _write_fake_checkpoint(tmp_path, backbone, qwen_module)
        backbone.init_disk_streaming(tmp_path, weight_map)
    else:
        backbone.embed_tokens.weight = torch.full((32, 8), 5, dtype=torch.bfloat16)

    def block_forward(input_ids, *args):
        events.append("block")
        if fail_forward:
            raise RuntimeError("injected forward failure")
        return torch.full((input_ids.numel(), 8), 7, dtype=torch.bfloat16)

    monkeypatch.setattr(backbone, "_forward_with_block_offload", Mock(side_effect=block_forward))
    for name in ("to_cuda", "to_cpu", "init_block_offload", "release_block_offload_buffers"):
        monkeypatch.setattr(backbone, name, Mock())
    for name in ("_forward_streaming_embedding", "load_streaming_layer", "release_disk_streaming_buffer"):
        monkeypatch.setattr(backbone, name, Mock(wraps=getattr(backbone, name)))

    prompt = "A cat walking on the grass."
    if fail_forward:
        with pytest.raises(RuntimeError, match="injected forward failure"):
            encoder.infer(prompt)
    else:
        result = encoder.infer(prompt)
        assert set(result) == {"prompt_embeds", "text_token_tags"}
        embeds, tags = result["prompt_embeds"], result["text_token_tags"]
        assert isinstance(embeds, torch.Tensor) and isinstance(tags, torch.Tensor)
        assert embeds.shape == (3, 8) and embeds.dtype == torch.bfloat16
        assert embeds.device.type == "cpu" and embeds.is_contiguous()
        assert torch.isfinite(embeds).all() and torch.all(embeds == 7)
        assert tags.shape == (3,) and tags.dtype == torch.long
        assert tags.device == embeds.device
        assert torch.all(tags == qwen_module.MINIMAX_H3_TEXT_TAG)

    encoder.tokenizer.assert_called_once_with(prompt, add_special_tokens=False)
    assert backbone.to_cuda.call_count == int(mode == "resident")
    assert backbone.to_cpu.call_count == int(mode == "resident")
    assert backbone.init_block_offload.call_count == int(mode == "block")
    assert backbone.release_block_offload_buffers.call_count == int(mode == "block" and release_buffers)
    assert backbone.release_disk_streaming_buffer.call_count == int(mode == "disk" and release_buffers)
    assert backbone._forward_with_block_offload.call_count == int(mode == "block")
    if mode == "disk":
        backbone._forward_streaming_embedding.assert_called_once()
        assert [call.args[0] for call in backbone.load_streaming_layer.call_args_list] == [0, 1]
        assert (backbone.streaming_layer is None) == release_buffers
        assert backbone.embed_tokens.weight is None
        assert not hasattr(backbone.embed_tokens, "pin_weight")
    else:
        backbone._forward_streaming_embedding.assert_not_called()
        backbone.load_streaming_layer.assert_not_called()
