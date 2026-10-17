"""The cache rerun may stop after the final block's raw K/V projections."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightx2v.common.kvcache.rolling import RollingKVCachePool
from lightx2v.models.networks.wan.infer.edenlive.transformer_infer import EdenLiveARTransformerInfer
from test_cases.test_edenlive_cuda_graph import _Pre, _cpu_infer


class _Block:
    def __init__(self, index):
        self.index = index
        self.leaf = SimpleNamespace(has_lora_branch=False, has_diff=False)
        self.compute_phases = (SimpleNamespace(rope=SimpleNamespace(apply_single=lambda x, freqs: x + freqs)),)

    def named_weight_leaves(self):
        return [("weight", self.leaf)]


def _integration_infer(graph, skip, blocks, transposed=False):
    infer = _cpu_infer(graph)
    infer.config["skip_cache_update_suffix"] = skip
    infer.blocks_num = len(blocks)
    infer.compile_options = {}
    infer.transposed_v_cache = transposed
    infer._can_graph_stack = lambda pre: graph and pre.grid_sizes.tuple[0] == 3 and infer._visible_frames(3) == 18
    infer.calls = [0] * len(blocks)
    infer.kv_calls = []

    def raw_kv(block, x, pre):
        k = x.view(-1, 2, 4).clone() + block.index * 0.125
        return k, k + block.index + 1

    def compute(index, block):
        def full(x, pre, cross, cache):
            infer.calls[index] += 1
            k, v = raw_kv(block, x, pre)
            keys, values = cache
            keys[-x.shape[0] :].copy_(k + infer.query_freqs)
            if transposed:
                from lightx2v.models.networks.wan.infer.edenlive import transformer_infer as module

                module.update_transposed_v(v.unsqueeze(0), values, keys.shape[0] - x.shape[0])
                logical_v = values[..., _v_indices(keys.shape[0])].permute(0, 3, 2, 1).squeeze(0)
            else:
                values[-x.shape[0] :].copy_(v)
                logical_v = values
            x.add_((keys.mean(0) + logical_v.mean(0)).flatten() + cross[0] + pre.embed0.flatten()[0])
            return x, k, v

        return full

    functions = [compute(index, block) for index, block in enumerate(blocks)]
    infer.get_compiled_block = lambda index, block: functions[index]

    def kv_only(block, x, pre):
        infer.kv_calls.append(block.index)
        return raw_kv(block, x, pre)

    infer.infer_cache_update_kv = kv_only
    infer.get_compiled_cache_update_kv = lambda index, block: lambda x, pre: kv_only(block, x, pre)
    return infer


def _v_indices(length, start=0):
    position = torch.arange(start, start + length)
    return (position & ~15) | (position & 1) | ((position & 8) >> 2) | ((position & 2) << 1) | ((position & 4) << 1)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
def test_cache_only_suffix_preserves_all_layer_caches_next_chunks_and_requests(monkeypatch, graph, transposed):
    from lightx2v.models.networks.wan.infer.edenlive import transformer_infer as module

    def update_v(v, output, start):
        output[..., _v_indices(v.shape[1], start)] = v.permute(0, 3, 2, 1)

    def make_v(v, capacity):
        output = v.new_zeros(v.shape[0], v.shape[-1], v.shape[-2], (capacity + 127) // 128 * 128)
        update_v(v, output, 0)
        return output

    monkeypatch.setattr(module, "make_transposed_v", make_v)
    monkeypatch.setattr(module, "update_transposed_v", update_v)
    blocks = [_Block(index) for index in range(3)]
    reference = _integration_infer(graph, False, blocks, transposed)
    optimized = _integration_infer(graph, True, blocks, transposed)
    steady_replays = 0
    for request in range(2):
        for infer in (reference, optimized):
            pool = RollingKVCachePool(3, 18, 2, 4, torch.float32, torch.device("cpu"))
            pool._init_kv_buffer()
            infer.cache_pools[True] = pool
            infer.cross_kv_cache[True] = ((request, True), [(torch.tensor(float(request + layer + 1)),) for layer in range(3)])
            infer._graph_cross_generation += 1
        start = 0
        for frames in [3] * 10 + [1]:
            for commit in [False] * 4 + [True]:
                initial = torch.arange(frames * 8, dtype=torch.float32).view(frames, 8) / 32 + request
                outputs = []
                for infer in (reference, optimized):
                    infer.scheduler = SimpleNamespace(chunk_start=start, infer_condition=True, cache_update=commit)
                    key = (infer._visible_frames(frames), 1, 1, torch.device("cpu"))
                    if key not in infer.freqs_cache:
                        infer.freqs_cache[key] = torch.arange(key[0], dtype=torch.float32).view(-1, 1, 1) / 100
                    infer.key_freqs = infer.freqs_cache[key]
                    infer.query_freqs = infer.key_freqs[-frames:]
                    pre = _Pre(initial.clone(), torch.tensor([0.25]), SimpleNamespace(tuple=(frames, 1, 1)), {"text_segments": None})
                    infer.current_pre = pre
                    infer._cache_update_suffix_skipped = False
                    outputs.append(infer.infer_without_offload(blocks, pre.x, pre))
                    assert infer.current_pre is pre
                if not commit:
                    assert torch.equal(outputs[0], outputs[1])
                    assert not optimized._cache_update_suffix_skipped
                else:
                    assert optimized._cache_update_suffix_skipped
                    for layer in range(3):
                        a, b = reference.cache_pools[True], optimized.cache_pools[True]
                        assert a.get_local_end(layer) == b.get_local_end(layer)
                        assert a.get_global_end(layer) == b.get_global_end(layer)
                        assert torch.equal(a.k_cache(layer), b.k_cache(layer))
                        assert torch.equal(a.v_cache(layer), b.v_cache(layer))
                if frames == 3 and start >= 15:
                    steady_replays += 1
            start += frames
        reference.clear_request_state()
        optimized.clear_request_state()
    assert optimized.kv_calls == [2] * 22
    assert reference.calls[-1] - optimized.calls[-1] == 22
    assert reference.calls[:-1] == optimized.calls[:-1]
    if graph:
        assert optimized.cuda_graph_stack.capture_count == 1
        assert optimized.cuda_graph_stack.replay_count == steady_replays


@pytest.mark.parametrize(
    "override",
    [
        {"skip_cache_update_suffix": False},
        {"cpu_offload": True},
        {"lazy_load": True},
        {"tensor_parallel": True},
        {"enable_cfg": True},
        {"cfg_parallel": True},
        {"lora_configs": ["adapter"]},
        {"lora_dynamic_apply": True},
    ],
)
def test_config_guards_keep_full_suffix(override):
    blocks = [_Block(0), _Block(1)]
    infer = _integration_infer(False, True, blocks)
    pre = SimpleNamespace(adapter_args={"text_segments": None})
    assert infer._can_skip_cache_update_suffix(blocks, pre)
    infer.config.update(override)
    assert not infer._can_skip_cache_update_suffix(blocks, pre)


@pytest.mark.parametrize("attr,value", [("seq_parallel", True), ("has_post_adapter", True), ("has_image_context", True), ("reuse_history", False), ("compile_backend", "custom")])
def test_runtime_guards_keep_full_suffix(attr, value):
    blocks = [_Block(0), _Block(1)]
    infer = _integration_infer(False, True, blocks)
    setattr(infer, attr, value)
    assert not infer._can_skip_cache_update_suffix(blocks, SimpleNamespace(adapter_args={"text_segments": None}))


@pytest.mark.parametrize("attr,value", [("has_lora_branch", True), ("has_diff", True), ("weight_diff", None), ("bias_diff", None)])
def test_leaf_guards_keep_full_suffix(attr, value):
    blocks = [_Block(0), _Block(1)]
    setattr(blocks[-1].leaf, attr, value)
    infer = _integration_infer(False, True, blocks)
    assert not infer._can_skip_cache_update_suffix(blocks, SimpleNamespace(adapter_args={"text_segments": None}))


def test_segmented_prompts_and_partial_block_lists_keep_full_suffix():
    blocks = [_Block(0), _Block(1)]
    infer = _integration_infer(False, True, blocks)
    assert not infer._can_skip_cache_update_suffix(blocks, SimpleNamespace(adapter_args={"text_segments": [3]}))
    assert not infer._can_skip_cache_update_suffix(blocks[:1], SimpleNamespace(adapter_args={"text_segments": None}))


def test_unused_head_is_skipped_only_after_successful_commit_suffix(monkeypatch):
    from lightx2v.models.networks.wan.infer.edenlive.transformer_infer import EdenLiveTransformerInfer

    infer = _cpu_infer(False)
    x = torch.ones(3, 8)
    monkeypatch.setattr(EdenLiveTransformerInfer, "infer_non_blocks", lambda *args: x)
    infer._cache_update_suffix_skipped = False
    infer.scheduler.cache_update = True
    assert infer.infer_non_blocks(None, x, None) is x
    infer._cache_update_suffix_skipped = True
    assert infer.infer_non_blocks(None, x, None) is None
    infer.scheduler.cache_update = False
    assert infer.infer_non_blocks(None, x, None) is x
    infer.clear_request_state()
    assert not infer._cache_update_suffix_skipped
    assert not infer._skip_cache_update_suffix_active
    assert infer._skip_suffix_weight_key is None


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="Actual quantized projections require SM120")
@pytest.mark.parametrize("scheme", ["fp8", "int8"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("kv_group", [False, True])
def test_compiled_kv_only_matches_full_qkv_raw_values(scheme, packed, kv_group):
    from lightx2v.common.ops.mm.mm_weight import MMWeightWfp8channelAfp8channeldynamicSgl, MMWeightWint8ConvRot
    from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightSgl
    from lightx2v.models.networks.wan.infer.edenlive.convrot_projection import ConvRotProjectionGroup
    from lightx2v.models.networks.wan.infer.edenlive.packed_projection import FP8ProjectionGroup

    torch.manual_seed(42)
    torch.compiler.reset()
    infer = EdenLiveARTransformerInfer.__new__(EdenLiveARTransformerInfer)
    infer.num_heads, infer.head_dim = 2, 128
    infer.seq_parallel = False
    infer.modulate_func = lambda value, scale, shift: value * (1 + scale.squeeze()) + shift.squeeze()
    infer._self_attention = lambda phase, q, k, v, grid, cache: q.flatten(1)
    modules = []
    cls = MMWeightWfp8channelAfp8channeldynamicSgl if scheme == "fp8" else MMWeightWint8ConvRot
    group_cls = FP8ProjectionGroup if scheme == "fp8" else ConvRotProjectionGroup
    for name in ("q", "k", "v"):
        module = cls(name + ".weight", name + ".bias")
        module.infer_dtype = torch.bfloat16
        module.weight = (
            torch.randn(256, 256, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).t() if scheme == "fp8" else torch.randint(-127, 128, (256, 256), device="cuda", dtype=torch.int8)
        )
        module.weight_scale = torch.rand(256, 1, device="cuda", dtype=torch.float32) * 0.02 + 0.01
        module.bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        if scheme == "int8":
            module._convrot_groupsize = 256
        modules.append(module)
    norm = RMSWeightSgl("norm.weight", eps=1e-6)
    norm.weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    group = group_cls.from_modules(tuple(modules), packed=packed)
    assert group is not None
    phase = SimpleNamespace(
        modulation=SimpleNamespace(tensor=torch.randn(1, 6, 256, device="cuda", dtype=torch.bfloat16)),
        norm1=SimpleNamespace(apply=lambda value: F.layer_norm(value, (256,), eps=1e-6)),
        self_attn_q=modules[0],
        self_attn_k=modules[1],
        self_attn_v=modules[2],
        self_attn_norm_q=norm,
        self_attn_norm_k=norm,
        self_attn_o=SimpleNamespace(apply=lambda value: value),
        self_attn_projection_group=group,
    )
    if kv_group:
        phase.self_attn_cache_kv_projection_group = group_cls.from_modules(tuple(modules[1:]), packed=False)
        assert phase.self_attn_cache_kv_projection_group is not None
    block = SimpleNamespace(compute_phases=[phase])
    x = torch.randn(67, 256, device="cuda", dtype=torch.bfloat16)
    pre = _Pre(x, torch.randn(1, 6, 256, device="cuda", dtype=torch.bfloat16), SimpleNamespace(tuple=(67, 1, 1)), {})

    def full(value, pre):
        shift, scale, _, _, _, _ = pre.embed0.add(phase.modulation.tensor).chunk(6, dim=1)
        _, k, v = infer.infer_self_attn(phase, value, shift, scale, (67, 1, 1), self_attn_cache=(None, None))
        return k, v

    options = {"emulate_precision_casts": True}
    reference = torch.compile(full, fullgraph=True, options=options)
    optimized = torch.compile(lambda value, pre: infer.infer_cache_update_kv(block, value, pre), fullgraph=True, options=options)
    with torch.no_grad():
        expected, actual = reference(x, pre), optimized(x, pre)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    torch.compiler.reset()
