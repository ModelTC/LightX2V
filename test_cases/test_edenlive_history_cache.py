from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightx2v.common.kvcache.rolling import RollingKVCachePool
from lightx2v.models.networks.wan.infer.edenlive.transformer_infer import EdenLiveARTransformerInfer, EdenLiveTransformerInfer


def make_infer(reuse):
    infer = EdenLiveARTransformerInfer.__new__(EdenLiveARTransformerInfer)
    infer.config = {"ar_config": {"local_attn_size": 18, "num_frame_per_chunk": 3, "sink_size": 6}}
    infer.history_cache = {}
    infer.reuse_history = reuse
    infer.transposed_v_cache = False
    infer.num_heads, infer.head_dim = 2, 4
    infer.block_idx = 0
    infer.cache_pools = {}
    for branch in [True, False]:
        pool = RollingKVCachePool(1, 18, 2, 4, torch.float32, torch.device("cpu"))
        pool._init_kv_buffer()
        infer.cache_pools[branch] = pool
    return infer


def run_attention(infer, phase, q, k, v, grid_sizes, compute=None):
    cache = infer.prepare_self_attn_cache(phase, k, grid_sizes, 0)
    output = (compute or infer._self_attention)(phase, q, k, v, grid_sizes, cache)
    infer.commit_kv(infer.cache_pools[infer.scheduler.infer_condition], 0, k, v, grid_sizes)
    return output


def test_history_reuse_matches_ring_cache_through_eviction_and_cfg(monkeypatch):
    def attention(self, phase, q, k, v):
        output = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return output.transpose(0, 1).flatten(1)

    class Rope:
        def apply_single(self, x, freqs):
            values = torch.view_as_complex(x.reshape(*x.shape[:-1], 2, 2))
            return torch.view_as_real(values * freqs).flatten(-2)

        def apply(self, q, k, freqs):
            return self.apply_single(q, freqs), self.apply_single(k, freqs)

    monkeypatch.setattr(EdenLiveTransformerInfer, "_attention", attention)
    phase = SimpleNamespace(rope=Rope())
    torch.manual_seed(0)
    reference, cached = make_infer(False), make_infer(True)
    start = 0
    pointers = {}
    for frames in [3] * 12 + [1]:
        for commit in [False] * 4 + [True]:
            for branch in [True, False]:
                q, k, v = [torch.randn(frames, 2, 4) for _ in range(3)]
                for infer in [reference, cached]:
                    infer.scheduler = SimpleNamespace(chunk_start=start, infer_condition=branch, cache_update=commit)
                    length = min(start, 15) + frames
                    angle = torch.arange(length).view(-1, 1, 1) * torch.tensor([0.1, 0.01])
                    infer.key_freqs = torch.polar(torch.ones_like(angle), angle)
                    infer.query_freqs = infer.key_freqs[-frames:]
                expected = run_attention(reference, phase, q.clone(), k.clone(), v.clone(), (frames, 1, 1))
                with monkeypatch.context() as patch:
                    patch.setattr(torch, "cat", lambda *a, **kw: pytest.fail("persistent history must not materialize with cat"))
                    actual = run_attention(cached, phase, q.clone(), k.clone(), v.clone(), (frames, 1, 1))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                _, key_storage, value_storage = cached.history_cache[branch, 0]
                current_pointers = (key_storage.data_ptr(), value_storage.data_ptr())
                assert current_pointers == pointers.setdefault(branch, current_pointers)
                if commit:
                    a, b = reference.cache_pools[branch], cached.cache_pools[branch]
                    end = a.get_local_end(0)
                    assert end == b.get_local_end(0)
                    torch.testing.assert_close(a.k_cache(0, 0, end), b.k_cache(0, 0, end), rtol=0, atol=0)
                    torch.testing.assert_close(a.v_cache(0, 0, end), b.v_cache(0, 0, end), rtol=0, atol=0)
        start += frames
    cached.clear_request_state()
    assert cached.history_cache == {} and cached.cache_pools == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "backend,compile_attention,reuse_history",
    [
        ("flash_attn3", False, True),
        ("sage_attn2", False, True),
        ("sage_attn2_sm90_fused", False, True),
        ("sage_attn2", True, True),
        ("sage_attn2_sm90_fused", True, True),
        ("sage_attn2", True, False),
        ("sage_attn2_sm90_fused", True, False),
    ],
)
def test_persistent_gpu_workspace_matches_materialized_history(backend, compile_attention, reuse_history):
    from lightx2v.common.ops.rope.flashinfer_rope import FlashInferRope
    from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

    if backend == "sage_attn2_sm90_fused" and torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    reference, cached = make_infer(False), make_infer(reuse_history)
    for infer in [reference, cached]:
        infer.num_heads, infer.head_dim = 2, 128
        infer.transposed_v_cache = backend == "sage_attn2_sm90_fused"
        for branch in [True, False]:
            pool = RollingKVCachePool(1, 18 * 15, 2, 128, torch.bfloat16, torch.device("cuda"))
            pool._init_kv_buffer()
            infer.cache_pools[branch] = pool
    phase = SimpleNamespace(rope=FlashInferRope(), self_attn_1=ATTN_WEIGHT_REGISTER[backend]())
    if compile_attention:
        torch.compiler.reset()
    compute = torch.compile(cached._self_attention, fullgraph=True, dynamic=None, options={"emulate_precision_casts": True}) if compile_attention else cached._self_attention
    pointers = {}
    torch.manual_seed(7)
    start = 0
    with torch.no_grad():
        for frames in [3] * 10 + [2]:
            for commit in [False] * 4 + [True]:
                for branch in [True, False]:
                    q, k, v = [torch.randn(frames * 15, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
                    for infer in [reference, cached]:
                        infer.scheduler = SimpleNamespace(chunk_start=start, infer_condition=branch, cache_update=commit)
                        length = (min(start, 15) + frames) * 15
                        angle = torch.arange(length, device="cuda").view(-1, 1, 1) * torch.linspace(0.001, 0.1, 64, device="cuda")
                        infer.key_freqs = phase.rope.prepare_freqs(torch.polar(torch.ones_like(angle), angle))
                        infer.query_freqs = infer.key_freqs[-frames * 15 :]
                    expected = run_attention(reference, phase, q.clone(), k.clone(), v.clone(), (frames, 3, 5))
                    raw_k, raw_v = k.clone(), v.clone()
                    actual = run_attention(cached, phase, q.clone(), raw_k, raw_v, (frames, 3, 5), compute)
                    torch.testing.assert_close(raw_k, k, rtol=0, atol=0)
                    torch.testing.assert_close(raw_v, v, rtol=0, atol=0)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"start={start}, frames={frames}, commit={commit}, branch={branch}")
                    if reuse_history:
                        _, keys, values = cached.history_cache[branch, 0]
                        current = (keys.data_ptr(), values.data_ptr())
                        assert current == pointers.setdefault(branch, current)
                    if commit:
                        a, b = reference.cache_pools[branch], cached.cache_pools[branch]
                        torch.testing.assert_close(a.k_cache(0), b.k_cache(0), rtol=0, atol=0)
                        torch.testing.assert_close(a.v_cache(0), b.v_cache(0), rtol=0, atol=0)
            start += frames


def test_cross_kv_refreshes_for_prompt_and_guidance_branch(monkeypatch):
    from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer

    infer = EdenLiveARTransformerInfer.__new__(EdenLiveARTransformerInfer)
    infer.reuse_cross_kv = True
    infer.cross_kv_cache = {}
    infer.has_image_context = False
    infer.scheduler = SimpleNamespace(infer_condition=True)
    phases = [object(), object()]
    weights = SimpleNamespace(blocks=[SimpleNamespace(compute_phases=[None, phase]) for phase in phases])
    calls = []

    def compute(phase, context):
        calls.append(phase)
        return context.clone(), context.clone()

    infer.cross_kv_compute = compute
    pre = SimpleNamespace(context=torch.ones(4, 8), adapter_args={"context_id": 1})
    infer.prepare_cross_kv(weights, pre)
    infer.prepare_cross_kv(weights, pre)
    assert len(calls) == 2
    infer.scheduler.infer_condition = False
    infer.prepare_cross_kv(weights, pre)
    assert len(calls) == 4
    infer.scheduler.infer_condition = True
    pre.context = torch.zeros(4, 8)
    pre.adapter_args["context_id"] = 2
    infer.prepare_cross_kv(weights, pre)
    assert len(calls) == 6
    torch.testing.assert_close(infer.cross_kv_cache[True][1][0][0], pre.context)
    torch.testing.assert_close(infer.cross_kv_cache[False][1][0][0], torch.ones_like(pre.context))
    cache = object()
    infer.config = {"ar_config": {}}
    infer.cache_pools = {True: object()}
    infer.prepare_self_attn_cache = lambda *args: cache
    infer.commit_kv = lambda *args: None
    pre.grid_sizes = SimpleNamespace(tuple=(1, 1, 1))
    captured = []

    def run_block(self, index, block, *args):
        captured.append(args)
        return args[0], "raw_k", "raw_v"

    monkeypatch.setattr(BaseTransformerInfer, "run_block", run_block)
    assert infer.run_block(1, weights.blocks[1], "x", pre) == "x"
    assert captured[0][-2] is infer.cross_kv_cache[True][1][1]
    assert captured[0][-1] is cache
