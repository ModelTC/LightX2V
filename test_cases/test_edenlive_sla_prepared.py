import pytest
import torch
import torch.nn.functional as F

from lightx2v.common.ops.attn.general_sparse_attn import GeneralSparseAttnWeight
from lightx2v.common.ops.attn.sparse_mask_generator import SlaMaskGenerator
from lightx2v.common.ops.attn.sparse_operator import SlaTritonOperator, SparseSageAttentionV2Operator
from lightx2v.models.networks.wan.infer.edenlive.cuda_graph import EdenLiveCUDAGraphStack

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _attention(generator, operator):
    attention = GeneralSparseAttnWeight.__new__(GeneralSparseAttnWeight)
    attention.mask_generator = generator
    attention.operator = operator
    return attention


@pytest.mark.parametrize("ratio", [-1, -0.01, 1, 1.1, float("nan"), float("inf")])
def test_sla_rejects_invalid_sparsity(ratio):
    with pytest.raises(ValueError, match="sparsity_ratio"):
        SlaMaskGenerator(sparse_setting={"sparsity_ratio": ratio})


def test_general_non_sla_retains_reorganization_and_restore():
    calls = []

    class Generator:
        def __call__(self, q, k):
            calls.append("mask")
            return "mask"

        def reorg(self, q, k, v):
            calls.append("reorg")
            return q + 1, k + 1, v + 1

        def restore(self, value):
            calls.append("restore")
            return value + 2

    def operator(q, k, v, mask, **kwargs):
        calls.append("operator")
        assert mask == "mask"
        return q + k + v

    attention = _attention(Generator(), operator)
    result = attention.apply(torch.ones(1), torch.ones(1), torch.ones(1))
    torch.testing.assert_close(result, torch.full((1,), 8.0))
    assert calls == ["mask", "reorg", "operator", "restore"]


def test_prepared_sage_reuses_router_qk_and_does_not_regenerate_topk(monkeypatch):
    generator = SlaMaskGenerator(128, 64, {"sparsity_ratio": 0.8})
    seen = {}

    def router(q, k, **kwargs):
        seen["q"], seen["k"] = q, k
        return torch.ones(1, 2, 1, 1, dtype=torch.int8), torch.zeros(1, 2, 1, 1, dtype=torch.int64), 1

    generator._get_block_map = router
    operator = SparseSageAttentionV2Operator.__new__(SparseSageAttentionV2Operator)
    operator.q_block_size, operator.k_block_size, operator.arch = 128, 64, "sm120"
    operator._block_map_incremental_lut_triton = lambda mask: (mask.int(), torch.ones(mask.shape[:-1], dtype=torch.int32))

    def compute(q, k, v, lut, valid, block_q, block_k, arch):
        assert q is seen["q"] and k is seen["k"]
        assert (block_q, block_k, arch) == (128, 64, "sm120")
        assert v.shape == (1, 2, 7, 4)
        return q + 1

    operator._sage2_prepared = compute
    monkeypatch.setattr(torch.Tensor, "item", lambda self, *a, **kw: pytest.fail("prepared SLA must not synchronize through item"))
    output = _attention(generator, operator).apply(torch.zeros(3, 2, 4), torch.zeros(7, 2, 4), torch.zeros(7, 2, 4))
    assert output.shape == (3, 8)


@cuda_required
@pytest.mark.parametrize("query_tokens,key_tokens,dim", [(1, 1, 64), (65, 193, 64), (129, 777, 128)])
def test_prepared_triton_sla_matches_masked_dense_with_unequal_lengths(query_tokens, key_tokens, dim, monkeypatch):
    torch.manual_seed(217)
    q = torch.randn(query_tokens, 2, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(key_tokens, 2, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    generator = SlaMaskGenerator(128, 128, {"sparsity_ratio": 0.8})
    operator = SlaTritonOperator()
    attention = _attention(generator, operator)
    mask, lut, topk = generator.prepare(q, k)
    key_blocks = (key_tokens + 127) // 128
    assert topk == max(1, int((1 - 0.8) * key_blocks))
    assert torch.all(mask.sum(-1) == topk)
    dense_mask = mask.bool().repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)[..., :query_tokens, :key_tokens]
    expected = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), attn_mask=dense_mask)
    expected = expected.squeeze(0).transpose(0, 1).reshape(query_tokens, -1)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", lambda self, *a, **kw: pytest.fail("prepared SLA must not synchronize through item"))
        actual = attention.apply(q, k, v, max_seqlen_q=query_tokens, max_seqlen_kv=key_tokens)
    assert actual.shape == (query_tokens, 2 * dim)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.015)


@cuda_required
def test_sla_full_window_keeps_requested_80_percent_block_sparsity():
    torch.manual_seed(218)
    q = torch.randn(2640, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(15840, 1, 128, device="cuda", dtype=torch.bfloat16)
    generator = SlaMaskGenerator(128, 128, {"sparsity_ratio": 0.8})
    mask, lut, topk = generator.prepare(q, k)
    assert mask.shape == (1, 1, 21, 124)
    assert lut.shape == (1, 1, 21, 24) and topk == 24
    assert torch.all(mask.sum(-1) == 24)
    assert 1 - topk / mask.shape[-1] >= 0.8


@cuda_required
def test_prepared_sla_compiles_fullgraph_without_scalar_sync():
    generator = SlaMaskGenerator(128, 128, {"sparsity_ratio": 0.8})
    attention = _attention(generator, SlaTritonOperator())
    compute = torch.compile(attention.apply, fullgraph=True, dynamic=False)
    torch.manual_seed(219)
    q = torch.randn(65, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(257, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    with torch.no_grad():
        expected = attention.apply(q, k, v)
        actual = compute(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.015)


@cuda_required
@pytest.mark.parametrize("compiled", [False, True])
def test_prepared_sage2_matches_existing_mask_operator_without_host_sync(compiled, monkeypatch):
    pytest.importorskip("spas_sage_attn")
    operator = SparseSageAttentionV2Operator()
    generator = SlaMaskGenerator(operator.q_block_size, operator.k_block_size, {"sparsity_ratio": 0.8})
    attention = _attention(generator, operator)
    compute = torch.compile(attention.apply, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}) if compiled else attention.apply
    torch.manual_seed(220)
    q = torch.randn(65, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(257, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    with torch.no_grad():
        mask = generator(q, k)
        expected = operator(q, k, v, mask, max_seqlen_q=q.shape[0], max_seqlen_kv=k.shape[0])
        if compiled:
            # Compilation may inspect constant tensors; forbid scalar reads
            # during the actual inference invocation after compilation.
            compute(q, k, v)
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "item", lambda self, *a, **kw: pytest.fail("prepared SLA must not synchronize through item"))
            actual = compute(q, k, v)
    assert actual.shape == (65, 256)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@cuda_required
@pytest.mark.parametrize("backend", ["triton", "sage2"])
def test_whole_sla_custom_op_growing_history_compiles_once(backend):
    from torch._dynamo.testing import CompileCounterWithBackend

    if backend == "sage2":
        pytest.importorskip("spas_sage_attn")
    operator = SlaTritonOperator() if backend == "triton" else SparseSageAttentionV2Operator()
    generator = SlaMaskGenerator(operator.q_block_size, operator.k_block_size, {"sparsity_ratio": 0.8})
    attention = _attention(generator, operator)
    counter = CompileCounterWithBackend("inductor")
    compute = torch.compile(attention.apply, fullgraph=True, dynamic=True, backend=counter)
    torch.manual_seed(221)
    q = torch.randn(7, 2, 128, device="cuda", dtype=torch.bfloat16)
    initial_frames = None
    with torch.no_grad():
        for length in [129, 257, 513, 1025]:
            k = torch.randn(length, 2, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            expected = attention.apply(q, k, v)
            actual = compute(q, k, v)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if initial_frames is None:
                initial_frames = counter.frame_count
            assert counter.frame_count == initial_frames


@cuda_required
@pytest.mark.parametrize("backend", ["triton", "sage2"])
def test_whole_sla_custom_op_cuda_graph_replay(backend):
    if backend == "sage2":
        pytest.importorskip("spas_sage_attn")
    operator = SlaTritonOperator() if backend == "triton" else SparseSageAttentionV2Operator()
    generator = SlaMaskGenerator(operator.q_block_size, operator.k_block_size, {"sparsity_ratio": 0.8})
    attention = _attention(generator, operator)
    compute = torch.compile(attention.apply, fullgraph=True, dynamic=True)
    graph = EdenLiveCUDAGraphStack()
    torch.manual_seed(222)
    k = torch.randn(257, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    def runner(x, embed, cross):
        return compute((x + embed).view(7, 2, 128), k, v)

    with torch.no_grad():
        for _ in range(3):
            x = torch.randn(7, 256, device="cuda", dtype=torch.bfloat16)
            embed = torch.randn(1, device="cuda", dtype=torch.bfloat16)
            expected = runner(x, embed, ())
            actual = graph.run("steady", runner, x, embed, bindings=(k, v))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert graph.capture_count == 1 and graph.replay_count == 3
