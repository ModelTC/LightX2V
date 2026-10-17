import pytest
import torch

from lightx2v.common.ops.attn.sage_sm90_fused import make_transposed_v, update_transposed_v
from lightx2v.common.ops.attn.sla_sm120_cached import SlaSage2Sm120CachedWeight
from lightx2v.common.ops.attn.sparse_operator import sla_sage2_attention
from lightx2v.models.networks.wan.infer.edenlive.cuda_graph import EdenLiveCUDAGraphStack

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("ratio", [-0.01, 1, 1.5, float("nan")])
def test_cached_sla_rejects_invalid_sparsity(ratio):
    attention = SlaSage2Sm120CachedWeight.__new__(SlaSage2Sm120CachedWeight)
    with pytest.raises(ValueError, match="sparsity_ratio"):
        attention.set_config({"sparsity_ratio": ratio})


@cuda_required
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("queries,keys", [(1, 1), (65, 257), (129, 777), (2640, 15840)])
def test_cached_sla_matches_original_sparse_sage_layout_bitexact(dtype, queries, keys):
    pytest.importorskip("spas_sage_attn")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    attention = SlaSage2Sm120CachedWeight({"sparsity_ratio": 0.8})
    torch.manual_seed(301)
    q = torch.randn(queries, 2, 128, device="cuda", dtype=dtype)
    k = torch.randn(keys, 2, 128, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    workspace = make_transposed_v(v.unsqueeze(0), keys)
    original = [value.clone() for value in (q, k, workspace)]
    with torch.no_grad():
        expected = sla_sage2_attention(q, k, v, 1 - 0.8, 128, 64, "sm120")
        actual = attention.apply_pretransposed(q, k, workspace)
        torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_input, original_input in zip((q, k, workspace), original):
        torch.testing.assert_close(actual_input, original_input, rtol=0, atol=0)


@cuda_required
def test_cached_sla_tracks_partial_current_v_writes_without_layout_rebuild():
    pytest.importorskip("spas_sage_attn")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    attention = SlaSage2Sm120CachedWeight()
    torch.manual_seed(302)
    k = torch.randn(257, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    workspace = make_transposed_v(v.unsqueeze(0), k.shape[0])
    pointer = workspace.data_ptr()
    for start, count in [(15, 65), (129, 128), (256, 1)]:
        q = torch.randn(count, 2, 64, device="cuda", dtype=torch.bfloat16)
        current = torch.randn(count, 2, 64, device="cuda", dtype=torch.bfloat16)
        v[start : start + count].copy_(current)
        update_transposed_v(current.unsqueeze(0), workspace, start)
        expected = sla_sage2_attention(q, k, v, 1 - 0.8, 128, 64, "sm120")
        actual = attention.apply_pretransposed(q, k, workspace)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert workspace.data_ptr() == pointer


@cuda_required
def test_cached_sla_compiled_graph_replay_refreshes_inputs_bitexact():
    pytest.importorskip("spas_sage_attn")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    attention = SlaSage2Sm120CachedWeight()
    compute = torch.compile(attention.apply_pretransposed, fullgraph=True, dynamic=True)
    graph = EdenLiveCUDAGraphStack()
    torch.manual_seed(303)
    k = torch.randn(257, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    workspace = make_transposed_v(v.unsqueeze(0), k.shape[0])

    def runner(x, embed, cross):
        return compute((x + embed).view(65, 2, 128), k, workspace)

    for _ in range(3):
        x = torch.randn(65, 256, device="cuda", dtype=torch.bfloat16)
        embed = torch.randn(1, device="cuda", dtype=torch.bfloat16)
        v[-65:] = torch.randn_like(v[-65:])
        update_transposed_v(v[-65:].unsqueeze(0), workspace, v.shape[0] - 65)
        expected = runner(x, embed, ())
        actual = graph.run("cached_sla", runner, x, embed, bindings=(k, workspace))
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert graph.capture_count == 1 and graph.replay_count == 3
