"""Text K/V cache lifecycle and native SM120 numerical parity."""

from types import SimpleNamespace

import pytest
import torch

from lightx2v.models.networks.wan.infer.edenlive.transformer_infer import EdenLiveARTransformerInfer


class _Attention:
    def __init__(self):
        self.prepare_calls = []
        self.prepared_calls = []
        self.raw_calls = []

    def prepare_kv(self, k, v):
        self.prepare_calls.append((k, v))
        return k.clone(), v.clone(), k.new_ones(1), v.new_ones(1)

    def apply_prepared(self, q, prepared):
        self.prepared_calls.append(prepared)
        return (q + prepared[0].mean()).flatten(1)

    def apply(self, q, k, v, **kwargs):
        self.raw_calls.append((q, k, v))
        return (q + k.mean()).flatten(1)


def _make_cache():
    infer = EdenLiveARTransformerInfer.__new__(EdenLiveARTransformerInfer)
    infer.reuse_cross_kv = True
    infer.reuse_quantized_cross_kv = True
    infer.cross_kv_cache = {}
    infer.has_image_context = False
    infer.num_heads, infer.head_dim = 2, 4
    infer.text_len = 4
    infer.scheduler = SimpleNamespace(infer_condition=True)
    identity = SimpleNamespace(apply=lambda x: x)
    phases = [
        SimpleNamespace(
            cross_attn_1=_Attention(),
            norm3=identity,
            cross_attn_q=identity,
            cross_attn_norm_q=identity,
            cross_attn_o=identity,
        )
        for _ in range(2)
    ]
    weights = SimpleNamespace(blocks=[SimpleNamespace(compute_phases=[None, phase]) for phase in phases])
    compute_calls = []

    def compute(phase, context):
        compute_calls.append((phase, context))
        return context.reshape(-1, 2, 4).clone(), context.reshape(-1, 2, 4).clone()

    infer.cross_kv_compute = compute
    pre = SimpleNamespace(
        context=torch.ones(4, 8),
        adapter_args={"context_id": 101, "text_segments": None},
        grid_sizes=SimpleNamespace(tuple=(2, 1, 1)),
    )
    return infer, weights, phases, compute_calls, pre


def test_quantized_text_cache_reuses_prompt_and_keeps_cfg_branches_separate():
    infer, weights, phases, compute_calls, pre = _make_cache()
    infer.prepare_cross_kv(weights, pre)
    conditional = infer.cross_kv_cache[True][1]
    for _ in range(5):
        infer.prepare_cross_kv(weights, pre)
    assert infer.cross_kv_cache[True][1] is conditional
    assert len(compute_calls) == 2
    assert all(len(phase.cross_attn_1.prepare_calls) == 1 for phase in phases)

    # The guidance branch must remain independent even with the same context ID.
    infer.scheduler.infer_condition = False
    pre.context = -torch.ones_like(pre.context)
    infer.prepare_cross_kv(weights, pre)
    unconditional = infer.cross_kv_cache[False][1]
    assert unconditional is not conditional
    torch.testing.assert_close(unconditional[0][0], -conditional[0][0], rtol=0, atol=0)
    infer.scheduler.infer_condition = True
    infer.prepare_cross_kv(weights, pre)
    assert infer.cross_kv_cache[True][1] is conditional
    assert len(compute_calls) == 4


def test_context_change_and_segmented_fallback_rebuild_the_right_cache():
    infer, weights, phases, compute_calls, pre = _make_cache()
    infer.prepare_cross_kv(weights, pre)
    old = infer.cross_kv_cache[True][1]
    pre.context.zero_()
    pre.adapter_args["context_id"] = 202
    infer.prepare_cross_kv(weights, pre)
    updated = infer.cross_kv_cache[True][1]
    assert updated is not old
    assert torch.count_nonzero(updated[0][0]) == 0
    assert len(compute_calls) == 4

    pre.adapter_args["text_segments"] = [1, 1]
    infer.prepare_cross_kv(weights, pre)
    segmented = infer.cross_kv_cache[True][1]
    assert all(len(kv) == 2 for kv in segmented)
    assert len(compute_calls) == 6
    assert all(len(phase.cross_attn_1.prepare_calls) == 2 for phase in phases)
    pre.adapter_args["text_segments"] = [2]
    infer.prepare_cross_kv(weights, pre)
    assert infer.cross_kv_cache[True][1] is segmented
    assert len(compute_calls) == 6

    pre.adapter_args["text_segments"] = None
    infer.prepare_cross_kv(weights, pre)
    assert all(len(kv) == 4 for kv in infer.cross_kv_cache[True][1])
    assert len(compute_calls) == 8
    assert all(len(phase.cross_attn_1.prepare_calls) == 3 for phase in phases)


def test_prepared_dispatch_preserves_changing_queries_and_raw_fallback():
    infer, weights, phases, _, pre = _make_cache()
    infer.current_pre = pre
    infer.cross_attn_cu_seqlens_q = torch.tensor([0, 2], dtype=torch.int32)
    infer.cross_attn_cu_seqlens_kv = torch.tensor([0, 4], dtype=torch.int32)
    infer.prepare_cross_kv(weights, pre)
    prepared = infer.cross_kv_cache[True][1][0]
    phase = phases[0]
    for value in (0.0, 3.0):
        x = torch.full((2, 8), value)
        actual = infer.infer_cross_attn_output(phase, x, pre.context, prepared)
        torch.testing.assert_close(actual, x + 1, rtol=0, atol=0)
    assert len(phase.cross_attn_1.prepared_calls) == 2
    assert all(value is prepared for value in phase.cross_attn_1.prepared_calls)
    assert not phase.cross_attn_1.raw_calls

    # Two video frames use separate text segments and their independent K means.
    pre.context = torch.cat((torch.ones(4, 8), 2 * torch.ones(4, 8)))
    pre.adapter_args["context_id"] = 303
    pre.adapter_args["text_segments"] = [1, 1]
    infer.prepare_cross_kv(weights, pre)
    raw = infer.cross_kv_cache[True][1][0]
    actual = infer.infer_cross_attn_output(phase, torch.zeros(2, 8), pre.context, raw)
    torch.testing.assert_close(actual, torch.stack((torch.ones(8), 2 * torch.ones(8))), rtol=0, atol=0)
    assert len(phase.cross_attn_1.raw_calls) == 2
    assert len(phase.cross_attn_1.prepared_calls) == 2


def test_request_cleanup_drops_prepared_text_kv_and_context_state():
    infer, weights, _, _, pre = _make_cache()
    infer.prepare_cross_kv(weights, pre)
    infer.current_pre = pre
    infer.key_freqs = infer.query_freqs = torch.ones(1)
    infer.freqs_cache = {"request": torch.ones(1)}
    infer.cache_pools = {True: object()}
    infer.history_cache = {(True, 0): object()}
    infer.clear_request_state()
    assert infer.cross_kv_cache == {}
    assert infer.cache_pools == {} and infer.history_cache == {} and infer.freqs_cache == {}
    assert infer.current_pre is None and infer.key_freqs is None and infer.query_freqs is None
    infer.prepare_cross_kv(weights, pre)
    assert len(infer.cross_kv_cache[True][1][0]) == 4


def test_pipeline_failure_clears_prepared_cache_and_closes_recording_reader():
    from lightx2v.models.runners.wan.edenlive_runner import EdenLiveRunner

    class FailingPipeline:
        def run_pipeline(self, input_info):
            raise RuntimeError("inference interrupted")

    class Runner(EdenLiveRunner, FailingPipeline):
        pass

    infer, weights, _, _, pre = _make_cache()
    infer.prepare_cross_kv(weights, pre)
    infer.cache_pools = {True: object()}
    infer.history_cache = {(True, 0): object()}
    closed = []
    runner = Runner.__new__(Runner)
    runner.model = SimpleNamespace(transformer_infer=infer)
    runner.recording_reader = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(RuntimeError, match="inference interrupted"):
        runner.run_pipeline(SimpleNamespace())
    assert infer.cross_kv_cache == {} and infer.cache_pools == {} and infer.history_cache == {}
    assert runner.recording_reader is None and closed == [True]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="SM120 required")
@pytest.mark.parametrize(
    "queries,keys,dim,dtype,compiled",
    [
        (65, 77, 64, torch.bfloat16, False),
        (129, 127, 128, torch.bfloat16, False),
        (65, 512, 128, torch.bfloat16, True),
        (33, 77, 64, torch.float16, True),
    ],
)
def test_prepared_sm120_attention_matches_native_exactly(queries, keys, dim, dtype, compiled):
    sageattention = pytest.importorskip("sageattention")
    from lightx2v.common.ops.attn.sage_sm120_cached import SageAttn2Sm120CachedWeight

    torch.manual_seed(42)
    heads = 4
    # Exercise non-contiguous source K/V and Q without changing numerical values.
    q = torch.randn(queries * 2, heads, dim, device="cuda", dtype=dtype)[::2]
    k = torch.randn(keys * 2, heads, dim, device="cuda", dtype=dtype)[::2]
    v = torch.randn(keys * 2, heads, dim, device="cuda", dtype=dtype)[::2]
    weight = SageAttn2Sm120CachedWeight()
    prepared = weight.prepare_kv(k, v)
    originals = [tensor.clone() for tensor in prepared]
    apply = torch.compile(weight.apply_prepared, fullgraph=True) if compiled else weight.apply_prepared
    for query in (q, q * 0.5):
        expected = sageattention.sageattn(query.contiguous().unsqueeze(0), k.contiguous().unsqueeze(0), v.contiguous().unsqueeze(0), tensor_layout="NHD").reshape(queries, heads * dim)
        actual = apply(query, prepared)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for tensor, original in zip(prepared, originals):
        # float8 comparisons are not implemented by all PyTorch builds.
        torch.testing.assert_close(tensor.float(), original.float(), rtol=0, atol=0)
