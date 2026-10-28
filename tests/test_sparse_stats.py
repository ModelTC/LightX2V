"""Exercise production sparse entries and consumers; replace native calls without claiming CUDA validation."""

import json
from types import SimpleNamespace

import pytest
import torch

from lightx2v.utils import workload_stats


def read_rows(path, name="core_ops.jsonl"):
    return [json.loads(line) for line in (path / name).read_text().splitlines()]


def test_dynamic_sla_passes_actual_shape_and_route_facts(tmp_path, monkeypatch):
    from math import ceil

    from lightx2v.common.ops.attn.dynamic_sparse_attn import DynamicSparseAttnWeight
    from lightx2v.common.ops.attn.utils import sla_util

    def pool(value, block, center=None):
        if center is not None:
            value = value - center
        length = value.shape[-2]
        return torch.stack([value[..., start : min(start + block, length), :].mean(-2) for start in range(0, length, block)], dim=-2)

    consumed = []

    class Native:
        @staticmethod
        def apply(q, k, v, mask, lut, topk, block_q, block_k):
            consumed.append((mask.clone(), topk, block_q, block_k))
            return q

    monkeypatch.setattr(sla_util, "mean_pool", pool)
    op = DynamicSparseAttnWeight({"operator": "triton", "sparsity_ratio": 0.5})
    op._attention = Native
    length, heads, dim = 257, 2, 64  # The production Triton kernel requires equal Q/K/V lengths and a supported head dimension.
    q, k, v = (torch.randn(length, heads, dim, dtype=torch.bfloat16) for _ in range(3))
    with workload_stats.WorkloadStats(tmp_path / "stats", families=["attention"]):
        output = op.apply(q, k, v)
    (core,) = read_rows(tmp_path / "stats")
    mask, topk, bq, bk = consumed[0]
    pairs = int(mask.repeat_interleave(bq, -2).repeat_interleave(bk, -1)[..., :length, :length].sum())
    assert core["sparse"]["sparsity"] == 0.5
    assert core["work"]["selected_logical_flops"] == pairs * 4 * dim
    assert core["work"]["scheduled_tile_flops"] == heads * ceil(length / bq) * topk * bq * bk * 4 * dim
    assert core["work"]["dense_equivalent_flops"] == heads * length * length * 4 * dim
    assert torch.equal(output, q.reshape(length, -1))


def test_magi_failed_native_does_not_export_or_consume_sample(tmp_path, monkeypatch):
    from lightx2v.common.ops.attn import dynamic_sparse_attn as module

    op = module.DynamicSparseAttnWeight({"operator": "magi"})
    op.BLKQ = op.BLKK = 2
    mask = torch.eye(3, dtype=torch.bool).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    op._get_block_map = lambda *args, **kwargs: (mask, None, 1)
    inputs = torch.arange(240, dtype=torch.float32).reshape(3, 5, 2, 8)[..., ::2]
    consumed, exported = [], []

    def native(q, k, v, **options):
        consumed.append(((q, k, v), options))
        if len(consumed) == 1:
            raise RuntimeError("native failure")
        return (q.clone(),)

    monkeypatch.setattr(module, "magi_ffa_func", native)
    with workload_stats.WorkloadStats(tmp_path / "stats", max_calls=1, on_sparse_inputs=exported.append):
        with pytest.raises(RuntimeError, match="native failure"):
            op.apply(*inputs)
        assert exported == []
        output = op.apply(*inputs)
    assert len(exported) == len(read_rows(tmp_path / "stats")) == 1
    assert read_rows(tmp_path / "stats")[0]["call"] == 2
    assert len(read_rows(tmp_path / "stats", "workloads.jsonl")) == 1
    assert torch.equal(output, inputs[0].reshape(5, -1))
    for name, logical, packed in zip(("q", "k", "v"), inputs, consumed[1][0]):
        view = exported[0][name]
        assert torch.equal(view, logical)
        assert view.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
        assert view.untyped_storage().data_ptr() != logical.untyped_storage().data_ptr()
    for actual, name in zip(exported[0]["ranges"], ("q_ranges", "k_ranges")):
        assert torch.equal(actual, consumed[1][1][name])


def general_with(mask, consumer):
    from lightx2v.common.ops.attn.general_sparse_attn import GeneralSparseAttnWeight

    class Generator:
        def __call__(self, q, k):
            return mask

        def reorg(self, q, k, v):
            return q, k, v

        def restore(self, output):
            return output

    op = GeneralSparseAttnWeight.__new__(GeneralSparseAttnWeight)
    op.mask_generator, op.operator = Generator(), consumer
    return op


@pytest.mark.parametrize("lengths", [(1, 1), (129, 257)])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_flashinfer_variable_plan_keeps_logical_work_and_unknown_tiles(monkeypatch, tmp_path, lengths, warm_cache):
    from math import ceil

    from lightx2v.common.ops.attn.sparse_operator import FlashinferOperator

    plans = []

    def plan(**kwargs):
        plans.append(kwargs)

    wrapper = SimpleNamespace(plan=plan, run=lambda q, k, v: q.clone())
    monkeypatch.setattr(FlashinferOperator, "sparse_wrapper", wrapper)
    monkeypatch.setattr(FlashinferOperator, "mask", None)
    monkeypatch.setattr(FlashinferOperator, "plan_key", None)
    sq, sk = lengths
    q, k, v = torch.randn(sq, 2, 64), torch.randn(sk, 2, 64), torch.randn(sk, 2, 64)
    mask = torch.eye(ceil(sq / 128), ceil(sk / 128), dtype=torch.bool).expand(1, 2, -1, -1).clone()
    mask[..., -1, -1] = True
    op = general_with(mask, FlashinferOperator())
    kwargs = {"max_seqlen_q": sq, "max_seqlen_kv": sk}
    if warm_cache:
        assert torch.equal(op.apply(q, k, v, **kwargs), q.reshape(sq, -1))
    target = tmp_path / "stats"
    with workload_stats.WorkloadStats(target, families=["attention"], max_calls=2):
        for _ in range(2):
            assert torch.equal(op.apply(q, k, v, **kwargs), q.reshape(sq, -1))
    assert len(plans) == 1
    expected_pairs = int(mask.repeat_interleave(128, -2).repeat_interleave(128, -1)[..., :sq, :sk].sum())
    cores, leaves = read_rows(target), read_rows(target, "workloads.jsonl")
    assert len(cores) == len(leaves) == 2
    for core, leaf in zip(cores, leaves):
        assert core["work"]["selected_logical_flops"] == leaf["work"]["selected_logical_flops"] == expected_pairs * 256
        assert core["work"]["scheduled_tile_flops"] is None and leaf["work"]["scheduled_tile_flops"] is None
        assert leaf["routing"]["q_block_sizes"] == [plans[0]["block_row_sz"].tolist()]
        assert leaf["routing"]["k_block_sizes"] == [plans[0]["block_col_sz"].tolist()]


@pytest.mark.parametrize("family", ["radial", "nbhd"])
def test_static_ranges_use_actual_lengths_and_head_replication(monkeypatch, tmp_path, family):
    from lightx2v.common.ops.attn import nbhd_attn, radial_attn

    module = radial_attn if family == "radial" else nbhd_attn
    cls = module.RadialAttnWeight if family == "radial" else module.NbhdAttnWeight
    op = cls.__new__(cls)
    op.prepare_mask = lambda **kwargs: None
    op.q_ranges = torch.tensor([[0, 2], [2, 3]], dtype=torch.int32)
    op.k_ranges = torch.tensor([[0, 1], [1, 3]], dtype=torch.int32)
    expected_pairs = 4  # Handwritten ranges only test Radial per-head reuse.
    if family == "nbhd":
        # Use actual CPU producer head/tail coordinates; preload only to bypass prepare_mask CUDA transfers.
        mask = nbhd_attn.generate_nbhd_mask(1, 2, 2, coefficient=[1.0], device="cpu")
        repeated = mask.unsqueeze(0).repeat(2, 1, 1)
        op.q_ranges, op.k_ranges = nbhd_attn.generate_qk_ranges(repeated, 2, 2, 3)
        expected_pairs = int(mask.repeat_interleave(2, 0).repeat_interleave(2, 1)[:3, :3].sum()) * 2
    op.attn_type_map = torch.zeros(len(op.q_ranges), dtype=torch.int32)
    consumed = []

    def native(q, *args, q_ranges, k_ranges, **kwargs):
        consumed.append((q_ranges, k_ranges))
        return (q,)

    monkeypatch.setattr(module, "magi_ffa_func", native, raising=False)
    q = torch.randn(3, 2, 4)
    baseline = op.apply(q, q, q)
    target = tmp_path / "stats"
    with workload_stats.WorkloadStats(target, families=["attention"]):
        assert torch.equal(op.apply(q, q, q), baseline)
    (core,) = read_rows(target)
    assert core["work"]["selected_logical_flops"] == expected_pairs * 16 * (2 if family == "radial" else 1)
    assert core["work"]["scheduled_tile_flops"] is None
    assert "未知（内部 tile/padding 未提供）" in (target / "core_ops.md").read_text()
    assert "None FLOPs" not in (target / "core_ops.md").read_text()
    if family == "nbhd":
        actual = torch.zeros(6, 6, dtype=torch.bool)
        for (q0, q1), (k0, k1) in zip(*consumed[0]):
            actual[q0:q1, k0:k1] = True
        token_mask = mask.repeat_interleave(2, 0).repeat_interleave(2, 1)[:3, :3]
        assert torch.equal(actual, torch.block_diag(token_mask, token_mask))


def test_general_triton_uses_consumed_lut_and_lazy_mask_cap(monkeypatch, tmp_path):
    from lightx2v.common.ops.attn.sparse_operator import SlaTritonOperator

    q = torch.randn(384, 1, 4)
    mask = torch.tensor([[[[1, 0, 0], [1, 1, 1], [0, 1, 0]]]], dtype=torch.bool)
    consumed = []

    class CPUConsumer:
        @staticmethod
        def apply(q, k, v, mask, lut, topk, *args):
            consumed.append(lut)
            return q

    operator = SlaTritonOperator.__new__(SlaTritonOperator)
    operator.q_block_size = operator.k_block_size = 128
    operator._attention = CPUConsumer

    op = general_with(mask, operator)
    reference = op.apply(q, q, q, max_seqlen_q=384)
    allocations = []
    original = torch.zeros_like

    def zeros_like(*args, **kwargs):
        allocations.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros_like", zeros_like)
    target = tmp_path / "stats"
    with workload_stats.WorkloadStats(target, families=["attention"], max_calls=1):
        for _ in range(2):
            assert torch.equal(op.apply(q, q, q, max_seqlen_q=384), reference)
    (core,) = read_rows(target)
    (leaf,) = read_rows(target, "workloads.jsonl")
    assert len(allocations) == 1 and len(consumed) == 3  # The cap prevents constructing a second mask.
    assert int(mask.sum()) == 5 and consumed[0].numel() == 9
    assert len(core["actual_backend_routes"]) == 1
    assert core["work"]["selected_logical_flops"] == leaf["work"]["selected_logical_flops"] == 2359296
    assert leaf["routing"]["selected_tiles"] == 9


@pytest.mark.parametrize("entry", ["general", "rainfusion"])
def test_magi_consumer_records_ranges_without_guessing_internal_tiles(monkeypatch, tmp_path, entry):
    from lightx2v.common.ops.attn import rainfusion_attn, sparse_operator

    consumed = []

    def magi_cpu(q, k, v, *, q_ranges, k_ranges, auto_range_merge, **kwargs):
        assert auto_range_merge
        consumed.append((q.shape[0], q_ranges.clone(), k_ranges.clone()))
        return (q.clone(),)  # Actual Python path, not evidence of Magi CUDA execution.

    monkeypatch.setattr(sparse_operator, "magi_ffa_func", magi_cpu)
    consumer = sparse_operator.MagiOperator()
    mask = torch.eye(3, dtype=torch.bool).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    q = torch.randn(257 if entry == "general" else 384, 2, 4)
    if entry == "general":
        op = general_with(mask, consumer)
        arguments = {}
    else:
        cls = rainfusion_attn.RainfusionAttnWeight
        for name, value in dict(
            pool_size=128, sparsity=0.8, skip_timesteps=-1, text_len=0, txt_first=False, backend="cuda_sparse", sparse_operator="magi_operator", dense_attn_type="torch_sdpa", operator_setting={}
        ).items():
            monkeypatch.setattr(cls, name, value)
        adapter = rainfusion_attn.CudaRainfusionSparseOperator.__new__(rainfusion_attn.CudaRainfusionSparseOperator)
        adapter.sparse_operator = consumer
        monkeypatch.setattr(cls, "_get_operator", classmethod(lambda cls: adapter))
        monkeypatch.setattr(cls, "_grid_size", (3, 8, 16))
        monkeypatch.setattr(cls, "_base_blockmask", cls.build_blockwise_mask(mask.float()))
        op, arguments = cls(), {"grid_sizes": (3, 8, 16)}
    reference = op.apply(q, q, q, **arguments)
    target = tmp_path / "stats"
    with workload_stats.WorkloadStats(target, families=["attention"], max_calls=1):
        for _ in range(2):
            assert torch.equal(op.apply(q, q, q, **arguments), reference)
    (core,) = read_rows(target)
    (leaf,) = read_rows(target, "workloads.jsonl")
    packed_length, q_ranges, k_ranges = consumed[0]
    tokens = torch.zeros(packed_length, packed_length, dtype=torch.bool)
    for (q0, q1), (k0, k1) in zip(q_ranges, k_ranges):
        tokens[q0:q1, k0:k1] = True
    expected = int(tokens.sum()) * 4 * q.shape[-1]
    assert leaf["operator"] == "MagiOperator" and len(core["actual_backend_routes"]) == 1
    assert core["work"]["selected_logical_flops"] == leaf["work"]["selected_logical_flops"] == expected
    assert core["work"]["scheduled_tile_flops"] is None and leaf["work"]["scheduled_tile_flops"] is None
    assert "未知" in (target / "core_ops.md").read_text() and len(consumed) == 3


@pytest.mark.parametrize("mode", ["selected", "unselected", "capped"])
def test_fake_cuda_sparse_input_never_queries_cuda_or_builds_region(monkeypatch, tmp_path, mode):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        fake = torch.empty(1, 1, 3, 4, device="cuda")
    assert fake.is_cuda
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: pytest.fail("fake input must not query CUDA"))
    target = tmp_path / "stats"
    families = ["gemm"] if mode == "unselected" else ["attention"]
    with workload_stats.WorkloadStats(target, families=families, max_calls=1):
        if mode == "capped":
            real = torch.ones(1, 1, 3, 4)
            workload_stats.record_sparse_attn(real, real, real, torch.ones(1, 1, 1, 1), 3, 3)
        workload_stats.record_sparse_attn(fake, fake, fake, lambda: pytest.fail("fake input must not build region"))
    summary = json.loads((target / "session.json").read_text())
    assert summary["status"] == "complete"
    assert len(read_rows(target, "workloads.jsonl")) == (1 if mode == "capped" else 0)
    if mode == "selected":
        assert any("fake/meta" in reason for reason in summary["coverage_gaps"])


def test_draft_callback_uses_consumed_bucket_coordinates(monkeypatch, tmp_path):
    from lightx2v.common.ops.attn import draft_attn

    consumed, sampled = [], []

    def cpu_magi(q, k, v, **options):
        consumed.append((q.clone(), k.clone(), v.clone(), options["q_ranges"], options["k_ranges"]))
        return (q.clone(),)  # Replace only Magi native compute; keep production Python routing, reordering, and callbacks.

    monkeypatch.setattr(draft_attn, "magi_ffa_func", cpu_magi)
    op = draft_attn.DraftAttnWeight()
    q = torch.arange(512, dtype=torch.float32).reshape(512, 1, 1)
    scheduler = SimpleNamespace(latents=torch.empty(1, 2, 8, 32), patch_size=(1, 1, 1))
    with workload_stats.WorkloadStats(tmp_path / "draft", families=["attention"], on_sparse_inputs=sampled.append):
        op.apply(q, torch.cos(q / 7), q + 1000, block_idx=1, scheduler=scheduler)
    assert not torch.equal(op.reorg_idx_dict[(512, 8, 32)], torch.arange(512))
    row = sampled[0]
    for name, value in zip(("q", "k", "v"), consumed[0][:3]):
        assert torch.equal(row[name].transpose(0, 1).reshape_as(value), value)
    assert not torch.equal(row["q"], q)
    assert all(torch.equal(a, b) for a, b in zip(row["ranges"], consumed[0][3:]))
