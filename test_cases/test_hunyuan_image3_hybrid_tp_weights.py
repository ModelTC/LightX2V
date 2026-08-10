from types import SimpleNamespace

import torch

from lightx2v.models.networks.hunyuan_image3.weights.common import HunyuanImage3MoEWeights
from lightx2v.models.networks.hunyuan_image3.weights.hybrid_tp import (
    FUSED_GATE_UP_LAYOUT,
    GROUPED_QKV_LAYOUT,
    HunyuanImage3HybridTensorParallelLinear,
    select_fused_gate_up_storage_shard,
    select_grouped_qkv_storage_shard,
    select_row_storage_shard,
)


def _hybrid_linear(context, split_dim, layout="plain", qkv_group_width=None, storage_rank=0):
    return HunyuanImage3HybridTensorParallelLinear(
        weight_name="weight",
        bias_name=None,
        mm_type="Default",
        tp_group=None,
        tp_rank=storage_rank,
        tp_size=2,
        split_dim=split_dim,
        parallel_context=context,
        micro_shard_count=2,
        weight_layout=layout,
        qkv_group_width=qkv_group_width,
        reduce_output=False,
    )


def _same_storage(left, right):
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def test_fused_gate_up_is_micro_major_and_phase_switch_uses_views():
    gate = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    up = gate + 100
    resident_checkpoint = select_fused_gate_up_storage_shard(torch.cat((gate, up)), 0, 2, 2)
    assert resident_checkpoint[:, 0].tolist() == [0, 1, 100, 101, 2, 3, 102, 103]

    context = SimpleNamespace(active_tp_size=4, active_tp_group=None, local_micro_shard_id=1)
    linear = _hybrid_linear(context, "col", FUSED_GATE_UP_LAYOUT)
    linear._mm.weight = resident_checkpoint.t()
    linear._mm.bias = None

    ar_weight = linear.active_weight
    assert _same_storage(ar_weight, linear._mm.weight)
    assert linear.apply(torch.ones(1, 1)).tolist() == [[2, 3, 102, 103]]
    expected_ar_hidden = torch.tensor([[2.0, 3.0]]) * torch.nn.functional.silu(torch.tensor([[102.0, 103.0]]))
    assert torch.allclose(linear.apply_gate_up_activation(torch.ones(1, 1)), expected_ar_hidden)

    context.active_tp_size = 2
    denoise_weight = linear.active_weight
    assert _same_storage(denoise_weight, linear._mm.weight)
    assert linear.apply(torch.ones(1, 1)).tolist() == [[0, 1, 2, 3, 100, 101, 102, 103]]
    expected_denoise_hidden = torch.tensor([[0.0, 1.0, 2.0, 3.0]]) * torch.nn.functional.silu(torch.tensor([[100.0, 101.0, 102.0, 103.0]]))
    assert torch.allclose(linear.apply_gate_up_activation(torch.ones(1, 1)), expected_denoise_hidden)


def test_grouped_qkv_splits_only_at_complete_kv_groups():
    # Q32/KV8/head_dim1 gives six output values per KV group.
    checkpoint = torch.arange(48 * 2, dtype=torch.float32).reshape(48, 2)
    resident_checkpoint = select_grouped_qkv_storage_shard(checkpoint, 1, 2, 2, 32, 8, 1)
    assert torch.equal(resident_checkpoint, checkpoint[24:])

    context = SimpleNamespace(active_tp_size=4, active_tp_group=None, local_micro_shard_id=0)
    linear = _hybrid_linear(context, "col", GROUPED_QKV_LAYOUT, qkv_group_width=6, storage_rank=1)
    linear._mm.weight = resident_checkpoint.t()
    linear._mm.bias = None
    assert _same_storage(linear.active_weight, linear._mm.weight)
    assert torch.equal(linear.active_weight, checkpoint[24:36].t())


def test_down_projection_micro_view_matches_checkpoint_input_slice():
    checkpoint = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)
    resident_checkpoint = select_row_storage_shard(checkpoint, 1, 2)
    context = SimpleNamespace(active_tp_size=4, active_tp_group=None, local_micro_shard_id=1)
    linear = _hybrid_linear(context, "row", storage_rank=1)
    linear._mm.weight = resident_checkpoint.t()
    linear._mm.bias = None
    assert _same_storage(linear.active_weight, linear._mm.weight)
    assert torch.equal(linear.active_weight, checkpoint[:, 6:8].t())


def test_flashinfer_pack_is_phase_neutral_and_micro_views_are_contiguous():
    context = SimpleNamespace(active_tp_size=4, local_micro_shard_id=1)
    moe = HunyuanImage3MoEWeights.__new__(HunyuanImage3MoEWeights)
    moe.num_experts = 2
    moe.parallel_context = context
    moe.storage_tp_rank = 1
    moe.storage_tp_size = 2
    moe.micro_shard_count = 2
    moe.flashinfer_logical_tp_size = 4
    moe.moe_weight = None
    moe.moe_weight_2 = None
    moe._flashinfer_weights_initialized = False
    moe._flashinfer_weight_device = None
    moe._flashinfer_weight_dtype = None

    expected_gate_up = []
    expected_down = []
    experts = []
    for expert_index in range(2):
        base = expert_index * 100
        gate_up = (torch.arange(8 * 3).reshape(8, 3) + base).to(torch.bfloat16)
        down = (torch.arange(3 * 4).reshape(3, 4) + base).to(torch.bfloat16)
        expected_gate_up.append(gate_up.clone())
        expected_down.append(down.clone())
        experts.append(
            SimpleNamespace(
                gate_and_up_proj=SimpleNamespace(weight=gate_up.t(), bias=None, has_lora_branch=False, has_diff=False),
                down_proj=SimpleNamespace(weight=down.t(), bias=None, has_lora_branch=False, has_diff=False),
            )
        )
    moe.experts = experts

    pack_w1, pack_w2 = moe.ensure_flashinfer_weights("cpu", torch.bfloat16)
    assert pack_w1.shape == (2, 2, 4, 3)
    assert pack_w2.shape == (2, 2, 3, 2)
    for micro_id in range(2):
        for expert_id in range(2):
            assert torch.equal(pack_w1[micro_id, expert_id], expected_gate_up[expert_id].reshape(2, 4, 3)[micro_id])
            expected_w2 = expected_down[expert_id].reshape(3, 2, 2).permute(1, 0, 2)[micro_id]
            assert torch.equal(pack_w2[micro_id, expert_id], expected_w2)

    ar_shards = moe.active_flashinfer_weight_shards("cpu", torch.bfloat16)
    assert len(ar_shards) == 1
    micro_id, canonical_rank, ar_w1, ar_w2 = ar_shards[0]
    assert (micro_id, canonical_rank) == (1, 3)
    assert ar_w1.is_contiguous() and ar_w2.is_contiguous()
    assert _same_storage(ar_w1, pack_w1) and _same_storage(ar_w2, pack_w2)

    context.active_tp_size = 2
    denoise_shards = moe.active_flashinfer_weight_shards("cpu", torch.bfloat16)
    assert [(item[0], item[1]) for item in denoise_shards] == [(0, 2), (1, 3)]
    assert all(_same_storage(item[2], pack_w1) and _same_storage(item[3], pack_w2) for item in denoise_shards)

    multi_w1, multi_w2 = moe.active_flashinfer_multi_micro_weights("cpu", torch.bfloat16)
    assert multi_w1 is moe.moe_weight
    assert multi_w2 is moe.moe_weight_2
    assert multi_w1.shape == (2, 2, 4, 3)
    assert multi_w2.shape == (2, 2, 3, 2)
    assert multi_w1.is_contiguous() and multi_w2.is_contiguous()
    assert _same_storage(multi_w1, pack_w1)
    assert _same_storage(multi_w2, pack_w2)
    assert multi_w1.storage_offset() == pack_w1.storage_offset() == 0
    assert multi_w2.storage_offset() == pack_w2.storage_offset() == 0

    moe.micro_shard_count = 1
    try:
        moe.active_flashinfer_multi_micro_weights("cpu", torch.bfloat16)
    except RuntimeError as error:
        assert "exactly two" in str(error)
    else:
        raise AssertionError("multi-micro accessor must reject a resident pack with micro_shard_count != 2")
