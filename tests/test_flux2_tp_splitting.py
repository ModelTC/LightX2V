import torch

from lightx2v.models.networks.flux2.model import Flux2DevTransformerModel


def _model_stub():
    model = Flux2DevTransformerModel.__new__(Flux2DevTransformerModel)
    model.config = {
        "num_attention_heads": 4,
        "attention_head_dim": 2,
        "mlp_ratio": 2.0,
    }
    return model


def test_flux2_tp_split_type_classifies_transformer_weights():
    model = _model_stub()

    assert model._get_split_type("transformer_blocks.0.attn.to_q.weight") == "col"
    assert model._get_split_type("transformer_blocks.0.attn.add_k_proj.weight") == "col"
    assert model._get_split_type("transformer_blocks.0.ff.linear_in.weight") == "ff_fused_col"
    assert model._get_split_type("transformer_blocks.0.ff_context.linear_in.weight") == "ff_fused_col"
    assert model._get_split_type("transformer_blocks.0.attn.to_out.0.weight") == "row"
    assert model._get_split_type("transformer_blocks.0.attn.to_add_out.weight") == "row"
    assert model._get_split_type("transformer_blocks.0.ff.linear_out.weight") == "row"
    assert model._get_split_type("transformer_blocks.0.attn.norm_q.weight") is None
    assert model._get_split_type("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight") == "single_fused_col"
    assert model._get_split_type("single_transformer_blocks.0.attn.to_out.weight") == "single_fused_row"
    assert model._get_split_type("double_stream_modulation_img.linear.weight") is None


def test_flux2_tp_splits_single_block_fused_projection_by_logical_segments():
    model = _model_stub()
    tp_size = 4
    in_dim = 4
    inner_dim = 8
    ffn_dim = 16

    qkv_mlp_weight = torch.arange((3 * inner_dim + 2 * ffn_dim) * in_dim).reshape(3 * inner_dim + 2 * ffn_dim, in_dim)
    splits = model._split_weight_for_tp("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight", qkv_mlp_weight, tp_size)

    rank = 2
    expected = torch.cat(
        [
            qkv_mlp_weight[4:6],
            qkv_mlp_weight[12:14],
            qkv_mlp_weight[20:22],
            qkv_mlp_weight[32:36],
            qkv_mlp_weight[48:52],
        ],
        dim=0,
    )
    assert torch.equal(splits[rank], expected)
    assert splits[rank].shape == (14, in_dim)

    out_weight = torch.arange(7 * (inner_dim + ffn_dim)).reshape(7, inner_dim + ffn_dim)
    out_splits = model._split_weight_for_tp("single_transformer_blocks.0.attn.to_out.weight", out_weight, tp_size)
    expected_out = torch.cat([out_weight[:, 4:6], out_weight[:, 16:20]], dim=1)

    assert torch.equal(out_splits[rank], expected_out)
    assert out_splits[rank].shape == (7, 6)


def test_flux2_tp_splits_double_block_swiglu_projection_by_gate_up_segments():
    model = _model_stub()
    tp_size = 4
    in_dim = 3
    ffn_dim = 16

    weight = torch.arange((2 * ffn_dim) * in_dim).reshape(2 * ffn_dim, in_dim)
    splits = model._split_weight_for_tp("transformer_blocks.0.ff.linear_in.weight", weight, tp_size)

    rank = 2
    expected = torch.cat([weight[8:12], weight[24:28]], dim=0)

    assert model._get_split_type("transformer_blocks.0.ff.linear_in.weight") == "ff_fused_col"
    assert torch.equal(splits[rank], expected)
    assert splits[rank].shape == (8, in_dim)


def test_flux2_tp_double_block_swiglu_linear_matches_single_rank_math():
    model = _model_stub()
    tp_size = 4
    tokens = 5
    in_dim = 3
    ffn_dim = 16
    out_dim = 7

    x = torch.randn(tokens, in_dim)
    linear_in = torch.randn(2 * ffn_dim, in_dim)
    linear_out = torch.randn(out_dim, ffn_dim)

    full_hidden = x @ linear_in.t()
    full_gate, full_up = full_hidden.chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(full_gate) * full_up) @ linear_out.t()

    in_splits = model._split_weight_for_tp("transformer_blocks.0.ff.linear_in.weight", linear_in, tp_size)
    out_splits = model._split_weight_for_tp("transformer_blocks.0.ff.linear_out.weight", linear_out, tp_size)

    partials = []
    for in_weight, out_weight in zip(in_splits, out_splits):
        local_hidden = x @ in_weight.t()
        local_gate, local_up = local_hidden.chunk(2, dim=-1)
        local_mlp = torch.nn.functional.silu(local_gate) * local_up
        partials.append(local_mlp @ out_weight.t())

    actual = sum(partials)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
