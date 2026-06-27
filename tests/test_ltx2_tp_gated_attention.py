from lightx2v.models.networks.ltx2.model import LTX2Model
from lightx2v.models.networks.ltx2.weights.transformer_weights import LTX2AttentionTP


def _tp_attention_config():
    return {
        "attn_type": "torch_sdpa",
        "rms_norm_type": "torch",
        "apply_gated_attention": True,
    }


def test_ltx2_attention_tp_registers_gated_attention_projection():
    attn = LTX2AttentionTP(
        block_index=0,
        attn_prefix="attn1",
        block_prefix="transformer_blocks",
        task="ltx2_s2v",
        mm_type="Default",
        config=_tp_attention_config(),
        tp_group=None,
        tp_rank=0,
        tp_size=8,
    )

    assert hasattr(attn, "to_gate_logits")
    assert attn.to_gate_logits.split_dim == "col"


def test_ltx2_tp_split_classifies_gated_attention_projection_as_column_split():
    model = LTX2Model.__new__(LTX2Model)
    key = "model.diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight"

    assert model._is_tp_weight(key)
    assert model._get_split_type(key) == "col"
