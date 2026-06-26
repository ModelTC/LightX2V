import json
from pathlib import Path
from types import SimpleNamespace

import torch

from lightx2v.models.networks.longcat_image.infer.module_io import LongCatImagePreInferModuleOutput
from lightx2v.models.networks.longcat_image.model import LongCatImageTransformerModel
from lightx2v.models.networks.longcat_image.weights import transformer_weights as longcat_weights
from lightx2v.models.schedulers.longcat_image.scheduler import LongCatImageScheduler


class _DummyModule:
    def __init__(self, *args, **kwargs):
        pass


def test_longcat_seq_parallel_splits_image_tokens_and_trims_gather(monkeypatch):
    group = object()
    model = LongCatImageTransformerModel.__new__(LongCatImageTransformerModel)
    model.seq_p_group = group
    model.scheduler = SimpleNamespace(latents=torch.empty(1, 5, 2))

    monkeypatch.setattr(longcat_weights, "ATTN_WEIGHT_REGISTER", longcat_weights.ATTN_WEIGHT_REGISTER)
    monkeypatch.setattr("lightx2v.models.networks.longcat_image.model.dist.get_world_size", lambda used_group: 3)
    monkeypatch.setattr("lightx2v.models.networks.longcat_image.model.dist.get_rank", lambda used_group: 1)

    pre_infer_out = LongCatImagePreInferModuleOutput(
        hidden_states=torch.arange(10, dtype=torch.float32).reshape(5, 2),
        encoder_hidden_states=torch.empty(4, 2),
        temb=torch.empty(1, 2),
        image_rotary_emb=(torch.empty(1), torch.empty(1)),
    )

    sharded = model._seq_parallel_pre_process(pre_infer_out)

    assert torch.equal(sharded.hidden_states, torch.tensor([[4.0, 5.0], [6.0, 7.0]]))

    def fake_all_gather(output_tensors, input_tensor, group=None):
        assert group is model.seq_p_group
        for rank, output_tensor in enumerate(output_tensors):
            output_tensor.copy_(torch.full_like(input_tensor, rank + 1))

    monkeypatch.setattr("lightx2v.models.networks.longcat_image.model.dist.all_gather", fake_all_gather)

    gathered = model._seq_parallel_post_process(torch.zeros(1, 2, 2))

    assert gathered.shape == (1, 5, 2)
    assert torch.equal(gathered[0, :, 0], torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0]))


def test_longcat_weights_register_parallel_attention_when_seq_parallel(monkeypatch):
    monkeypatch.setitem(longcat_weights.MM_WEIGHT_REGISTER, "dummy_mm", _DummyModule)
    monkeypatch.setitem(longcat_weights.RMS_WEIGHT_REGISTER, "dummy_rms", _DummyModule)
    monkeypatch.setitem(longcat_weights.ATTN_WEIGHT_REGISTER, "dummy_attn", _DummyModule)
    monkeypatch.setitem(longcat_weights.ATTN_WEIGHT_REGISTER, "dummy_parallel_attn", _DummyModule)

    config = {
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "dit_quant_scheme": "dummy_mm",
        "rms_norm_type": "dummy_rms",
        "attn_type": "dummy_attn",
        "seq_parallel": True,
        "parallel": {"seq_p_attn_type": "dummy_parallel_attn"},
    }

    double_block = longcat_weights.LongCatImageDoubleBlockWeights(config, 0)
    single_block = longcat_weights.LongCatImageSingleBlockWeights(config, 0)

    assert isinstance(double_block.calculate_parallel, _DummyModule)
    assert isinstance(single_block.calculate_parallel, _DummyModule)


def test_longcat_scheduler_chunks_only_image_rope_for_seq_parallel(monkeypatch):
    group = object()
    scheduler = LongCatImageScheduler.__new__(LongCatImageScheduler)
    scheduler.config = {"rope_type": "torch"}
    scheduler.seq_p_group = group

    monkeypatch.setattr("lightx2v.models.schedulers.longcat_image.scheduler.dist.get_world_size", lambda used_group: 2)
    monkeypatch.setattr("lightx2v.models.schedulers.longcat_image.scheduler.dist.get_rank", lambda used_group: 1)

    cos = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    sin = cos + 100

    chunked_cos, chunked_sin = scheduler._seq_parallel_rope((cos, sin), txt_seq_len=2)

    expected_cos = torch.tensor([[0.0, 1.0], [2.0, 3.0], [8.0, 9.0], [0.0, 0.0]])
    expected_sin = torch.tensor([[100.0, 101.0], [102.0, 103.0], [108.0, 109.0], [0.0, 0.0]])
    assert torch.equal(chunked_cos, expected_cos)
    assert torch.equal(chunked_sin, expected_sin)


def test_longcat_nvidia_dist8_launcher_uses_torchrun():
    script = Path("scripts/platforms/nvidia/dist_8/longcat_image_t2i.sh").read_text(encoding="utf-8")

    assert 'torchrun --nproc_per_node="${gpus}" -m lightx2v.infer' in script
    assert "gpus=${GPUS:-8}" in script
    assert "python -m lightx2v.infer" not in script


def test_longcat_nvidia_dist8_config_matches_qwen_image_parallel_layout():
    with open("configs/platforms/nvidia/dist_8/longcat_image_t2i.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    assert config["parallel"] == {
        "seq_p_size": 4,
        "seq_p_attn_type": "ulysses",
        "cfg_p_size": 2,
    }
