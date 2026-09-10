import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


def _load_checkpoint_class():
    module_path = Path(__file__).parents[3] / "lightx2v/models/networks/minimax_h3/checkpoint.py"
    spec = importlib.util.spec_from_file_location("minimax_h3_checkpoint", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MiniMaxH3ShardCheckpoint


MiniMaxH3ShardCheckpoint = _load_checkpoint_class()


def _write_fake_checkpoint(tmp_path):
    shard_1 = {
        "proj_in.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "transformer_blocks.0.attn.to_q.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "transformer_blocks.1.attn.to_q.weight": torch.full((2, 2), 3, dtype=torch.bfloat16),
    }
    shard_2 = {
        "transformer_blocks.0.ff.net.2.weight": torch.full((2, 2), 2, dtype=torch.bfloat16),
        "norm_out.linear.weight": torch.full((2, 2), 4, dtype=torch.float32),
    }
    save_file(shard_1, tmp_path / "model-00001-of-00002.safetensors")
    save_file(shard_2, tmp_path / "model-00002-of-00002.safetensors")

    weight_map = {name: "model-00001-of-00002.safetensors" for name in shard_1} | {name: "model-00002-of-00002.safetensors" for name in shard_2}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return shard_1, shard_2


def test_tensor_block_and_non_block_names_are_deterministic(tmp_path):
    _write_fake_checkpoint(tmp_path)
    checkpoint = MiniMaxH3ShardCheckpoint(tmp_path)

    assert checkpoint.tensor_names == (
        "norm_out.linear.weight",
        "proj_in.weight",
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.ff.net.2.weight",
        "transformer_blocks.1.attn.to_q.weight",
    )
    assert checkpoint.block_indices == (0, 1)
    assert checkpoint.tensor_names_for_block(0) == (
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.ff.net.2.weight",
    )
    assert checkpoint.non_block_tensor_names() == (
        "norm_out.linear.weight",
        "proj_in.weight",
    )
    assert checkpoint.block_names(0) == list(checkpoint.tensor_names_for_block(0))
    assert checkpoint.non_block_names() == list(checkpoint.non_block_tensor_names())
    assert checkpoint.shard_for_tensor("transformer_blocks.0.attn.to_q.weight") == "model-00001-of-00002.safetensors"
    assert checkpoint.shard_for_tensor("transformer_blocks.0.ff.net.2.weight") == "model-00002-of-00002.safetensors"


def test_load_tensors_reads_requested_block_tensors_across_shards(tmp_path):
    shard_1, shard_2 = _write_fake_checkpoint(tmp_path)
    checkpoint = MiniMaxH3ShardCheckpoint(tmp_path)

    tensors = checkpoint.load_tensors(checkpoint.tensor_names_for_block(0))

    assert set(tensors) == set(checkpoint.tensor_names_for_block(0))
    assert torch.equal(tensors["transformer_blocks.0.attn.to_q.weight"], shard_1["transformer_blocks.0.attn.to_q.weight"])
    assert torch.equal(tensors["transformer_blocks.0.ff.net.2.weight"], shard_2["transformer_blocks.0.ff.net.2.weight"])


def test_load_tensors_rejects_unknown_tensor(tmp_path):
    _write_fake_checkpoint(tmp_path)
    checkpoint = MiniMaxH3ShardCheckpoint(tmp_path)

    with pytest.raises(KeyError, match="missing requested tensors"):
        checkpoint.load_tensors(["transformer_blocks.9.attn.to_q.weight"])


def test_shard_for_tensor_rejects_unknown_tensor(tmp_path):
    _write_fake_checkpoint(tmp_path)
    checkpoint = MiniMaxH3ShardCheckpoint(tmp_path)

    with pytest.raises(KeyError, match="missing requested tensor"):
        checkpoint.shard_for_tensor("missing")


def test_missing_index_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="safetensors index not found"):
        MiniMaxH3ShardCheckpoint(tmp_path)


def test_invalid_weight_map_shard_name_raises_value_error(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"proj_in.weight": ""}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid shard file names"):
        MiniMaxH3ShardCheckpoint(tmp_path)


def test_missing_referenced_shard_raises_file_not_found(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"proj_in.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing shard files"):
        MiniMaxH3ShardCheckpoint(tmp_path)
