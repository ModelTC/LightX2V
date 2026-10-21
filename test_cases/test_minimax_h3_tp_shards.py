import torch

from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model


def test_column_tp_shard_owns_only_local_storage():
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    model.config = {"tensor_parallel": True}
    model.tp_rank = 2
    model.tp_size = 4

    source = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    shard = model._select_tensor_parallel_shard(
        "transformer_blocks.0.attn.to_q.weight",
        source,
    )

    torch.testing.assert_close(shard, source[4:6])
    assert shard.is_contiguous()
    assert shard.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert shard.untyped_storage().nbytes() == shard.numel() * shard.element_size()
