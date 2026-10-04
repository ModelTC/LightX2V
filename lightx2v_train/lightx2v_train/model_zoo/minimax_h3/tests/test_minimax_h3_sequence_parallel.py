import copy
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _configure_cpu_sp(rank, world_size):
    import lightx2v_train.runtime.distributed as runtime_dist

    runtime_dist._SP_GROUP = dist.group.WORLD
    runtime_dist._SP_RANK = rank
    runtime_dist._SP_WORLD_SIZE = world_size
    runtime_dist._DP_GROUP = None
    runtime_dist._DP_RANK = 0
    runtime_dist._DP_WORLD_SIZE = 1


def _distributed_sp_worker(rank, world_size, init_file):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _configure_cpu_sp(rank, world_size)
        from lightx2v_train.runtime.sequence_parallel import (
            all_gather_variable_sequence,
            all_to_all_4d_variable,
            balanced_sequence_lengths,
            balanced_sequence_slice,
            sync_sequence_parallel_parameters,
        )

        lengths = balanced_sequence_lengths(5)
        assert lengths == (3, 2)
        start, end = balanced_sequence_slice(5)
        global_tensor = torch.arange(1 * 5 * 4 * 3, dtype=torch.float32).reshape(1, 5, 4, 3)
        local = global_tensor[:, start:end].clone().requires_grad_(True)

        head_shard = all_to_all_4d_variable(
            local,
            scatter_dim=2,
            gather_dim=1,
            sequence_lengths=lengths,
        )
        torch.testing.assert_close(head_shard, global_tensor[:, :, rank * 2 : (rank + 1) * 2])
        restored = all_to_all_4d_variable(
            head_shard,
            scatter_dim=1,
            gather_dim=2,
            sequence_lengths=lengths,
        )
        torch.testing.assert_close(restored, local)
        restored.square().sum().backward()
        torch.testing.assert_close(local.grad, 2 * local.detach())

        gathered = all_gather_variable_sequence(local.detach().requires_grad_(True), lengths, dim=1)
        torch.testing.assert_close(gathered, global_tensor)
        gathered.sum().backward()

        parameter = torch.nn.Parameter(torch.tensor(float(rank + 1)))
        sync_sequence_parallel_parameters((parameter,))
        torch.testing.assert_close(parameter, torch.tensor(1.0))
    finally:
        dist.destroy_process_group()


def _distributed_h3_parity_worker(rank, world_size, init_file):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _configure_cpu_sp(rank, world_size)
        from diffusers import MiniMaxH3Transformer3DModel

        from lightx2v_train.model_zoo.native.minimax_h3.sequence_parallel import (
            install_minimax_h3_sequence_parallel,
        )

        torch.manual_seed(1234)
        dense = MiniMaxH3Transformer3DModel(
            num_attention_heads=4,
            attention_head_dim=8,
            hidden_size=16,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=2,
            audio_in_channels=3,
            patch_size=(1, 1, 1),
            text_dim=6,
            freq_dim=8,
            time_embed_hidden_dim=16,
            time_embed_dim=8,
            rope_freq_dim=1,
        )
        sequence_parallel = copy.deepcopy(dense)
        install_minimax_h3_sequence_parallel(sequence_parallel)
        dense.enable_gradient_checkpointing()
        sequence_parallel.enable_gradient_checkpointing()

        video_indices = torch.tensor([1, 2, 5])
        audio_indices = torch.tensor([3, 6])
        text_indices = torch.tensor([0, 4])
        token_tags = torch.tensor([1, 0, 0, 2, 1, 0, 2])
        inputs = dict(
            hidden_states=torch.randn(1, 3, 2),
            audio_hidden_states=torch.randn(1, 2, 3),
            encoder_hidden_states=torch.randn(1, 2, 6),
            timestep=torch.tensor([0.2, 0.8]),
            timestep_indices=torch.tensor([1, 0, 0, 0, 1, 0, 0]),
            token_tags=token_tags,
            position_ids=torch.tensor(
                [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2]],
                dtype=torch.float32,
            ),
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            return_dict=False,
        )

        dense_video, dense_audio = dense(**inputs)
        sp_video, sp_audio = sequence_parallel(**inputs)
        torch.testing.assert_close(sp_video, dense_video, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(sp_audio, dense_audio, rtol=2e-5, atol=2e-5)

        (dense_video.square().sum() + dense_audio.square().sum()).backward()
        (sp_video.square().sum() + sp_audio.square().sum()).backward()
        for (_, dense_param), (_, sp_param) in zip(dense.named_parameters(), sequence_parallel.named_parameters()):
            if sp_param.grad is None:
                assert dense_param.grad is None
                continue
            dist.all_reduce(sp_param.grad)
            torch.testing.assert_close(sp_param.grad, dense_param.grad, rtol=3e-4, atol=3e-4)
    finally:
        dist.destroy_process_group()


def _distributed_h3_mixed_route_worker(rank, world_size, init_file):
    """Exercise overlapping SP and DP collectives with unequal route lengths."""

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from diffusers import MiniMaxH3Transformer3DModel
        from torch.nn.parallel import DistributedDataParallel

        import lightx2v_train.runtime.distributed as runtime_dist
        from lightx2v_train.model_zoo.native.minimax_h3.sequence_parallel import (
            install_minimax_h3_sequence_parallel,
        )
        from lightx2v_train.runtime.sequence_parallel import (
            sync_sequence_parallel_gradients,
        )

        # Row-major (DP=2, SP=2): SP peers share a route, while each DDP group
        # crosses the two logical routes at one fixed SP coordinate.
        sp_groups = [
            dist.new_group((0, 1)),
            dist.new_group((2, 3)),
        ]
        dp_groups = [
            dist.new_group((0, 2)),
            dist.new_group((1, 3)),
        ]
        dp_rank, sp_rank = divmod(rank, 2)
        runtime_dist._SP_GROUP = sp_groups[dp_rank]
        runtime_dist._SP_RANK = sp_rank
        runtime_dist._SP_WORLD_SIZE = 2
        runtime_dist._DP_GROUP = dp_groups[sp_rank]
        runtime_dist._DP_RANK = dp_rank
        runtime_dist._DP_WORLD_SIZE = 2

        torch.manual_seed(4321)
        model = MiniMaxH3Transformer3DModel(
            num_attention_heads=4,
            attention_head_dim=8,
            hidden_size=16,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=2,
            audio_in_channels=3,
            patch_size=(1, 1, 1),
            text_dim=6,
            freq_dim=8,
            time_embed_hidden_dim=16,
            time_embed_dim=8,
            rope_freq_dim=1,
        )
        install_minimax_h3_sequence_parallel(model)
        model = DistributedDataParallel(model, process_group=dp_groups[sp_rank])

        if dp_rank == 0:
            video_count, audio_count, text_count = 3, 2, 2
            video_indices = torch.tensor([1, 2, 5])
            audio_indices = torch.tensor([3, 6])
            text_indices = torch.tensor([0, 4])
            token_tags = torch.tensor([1, 0, 0, 2, 1, 0, 2])
        else:
            # FL-like route: extra condition rows produce a different packed
            # sequence length while the denoiser parameter order stays equal.
            video_count, audio_count, text_count = 4, 3, 3
            video_indices = torch.tensor([1, 2, 5, 9])
            audio_indices = torch.tensor([3, 6, 7])
            text_indices = torch.tensor([0, 4, 8])
            token_tags = torch.tensor([1, 0, 0, 2, 1, 0, 2, 2, 1, 0])
        packed_length = video_count + audio_count + text_count
        generator = torch.Generator().manual_seed(9000 + dp_rank)
        inputs = dict(
            hidden_states=torch.randn(1, video_count, 2, generator=generator),
            audio_hidden_states=torch.randn(1, audio_count, 3, generator=generator),
            encoder_hidden_states=torch.randn(1, text_count, 6, generator=generator),
            timestep=torch.tensor([0.2, 0.8]),
            timestep_indices=torch.arange(packed_length) % 2,
            token_tags=token_tags,
            position_ids=torch.stack(
                (
                    torch.arange(packed_length),
                    torch.zeros(packed_length),
                    torch.arange(packed_length) % 3,
                ),
                dim=1,
            ).to(torch.float32),
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            return_dict=False,
        )
        video, audio = model(**inputs)
        (video.square().mean() + audio.square().mean()).backward()
        sync_sequence_parallel_gradients(model.parameters())
        torch.optim.SGD(model.parameters(), lr=0.01).step()

        checksum = torch.stack([parameter.detach().float().sum() for parameter in model.parameters()]).sum()
        checksums = [torch.empty_like(checksum) for _ in range(world_size)]
        dist.all_gather(checksums, checksum)
        for peer_checksum in checksums[1:]:
            torch.testing.assert_close(peer_checksum, checksums[0])
    finally:
        dist.destroy_process_group()


class MiniMaxH3SequenceParallelTest(unittest.TestCase):
    @staticmethod
    def _spawn(worker):
        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "process_group")
            mp.spawn(worker, args=(2, init_file), nprocs=2, join=True)

    def test_uneven_all_to_all_and_variable_gather(self):
        self._spawn(_distributed_sp_worker)

    def test_tiny_h3_forward_and_gradient_parity(self):
        self._spawn(_distributed_h3_parity_worker)

    def test_tiny_h3_mixed_route_sp2_dp2_forward_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "process_group")
            mp.spawn(
                _distributed_h3_mixed_route_worker,
                args=(4, init_file),
                nprocs=4,
                join=True,
            )

    def test_packed_indices_must_be_ordered_disjoint_and_complete(self):
        from lightx2v_train.model_zoo.native.minimax_h3.sequence_parallel import (
            _validate_packed_inputs,
        )

        common = dict(
            hidden_states=torch.zeros(1, 2, 2),
            audio_hidden_states=torch.zeros(1, 1, 3),
            encoder_hidden_states=torch.zeros(1, 0, 6),
            timestep_indices=torch.zeros(3, dtype=torch.long),
            token_tags=torch.tensor([0, 2, 0]),
            position_ids=torch.zeros(3, 3),
            text_indices=torch.empty(0, dtype=torch.long),
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _validate_packed_inputs(
                **common,
                video_indices=torch.tensor([2, 0]),
                audio_indices=torch.tensor([1]),
            )
        with self.assertRaisesRegex(ValueError, "disjoint.*cover"):
            _validate_packed_inputs(
                **common,
                video_indices=torch.tensor([0, 1]),
                audio_indices=torch.tensor([1]),
            )


if __name__ == "__main__":
    unittest.main()
