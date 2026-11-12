"""Manual two-GPU regression; intentionally excluded from CPU test discovery.

PYTHONPATH=lightx2v_train CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone \
    --nproc_per_node=2 -m lightx2v_train.model_zoo.minimax_h3_image.tests.verify_fsdp

Rank-local score sigmas must not mix PDD-head gradients. The tiny checkpoint
is generated locally; no pretrained model download is required.
"""

import copy
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from lightx2v_train.model_zoo.minimax_h3_image.model import MiniMaxH3ImageModel
from lightx2v_train.model_zoo.minimax_h3_image.tests.test_model import model_config, save_tiny_model
from lightx2v_train.model_zoo.native.minimax_h3.sharded_loading import stream_load_minimax_h3_transformer
from lightx2v_train.model_zoo.native.minimax_h3_image import load_transformer
from lightx2v_train.runtime.fsdp import _build_mp_policy, _iter_shard_plan
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


def main():
    if torch.cuda.device_count() < 2 or int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("Run this regression explicitly with torchrun --nproc_per_node=2 and two visible CUDA GPUs.")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90), device_id=torch.device("cuda", rank))
    fixture = tempfile.TemporaryDirectory() if rank == 0 else None
    try:
        paths = [fixture.name if fixture is not None else None]
        dist.broadcast_object_list(paths, src=0, device=torch.device("cuda", rank))
        directory = paths[0]
        if rank == 0:
            save_tiny_model(directory)
        dist.barrier()
        mesh = init_device_mesh("cuda", (2,))
        config = model_config(directory)
        model = MiniMaxH3ImageModel(config)
        model.load_components(load_transformer=False, load_vae=False, load_condition_encoder=False)
        reference = copy.copy(model)
        reference.transformer = load_transformer(directory, torch_dtype=torch.float32).cuda()
        reference.transformer.requires_grad_(True)
        model.transformer = load_transformer(directory, torch_dtype=torch.float32, meta=True)
        model.transformer.requires_grad_(True)
        for modules, reshard, overrides in _iter_shard_plan(model.fsdp2_shard_plan({})):
            policy = {"param_dtype": "fp32", "reduce_dtype": "fp32", "cast_forward_inputs": False, **(overrides or {})}
            fully_shard(modules, mesh=mesh, reshard_after_forward=reshard, mp_policy=_build_mp_policy(policy))
        stream_load_minimax_h3_transformer(model.transformer, Path(directory) / "transformer_diffusers", device=model.device)
        optimizer = torch.optim.AdamW(model.transformer.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.AdamW(reference.transformer.parameters(), lr=1e-3)
        # Ranks choose different output heads, then switch to a different pair
        # on the next step, as the independent PDMD score-time sampler does.
        for step in range(2):
            torch.manual_seed(100 + rank + step * 10)
            latent = torch.randn(1, 2, 1, 4, 4, device=model.device)
            condition = {"prompt_embeds": torch.randn(1, 3, 12, device=model.device)}
            sigma = torch.tensor([((1.0, 0.75), (0.95, 0.1))[step][rank]], device=model.device)
            expected = reference.predict_denoiser_output(latent, sigma, condition)
            expected.square().mean().backward()
            # Explicit reference DP averages zeros for heads unused on a rank.
            for parameter in reference.transformer.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(2)
            actual = model.predict_denoiser_output(latent, sigma, condition)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            actual.square().mean().backward()
            for (name, parameter), (reference_name, reference_parameter) in zip(model.transformer.named_parameters(), reference.transformer.named_parameters(), strict=True):
                assert name == reference_name
                gradient = parameter.grad.full_tensor() if parameter.grad is not None else torch.zeros_like(reference_parameter)
                torch.testing.assert_close(gradient, reference_parameter.grad, rtol=4e-4, atol=2e-6, msg=f"rank={rank} step={step} gradient={name}")
            optimizer.step()
            reference_optimizer.step()
            # Zero reference gradients rather than AdamW's None handling for
            # globally unused heads, matching the grouped FSDP representation.
            for name, parameter in model.transformer.named_parameters():
                expected_parameter = dict(reference.transformer.named_parameters())[name]
                torch.testing.assert_close(parameter.full_tensor(), expected_parameter, rtol=5e-4, atol=2e-5, msg=f"rank={rank} step={step} optimizer={name}")
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            print("H3-Image FSDP: different rank-local PDD heads preserve forward, averaged gradients and AdamW updates.", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
        if fixture is not None:
            fixture.cleanup()


if __name__ == "__main__":
    main()
