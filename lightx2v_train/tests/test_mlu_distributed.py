"""Small MLU integration checks; run each mode in a fresh torchrun job.

Example from lightx2v_train/ (with the Neuware library path configured)::

    PLATFORM=cambricon_mlu MLU_VISIBLE_DEVICES=6,7 \
        torchrun --standalone --nproc_per_node=2 tests/test_mlu_distributed.py \
        --mode fsdp2 --require-mlu

Modes ddp/fsdp2 train for three steps and verify a DCP model/optimizer resume.
Mode sp checks sequence collectives and their gradients. Without --require-mlu,
missing hardware or a non-distributed launch is explicitly reported as SKIP.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _skip(reason, required):
    print(f"SKIP MLU distributed integration: {reason}", flush=True)
    if required:
        raise RuntimeError(reason)


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _check_collectives(torch, dist, device, barrier):
    barrier()
    value = torch.tensor(float(dist.get_rank() + 1), device=device)
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    torch.testing.assert_close(value.cpu(), torch.tensor(expected))
    barrier()


def _train_and_resume(torch, dist, backend, mode, dtype, config, barrier):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

    from lightx2v_train.runtime.ddp import apply_ddp, unwrap_ddp_module
    from lightx2v_train.runtime.fsdp import apply_fsdp2, is_fsdp2_module

    class ModelAdapter:
        def __init__(self):
            self.transformer = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.SiLU(), torch.nn.Linear(32, 8)).to(backend.device)

        def denoiser_module(self):
            return self.transformer

        def is_fsdp2_wrapped(self):
            return is_fsdp2_module(self.transformer)

        def fsdp2_shard_plan(self, fsdp_config):
            yield {
                "modules": [self.transformer[0], self.transformer[2], self.transformer],
                "reshard_after_forward": True,
            }

    model = ModelAdapter()
    if mode == "ddp":
        apply_ddp(model, config)
    else:
        apply_fsdp2(model, config)
        assert is_fsdp2_module(model.transformer)
    module = model.transformer
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    inputs = torch.linspace(-1, 1, 64).reshape(4, 16).to(backend.device)
    targets = torch.linspace(-0.5, 0.5, 32).reshape(4, 8).to(backend.device)
    before = [_local(param).detach().cpu().clone() for param in module.parameters()]

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with backend.autocast(dtype):
            prediction = module(inputs)
            loss = (prediction.float() - targets).square().mean()
        assert torch.isfinite(loss).item(), "non-finite loss"
        loss.backward()
        for param in module.parameters():
            assert param.grad is not None, "missing gradient"
            assert torch.isfinite(_local(param.grad)).all().item(), "non-finite gradient"
        optimizer.step()
        return float(loss.item())

    losses = [train_step() for _ in range(3)]
    assert any(not torch.equal(initial, _local(param).detach().cpu()) for initial, param in zip(before, module.parameters())), "optimizer did not update model parameters"

    # Match the trainer's DCP options and plain DDP state-module selection.
    state_module = unwrap_ddp_module(module)
    options = StateDictOptions(ignore_frozen_params=True, strict=False)
    checkpoint_dirs = [tempfile.mkdtemp(prefix=f"lightx2v-mlu-{mode}-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(checkpoint_dirs, src=0, device=backend.device)
    checkpoint_dir = checkpoint_dirs[0]
    model_state, optimizer_state = get_state_dict(state_module, optimizer, options=options)
    dcp.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=checkpoint_dir)
    barrier()

    # A resumed next optimizer step must exactly match the uninterrupted step.
    expected_loss = train_step()
    expected_params = [_local(param).detach().cpu().clone() for param in module.parameters()]
    model_state, optimizer_state = get_state_dict(state_module, optimizer, options=options)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=checkpoint_dir)
    set_state_dict(
        state_module,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=options,
    )
    resumed_loss = train_step()
    torch.testing.assert_close(torch.tensor(resumed_loss), torch.tensor(expected_loss), rtol=0, atol=0)
    for expected, param in zip(expected_params, module.parameters()):
        torch.testing.assert_close(_local(param).detach().cpu(), expected, rtol=0, atol=0)
    barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(checkpoint_dir)
        print(f"PASS {mode}: {len(losses)} {dtype} steps, finite gradients, parameter updates, DCP resume; losses={losses}", flush=True)
    barrier()


def _check_sequence_parallel(torch, dist, backend, barrier):
    from lightx2v_train.runtime.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size
    from lightx2v_train.runtime.sequence_parallel import all_gather_sequence, all_to_all_4d, shrink_sequence

    world_size = get_sequence_parallel_world_size()
    rank = get_sequence_parallel_rank()
    full = torch.arange(2 * 3 * world_size * 2 * world_size * 4, dtype=torch.float32)
    full = full.reshape(2, 3 * world_size, 2 * world_size, 4).to(backend.device)
    local = shrink_sequence(full).detach().requires_grad_()
    exchanged = all_to_all_4d(local, scatter_dim=2, gather_dim=1)
    torch.testing.assert_close(exchanged.cpu(), full[:, :, rank * 2 : (rank + 1) * 2].cpu())
    restored = all_to_all_4d(exchanged, scatter_dim=1, gather_dim=2)
    torch.testing.assert_close(restored.cpu(), local.detach().cpu())
    restored.square().sum().backward()
    torch.testing.assert_close(local.grad.cpu(), (2 * local.detach()).cpu())

    gathered_input = local.detach().requires_grad_()
    gathered = all_gather_sequence(gathered_input)
    torch.testing.assert_close(gathered.cpu(), full.cpu())
    gathered.sum().backward()
    torch.testing.assert_close(gathered_input.grad.cpu(), torch.ones_like(gathered_input).cpu())
    barrier()
    if dist.get_rank() == 0:
        print("PASS sp: DeviceMesh, sequence all_to_all forward/inverse/backward, all_gather forward/backward", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ddp", "fsdp2", "sp"), default="ddp")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--require-mlu", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("PLATFORM", "cambricon_mlu")
    try:
        import torch
        import torch_mlu  # noqa: F401
    except (ImportError, RuntimeError) as error:
        _skip(f"MLU extension could not load: {error}", args.require_mlu)
        return
    if not torch.mlu.is_available():
        _skip("torch.mlu.is_available() is false", args.require_mlu)
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        _skip("launch with torchrun --nproc_per_node=2 (or more)", args.require_mlu)
        return
    if torch.mlu.device_count() < int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size))):
        _skip("fewer visible MLU devices than local workers", args.require_mlu)
        return
    if os.environ["PLATFORM"] != "cambricon_mlu":
        raise RuntimeError("This integration check requires PLATFORM=cambricon_mlu")

    import torch.distributed as dist

    from lightx2v_train.runtime.backend import get_backend
    from lightx2v_train.runtime.distributed import barrier, cleanup_distributed, init_distributed

    config = {
        "runtime": {"seed": 1234},
        "distributed": {
            "backend": "auto",
            "timeout_minutes": 2,
            "dp": {"enabled": args.mode == "ddp"},
            "fsdp2": {
                "enabled": args.mode == "fsdp2",
                "size": 1 if args.mode == "sp" else world_size,
                "mixed_precision": {
                    "param_dtype": args.dtype,
                    "reduce_dtype": "fp32",
                    "output_dtype": "fp32",
                    "cast_forward_inputs": True,
                },
            },
            "sequence_parallel": {"enabled": args.mode == "sp", "size": world_size},
        },
    }
    try:
        init_distributed(config)
        backend = get_backend()
        assert backend.name == "cambricon_mlu" and backend.device_type == "mlu"
        assert dist.get_backend() == "cncl"
        _check_collectives(torch, dist, backend.device, barrier)
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        if args.mode == "sp":
            _check_sequence_parallel(torch, dist, backend, barrier)
        else:
            _train_and_resume(torch, dist, backend, args.mode, dtype, config, barrier)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
