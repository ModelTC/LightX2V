"""Opt-in real-weight, real-video DDP encoder-distillation smoke test.

Run with torchrun on idle GPUs. This does not create or resume training runs.
"""

import argparse
import json
import os
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.distributed as dist

from lightx2v_train.model_capabilities import VAEDistillationCapability, VAEDistillationStepContext
from lightx2v_train.model_zoo.minimax_h3.minimax_h3_pruned_encoder import MiniMaxH3PrunedEncoderModel
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_vae_distillation_capability import _SampleGroup
from lightx2v_train.runtime.config import load_config
from lightx2v_train.runtime.ddp import apply_ddp
from lightx2v_train.trainers.vae.adversarial import VAEAdversarialObjective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--route", choices=("single", "quad", "temporal_pair", "temporal_quad"), default="single")
    parser.add_argument("--iterations", type=int, nargs="+", default=[0, 1000])
    parser.add_argument("--video", default="/data/nvme6/gushiqiao/datasets/minimax_h3_teacher_api_5s/videos/teacher_id_0_e358f4ddd79a.mp4")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    try:
        root = Path(__file__).resolve().parents[1]
        phase = "search" if args.search else "recover"
        config = load_config(str(root / f"configs/train/vae/minimax_h3_encoder_prune_{phase}_keep3_2gpu_ddp.yaml"))
        if not args.search:
            config["model"]["pruned_encoder"].pop("selection_path")
            config["model"]["pruned_encoder"]["kept_residual_indices"] = [0, 4, 10]
        torch.manual_seed(2026)
        model = MiniMaxH3PrunedEncoderModel(config)
        model.load_components(load_transformer=True, load_vae=True, load_condition_encoder=False)
        model.set_full_trainable()
        model.enable_gradient_checkpointing()
        apply_ddp(model, config)
        model.ensure_capabilities()
        capability = model.capabilities.require(VAEDistillationCapability)
        counts = {
            "single": (1, 1, 1), "quad": (1, 2, 2),
            "temporal_pair": (2, 1, 1), "temporal_quad": (4, 1, 1),
        }
        t_count, h_count, w_count = counts[args.route]
        # An interior spatial group exercises incoming/outgoing overlap halos.
        capability._sample_route = lambda *unused: args.route
        capability._sample_group = lambda *unused: _SampleGroup(args.route, 1, t_count, 1, h_count, 1, w_count)
        with av.open(args.video) as container:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0).float().div_(255)
        sample = {"inputs": {"video": video}, "meta": {"source_num_frames": torch.tensor([video.shape[2]])}}
        parameters = list(model.trainable_parameters())
        optimizer = torch.optim.AdamW(parameters, lr=1e-6)
        adversarial = None
        if not args.search:
            adversarial = VAEAdversarialObjective(config["training"]["vae_distillation"]["gan"], device=model.device,
                                                latent_channels=model.latent_channels, gradient_accumulation_iters=1)
        records = []
        for index in args.iterations:
            if args.search:
                model.denoiser_module().prepare_search_step(1)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            print(f"rank={rank} phase={phase} route={args.route} step={index} forward", flush=True)
            result = capability.compute_loss(sample, VAEDistillationStepContext(
                running_dtype=model.running_dtype, iteration=index, micro_step=0, adversarial_objective=adversarial,
            ))
            print(f"rank={rank} loss={result.loss.item():.6f} backward", flush=True)
            result.loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
                raise AssertionError("Missing or nonfinite encoder gradients.")
            if not any(p.grad.abs().sum() > 0 for p in parameters):
                raise AssertionError("No encoder gradient.")
            if any(p.requires_grad or p.grad is not None for p in model.teacher_vae.parameters()):
                raise AssertionError("Teacher encoder/decoder must remain frozen.")
            if adversarial is not None:
                adversarial.step()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            record = {"phase": phase, "route": args.route, "iteration": index, "rank": rank,
                      "seconds": time.perf_counter() - start,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                      "metrics": {key: float(value.detach()) if torch.is_tensor(value) else value for key, value in result.metrics.items()}}
            records.append(record)
            print(json.dumps(record), flush=True)
            del result
        if args.output is not None:
            gathered = [None] * dist.get_world_size() if rank == 0 else None
            dist.gather_object(records, gathered, dst=0)
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(gathered, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
