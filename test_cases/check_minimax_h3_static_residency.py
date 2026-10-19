"""Compare all real H3 blocks against legacy offload on a small packed AV input."""

import argparse
import json
from pathlib import Path

import torch

from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.utils.set_config import build_startup_config
from lightx2v_platform.base.global_var import AI_DEVICE
from tools.estimate_offload_residency import build_minimax_h3_case, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layout", choices=("per_tensor", "contiguous"), default="per_tensor")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    deployment = json.loads(Path(args.config).read_text())
    deployment.update(
        model_cls="minimax_h3",
        model_variant="fl2av",
        task="t2av",
        model_path=args.model_path,
        size=[128, 128],
        num_frames=22,
        cpu_offload_layout=args.layout,
        use_event_offload=True,
        offload_resident_blocks=0,
    )
    config = build_startup_config(deployment)
    case = build_minimax_h3_case(config, "A fox walks through the snow. Soft wind and quiet music.")
    legacy = dict(config)
    legacy.pop("offload_resident_blocks")
    legacy["use_event_offload"] = False
    model = MiniMaxH3Model(args.model_path, legacy, torch.device("cpu"))
    weights = model.transformer_weights
    module = getattr(torch, AI_DEVICE)
    scheduler = case.infer.scheduler

    def run(infer, step):
        infer.set_scheduler(scheduler)
        scheduler.step_pre(step)
        pre = case.pre
        pre.timestep_indices = scheduler.timestep_indices
        pre.adaln_indices = scheduler.timestep_indices * 3 + scheduler.layout.token_tags.clamp(min=0)
        infer._prepare_adaln_cache(pre)
        result = infer.infer_with_blocks_offload(weights.blocks, pre.hidden_states.clone(), pre)
        module.synchronize()
        return result

    reference = [run(model.transformer_infer, step) for step in (0, 1)]
    assert all(torch.isfinite(tensor).all() for tensor in reference)
    results = []
    total = len(weights.blocks)
    for count in (0, 1, total // 2, total - 1, total):
        infer = MiniMaxH3OffloadTransformerInfer({**config, "offload_resident_blocks": count})
        manager = infer.offload_manager
        manager.init_cuda_buffer(weights.offload_block_cuda_buffers)
        if args.layout == "contiguous":
            manager.init_contiguous_group(weights.offload_group)
        try:
            for step in (0, 1):
                torch.testing.assert_close(run(infer, step), reference[step], rtol=0, atol=0)
            if args.layout == "per_tensor" and count == 1:
                model.transformer_infer = infer
                model.release_block_offload_buffers()
                model.ensure_block_offload_buffers()
                torch.testing.assert_close(run(infer, 1), reference[1], rtol=0, atol=0)
            results.append({"resident_blocks": count, "streamed_blocks": total - count, "indices": infer.resident_blocks.indices, "bitwise_equal": True})
        finally:
            infer.release_resident_blocks()
            if manager.contiguous_transfer is not None:
                manager.contiguous_transfer.close()
        assert not infer.resident_blocks.active and not infer.resident_blocks.originals
    report = {"layout": args.layout, "blocks": total, "grid": case.grid, "steps": [0, 1], "video_tokens": case.video_tokens, "audio_tokens": case.audio_tokens, "checks": results}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
