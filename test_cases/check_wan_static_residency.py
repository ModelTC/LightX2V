"""Compare real Wan blocks with legacy offload; explicitly invoked GPU check."""

import argparse
import json

import torch

from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.utils.set_config import build_startup_config
from lightx2v_platform.base.global_var import AI_DEVICE
from tools.estimate_offload_residency import build_wan_case, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--layout", choices=("per_tensor", "contiguous"), default="per_tensor")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    deployment = json.loads(open(args.config).read())
    deployment.update(model_cls="wan2.1", task="i2v", model_path=args.model_path, size=[64, 64], num_frames=1, cpu_offload_layout=args.layout, use_event_offload=True, offload_resident_blocks=0)
    config = build_startup_config(deployment)
    case = build_wan_case(config, args.image_path)
    legacy_config = dict(config)
    legacy_config.pop("offload_resident_blocks")
    legacy_config["use_event_offload"] = False
    model = WanModel(args.model_path, legacy_config, torch.device("cpu"))
    weights = model.transformer_weights
    module = getattr(torch, AI_DEVICE)

    def setup(infer):
        infer.set_scheduler(case.infer.scheduler)
        infer.cos_sin, infer.rope_positions = case.infer.cos_sin, case.infer.rope_positions
        infer.reset_infer_states(case.pre.x, case.pre.context)

    setup(model.transformer_infer)
    reference = model.transformer_infer.infer_with_blocks_offload(weights.blocks, case.pre.x.clone(), case.pre)
    module.synchronize()
    assert torch.isfinite(reference).all()
    results = []
    for count in (0, 1, 30, 39, 40):
        infer = WanOffloadTransformerInfer({**config, "offload_resident_blocks": count})
        setup(infer)
        manager = infer.offload_manager
        manager.init_cuda_buffer(weights.offload_block_cuda_buffers)
        if args.layout == "contiguous":
            manager.init_contiguous_group(weights.offload_group)
        try:
            for _ in range(2):
                actual = infer.infer_with_blocks_offload(weights.blocks, case.pre.x.clone(), case.pre)
                module.synchronize()
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            results.append({"resident_blocks": count, "streamed_blocks": len(weights.blocks) - count, "indices": infer.resident_blocks.indices, "bitwise_equal": True})
        finally:
            infer.release_resident_blocks()
        assert not infer.resident_blocks.active and not infer.resident_blocks.originals
    report = {"layout": args.layout, "blocks": len(weights.blocks), "grid": case.grid, "passes_per_count": 2, "checks": results}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
