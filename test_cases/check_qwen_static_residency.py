"""Compare real Qwen Image blocks with legacy offload; explicitly invoked GPU check."""

import argparse
import json
from pathlib import Path

import torch

from lightx2v.models.networks.qwen_image.infer.offload.transformer_infer import QwenImageOffloadTransformerInfer
from lightx2v.models.networks.qwen_image.model import QwenImageTransformerModel
from lightx2v.utils.set_config import build_startup_config
from lightx2v_platform.base.global_var import AI_DEVICE
from tools.estimate_offload_residency import build_qwen_case, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layout", choices=("per_tensor", "contiguous"), default="per_tensor")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    deployment = json.loads(Path(args.config).read_text())
    deployment.update(model_cls="qwen_image", task="t2i", model_path=args.model_path, size=[256, 256], cpu_offload_layout=args.layout, use_event_offload=True, offload_resident_blocks=0)
    config = build_startup_config(deployment)
    case = build_qwen_case(config, "A coffee shop entrance.")
    legacy_config = dict(config)
    legacy_config.pop("offload_resident_blocks")
    legacy_config["use_event_offload"] = False
    model = QwenImageTransformerModel(str(Path(args.model_path) / "transformer"), legacy_config, torch.device("cpu"))
    weights = model.transformer_weights
    module = getattr(torch, AI_DEVICE)

    def run(infer):
        final_text = []
        original = infer.run_block

        def capture(index, block, *inputs):
            result = original(index, block, *inputs)
            if index == len(weights.blocks) - 1:
                final_text.append(result[0])
            return result

        infer.run_block = capture
        pre = case.pre
        try:
            image = infer.infer_with_blocks_offload(
                weights.blocks, pre.hidden_states.clone(), pre.encoder_hidden_states.clone(), pre.temb_img_silu, pre.temb_txt_silu, pre.image_rotary_emb, pre.image_rotary_positions, None
            )
            module.synchronize()
            return final_text[0], image
        finally:
            infer.run_block = original

    reference = run(model.transformer_infer)
    assert all(torch.isfinite(tensor).all() for tensor in reference)
    results = []
    total = len(weights.blocks)
    for count in (0, 1, total // 2, total - 1, total):
        infer = QwenImageOffloadTransformerInfer({**config, "offload_resident_blocks": count})
        manager = infer.offload_manager
        manager.init_cuda_buffer(weights.offload_block_cuda_buffers)
        if args.layout == "contiguous":
            manager.init_contiguous_group(weights.offload_group)
        try:
            for _ in range(2):
                actual = run(infer)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            results.append({"resident_blocks": count, "streamed_blocks": total - count, "indices": infer.resident_blocks.indices, "bitwise_equal": True})
        finally:
            infer.release_resident_blocks()
        assert not infer.resident_blocks.active and not infer.resident_blocks.originals
    report = {"layout": args.layout, "blocks": total, "grid": case.grid, "passes_per_count": 2, "checks": results}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
