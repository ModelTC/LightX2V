"""Compare real Qwen 2.1 prefill KV and denoising against legacy block offload."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from lightx2v.common.kvcache.static import StaticKVCachePool
from lightx2v.models.networks.qwen_image_21.infer.offload.transformer_infer import QwenImage21OffloadTransformerInfer
from lightx2v.models.networks.qwen_image_21.model import QwenImage21TransformerModel
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.set_config import build_startup_config
from lightx2v_platform.base.global_var import AI_DEVICE
from tools.estimate_offload_residency import build_qwen_image_21_case, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layout", choices=("per_tensor", "contiguous"), default="per_tensor")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    deployment = json.loads(Path(args.config).read_text())
    deployment.update(model_cls="qwen_image_21", model_path=args.model_path, task="t2i", size=[128, 128], resolution=128, cpu_offload_layout=args.layout)
    config = build_startup_config(deployment)
    legacy = dict(config)
    legacy.pop("offload_resident_blocks")
    legacy["use_event_offload"] = False
    model = QwenImage21TransformerModel(str(Path(args.model_path) / "transformer"), legacy, torch.device("cpu"))
    weights = model.transformer_weights
    module = getattr(torch, AI_DEVICE)
    total = len(weights.blocks)
    results = []
    for task in ("t2i", "i2i"):
        case = build_qwen_image_21_case({**config, "task": task}, "A boat sailing on a calm blue sea.", args.image_path)
        target = case.pre
        layout = target.layout
        prefix = replace(
            target,
            hidden_states=torch.randn(layout.prefix_len, target.hidden_states.shape[-1], device=AI_DEVICE, dtype=GET_DTYPE()) * 0.05,
            rotary=layout.rotary[: layout.prefix_len],
            rotary_positions=layout.rotary_positions[: layout.prefix_len] if layout.rotary_positions is not None else None,
        )

        def run(infer):
            cache = StaticKVCachePool(total, layout.prefix_len, config["num_attention_heads"], config["attention_head_dim"], GET_DTYPE(), torch.device(AI_DEVICE))
            infer.prefill(weights, prefix, cache)
            outputs = [infer.infer(weights, replace(target, modulation=target.modulation * scale), cache) for scale in (1, 0.5)]
            module.synchronize()
            assert all(torch.isfinite(tensor).all() for tensor in outputs)
            return cache, outputs

        # Other managers share these slots; legacy's cached block-0 assumption
        # no longer holds when switching back for the next task's reference.
        model.transformer_infer.offload_manager.need_init_first_buffer = True
        reference_kv, reference = run(model.transformer_infer)
        for count in (0, 1, total // 2, total - 1, total):
            infer = QwenImage21OffloadTransformerInfer({**config, "offload_resident_blocks": count})
            manager = infer.offload_manager
            manager.init_cuda_buffer(weights.offload_block_cuda_buffers)
            if args.layout == "contiguous":
                manager.init_contiguous_group(weights.offload_group)
            try:
                for _ in range(2):
                    cache, outputs = run(infer)
                    for index in range(total):
                        torch.testing.assert_close(cache.k_cache(index), reference_kv.k_cache(index), rtol=0, atol=0)
                        torch.testing.assert_close(cache.v_cache(index), reference_kv.v_cache(index), rtol=0, atol=0)
                    for actual, expected in zip(outputs, reference):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    infer.release_resident_blocks()
                    assert not infer.resident_blocks.active and not infer.resident_blocks.originals
                results.append(
                    {"task": task, "resident_blocks": count, "indices": infer.resident_blocks.indices, "prefix_tokens": layout.prefix_len, "target_tokens": layout.target_len, "bitwise_equal": True}
                )
            finally:
                infer.release_resident_blocks()
                if manager.contiguous_transfer is not None:
                    manager.contiguous_transfer.close()
        if case.manager.contiguous_transfer is not None:
            case.manager.contiguous_transfer.close()
    report = {"layout": args.layout, "blocks": total, "requests_per_case": 2, "denoise_steps_per_request": 2, "checks": results}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
