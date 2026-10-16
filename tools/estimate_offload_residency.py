"""Measure one Wan block and write a static resident-block count into a config.

Run with python -m tools.estimate_offload_residency --help.
"""

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from lightx2v.common.offload.block_loader import OffloadGroup, prepare_contiguous_group, validate_contiguous_config
from lightx2v.common.offload.event_manager import EventSlotWeightAsyncStreamManager
from lightx2v.common.offload.resident import uniform_resident_indices
from lightx2v.common.offload.safetensors_checkpoint import SafetensorsCheckpoint
from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.models.networks.wan.infer.offload.transformer_infer import validate_wan_residency_config
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.models.networks.wan.infer.utils import image_latent_size
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerAttentionBlock
from lightx2v.models.schedulers.wan.scheduler_factory import create_wan_scheduler
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v.utils.set_config import build_startup_config
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.ops.offload import get_block_offload_backend
from lightx2v_platform.ops.offload.weight_storage import TensorMetadata

WARMUP = 3
REPEATS = 10
torch_device_module = getattr(torch, AI_DEVICE)


def estimate_resident_count(num_blocks, compute_ms, transfer_ms):
    if type(num_blocks) is not int or num_blocks <= 0:
        raise ValueError("num_blocks must be a positive integer")
    if not all(math.isfinite(value) and value > 0 for value in (compute_ms, transfer_ms)):
        raise ValueError("Block computation and transfer times must be finite and positive")
    if transfer_ms <= compute_ms:
        return 0
    return num_blocks - math.floor(num_blocks * compute_ms / transfer_ms)


def build_wan_case(config, image_path=None):
    """Load only blocks.0 and use the production operators with synthetic inputs."""
    validate_wan_residency_config(config)
    validate_contiguous_config(config)
    backend = get_block_offload_backend(required=False)
    if backend is not None:
        backend.prepare()
    checkpoint_key = "dit_quantized_ckpt" if config.get("dit_quantized", False) else "dit_original_ckpt"
    checkpoint_path = config.get(checkpoint_key) or config["model_path"]
    checkpoint = SafetensorsCheckpoint(checkpoint_path)
    names = [name for name in checkpoint.weight_map if name.startswith("blocks.0.")]
    if not names:
        raise ValueError("Checkpoint contains no Wan blocks.0 weights")
    weights = checkpoint.load_tensors(names)
    metadata = {name: TensorMetadata(tuple(value.shape), value.dtype) for name, value in weights.items()}
    for name, value in weights.items():
        if value.dtype in (torch.float16, torch.bfloat16, torch.float32):
            dtype = GET_SENSITIVE_DTYPE() if any(key in name for key in ("norm", "embedding", "modulation", "time", "before_proj", "after_proj")) else GET_DTYPE()
            weights[name] = value.to(dtype)
    kwargs = dict(block_index=0, task=config["task"], mm_type=config.get("dit_quant_scheme", "Default"), config=config)
    block = WanTransformerAttentionBlock(**kwargs)
    slots = [WanTransformerAttentionBlock(**kwargs, create_cuda_buffer=True) for _ in range(2)]
    blocks = [block]
    group = OffloadGroup(blocks, slots, ("blocks.0.",))
    if config.get("cpu_offload_layout", "per_tensor") == "contiguous":
        prepare_contiguous_group(group, weights, metadata)
    for item in (*slots, block):
        item.load(dict(weights))
    del weights
    manager = EventSlotWeightAsyncStreamManager("block")
    manager.init_cuda_buffer(slots)
    if config.get("cpu_offload_layout", "per_tensor") == "contiguous":
        manager.init_contiguous_group(group)

    stride, patch = config["vae_stride"], config["patch_size"]
    if config["task"] == "i2v":
        if image_path is None:
            raise ValueError("I2V measurement needs --image-path to match the inference aspect ratio")
        with Image.open(image_path) as image:
            latent_h, latent_w = image_latent_size((image.height, image.width), config["size"], stride, patch)
    else:
        latent_h, latent_w = config["size"][0] // stride[1], config["size"][1] // stride[2]
    latent_t = (config["num_frames"] - 1) // stride[0] + 1
    grid = (latent_t // patch[0], latent_h // patch[1], latent_w // patch[2])
    if min(grid) <= 0:
        raise ValueError(f"Input dimensions produce an empty block grid: {grid}")
    dim, dtype = config["dim"], GET_DTYPE()
    torch.manual_seed(42)
    x = torch.randn(math.prod(grid), dim, device=AI_DEVICE, dtype=dtype) * 0.05
    context_len = config["text_len"] + (257 if config["task"] == "i2v" and config.get("use_image_encoder", True) else 0)
    context = torch.randn(context_len, dim, device=AI_DEVICE, dtype=dtype) * 0.05
    pre = WanPreInferModuleOutput(
        embed=torch.zeros(1, dim, device=AI_DEVICE, dtype=GET_SENSITIVE_DTYPE()),
        grid_sizes=GridOutput(torch.tensor([grid], device=AI_DEVICE), grid),
        x=x,
        embed0=torch.randn(1, 6, dim, device=AI_DEVICE, dtype=GET_SENSITIVE_DTYPE()) * 0.05,
        context=context,
    )
    pre_infer = WanPreInfer(config)
    pre_infer.set_rope(block.compute_phases[0].rope)
    pre.cos_sin = pre_infer.prepare_rope_cache(pre_infer.prepare_cos_sin(grid, pre_infer.freqs))
    pre.rope_positions = pre_infer.rope_positions
    infer = WanTransformerInfer(config)
    infer.block_idx = 0
    infer.set_scheduler(create_wan_scheduler(config))
    infer.scheduler.step_index = 0
    infer.scheduler.infer_condition = True
    infer.cos_sin, infer.rope_positions = pre.cos_sin, pre.rope_positions
    infer.reset_infer_states(x, context)
    if infer._sol_global_morton_enabled:
        raise ValueError("The single-block estimator does not implement Sol-Attn Morton input reordering")
    sources = block.state_dict()
    block_bytes = sum(value.numel() * value.element_size() for value in sources.values() if isinstance(value, torch.Tensor))
    if manager.contiguous_transfer is not None:
        block_bytes = block.block_buffer.layout.nbytes
    return SimpleNamespace(blocks=blocks, manager=manager, infer=infer, pre=pre, block_bytes=block_bytes, grid=grid, pixel_size=[latent_h * stride[1], latent_w * stride[2]])


def measure_ms(action, stream, prepare=None):
    samples = []
    result = None
    start = torch_device_module.Event(enable_timing=True)
    end = torch_device_module.Event(enable_timing=True)
    for iteration in range(WARMUP + REPEATS):
        with torch_device_module.stream(stream):
            if prepare is not None:
                prepare()
            start.record(stream)
            result = action()
            end.record(stream)
        end.synchronize()
        if iteration >= WARMUP:
            samples.append(start.elapsed_time(end))
    return statistics.median(samples), samples, result


def measure_block(case):
    manager, pre = case.manager, case.pre
    torch_device_module.synchronize()
    d, transfer_samples, _ = measure_ms(lambda: manager.copy_block_to_buffer(manager.cuda_buffers[0], 0, case.blocks), manager.cuda_load_stream)
    hidden = torch.empty_like(pre.x)
    torch_device_module.synchronize()
    c, compute_samples, result = measure_ms(
        lambda: case.infer.run_block(0, manager.cuda_buffers[0], hidden, pre),
        manager.compute_stream,
        prepare=lambda: hidden.copy_(pre.x),
    )
    if not torch.isfinite(result).all():
        raise RuntimeError("The representative block produced non-finite output")
    return c, d, compute_samples, transfer_samples


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False)
    try:
        with temporary:
            json.dump(content, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
        os.replace(temporary.name, path)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cls", choices=("wan2.1",), default="wan2.1")
    parser.add_argument("--task", choices=("t2v", "i2v"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-path")
    parser.add_argument("--output-config", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output_config or args.config.with_name(args.config.stem + ".resident.json")
    deployment = json.loads(args.config.read_text())
    deployment.update(model_cls=args.model_cls, model_path=args.model_path, task=args.task, cpu_offload=True, offload_granularity="block", use_event_offload=True, offload_resident_blocks=0)
    config = build_startup_config(deployment)
    case = build_wan_case(config, args.image_path)
    try:
        c, d, compute_samples, transfer_samples = measure_block(case)
        count = estimate_resident_count(config["num_layers"], c, d)
        deployment["offload_resident_blocks"] = count
        # Match normalized frame counts used to construct the measured input.
        deployment["num_frames"] = config["num_frames"]
        report = {
            "model_cls": args.model_cls,
            "model_path": args.model_path,
            "device": torch_device_module.get_device_name(),
            "dtype": str(GET_DTYPE()),
            "quant_scheme": config.get("dit_quant_scheme", "Default"),
            "layout": config.get("cpu_offload_layout", "per_tensor"),
            "num_frames": config["num_frames"],
            "pixel_size": case.pixel_size,
            "block_grid": case.grid,
            "text_tokens": config["text_len"],
            "num_blocks": config["num_layers"],
            "compute_ms": c,
            "transfer_ms": d,
            "resident_blocks": count,
            "streamed_blocks": config["num_layers"] - count,
            "resident_indices": uniform_resident_indices(config["num_layers"], count),
            "resident_weight_bytes": count * case.block_bytes,
            "execution_buffer_bytes": 2 * case.block_bytes,
            "warmup": WARMUP,
            "repeats": REPEATS,
            "compute_samples_ms": compute_samples,
            "transfer_samples_ms": transfer_samples,
            "elapsed_seconds": time.perf_counter() - started,
            "output_config": str(output),
        }
        write_json(output.with_suffix(".profile.json"), report)
        write_json(output, deployment)
        print(json.dumps(report, indent=2))
    finally:
        case.manager.compute_stream.synchronize()
        case.manager.cuda_load_stream.synchronize()
        if case.manager.contiguous_transfer is not None:
            case.manager.contiguous_transfer.close()


if __name__ == "__main__":
    main()
