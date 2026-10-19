"""Measure one model block and write a static resident-block count into a config.

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
from lightx2v.models.networks.qwen_image.infer.module_io import QwenPreInferModuleOutput
from lightx2v.models.networks.qwen_image.infer.offload.transformer_infer import validate_qwen_residency_config
from lightx2v.models.networks.qwen_image.infer.transformer_infer import QwenImageTransformerInfer
from lightx2v.models.networks.qwen_image.infer.utils import get_qwen_image_size
from lightx2v.models.networks.qwen_image.weights.transformer_weights import QwenImageTransformerAttentionBlock
from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.models.networks.wan.infer.offload.transformer_infer import validate_wan_residency_config
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.models.networks.wan.infer.utils import image_latent_size
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerAttentionBlock
from lightx2v.models.schedulers.qwen_image.scheduler import QwenEmbedRope
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


def load_block(config, checkpoint_path, block_class, prefix, sensitive_layers=(), *, block_kwargs=None, exclude_keys=()):
    """Load one CPU block and the same two device slots used by inference."""
    validate_contiguous_config(config)
    backend = get_block_offload_backend(required=False)
    if backend is not None:
        backend.prepare()
    checkpoint = SafetensorsCheckpoint(checkpoint_path)
    names = [name for name in checkpoint.weight_map if name.startswith(prefix) and not any(key in name for key in exclude_keys)]
    if not names:
        raise ValueError(f"Checkpoint contains no {prefix} weights")
    weights = checkpoint.load_tensors(names)
    metadata = {name: TensorMetadata(tuple(value.shape), value.dtype) for name, value in weights.items()}
    for name, value in weights.items():
        if value.dtype in (torch.float16, torch.bfloat16, torch.float32):
            dtype = GET_SENSITIVE_DTYPE() if any(key in name for key in sensitive_layers) else GET_DTYPE()
            weights[name] = value.to(dtype)
    kwargs = block_kwargs if block_kwargs is not None else dict(block_index=0, task=config["task"], mm_type=config.get("dit_quant_scheme", "Default"), config=config)
    block = block_class(**kwargs)
    slots = [block_class(**kwargs, create_cuda_buffer=True) for _ in range(2)]
    blocks = [block]
    group = OffloadGroup(blocks, slots, (prefix,))
    if config.get("cpu_offload_layout", "per_tensor") == "contiguous":
        prepare_contiguous_group(group, weights, metadata)
    for item in (*slots, block):
        item.load(dict(weights))
    del weights
    manager = EventSlotWeightAsyncStreamManager("block")
    manager.init_cuda_buffer(slots)
    if config.get("cpu_offload_layout", "per_tensor") == "contiguous":
        manager.init_contiguous_group(group)

    sources = block.state_dict()
    block_bytes = sum(value.numel() * value.element_size() for value in sources.values() if isinstance(value, torch.Tensor))
    if manager.contiguous_transfer is not None:
        block_bytes = block.block_buffer.layout.nbytes
    return blocks, manager, block_bytes


def build_wan_case(config, image_path=None):
    """Load only blocks.0 and use the production operators with synthetic inputs."""
    validate_wan_residency_config(config)
    checkpoint_key = "dit_quantized_ckpt" if config.get("dit_quantized", False) else "dit_original_ckpt"
    checkpoint_path = config.get(checkpoint_key) or config["model_path"]
    blocks, manager, block_bytes = load_block(config, checkpoint_path, WanTransformerAttentionBlock, "blocks.0.", ("norm", "embedding", "modulation", "time", "before_proj", "after_proj"))
    block = blocks[0]
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
    hidden = torch.empty_like(x)
    return SimpleNamespace(
        blocks=blocks,
        manager=manager,
        infer=infer,
        pre=pre,
        block_bytes=block_bytes,
        grid=grid,
        pixel_size=[latent_h * stride[1], latent_w * stride[2]],
        text_tokens=config["text_len"],
        compute=lambda: (infer.run_block(0, manager.cuda_buffers[0], hidden, pre),),
        reset=lambda: hidden.copy_(pre.x),
    )


def build_qwen_case(config, prompt, negative_prompt=" "):
    """Use real block weights, tokenizer lengths and production RoPE with synthetic activations."""
    from transformers import Qwen2Tokenizer

    validate_qwen_residency_config(config)
    size = get_qwen_image_size(config, config.get("size", []))
    if size is None:
        raise ValueError("Qwen measurement requires a valid aspect_ratio or size")
    width, height = size
    grid = (1, height // config["vae_scale_factor"] // 2, width // config["vae_scale_factor"] // 2)
    tokenizer = Qwen2Tokenizer.from_pretrained(config.get("qwen25vl_tokenizer_path", os.path.join(config["model_path"], "tokenizer")))
    prompts = [prompt, negative_prompt] if config["enable_cfg"] else [prompt]
    tokens = tokenizer([config["prompt_template_encode"].format(text) for text in prompts], add_special_tokens=True)["input_ids"]
    lengths = [len(ids) - config["prompt_template_encode_start_idx"] for ids in tokens]
    if min(lengths) <= 0 or max(lengths) > config.get("tokenizer_max_length", 4096):
        raise ValueError(f"Qwen text lengths outside the supported tokenizer range: {lengths}")
    # The shorter CFG branch generally has less computation available to hide transfers.
    text_tokens = min(lengths)
    checkpoint_key = "dit_quantized_ckpt" if config.get("dit_quantized", False) else "dit_original_ckpt"
    checkpoint_path = config.get(checkpoint_key) or config.get("transformer_model_path") or os.path.join(config["model_path"], "transformer")
    blocks, manager, block_bytes = load_block(config, checkpoint_path, QwenImageTransformerAttentionBlock, "transformer_blocks.0.")
    dim = config["num_attention_heads"] * config["attention_head_dim"]
    torch.manual_seed(42)
    image = torch.randn(math.prod(grid), dim, device=AI_DEVICE, dtype=GET_DTYPE()) * 0.05
    text = torch.randn(text_tokens, dim, device=AI_DEVICE, dtype=GET_DTYPE()) * 0.05
    temb = torch.randn(1, dim, device=AI_DEVICE, dtype=GET_DTYPE()) * 0.05
    rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
    sources = rope([[grid]], text_tokens, AI_DEVICE)
    operators = [blocks[0].compute_phases[i].rope for i in (0, 1)]
    freqs = tuple(op.prepare_freqs(source) for op, source in zip(operators, sources))
    positions = tuple(op.prepare_positions(freq) for op, freq in zip(operators, freqs))
    pre = QwenPreInferModuleOutput(image, text, temb, temb, freqs, positions)
    infer = QwenImageTransformerInfer(config)
    hidden, encoder_hidden = torch.empty_like(image), torch.empty_like(text)

    def reset():
        hidden.copy_(image)
        encoder_hidden.copy_(text)

    return SimpleNamespace(
        blocks=blocks,
        manager=manager,
        infer=infer,
        pre=pre,
        block_bytes=block_bytes,
        grid=grid,
        pixel_size=[height, width],
        text_tokens=text_tokens,
        text_lengths=lengths,
        compute=lambda: infer.run_block(0, manager.cuda_buffers[0], hidden, encoder_hidden, temb, temb, freqs, positions, None),
        reset=reset,
    )


def build_minimax_h3_case(config, prompt):
    """Measure the joint text/audio/video block using H3's packed layout and AdaLN cache."""
    from transformers import Qwen2TokenizerFast

    from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3PreInferOutput
    from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import validate_minimax_h3_residency_config
    from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer
    from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
    from lightx2v.models.networks.minimax_h3.packing import TEXT_TAG, align_num_frames
    from lightx2v.models.networks.minimax_h3.weights.transformer_weights import MiniMaxH3TransformerBlockWeights
    from lightx2v.models.schedulers.minimax_h3.scheduler import MiniMaxH3Scheduler

    validate_minimax_h3_residency_config(config)
    tokenizer_path = config.get("tokenizer_path") or os.path.join(config["model_path"], "tokenizer")
    tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
    text_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if text_tokens == 0:
        raise ValueError("MiniMax-H3 measurement requires a nonempty --prompt matching inference")
    height, width = config["size"]
    config["num_frames"] = align_num_frames(config["num_frames"])
    scheduler = MiniMaxH3Scheduler(config)
    scheduler.prepare(42, config["num_frames"], height, width, torch.full((text_tokens,), TEXT_TAG, dtype=torch.long))
    sample_step = min(1, scheduler.infer_steps - 1)
    scheduler.step_pre(sample_step)
    layout = scheduler.layout
    torch.manual_seed(42)
    source = torch.randn(layout.sequence_length, config["hidden_size"], device=AI_DEVICE, dtype=GET_DTYPE()) * 0.05
    pre_infer = MiniMaxH3PreInfer(config)
    pre = MiniMaxH3PreInferOutput(
        hidden_states=source,
        temb=None,
        timestep_indices=scheduler.timestep_indices,
        adaln_indices=scheduler.timestep_indices * 3 + layout.token_tags.clamp(min=0),
        rotary_emb=pre_infer._rotary_embedding(layout.position_ids),
        video_indices=layout.video_indices,
        audio_indices=layout.audio_indices,
        text_indices=layout.text_indices,
    )
    infer = MiniMaxH3TransformerInfer(config)
    infer.set_scheduler(scheduler)
    infer.block_idx = 0
    infer._prepare_adaln_cache(pre)
    checkpoint_path = config.get("dit_original_ckpt") or os.path.join(config["model_path"], "transformer")
    blocks, manager, block_bytes = load_block(
        config,
        checkpoint_path,
        MiniMaxH3TransformerBlockWeights,
        "transformer_blocks.0.",
        block_kwargs=dict(index=0, config=config),
        exclude_keys=(".adaln_proj.",),
    )
    hidden = torch.empty_like(source)
    patch = config["patch_size"]
    return SimpleNamespace(
        blocks=blocks,
        manager=manager,
        infer=infer,
        pre=pre,
        block_bytes=block_bytes,
        grid=(scheduler.num_latent_frames // patch[0], scheduler.latent_height // patch[1], scheduler.latent_width // patch[2]),
        pixel_size=[height, width],
        text_tokens=text_tokens,
        video_tokens=layout.video_indices.numel(),
        audio_tokens=layout.audio_indices.numel(),
        sample_step=sample_step,
        compute=lambda: (infer.run_block(0, manager.cuda_buffers[0], hidden, pre),),
        reset=lambda: hidden.copy_(source),
    )


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
    manager = case.manager
    torch_device_module.synchronize()
    d, transfer_samples, _ = measure_ms(lambda: manager.copy_block_to_buffer(manager.cuda_buffers[0], 0, case.blocks), manager.cuda_load_stream)
    torch_device_module.synchronize()
    c, compute_samples, result = measure_ms(case.compute, manager.compute_stream, prepare=case.reset)
    if not all(torch.isfinite(tensor).all() for tensor in result):
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
    parser.add_argument("--model-cls", choices=("wan2.1", "qwen_image", "minimax_h3"), default="wan2.1")
    parser.add_argument("--model-variant", choices=("fl2av",))
    parser.add_argument("--task", choices=("t2v", "i2v", "t2i", "t2av"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-path")
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--output-config", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output_config or args.config.with_name(args.config.stem + ".resident.json")
    deployment = json.loads(args.config.read_text())
    deployment.update(model_cls=args.model_cls, model_path=args.model_path, task=args.task, cpu_offload=True, offload_granularity="block", use_event_offload=True, offload_resident_blocks=0)
    if args.model_cls == "minimax_h3":
        if args.task != "t2av" or not args.prompt:
            parser.error("MiniMax-H3 measurement requires --task t2av and --prompt matching inference")
        deployment["model_variant"] = args.model_variant or deployment.get("model_variant", "fl2av")
    config = build_startup_config(deployment)
    if args.model_cls == "qwen_image":
        if args.prompt is None:
            parser.error("Qwen measurement requires --prompt matching inference")
        case = build_qwen_case(config, args.prompt, args.negative_prompt)
    elif args.model_cls == "minimax_h3":
        case = build_minimax_h3_case(config, args.prompt)
    else:
        case = build_wan_case(config, args.image_path)
    try:
        c, d, compute_samples, transfer_samples = measure_block(case)
        count = estimate_resident_count(config["num_layers"], c, d)
        deployment["offload_resident_blocks"] = count
        # Match normalized frame counts used to construct the measured input.
        if args.model_cls in ("wan2.1", "minimax_h3"):
            deployment["num_frames"] = config["num_frames"]
        report = {
            "model_cls": args.model_cls,
            "model_path": args.model_path,
            "device": torch_device_module.get_device_name(),
            "dtype": str(GET_DTYPE()),
            "quant_scheme": config.get("dit_quant_scheme", "Default"),
            "layout": config.get("cpu_offload_layout", "per_tensor"),
            "pixel_size": case.pixel_size,
            "block_grid": case.grid,
            "text_tokens": case.text_tokens,
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
        if args.model_cls == "qwen_image":
            report["text_lengths"] = case.text_lengths
        else:
            report["num_frames"] = config["num_frames"]
        if args.model_cls == "minimax_h3":
            report.update(model_variant=config["model_variant"], video_tokens=case.video_tokens, audio_tokens=case.audio_tokens, sample_step=case.sample_step)
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
