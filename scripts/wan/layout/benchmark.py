"""Standalone Wan layout comparison. See README.md for checkpoint requirements."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def stats(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "samples": values}


def device_setup():
    backend = os.environ.get("PLATFORM", "cuda")
    if backend == "ascend_npu":
        import torch_npu  # noqa: F401

        device = torch.device("npu:0")
        # Both variants use ND storage; private formats can invalidate byte views.
        torch.npu.config.allow_internal_format = False
    elif backend == "cuda":
        device = torch.device("cuda:0")
    else:
        raise ValueError("This benchmark supports PLATFORM=cuda or ascend_npu")
    module = getattr(torch, device.type)
    module.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run this benchmark with one device and one process, without torchrun")
    return device, module


def preflight(device, module):
    from lightx2v.common.offload.block_layout import BlockBuffer, BlockLayout

    layout = BlockLayout.build([(("a", "weight"), "a", (4, 6), torch.bfloat16, True), (("b", "scale"), "b", (4,), torch.float32, False), (("c", "weight"), "c", (4, 6), torch.int8, True)])
    source, target = BlockBuffer(layout, "cpu"), BlockBuffer(layout, device)
    source.storage.fill_(1)
    if not source.storage.is_pinned():
        raise RuntimeError("The active backend did not allocate pinned host memory")
    stream = module.Stream()
    stream.wait_stream(module.current_stream())
    with module.stream(stream):
        target.storage.copy_(source.storage, non_blocking=True)
        target.storage.record_stream(stream)
    stream.synchronize()
    for spec in layout.tensors:
        actual = spec.view(target.storage, operator=True)
        expected = spec.view(source.storage, operator=True)
        if actual.untyped_storage().data_ptr() != target.storage.data_ptr() or not torch.equal(actual.cpu(), expected):
            raise RuntimeError(f"Backend failed mixed-dtype block view validation: {spec.name}")


def metadata_for_blocks(blocks, contiguous):
    rows = []
    for block in blocks:
        state = [tensor for tensor in block.state_dict().values() if tensor is not None]
        host = [tensor for tensor in state if tensor.device.type == "cpu"]
        rows.append(
            {
                "h2d_bytes": block.block_buffer.layout.nbytes if contiguous else sum(t.numel() * t.element_size() for t in host),
                "logical_bytes": sum(t.numel() * t.element_size() for t in host),
                "h2d_copy_calls": 1 if contiguous else len(host),
                "host_tensors": len(host),
                "pinned_tensors": sum(t.is_pinned() for t in host),
                "d2d_tensors": len(state) - len(host),
            }
        )
    return rows


def profile_copies(callback, output, device):
    from scripts.wan.layout.diagnostics import device_profiler

    with device_profiler(output, device):
        callback()
        getattr(torch, device.type).synchronize()


def checkpoint_blocks(path):
    """Index only transformer blocks; do not load encoders or non-block weights."""
    from safetensors import safe_open

    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    blocks = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                match = re.fullmatch(r"blocks\.(\d+)\.(.+)", name)
                if match:
                    block = blocks.setdefault(int(match[1]), {})
                    if name in block:
                        raise ValueError(f"Duplicate checkpoint weight: {name}")
                    block[name] = file
    if not blocks:
        raise ValueError(f"No blocks.<index> weights found in {path}")
    return blocks


def validate_npu_checkpoint(path, scheme):
    from safetensors import safe_open

    by_file = {}
    for entries in checkpoint_blocks(Path(path)).values():
        for name, file in entries.items():
            by_file.setdefault(file, []).append(name)
    expected = ("I8",) if scheme == "int8-npu" else ("BF16", "F16", "F32")
    for file, names in by_file.items():
        with safe_open(file, framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_slice(name)
                if name.endswith(".weight") and len(tensor.get_shape()) == 2 and tensor.get_dtype() not in expected:
                    raise ValueError(f"{scheme} requires {expected} matrix weights; {name} in {file} is {tensor.get_dtype()}. No implicit FP8 conversion is performed.")


def run_copy(args, output, device, module):
    from safetensors import safe_open

    from lightx2v.common.offload.block_layout import BlockBuffer, BlockLayout, ContiguousBlockTransfer
    from lightx2v.common.offload.timing import TransferTimer

    start = perf_counter()
    blocks, sources, metadata, manifest = [], [], [], []
    contiguous = args.variant == "solution2"
    layout = None
    for index, entries in sorted(checkpoint_blocks(Path(args.checkpoint)).items()):
        weights = {}
        for file in set(entries.values()):
            with safe_open(file, framework="pt", device="cpu") as handle:
                weights.update({name: handle.get_tensor(name) for name, location in entries.items() if location == file})
        fp8 = any(t.dtype == torch.float8_e4m3fn for t in weights.values())
        # Match native FP8 scale conversion, which may lose its initial pinning.
        pinned = {name: not (fp8 and name.endswith("weight_scale") and tensor.dtype != torch.float32) for name, tensor in weights.items()}
        weights = {name: tensor.float() if name.endswith("weight_scale") else tensor for name, tensor in weights.items()}
        current = BlockLayout.build([((name.split(".", 2)[2], "tensor"), name, tuple(t.shape), t.dtype, False) for name, t in weights.items()])
        if layout is not None and layout != current:
            raise ValueError("Copy benchmark requires identical block layouts")
        layout = current
        if contiguous:
            buffer = BlockBuffer(layout, "cpu")
            for name, tensor in weights.items():
                buffer.views[name].copy_(tensor)
            blocks.append(SimpleNamespace(block_buffer=buffer, block_auxiliary={}))
            host = [buffer.storage]
        else:
            host = []
            for spec in layout.tensors:
                tensor = torch.empty(spec.shape, dtype=spec.dtype, pin_memory=pinned[spec.name])
                tensor.copy_(weights[spec.name])
                host.append(tensor.view(torch.uint8).reshape(-1))
        sources.append(host)
        metadata.append(
            {
                "h2d_bytes": sum(t.numel() for t in host),
                "logical_bytes": sum(s.nbytes for s in layout.tensors),
                "h2d_copy_calls": len(host),
                "host_tensors": len(layout.tensors),
                "pinned_tensors": len(layout.tensors) if contiguous else sum(pinned.values()),
                "d2d_tensors": 0,
            }
        )
        manifest.append({"checkpoint_block": index, "tensors": [{"name": s.name, "shape": s.shape, "dtype": str(s.dtype), "nbytes": s.nbytes} for s in layout.tensors]})
        del weights

    if contiguous:
        slots = [SimpleNamespace(block_buffer=SimpleNamespace(layout=layout, storage=torch.empty(layout.nbytes, dtype=torch.uint8, device=device)), block_auxiliary={}) for _ in range(2)]
        transfer = ContiguousBlockTransfer(blocks, slots)

        def copy(index):
            transfer.copy(index, slots[index % 2])
    else:
        slots = [[torch.empty(t.shape, dtype=torch.uint8, device=device) for t in sources[0]] for _ in range(2)]

        def copy(index):
            for target, source in zip(slots[index % 2], sources[index]):
                target.copy_(source, non_blocking=True)

    stream = module.Stream()
    stream.wait_stream(module.current_stream())
    module.synchronize()
    initialization_s = perf_counter() - start
    versions = [[t._version for t in block] for block in sources]
    for index, host in enumerate(sources):
        with module.stream(stream):
            copy(index)
        stream.synchronize()
        targets = [slots[index % 2].block_buffer.storage] if contiguous else slots[index % 2]
        if not all(torch.equal(dst.cpu(), src) for dst, src in zip(targets, host)):
            raise AssertionError(f"Transfer corrupted checkpoint block {index}")

    def traversal(timer=None):
        with module.stream(stream):
            for index in range(len(sources)):
                ticket = timer.start(index, "copy", stream) if timer else None
                copy(index)
                if timer:
                    timer.stop(ticket)
        stream.synchronize()

    for _ in range(args.warmup):
        traversal()
    durations = []
    for _ in range(args.repeats):
        start = perf_counter()
        traversal()
        durations.append(perf_counter() - start)
    timer = TransferTimer(module, metadata, limit=len(sources))
    traversal(timer)
    rows = timer.collect()
    if args.profile:
        profile_copies(traversal, output / "profiler", device)
    if versions != [[t._version for t in block] for block in sources]:
        raise AssertionError("CPU weights changed during measurement")
    if contiguous:
        transfer.close()
    write_json(output / "weights.json", manifest)
    return {
        "kind": "byte_layout_microbenchmark",
        "initialization_s": initialization_s,
        "traversal_s": stats(durations),
        "block_count": len(sources),
        "transfer_verified": True,
        "runtime_pack": None,
    }, rows


def run_infer(args, output, device, module):
    from lightx2v.common.offload.timing import TransferTimer
    from lightx2v.models.runners.runner_factory import build_runner
    from lightx2v.utils.set_config import build_startup_config
    from lightx2v.utils.utils import validate_config_paths

    config_path = args.baseline_config if args.variant == "baseline" else args.solution2_config
    config = build_startup_config({"model_cls": "wan2.1", "task": "i2v", "model_path": args.model_path, "config_json": config_path})
    scheme = config.get("dit_quant_scheme", "Default")
    if device.type == "npu" and (scheme not in ("Default", "int8-npu") or config.get("t5_quantized") or config.get("clip_quantized")):
        raise ValueError("NPU inference requires BF16 or int8-npu DiT and unquantized T5/CLIP; use the NPU configs and matching checkpoints")
    if device.type == "npu":
        validate_npu_checkpoint(config.get("dit_quantized_ckpt") if config.get("dit_quantized") else config.get("dit_original_ckpt") or args.model_path, scheme)
    if not config.get("cpu_offload") or config.get("offload_granularity") != "block" or config.get("lazy_load") or config.get("unload_modules") or config.get("parallel"):
        raise ValueError("Benchmark requires persistent, single-device CPU block offload")
    if args.mode == "diagnose" and not 3 <= args.steps <= config["infer_steps"]:
        raise ValueError("diagnose requires 3 <= --steps <= configured infer_steps")
    validate_config_paths(config)
    write_json(output / "config.json", config)
    module.synchronize()
    start = perf_counter()
    runner = build_runner(config)
    module.synchronize()
    initialization_s = perf_counter() - start
    manager = runner.model.transformer_infer.offload_manager
    blocks = runner.model.transformer_weights.blocks
    metadata = metadata_for_blocks(blocks, args.variant == "solution2")
    host_tensors = [tensor for block in blocks for tensor in block.state_dict().values() if tensor is not None and tensor.device.type == "cpu"]
    host_versions = [(tensor.data_ptr(), tensor._version) for tensor in host_tensors]
    write_json(output / "blocks.json", metadata)
    request = {"task": "i2v", "seed": args.seed, "prompt": args.prompt, "negative_prompt": args.negative_prompt, "image_path": args.image_path}
    if args.mode == "diagnose":
        from scripts.wan.layout.diagnostics import run_diagnosis

        result, rows = run_diagnosis(runner, request, metadata, args, output, device, module)
        result["step_wall_s"] = stats(result["step_wall_s"])
        result.update(initialization_s=initialization_s, precision=scheme)
        if host_versions != [(tensor.data_ptr(), tensor._version) for tensor in host_tensors]:
            raise AssertionError("CPU weights changed during diagnostics")
        return result, rows
    denoise_times, retained_latents = [], []
    original = runner.run_segment

    def timed_segment(*positional, **keywords):
        module.synchronize()
        start = perf_counter()
        result = original(*positional, **keywords)
        module.synchronize()
        denoise_times.append(perf_counter() - start)
        retained_latents[:] = [result]
        return result

    runner.run_segment = timed_segment

    def request_once():
        input_info = runner.prepare_request(request)
        denoise_times.clear()
        retained_latents.clear()
        module.synchronize()
        start = perf_counter()
        runner.run_request(input_info)
        module.synchronize()
        return perf_counter() - start, sum(denoise_times)

    for _ in range(args.warmup):
        print(f"[{args.variant}][warmup] request {_ + 1}/{args.warmup}", flush=True)
        request_once()
    requests, denoises = [], []
    for _ in range(args.repeats):
        print(f"[{args.variant}][measure] request {_ + 1}/{args.repeats}", flush=True)
        request_s, denoise_s = request_once()
        requests.append(request_s)
        denoises.append(denoise_s)
        write_json(output / "measurements.json", {"request_s": requests, "denoise_s": denoises})
    # Saving and comparing latents is outside all measured intervals.
    latent = retained_latents[0].detach().cpu()
    torch.save(latent, output / "latents.pt")
    digest = hashlib.sha256(latent.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    timer = TransferTimer(module, metadata, args.samples)
    manager.transfer_timer = timer
    try:
        from scripts.wan.layout.diagnostics import run_prefix

        calls_per_step = len(blocks) * (2 if config["enable_cfg"] else 1)
        sample_steps = min(runner.model.scheduler.infer_steps, math.ceil(args.samples / calls_per_step))
        print(f"[{args.variant}][sample] {sample_steps} steps, at most {args.samples} block loads", flush=True)
        run_prefix(runner, request, sample_steps)
        module.synchronize()
    finally:
        manager.transfer_timer = None
    rows = timer.collect()
    if args.profile:
        # Profile a bounded traversal of the actual native loading path.
        def prefetch():
            for index in range(len(blocks)):
                manager.prefetch_weights(index, blocks)

        profile_copies(prefetch, output / "profiler", device)
    if host_versions != [(tensor.data_ptr(), tensor._version) for tensor in host_tensors]:
        raise AssertionError("CPU weights changed during inference")
    return {
        "kind": "native_inference",
        "precision": scheme,
        "initialization_s": initialization_s,
        "request_s": stats(requests),
        "denoise_s": stats(denoises),
        "latent_sha256": digest,
        "runtime_pack": None,
    }, rows


def main():
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("copy", "infer", "diagnose"))
    parser.add_argument("--checkpoint", help="Wan block safetensors file/directory, required for copy")
    parser.add_argument("--model-path", help="Full Wan2.1 I2V model directory, required for infer")
    parser.add_argument("--baseline-config", default=str(ROOT / "configs/wan/layout/baseline_npu.json"))
    parser.add_argument("--solution2-config", default=str(ROOT / "configs/wan/layout/solution2_npu.json"))
    parser.add_argument("--image-path", default=str(ROOT / "assets/inputs/imgs/img_0.jpg"))
    parser.add_argument("--prompt", default="A white cat wearing sunglasses sits on a surfboard on a sunny beach.")
    parser.add_argument("--negative-prompt", default="过曝，静态，模糊，字幕，低质量，畸形")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=160, help="Maximum block loads sampled in a separate short inference prefix")
    parser.add_argument("--profile", action="store_true", help="Export a separate device-profiler copy trace")
    parser.add_argument("--steps", type=int, default=4, help="diagnose only: execute this prefix of the unchanged timestep grid; first step warms up, last step is profiled")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", choices=("baseline", "solution2"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.samples < 1:
        parser.error("repeats/samples must be positive and warmup must be nonnegative")
    if args.mode == "diagnose" and args.steps < 3:
        parser.error("diagnose needs at least 3 steps: warmup, unprofiled sampling, and profiling")
    if not (args.checkpoint if args.mode == "copy" else args.model_path):
        parser.error("copy requires --checkpoint; infer/diagnose require --model-path")
    os.environ["PROFILING_DEBUG_LEVEL"] = "0"
    os.environ.setdefault("DTYPE", "BF16")
    os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "None")
    if args.output is None:
        args.output = ROOT / "save_results/wan_layout" / datetime.now(timezone.utc).strftime("benchmark_%Y%m%d_%H%M%S_%f")
    args.output = args.output.resolve()
    if args.variant:
        output = args.output / args.variant
        output.mkdir()
        device, module = device_setup()
        preflight(device, module)
        module.synchronize()
        with torch.no_grad():
            result, rows = (run_copy if args.mode == "copy" else run_infer)(args, output, device, module)
        result.update(
            {
                "variant": args.variant,
                "device": str(device),
                "device_name": module.get_device_name(device),
                "torch": torch.__version__,
                "compute_dtype": os.environ["DTYPE"],
                "warmup": 1 if args.mode == "diagnose" else args.warmup,
                "repeats": None if args.mode == "diagnose" else args.repeats,
                "sampled_load_stream_ms": sum(row["load_stream_ms"] for row in rows),
                "sampled_blocks": len(rows),
                "dma_ms": None,
            }
        )
        if device.type == "npu":
            import torch_npu

            result["torch_npu"] = torch_npu.__version__
            if shutil.which("npu-smi"):
                info = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15)
                (output / "npu-smi.txt").write_text(info.stdout + info.stderr)
            toolkit = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
            result["cann"] = {str(path): path.read_text() for path in toolkit.glob("*/ascend_toolkit_install.info")}
        write_json(output / "result.json", result)
        write_csv(output / "transfers.csv", rows)
        return

    if args.mode != "copy":
        baseline = json.loads(Path(args.baseline_config).read_text())
        solution = json.loads(Path(args.solution2_config).read_text())
        if baseline.pop("cpu_offload_layout", "per_tensor") != "per_tensor" or solution.pop("cpu_offload_layout", None) != "contiguous" or baseline != solution:
            raise ValueError("The two configs must differ only in cpu_offload_layout")
    args.output.mkdir(parents=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    changes = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    (args.output / "worktree.txt").write_text(changes)
    write_json(
        args.output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "revision": revision,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "environment": {k: v for k, v in os.environ.items() if k.startswith(("ASCEND", "CANN", "OMP", "MKL", "TASK_QUEUE", "PLATFORM", "DTYPE", "SENSITIVE", "PROFILING"))},
        },
    )
    results = []
    for variant in ("baseline", "solution2"):
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--output", str(args.output), "--variant", variant]
        print(f"Running {variant}; log: {args.output / (variant + '.log')}", flush=True)
        with (args.output / f"{variant}.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f"{variant} failed; see {args.output / (variant + '.log')}")
        results.append(json.loads((args.output / variant / "result.json").read_text()))
    summary = {"results": results, "dma_note": "load_stream_ms includes submission gaps and any D2D; pure DMA durations require the optional profiler trace"}
    metric = {"infer": "denoise_s", "copy": "traversal_s", "diagnose": "step_wall_s"}[args.mode]
    summary["speedup"] = results[0][metric]["median"] / results[1][metric]["median"]
    if args.mode != "copy":
        a = torch.load(args.output / "baseline/latents.pt", weights_only=True)
        b = torch.load(args.output / "solution2/latents.pt", weights_only=True)
        summary["latents_equal"] = torch.equal(a, b)
        summary["latent_max_abs_error"] = (a.float() - b.float()).abs().max().item()
        summary["latents_close"] = torch.allclose(a.float(), b.float(), rtol=1e-2, atol=1e-2)
    write_json(args.output / "summary.json", summary)
    write_csv(
        args.output / "summary.csv",
        [
            {
                "variant": r["variant"],
                "metric": metric,
                "median_s": r[metric]["median"],
                "min_s": r[metric]["min"],
                "max_s": r[metric]["max"],
                "sampled_load_stream_ms": r["sampled_load_stream_ms"],
                "sampled_blocks": r["sampled_blocks"],
            }
            for r in results
        ],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results: {args.output}")
    if args.mode != "copy" and not summary["latents_close"]:
        raise AssertionError("Baseline and solution2 latents differ beyond rtol=atol=0.01; inspect summary.json")


if __name__ == "__main__":
    main()
