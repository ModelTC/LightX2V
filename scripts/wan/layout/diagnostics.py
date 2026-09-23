"""Short native denoise diagnostics; the scheduler keeps its full timestep grid."""

import csv
import json
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
from time import perf_counter
from unittest.mock import patch

import torch

from lightx2v.common.offload.timing import InferenceTimer, TransferTimer
from lightx2v.utils.utils import seed_all


def device_profiler(output, device, record_shapes=False):
    if device.type == "npu":
        import torch_npu

        profiler = torch_npu.profiler
        activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
        options = {"experimental_config": profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level1)}
    else:
        profiler = torch.profiler
        activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
        options = {}
    return profiler.profile(activities=activities, record_shapes=record_shapes, on_trace_ready=profiler.tensorboard_trace_handler(str(output)), **options)


def run_prefix(runner, request, count, step_context=None):
    """Reuse native input preparation and step functions without decoding a partial video."""
    if not 1 <= count <= runner.model.scheduler.infer_steps:
        raise ValueError("Diagnostic steps must be within the configured scheduler length")
    runner.input_info = runner.prepare_request(request)
    seed_all(runner.input_info.seed)
    runner.inputs = runner.run_input_encoder()
    runner.init_run()
    if runner.video_segment_num != 1:
        raise ValueError("Layout diagnostics require a single video segment")
    runner.init_run_segment(0)
    try:
        for index in range(count):
            with step_context(index) if step_context else nullcontext():
                runner.model.scheduler.step_pre(step_index=index)
                runner.model.infer(runner.inputs)
                runner.model.scheduler.step_post()
        return runner.model.scheduler.latents
    finally:
        del runner.inputs


@contextmanager
def instrument(runner, timer, transfers):
    """Scope diagnostic wrappers to this runner, restoring methods even on failure."""
    model = runner.model
    infer = model.transformer_infer
    manager = infer.offload_manager
    transfers.context = timer.context

    def timed(method, stage):
        @wraps(method)
        def call(*args, **kwargs):
            with timer.measure(stage):
                return method(*args, **kwargs)

        return call

    original_branch = model._infer_cond_uncond
    original_block = infer.run_block

    def branch(*args, **kwargs):
        previous = timer.context.copy()
        timer.context.update(branch="conditional" if kwargs["infer_condition"] else "unconditional", block_index=None)
        try:
            with timer.measure(timer.context["branch"]):
                return original_branch(*args, **kwargs)
        finally:
            timer.context.update(previous)

    def block(block_index, *args, **kwargs):
        timer.context["block_index"] = block_index
        with timer.measure("block_compute"):
            return original_block(block_index, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(model, "_infer_cond_uncond", branch))
        stack.enter_context(patch.object(infer, "run_block", block))
        for owner, name, label in (
            (model.scheduler, "step_pre", "scheduler_pre"),
            (model.scheduler, "step_post", "scheduler_post"),
            (infer, "infer_self_attn", "self_attention"),
            (infer, "infer_cross_attn", "cross_attention"),
            (infer, "infer_ffn", "ffn"),
            (model.pre_weight, "to_cuda", "pre_weights_h2d"),
            (model.pre_weight, "to_cpu", "pre_weights_offload"),
            (model.transformer_weights, "non_block_weights_to_cuda", "non_block_weights_h2d"),
            (model.transformer_weights, "non_block_weights_to_cpu", "non_block_weights_offload"),
        ):
            stack.enter_context(patch.object(owner, name, timed(getattr(owner, name), label)))
        stack.enter_context(patch.object(manager, "diagnostic_timer", timer))
        stack.enter_context(patch.object(manager, "transfer_timer", transfers))
        yield


def run_diagnosis(runner, request, metadata, args, output, device, module):
    manager = runner.model.transformer_infer.offload_manager
    blocks = runner.model.transformer_weights.blocks
    calls_per_step = len(blocks) * (2 if runner.config["enable_cfg"] else 1)
    timer = InferenceTimer(module, capacity=4 * calls_per_step + 32)
    transfers = TransferTimer(module, metadata, limit=calls_per_step + 1)
    steps, stages, loads = [], [], []

    @contextmanager
    def step_context(index):
        timer.context.update(step=index + 1, branch="", block_index=None)
        timer.enabled = index > 0
        timer.profiling = index == args.steps - 1
        # First-step loads are warmup too. No transfer events enter that step.
        manager.transfer_timer = transfers if timer.enabled else None
        prof = device_profiler(output / "inference_profiler", device, record_shapes=True) if timer.profiling else nullcontext()
        with prof:
            module.synchronize()
            start = perf_counter()
            yield
            module.synchronize()
            elapsed = perf_counter() - start
        step_stages = timer.collect()
        step_loads = transfers.collect()
        transfers.pending.clear()
        row = {
            "step": index + 1,
            "warmup": index == 0,
            "profiled": timer.profiling,
            "step_wall_s": elapsed,
            "h2d_bytes": sum(r["h2d_bytes"] for r in step_loads),
            "load_stream_ms": sum(r["load_stream_ms"] for r in step_loads),
            "block_compute_span_ms": sum(r["device_span_ms"] for r in step_stages if r["stage"] == "block_compute"),
            "wait_load_host_ms": sum(r["host_ms"] for r in step_stages if r["stage"] == "wait_load"),
            "wait_compute_host_ms": sum(r["host_ms"] for r in step_stages if r["stage"] == "wait_compute"),
        }
        steps.append(row)
        stages.extend(step_stages)
        loads.extend(step_loads)
        for filename, records in (("steps.csv", steps), ("stages.csv", stages), ("transfers.csv", loads)):
            if records:
                with (output / filename).open("w") as file:
                    writer = csv.DictWriter(file, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
        print(f"[{args.variant}][diagnose] " + json.dumps(row), flush=True)

    with instrument(runner, timer, transfers):
        latent = run_prefix(runner, request, args.steps, step_context).detach().cpu()
    torch.save(latent, output / "latents.pt")

    weight = manager.cuda_buffers[0].compute_phases[0].self_attn_q.weight
    runtime = {
        "scheduler_steps": runner.model.scheduler.infer_steps,
        "executed_steps": args.steps,
        "latent_shape": list(latent.shape),
        "query_tokens": int(runner.model.transformer_infer.self_attn_cu_seqlens_qkv[-1]),
        "weight_dtype": str(weight.dtype),
        "weight_shape": list(weight.shape),
        "weight_stride": list(weight.stride()),
        "weight_storage_offset": weight.storage_offset(),
        "torch_num_threads": torch.get_num_threads(),
    }
    if device.type == "npu":
        import torch_npu

        runtime["allow_internal_format"] = torch.npu.config.allow_internal_format
        runtime["weight_npu_format"] = torch_npu.get_npu_format(weight)
        runtime["jit_compile_false"] = torch.npu.is_jit_compile_false()
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")

    # Reuse the actual typed, already-converted model weights for isolated H2D.
    # Slots are no longer consumed by inference after this traversal.
    transfer_timer = TransferTimer(module, metadata, limit=len(blocks))
    module.synchronize()
    start = perf_counter()
    with patch.object(manager, "transfer_timer", transfer_timer):
        for index in range(len(blocks)):
            manager.prefetch_weights(index, blocks)
        module.synchronize()
    traversal_s = perf_counter() - start
    copy_rows = transfer_timer.collect()
    with (output / "native_copy.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(copy_rows[0]))
        writer.writeheader()
        writer.writerows(copy_rows)
    copied_bytes = sum(row["h2d_bytes"] for row in copy_rows)
    return {
        "kind": "native_diagnosis",
        "scheduler_steps": runner.model.scheduler.infer_steps,
        "executed_steps": args.steps,
        "steps": steps,
        "step_wall_s": [r["step_wall_s"] for r in steps if not r["warmup"] and not r["profiled"]],
        "native_copy": {"bytes": copied_bytes, "traversal_s": traversal_s, "effective_GBps": copied_bytes / traversal_s / 1e9},
        "runtime_pack": None,
        "timing_note": "Diagnostic spans include host submission gaps. Nested and overlapping spans and host waits must not be added. Profiled and warmup steps are excluded from step_wall_s.",
    }, loads
