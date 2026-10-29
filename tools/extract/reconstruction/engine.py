"""DDP reconstruction with bounded GPU residency and rolling disk caches."""

import gc
import json
import math
import os
import random
import shutil
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from ..export_student import ModelWeights, torch_dtype
from ..lora_extractor import ExtractionWeights, decompose_to_lora, select_layers
from .calibration import calibration_files
from .qwen_image_21 import TEMPORAL_LAYERS, build_unit, move, prepare_record, read_architecture, reference, trace_records, update_factors


class Runtime:
    def __init__(self, device):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device == "auto":
            device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.world > 1 and self.device.type == "cuda":
            if self.device.index is not None and self.device.index != local_rank:
                raise ValueError("DDP device must be auto or match LOCAL_RANK")
            self.device = torch.device("cuda", local_rank)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.owns_group = self.world > 1 and not dist.is_initialized()
        if self.owns_group:
            # Collection may have just destroyed its Gloo group in this same
            # torchrun launch. Isolate rendezvous keys so the new group cannot
            # consume stale keys from that phase, including on CPU/Gloo runs.
            store, rank, world = next(dist.rendezvous("env://"))
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo", store=dist.PrefixStore("lora_reconstruction/", store), rank=rank, world_size=world)
        if dist.is_initialized():
            self.rank, self.world = dist.get_rank(), dist.get_world_size()

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def mean(self, value):
        tensor = torch.tensor(value, device=self.device, dtype=torch.float64)
        if self.world > 1:
            dist.all_reduce(tensor)
        return float(tensor.item()) / self.world

    def log(self, text):
        if self.rank == 0:
            print(text, flush=True)

    def progress(self, iterable, **kwargs):
        return tqdm(iterable, disable=self.rank != 0, **kwargs)

    def close(self):
        if self.owns_group:
            dist.destroy_process_group()


def save_adapter(path, factors, config, stage):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    values = {key: value.detach().to(device="cpu", dtype=torch.float32 if key.endswith(".alpha") else torch_dtype(config.dtype)).contiguous() for key, value in factors.items()}
    try:
        save_file(values, str(temporary), metadata={"reconstruction_config": json.dumps(asdict(config)), "stage": stage})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _save(path, record):
    torch.save({key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in record.items()}, path)


def _release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def initialize(source, target, selected, config, runtime, work):
    if config.initialization == "lora":
        factors = load_file(config.initial_lora)
    else:
        local = {}
        for index in runtime.progress(range(runtime.rank, len(selected), runtime.world), desc=f"{config.initialization.upper()} initialization", unit="layer"):
            key = selected[index]
            module = key[:-7]
            shape = source.shapes[key]
            rank = min(config.rank, *shape)
            prefix = f"transformer.{module}"
            if config.initialization == "svd":
                delta = target[key].float() - source[key].float()
                if torch.count_nonzero(delta):
                    local.update(decompose_to_lora(delta, module, config.rank, config.alpha, "transformer", runtime.device))
                    del delta
                    continue
                del delta
            # Zero-delta layers must remain trainable: both factors zero would
            # have identically zero gradients. Also used for random initialization.
            generator = torch.Generator().manual_seed(config.seed + index)
            local[f"{prefix}.lora_down.weight"] = torch.randn(rank, shape[1], generator=generator) / math.sqrt(shape[1])
            local[f"{prefix}.lora_up.weight"] = torch.zeros(shape[0], rank)
            local[f"{prefix}.alpha"] = torch.tensor(config.alpha, dtype=torch.float32)
        save_file(local, str(work / f"initial_rank_{runtime.rank}.safetensors"))
        runtime.barrier()
        if runtime.rank == 0:
            merged = {}
            for rank in range(runtime.world):
                merged.update(load_file(str(work / f"initial_rank_{rank}.safetensors")))
            path = config.svd_output if config.initialization == "svd" else work / "initial.safetensors"
            save_adapter(path, merged, config, "initialization")
            runtime.log(f"Saved initialization: {path}")
            del merged
        del local
        runtime.barrier()
        path = config.svd_output if config.initialization == "svd" else work / "initial.safetensors"
        factors = load_file(str(path))
    expected = set()
    for key in selected:
        name = f"transformer.{key[:-7]}"
        rows, columns = source.shapes[key]
        rank = min(config.rank, rows, columns)
        for suffix, shape in (("lora_down.weight", (rank, columns)), ("lora_up.weight", (rows, rank)), ("alpha", ())):
            factor_key = f"{name}.{suffix}"
            expected.add(factor_key)
            if factor_key not in factors or tuple(factors[factor_key].shape) != shape:
                raise ValueError(f"Invalid initial LoRA shape/missing key: {factor_key}, expected {shape}")
            if not torch.isfinite(factors[factor_key]).all():
                raise ValueError(f"Non-finite initial LoRA: {factor_key}")
        if not math.isclose(float(factors[f"{name}.alpha"]), config.alpha):
            raise ValueError(f"Initial LoRA alpha differs from config: {name}")
    if set(factors) != expected:
        raise ValueError("Initial LoRA must contain exactly the configured layers and per-layer alpha")
    return {key: value.float() for key, value in factors.items()}


def prepare_calibration(config, architecture, runtime, work):
    files = calibration_files(config)
    counts = dict.fromkeys(("t2i", "i2i_single", "i2i_multi"), 0)
    local = work / f"rank_{runtime.rank}" / "calibration"
    local.mkdir(parents=True)
    rope = reference().QwenImage21Rope(10000, architecture["axes_dims_rope"])
    paths, times, provenance = [], set(), []
    total = 0
    for path in runtime.progress(files, desc="Preparing calibration", unit="trace"):
        data = _load(path)
        if "steps" in data:
            references = len(data["condition"]["source_img_shapes"])
            counts["t2i" if references == 0 else "i2i_single" if references == 1 else "i2i_multi"] += 1
            records = trace_records(data, config.step_indices, config.compute_dtype)
        elif "hidden_states" in data:
            references = len(data["img_shapes"]) - 1
            counts["t2i" if references == 0 else "i2i_single" if references == 1 else "i2i_multi"] += 1
            if config.calibration_counts is not None:
                raise ValueError("calibration_counts counts complete native traces, not individual prepared timestep records; omit it for prepared records")
            if config.step_indices is not None:
                raise ValueError("step_indices only applies to native traces")
            records = [data]
        else:
            raise ValueError(f"Not a native trace or prepared calibration record: {path}")
        start = total
        for record in records:
            times.add(float(record["timestep"].item()))
            if total % runtime.world == runtime.rank:
                prepared = prepare_record(record, architecture, rope)
                output = local / f"{total:08d}.pt"
                _save(output, prepared)
                paths.append(output)
            total += 1
        provenance.append({"path": str(path.resolve()), "model_path": data.get("model_path"), "records": total - start})
    if config.calibration_counts is not None and counts != config.calibration_counts:
        raise ValueError(f"Calibration request counts differ: found {counts}, configured {config.calibration_counts}")
    if total < runtime.world:
        raise ValueError(f"Need at least WORLD_SIZE={runtime.world} calibration samples, got {total}")
    # Equal local populations avoid DDP weighting bias and uneven execution.
    # Repeat the first global records only to fill the final partial shard.
    remainder = total % runtime.world
    if remainder and runtime.rank >= remainder:
        donor = work / f"rank_{runtime.rank - remainder}" / "calibration"
        runtime.barrier()
        original = min(donor.glob("*.pt"))
        output = local / f"{total + runtime.rank - remainder:08d}.pt"
        shutil.copyfile(original, output)
        paths.append(output)
    elif remainder:
        runtime.barrier()
    if runtime.rank == 0:
        (work / "calibration.json").write_text(
            json.dumps({"requests": counts, "samples": total, "padded_samples": math.ceil(total / runtime.world) * runtime.world, "timesteps": sorted(times), "traces": provenance}, indent=2)
        )
    runtime.log(f"Calibration: {total} samples, {len(paths)} per rank after padding, {len(times)} timesteps (+ condition t=0)")
    return paths, sorted(times | {0.0})


def mse(prediction, target, mask=None):
    difference = (prediction.float() - target.float()).square()
    if mask is not None:
        difference = difference[mask]
    return difference.mean()


def objective(prediction, record, outputs=None):
    if outputs:
        return sum(mse(prediction[key], record[f"teacher_{key}"]) for key in outputs)
    return mse(prediction, record["y"], record.get("loss_mask"))


def accumulation_steps(config, world_size):
    if config.batch_size is None:
        return config.gradient_accumulation
    if config.batch_size % world_size:
        raise ValueError(f"Global batch_size={config.batch_size} must be divisible by WORLD_SIZE={world_size}; each forward processes one sample per rank")
    return config.batch_size // world_size


def mix_training_input(record, student_prob, generator):
    """Mix paired prefix features elementwise without changing targets/caches."""
    if "x" not in record or student_prob == 1:
        return record
    student, teacher = record["x"], record["teacher_input_x"]
    if student.shape != teacher.shape:
        raise ValueError("QDrop requires matching teacher/student input shapes")
    if student_prob == 0:
        mixed = teacher
    else:
        mask = torch.rand(student.shape, device=student.device, generator=generator) < student_prob
        mixed = torch.where(mask, student, teacher)
    return {**record, "x": mixed}


def optimize(unit, paths, config, runtime, stage, work, iterations, outputs=None):
    parameters = [parameter for parameter in unit.parameters() if parameter.requires_grad]
    if not parameters:
        runtime.log(f"{stage}: no selected trainable layers; forwarding only")
        return
    dtype = torch_dtype(config.compute_dtype)
    device = runtime.device
    accumulation = accumulation_steps(config, runtime.world)
    wrapped = DistributedDataParallel(unit, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False) if runtime.world > 1 else unit
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)

    def evaluate():
        total = 0.0
        with torch.no_grad():
            for path in paths:
                record = move(_load(path), device, dtype)
                total += float(objective(unit(record), record, outputs))
        return runtime.mean(total / len(paths))

    initial = evaluate()
    if not math.isfinite(initial):
        raise FloatingPointError(f"Non-finite initial reconstruction loss: {stage}")
    original = [parameter.detach().cpu().clone() for parameter in parameters]
    generator = random.Random(config.seed + runtime.rank)
    mix_generator = torch.Generator(device=device).manual_seed(config.seed + runtime.rank) if config.qdrop_input_mix else None
    progress = runtime.progress(range(iterations), desc=stage, unit="iter")
    for iteration in progress:
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for micro_step in range(accumulation):
            record = move(_load(generator.choice(paths)), device, dtype)
            if config.qdrop_input_mix:
                record = mix_training_input(record, config.qdrop_student_prob, mix_generator)
            synchronize = micro_step == accumulation - 1
            context = wrapped.no_sync() if runtime.world > 1 and not synchronize else nullcontext()
            with context:
                loss = objective(wrapped(record), record, outputs)
                (loss / accumulation).backward()
            accumulated += float(loss.detach()) / accumulation
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        # Make the failure collective so every rank terminates this stage.
        valid = runtime.mean(float(math.isfinite(accumulated) and torch.isfinite(norm).item()))
        if valid != 1.0:
            raise FloatingPointError(f"Non-finite loss/gradient in {stage}, iteration {iteration}")
        optimizer.step()
        if (iteration + 1) % config.log_every == 0 or iteration == 0 or iteration + 1 == iterations:
            average = runtime.mean(accumulated)
            if runtime.rank == 0:
                progress.set_postfix(mse=f"{average:.6g}")
    final = evaluate()
    accepted = math.isfinite(final) and final <= initial
    if not accepted:
        with torch.no_grad():
            for parameter, saved in zip(parameters, original, strict=True):
                parameter.copy_(saved)
        runtime.log(f"{stage}: final calibration MSE increased; keeping this unit's initial factors")
    result = {"stage": stage, "initial_mse": initial, "optimized_mse": final, "retained_mse": final if accepted else initial, "accepted": accepted, "iterations": iterations}
    if runtime.rank == 0:
        with (work / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(result) + "\n")
    runtime.log(f"{stage}: MSE {initial:.6g} -> {result['retained_mse']:.6g}")


def temporal_reconstruction(source, target, selected, factors, architecture, times, config, runtime, work):
    dtype = torch_dtype(config.compute_dtype)
    folder = work / f"rank_{runtime.rank}" / "temporal"
    folder.mkdir()
    teacher = build_unit("temporal", architecture, target, runtime.device, dtype)
    paths = []
    with torch.no_grad():
        for index, time in enumerate(times):
            record = {"timestep": torch.tensor([time], device=runtime.device, dtype=dtype)}
            record.update({f"teacher_{key}": value for key, value in teacher(record).items()})
            path = folder / f"{index:04d}.pt"
            _save(path, record)
            paths.append(path)
    del teacher
    _release()
    if config.algorithm == "tfmq":
        # The shared modulation and final scale are the consumers of the time
        # embedding. Jointly fit these endpoints, letting embedding errors be
        # compensated inside this temporal unit (no extra embedding penalty).
        stages = [("temporal_joint", TEMPORAL_LAYERS, ("modulation", "scale"))]
    else:
        stages = [(f"temporal_{index}", (name,), (output,)) for index, (name, output) in enumerate(zip(TEMPORAL_LAYERS, ("first", "embedding", "modulation", "scale"), strict=True))]
    for name, active, outputs in stages:
        unit = build_unit("temporal", architecture, source, runtime.device, dtype, factors, set(active) & selected)
        optimize(unit, paths, config, runtime, name, work, config.temporal_iterations, outputs)
        update_factors(unit, factors)
        del unit
        _release()
    # Cache the two finalized temporal paths. t=0 is a distinct, fixed row for
    # text/reference tokens, not replaced with the target image's timestep.
    result = {}
    for label, weights, adapters in (("teacher", target, None), ("student", source, factors)):
        unit = build_unit("temporal", architecture, weights, runtime.device, dtype, adapters)
        with torch.no_grad():
            for time in times:
                values = unit({"timestep": torch.tensor([time, 0.0], device=runtime.device, dtype=dtype)})
                result[label, time] = {key: values[key].cpu() for key in ("modulation", "scale")}
        del unit
        _release()
    return result


def reconstruct_stage(kind, index, source, target, selected, factors, architecture, paths, temporal, config, runtime, work):
    dtype = torch_dtype(config.compute_dtype)
    stage = f"block_{index:02d}" if kind == "block" else kind
    folder = work / f"rank_{runtime.rank}" / stage
    folder.mkdir()
    train_paths = []
    teacher = build_unit(kind, architecture, target, runtime.device, dtype, block_index=index)
    with torch.no_grad():
        for path in runtime.progress(paths, desc=f"{stage}: teacher cache", unit="sample"):
            record = _load(path)
            time = float(record["timestep"].item())
            teacher_input = dict(record)
            if kind != "input":
                teacher_input["x"] = record["teacher_x"]
            teacher_input.update(temporal["teacher", time])
            output = teacher(move(teacher_input, runtime.device, dtype)).cpu()
            if kind != "input":
                record["x"] = record.pop("student_x")
                if config.qdrop_input_mix:
                    record["teacher_input_x"] = record["teacher_x"]
                del record["teacher_x"]
            record.update(temporal["student", time])
            record["y"] = output
            if kind != "output":
                record["loss_mask"] = record["key_valid"]
            new_path = folder / path.name
            _save(new_path, record)
            train_paths.append(new_path)
    del teacher
    _release()
    # Only factors in this unit are trainable. Temporal factors are already
    # frozen and represented by the detached cached modulation/scale tensors.
    unit = build_unit(kind, architecture, source, runtime.device, dtype, factors, selected, index, config.checkpointing)
    optimize(unit, train_paths, config, runtime, stage, work, config.iterations)
    update_factors(unit, factors)
    with torch.no_grad():
        for path in runtime.progress(train_paths, desc=f"{stage}: student cache", unit="sample"):
            record = _load(path)
            output = unit(move(record, runtime.device, dtype)).cpu()
            record["teacher_x"] = record.pop("y")
            record["student_x"] = output
            for key in ("x", "teacher_input_x", "modulation", "scale", "loss_mask", "hidden_states", "encoder_hidden_states", "repeats", "image_pad_mask", "target_tokens"):
                record.pop(key, None)
            _save(path, record)
    del unit
    _release()
    if not config.keep_cache:
        shutil.rmtree(paths[0].parent)
    return train_paths


def reconstruct(config):
    runtime = Runtime(config.device)
    try:
        accumulation = accumulation_steps(config, runtime.world)
        runtime.log(f"Global batch size: {runtime.world * accumulation}; samples per rank per forward: 1; accumulation: {accumulation}")
        torch.manual_seed(config.seed)
        if runtime.device.type == "cuda" and torch_dtype(config.compute_dtype) == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16; use compute_dtype=fp32")
        source, target = ExtractionWeights(ModelWeights(config.source_model)), ExtractionWeights(ModelWeights(config.target_model))
        architecture = read_architecture(source, config.model_config)
        reference()  # Fail before doing SVD if the required backend is unavailable.
        selected_keys = select_layers(source, target, config)
        selected = {key[:-7] for key in selected_keys}
        supported = set()
        with torch.device("meta"):
            from .qwen_image_21 import Unit

            units = [("temporal", None), ("input", None), ("output", None)] + [("block", i) for i in range(architecture["num_layers"])]
            for kind, index in units:
                unit = Unit(kind, architecture, index)
                supported.update(name for name, layer in unit.named_modules() if isinstance(layer, torch.nn.Linear))
                for key, tensor in unit.state_dict().items():
                    if key not in source or key not in target or source.shapes[key] != tuple(tensor.shape) or target.shapes[key] != tuple(tensor.shape):
                        raise ValueError(f"Source/target architecture mismatch: {key}")
        if selected - supported:
            raise ValueError(f"Unsupported selected layers: {sorted(selected - supported)}")
        run_id = [uuid.uuid4().hex if runtime.rank == 0 else None]
        if runtime.world > 1:
            dist.broadcast_object_list(run_id, src=0)
        work = Path(config.work_dir) / run_id[0]
        work.mkdir(parents=True, exist_ok=True)
        runtime.log(f"Reconstruction: {config.algorithm}, {len(selected)} Linear layers, rank={config.rank}, alpha={config.alpha}, DDP={runtime.world}\nRun directory: {work}")
        if config.qdrop_input_mix:
            runtime.log(f"QDrop input mixing: student probability={config.qdrop_student_prob}; training only, evaluation and prefix propagation use student inputs")
        if runtime.rank == 0:
            (work / "config.json").write_text(json.dumps(asdict(config), indent=2))
        paths, times = prepare_calibration(config, architecture, runtime, work)
        factors = initialize(source, target, selected_keys, config, runtime, work)
        temporal = temporal_reconstruction(source, target, selected, factors, architecture, times, config, runtime, work)
        for kind, index in [("input", None)] + [("block", i) for i in range(architecture["num_layers"])] + [("output", None)]:
            paths = reconstruct_stage(kind, index, source, target, selected, factors, architecture, paths, temporal, config, runtime, work)
        runtime.barrier()
        if runtime.rank == 0:
            save_adapter(config.output, factors, config, "reconstructed")
            runtime.log(f"Saved reconstructed LoRA: {config.output}\nMetrics: {work / 'metrics.jsonl'}")
        runtime.barrier()
        if not config.keep_cache:
            shutil.rmtree(work / f"rank_{runtime.rank}")
            (work / f"initial_rank_{runtime.rank}.safetensors").unlink(missing_ok=True)
        return str(work)
    finally:
        runtime.close()
