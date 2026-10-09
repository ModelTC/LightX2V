#!/usr/bin/env python3
"""Read-only CPU or multi-GPU NaN/Inf audit of condition_path JSONL caches.

Writes bad.jsonl (only abnormal rows) and summary.json to a NEW directory.
Checks every floating tensor recursively, including prompt_embeds and reference
latents. This is a numerical audit, not a complete Ref2AV schema validator.
Exit codes: 0 = all selected rows finite, 2 = bad rows found, 1 = incomplete scan.
Requires PyTorch; loads with weights_only=True, never unrestricted pickle.
With torchrun --ddp: each rank scans a fixed 1/world_size subset, using whole
tensors on its own GPU. No worker pools, padding, duplication or dropped rows.
DDP completes with exit 0 even when bad rows are found; see audit_exit_code in
summary.json (0 clean / 2 bad / 1 incomplete). Fatal failures still exit 1.
Use --gpus 0,1,2,3,4,5,6,7 for one worker per visible GPU. Files are loaded on
CPU and checked on GPU in bounded chunks; no models or NCCL are involved.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import shutil
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

COUNT_KEYS = (
    "float_tensors_checked",
    "float_elements_checked",
    "nan_count",
    "posinf_count",
    "neginf_count",
)
_SCAN_DEVICE = "cpu"


def json_line(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def init_worker(device="cpu"):
    global _SCAN_DEVICE
    # Parallelism is across files, not eight processes each using all CPU cores.
    torch.set_num_threads(1)
    _SCAN_DEVICE = device
    if device != "cpu":
        torch.cuda.set_device(torch.device(device))
        # Fail the worker at startup if its CUDA context cannot be initialized,
        # rather than reporting every cache file as corrupt.
        torch.empty(0, device=device)


def resolve_devices(args):
    if args.gpus is None:
        return ["cpu"]
    if not torch.cuda.is_available():
        raise ValueError("--gpus requested but CUDA is unavailable; check the driver/environment")
    count = torch.cuda.device_count()
    if max(args.gpus) >= count:
        raise ValueError(f"Requested GPU indices {args.gpus}, but only {count} CUDA devices are visible")
    return [f"cuda:{index}" for index in args.gpus]


def floating_tensors(value, field="$", active=None):
    if torch.is_tensor(value):
        if value.is_complex():
            raise ValueError(f"Unexpected complex tensor at {field}")
        if value.is_floating_point():
            yield field, value
        return
    if not isinstance(value, (dict, list, tuple)):
        return
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError(f"Recursive container at {field}")
    active.add(id(value))
    try:
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in items:
            suffix = f".{key}" if isinstance(value, dict) else f"[{key}]"
            yield from floating_tensors(child, field + suffix, active)
    finally:
        active.remove(id(value))


def chunked_device_reductions(flat, chunk_elements, device):
    """Device-side reduction used by CUDA scans; also testable on CPU."""
    counts = torch.zeros(3, dtype=torch.int64, device=device)
    reduction_dtype = torch.float64 if flat.dtype == torch.float64 else torch.float32
    maximum = torch.full((), -math.inf, dtype=reduction_dtype, device=device)
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start : start + chunk_elements].to(device=device)
        if chunk.dtype in (torch.bfloat16, torch.float16):
            chunk = chunk.float()
        counts[0] += torch.isnan(chunk).sum()
        counts[1] += torch.isposinf(chunk).sum()
        counts[2] += torch.isneginf(chunk).sum()
        absolute = chunk.abs().masked_fill_(~torch.isfinite(chunk), -math.inf)
        maximum = torch.maximum(maximum, absolute.max())
    counts = counts.cpu().tolist()
    maximum_value = maximum.item()
    return counts, maximum_value if math.isfinite(maximum_value) else None


def tensor_stats(field, tensor, chunk_elements, device=None):
    if tensor.layout != torch.strided or tensor.device.type != "cpu":
        raise ValueError(f"Expected dense CPU tensor at {field}, got {tensor.layout}/{tensor.device}")
    if tensor.numel() == 0:
        raise ValueError(f"Empty floating tensor at {field}")
    flat = tensor.detach().reshape(-1)
    stats = {
        "field": field,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "nan_count": 0,
        "posinf_count": 0,
        "neginf_count": 0,
        "finite_abs_max": None,
    }
    device = torch.device(_SCAN_DEVICE if device is None else device)
    if device.type == "cuda":
        # Accumulate on GPU; only copy four scalar results back per tensor.
        # Avoid a CPU/GPU synchronization after each individual chunk.
        counts, maximum = chunked_device_reductions(flat, chunk_elements, device)
        stats["nan_count"], stats["posinf_count"], stats["neginf_count"] = counts
        stats["finite_abs_max"] = maximum
        return stats
    if device.type != "cpu":
        raise ValueError(f"Unsupported scan device: {device}")
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start : start + chunk_elements]
        # BF16/FP16 expand exactly to FP32. Preserve FP64 without downcasting.
        if chunk.dtype in (torch.bfloat16, torch.float16):
            chunk = chunk.float()
        finite = torch.isfinite(chunk)
        if bool(finite.all()):
            maximum = chunk.abs().max().item()
        else:
            stats["nan_count"] += int(torch.isnan(chunk).sum().item())
            stats["posinf_count"] += int(torch.isposinf(chunk).sum().item())
            stats["neginf_count"] += int(torch.isneginf(chunk).sum().item())
            maximum = chunk[finite].abs().max().item() if bool(finite.any()) else None
        if maximum is not None:
            previous = stats["finite_abs_max"]
            stats["finite_abs_max"] = maximum if previous is None else max(previous, maximum)
    return stats


def scan_record(metadata_line, raw_line, metadata_root, chunk_elements):
    result = {
        "metadata_line": metadata_line,
        "condition_path": None,
        "scan_device": _SCAN_DEVICE,
        "source_id": None,
        "source_index": None,
        "status": "clean",
        **dict.fromkeys(COUNT_KEYS, 0),
        "finite_abs_max": None,
        "tensors": [],
    }
    try:
        row = json.loads(raw_line)
        if not isinstance(row, dict):
            raise ValueError("Metadata row must be an object")
        result["source_id"] = row.get("source_id", row.get("sample_id", row.get("id")))
        result["source_index"] = row.get("source_index")
        value = row.get("condition_path")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Missing/non-string condition_path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(metadata_root) / path
        result["condition_path"] = str(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            for key in ("source_id", "source_index"):
                if result[key] is None:
                    result[key] = payload.get(key)
        # Do not retain tensor values/embeddings in reports, only scalar stats.
        for field, tensor in floating_tensors(payload):
            stats = tensor_stats(field, tensor, chunk_elements)
            result["float_tensors_checked"] += 1
            result["float_elements_checked"] += stats["numel"]
            for key in ("nan_count", "posinf_count", "neginf_count"):
                result[key] += stats[key]
            maximum = stats["finite_abs_max"]
            if maximum is not None:
                previous = result["finite_abs_max"]
                result["finite_abs_max"] = maximum if previous is None else max(previous, maximum)
            if stats["nan_count"] + stats["posinf_count"] + stats["neginf_count"]:
                result["tensors"].append(stats)
        if result["float_tensors_checked"] == 0:
            raise ValueError("Cache contains no floating tensors")
        if result["tensors"]:
            result["status"] = "nonfinite"
    except Exception as error:
        # Missing files, corrupt archives and unsafe pickle types are visible
        # errors, never silently counted as clean or retried unsafely.
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def selected_rows(args):
    selected = 0
    index = 0
    with args.metadata.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            belongs = index % args.num_shards == args.shard_index
            index += 1
            if not belongs:
                continue
            yield line_number, line
            selected += 1
            if args.max_samples is not None and selected >= args.max_samples:
                break


def ddp_selected_rows(args, rank, world_size):
    """Exact static partition, unlike a sampler that pads to equal lengths.

    max_samples, if present, is a GLOBAL prefix before rank partitioning.
    Blank lines do not take up an index; malformed JSON does (reported later).
    """
    index = 0
    with args.metadata.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if args.max_samples is not None and index >= args.max_samples:
                break
            if index % world_size == rank:
                yield line_number, line
            index += 1


def write_summary(path, summary):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json_line(summary) + "\n", encoding="utf-8")
    temporary.replace(path)


def scan_ddp_rank(args, rank, world_size, local_rank, device):
    """Read and scan on the torchrun process itself, with no child workers."""
    output = args.output_dir / f"rank-{rank:05d}"
    output.mkdir(exist_ok=False)
    summary = {
        "complete": False,
        "mode": "ddp",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "scan_device": device,
        "metadata": str(args.metadata),
        "max_samples": args.max_samples,
        "selected_rows": 0,
        "scanned_rows": 0,
        "clean_rows": 0,
        "nonfinite_rows": 0,
        "error_rows": 0,
        **dict.fromkeys(COUNT_KEYS, 0),
        "finite_abs_max": None,
        "bad_report": str(output / "bad.jsonl"),
    }
    started = time.monotonic()
    lock = threading.RLock()
    stop = threading.Event()

    def progress(event):
        with lock:
            summary["elapsed_s"] = round(time.monotonic() - started, 2)
            fields = {
                key: summary[key]
                for key in (
                    "rank",
                    "world_size",
                    "scan_device",
                    "scanned_rows",
                    "clean_rows",
                    "nonfinite_rows",
                    "error_rows",
                    "elapsed_s",
                )
            }
            print(json_line({"event": event, **fields}), file=sys.stderr, flush=True)
            write_summary(output / "summary.json", summary)

    def heartbeat():
        while not stop.wait(args.log_interval):
            progress("heartbeat")

    progress("start")
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        with (output / "bad.jsonl").open("x", encoding="utf-8") as bad_file:
            for line_number, raw_line in ddp_selected_rows(args, rank, world_size):
                # One transfer/reduction per FULL tensor: no chunk-elements
                # splitting in DDP mode. Load .pt files on CPU as before.
                result = scan_record(line_number, raw_line, str(args.metadata.parent), sys.maxsize)
                result.update(rank=rank, local_rank=local_rank)
                with lock:
                    summary["selected_rows"] += 1
                    summary["scanned_rows"] += 1
                    summary[result["status"] + "_rows"] += 1
                    for key in COUNT_KEYS:
                        summary[key] += result[key]
                    maximum = result["finite_abs_max"]
                    if maximum is not None:
                        previous = summary["finite_abs_max"]
                        summary["finite_abs_max"] = maximum if previous is None else max(previous, maximum)
                    if result["status"] != "clean":
                        bad_file.write(json_line(result) + "\n")
                        bad_file.flush()
                    if summary["scanned_rows"] == 1 or summary["scanned_rows"] % args.log_every == 0:
                        progress("progress")
        # A rank with zero assigned rows is valid (e.g. a tiny smoke prefix).
        with lock:
            summary["complete"] = True
    except Exception as error:
        with lock:
            summary["fatal_error"] = f"{type(error).__name__}: {error}"
    finally:
        stop.set()
        thread.join()
        progress("rank_complete" if summary["complete"] else "rank_incomplete")
    return summary


def merge_ddp_reports(args, summaries, started):
    summary = {
        "complete": all(item["complete"] for item in summaries),
        "mode": "ddp",
        "metadata": str(args.metadata),
        "world_size": len(summaries),
        "partition": "nonempty_row_index_mod_world_size_no_padding_no_drop",
        "max_samples": args.max_samples,
        "whole_tensor": True,
        "bad_report": str(args.output_dir / "bad.jsonl"),
        "rows_by_rank": {str(item["rank"]): item["scanned_rows"] for item in summaries},
        "per_rank": summaries,
        "finite_abs_max": None,
        "elapsed_s": round(time.monotonic() - started, 2),
    }
    for key in ("selected_rows", "scanned_rows", "clean_rows", "nonfinite_rows", "error_rows", *COUNT_KEYS):
        summary[key] = sum(item[key] for item in summaries)
    maxima = [item["finite_abs_max"] for item in summaries if item["finite_abs_max"] is not None]
    summary["finite_abs_max"] = max(maxima) if maxima else None
    if not summary["scanned_rows"]:
        summary.update(complete=False, fatal_error="Manifest has no selected rows")
    summary["audit_exit_code"] = 1 if not summary["complete"] else 2 if summary["nonfinite_rows"] or summary["error_rows"] else 0
    temporary = args.output_dir / "bad.tmp"
    with temporary.open("x", encoding="utf-8") as combined:
        for item in summaries:
            with Path(item["bad_report"]).open(encoding="utf-8") as source:
                shutil.copyfileobj(source, combined)
    temporary.replace(args.output_dir / "bad.jsonl")
    write_summary(args.output_dir / "summary.json", summary)
    print(json_line(summary), flush=True)
    # A bad data row is an audit finding, not a crashed DDP worker. Returning
    # 2 here would make torchrun print ChildFailedError for a completed audit.
    return 0 if summary["complete"] else 1


def run_ddp(args):
    import torch.distributed as dist

    try:
        rank, world_size, local_rank = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    except (KeyError, ValueError) as error:
        raise ValueError("--ddp must be launched with torchrun (RANK/WORLD_SIZE/LOCAL_RANK required)") from error
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid torchrun rank environment")
    started = time.monotonic()
    # Only startup/completion metadata is synchronized. Tensor checks are
    # independent on each GPU; there is no model to wrap with DDP or NCCL work.
    dist.init_process_group("gloo", timeout=timedelta(hours=3))
    root_created = False
    try:
        preflight = {"error": None}
        try:
            args.metadata = args.metadata.expanduser().resolve(strict=True)
            if not args.metadata.is_file():
                raise ValueError(f"Metadata is not a file: {args.metadata}")
            args.output_dir = args.output_dir.expanduser().resolve()
            device = "cpu" if args.ddp_device == "cpu" else f"cuda:{local_rank}"
            if args.ddp_device == "cuda":
                if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
                    raise ValueError(f"Rank {rank} requires visible GPU {local_rank}, but CUDA/device is unavailable")
            init_worker(device)
            preflight.update(metadata=str(args.metadata), output_dir=str(args.output_dir), metadata_size=args.metadata.stat().st_size, max_samples=args.max_samples)
        except Exception as error:
            preflight["error"] = f"rank {rank}: {type(error).__name__}: {error}"
        checks = [None] * world_size
        dist.all_gather_object(checks, preflight)
        failures = [item["error"] for item in checks if item["error"]]
        if failures:
            raise ValueError("DDP preflight failed: " + "; ".join(failures))
        if any(item != checks[0] for item in checks):
            raise ValueError("DDP ranks disagree on metadata/output/max-samples")

        setup = [None]
        if rank == 0:
            try:
                args.output_dir.mkdir(parents=True, exist_ok=False)
                root_created = True
                write_summary(
                    args.output_dir / "summary.json",
                    {
                        "complete": False,
                        "mode": "ddp",
                        "world_size": world_size,
                        "metadata": str(args.metadata),
                        "audit_exit_code": None,
                        "note": "Ranks are scanning; inspect rank-*/summary.json for progress",
                    },
                )
            except Exception as error:
                setup[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(setup, src=0)
        if setup[0]:
            raise ValueError(setup[0])

        local_summary = scan_ddp_rank(args, rank, world_size, local_rank, device)
        summaries = [None] * world_size
        dist.all_gather_object(summaries, local_summary)
        outcome = [None]
        if rank == 0:
            try:
                outcome[0] = merge_ddp_reports(args, summaries, started)
            except Exception as error:
                write_summary(
                    args.output_dir / "summary.json",
                    {
                        "complete": False,
                        "mode": "ddp",
                        "audit_exit_code": 1,
                        "fatal_error": f"{type(error).__name__}: {error}",
                        "per_rank": summaries,
                    },
                )
                outcome[0] = 1
        dist.broadcast_object_list(outcome, src=0)
        return outcome[0]
    except Exception as error:
        if rank == 0 and root_created:
            write_summary(
                args.output_dir / "summary.json",
                {
                    "complete": False,
                    "mode": "ddp",
                    "audit_exit_code": 1,
                    "fatal_error": f"{type(error).__name__}: {error}",
                },
            )
        raise
    finally:
        dist.destroy_process_group()


def run(args):
    # Fail before opening outputs if the input is inaccessible. Refuse reuse
    # of a report directory so an old successful summary is never misleading.
    args.metadata = args.metadata.expanduser().resolve(strict=True)
    if not args.metadata.is_file():
        raise ValueError(f"Metadata is not a file: {args.metadata}")
    devices = resolve_devices(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_dir / "summary.json"
    bad_path = args.output_dir / "bad.jsonl"
    summary = {
        "complete": False,
        "metadata": str(args.metadata),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_samples": args.max_samples,
        "workers": args.workers,
        "devices": devices,
        "worker_rows_by_device": dict.fromkeys(devices, 0),
        "selected_rows": 0,
        "scanned_rows": 0,
        "clean_rows": 0,
        "nonfinite_rows": 0,
        "error_rows": 0,
        **dict.fromkeys(COUNT_KEYS, 0),
        "finite_abs_max": None,
        "bad_report": str(bad_path),
    }
    started = time.monotonic()
    last_log = started
    write_summary(summary_path, summary)

    def progress(event):
        nonlocal last_log
        now = time.monotonic()
        summary["elapsed_s"] = round(now - started, 2)
        counts = {
            key: summary[key]
            for key in (
                "selected_rows",
                "scanned_rows",
                "clean_rows",
                "nonfinite_rows",
                "error_rows",
            )
        }
        print(json_line({"event": event, **counts, "devices": devices, "elapsed_s": summary["elapsed_s"]}), file=sys.stderr, flush=True)
        write_summary(summary_path, summary)
        last_log = now

    def record(result, bad_file):
        summary["scanned_rows"] += 1
        summary["worker_rows_by_device"][result["scan_device"]] += 1
        summary[result["status"] + "_rows"] += 1
        for key in COUNT_KEYS:
            summary[key] += result[key]
        maximum = result["finite_abs_max"]
        if maximum is not None:
            previous = summary["finite_abs_max"]
            summary["finite_abs_max"] = maximum if previous is None else max(previous, maximum)
        if result["status"] != "clean":
            bad_file.write(json_line(result) + "\n")
            bad_file.flush()
        if summary["scanned_rows"] == 1 or summary["scanned_rows"] % args.log_every == 0 or time.monotonic() - last_log >= args.log_interval:
            progress("progress")

    progress("start")
    try:
        with bad_path.open("x", encoding="utf-8") as bad_file:
            # Bound pending work and memory regardless of manifest size. Even
            # --workers=1 uses a child, letting the parent log I/O heartbeats.
            with ExitStack() as stack:
                # Separate single-worker executors pin each GPU to one process
                # for the entire run. CPU mode keeps the original shared pool.
                pools = []
                limits = []
                for device in devices:
                    workers = args.workers if device == "cpu" else 1
                    pools.append(
                        stack.enter_context(
                            ProcessPoolExecutor(
                                max_workers=workers,
                                mp_context=multiprocessing.get_context("spawn"),
                                initializer=init_worker,
                                initargs=(device,),
                            )
                        )
                    )
                    limits.append(workers * 2)
                assigned = [0] * len(pools)
                rows = iter(selected_rows(args))
                pending = {}
                exhausted = False
                while pending or not exhausted:
                    for pool_index, executor in enumerate(pools):
                        while not exhausted and assigned[pool_index] < limits[pool_index]:
                            try:
                                line_number, raw_line = next(rows)
                            except StopIteration:
                                exhausted = True
                                break
                            future = executor.submit(scan_record, line_number, raw_line, str(args.metadata.parent), args.chunk_elements)
                            pending[future] = pool_index
                            assigned[pool_index] += 1
                            summary["selected_rows"] += 1
                    if not pending:
                        break
                    done, _ = wait(pending, timeout=min(args.log_interval, 1.0), return_when=FIRST_COMPLETED)
                    for future in done:
                        assigned[pending.pop(future)] -= 1
                        record(future.result(), bad_file)
                    if time.monotonic() - last_log >= args.log_interval:
                        progress("heartbeat")
        if summary["scanned_rows"] == 0:
            raise ValueError("No selected rows; check the manifest and shard options")
        summary["complete"] = True
    except BaseException as error:
        summary["fatal_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        progress("complete" if summary["complete"] else "incomplete")
    print(json_line(summary), flush=True)
    return 2 if summary["nonfinite_rows"] or summary["error_rows"] else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New report directory; never overwritten")
    parser.add_argument("--workers", type=int, help="CPU workers (default 4); with --gpus must equal GPU count")
    parser.add_argument("--gpus", help="Comma-separated visible GPU indices, e.g. 0,1,2,3,4,5,6,7; one worker per GPU")
    parser.add_argument("--ddp", action="store_true", help="torchrun: fixed per-rank data partitions and whole-tensor GPU checks")
    parser.add_argument("--ddp-device", choices=("cuda", "cpu"), help="DDP defaults to CUDA; CPU is for distributed smoke tests")
    parser.add_argument("--chunk-elements", type=int, help="Non-DDP only: elements per temporary check (default 1048576)")
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int, help="Non-DDP: zero-based shard by non-empty manifest row order")
    parser.add_argument("--max-samples", type=int, help="Prefix limit: global before partition in DDP; per shard otherwise")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-interval", type=float, default=30)
    args = parser.parse_args(argv)
    if args.ddp:
        if any(getattr(args, name) is not None for name in ("workers", "gpus", "chunk_elements", "num_shards", "shard_index")):
            parser.error("--ddp uses torchrun ranks and whole tensors; do not set workers/gpus/chunk-elements/shard options")
    elif args.ddp_device is not None:
        parser.error("--ddp-device requires --ddp")
    args.ddp_device = args.ddp_device or "cuda"
    if args.chunk_elements is None:
        args.chunk_elements = 1048576
    if args.num_shards is None:
        args.num_shards = 1
    if args.shard_index is None:
        args.shard_index = 0
    if args.gpus is not None:
        try:
            args.gpus = [int(part.strip()) for part in args.gpus.split(",")]
        except ValueError:
            parser.error("--gpus must be comma-separated non-negative integers")
        if min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
            parser.error("--gpus must contain unique non-negative indices")
        if args.workers is not None and args.workers != len(args.gpus):
            parser.error("--workers must equal the number of --gpus (one worker per GPU)")
        args.workers = len(args.gpus)
    elif args.workers is None:
        args.workers = 4
    if min(args.workers, args.chunk_elements, args.num_shards, args.log_every) < 1:
        parser.error("workers/chunk-elements/num-shards/log-every must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("max-samples must be positive")
    if not math.isfinite(args.log_interval) or args.log_interval <= 0:
        parser.error("log-interval must be finite and positive")
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
        # Guard against accidentally running eight independent worker pools
        # into the same output when --ddp was omitted from a torchrun command.
        if not args.ddp and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("Under torchrun with WORLD_SIZE > 1, pass --ddp")
        return run_ddp(args) if args.ddp else run(args)
    except KeyboardInterrupt:
        print("Scan interrupted; do not treat partial reports as a clean audit.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
