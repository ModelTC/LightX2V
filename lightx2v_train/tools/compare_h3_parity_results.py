#!/usr/bin/env python3
"""Compare CPU diagnostics produced by the old/new H3 parity harness.

Usage: python compare_h3_parity_results.py --old OLD --new NEW --output report.json

This reads rank*.pt files without changing them. The report does not impose a
numerical tolerance or declare model parity: full tensors are compared directly,
but gradient differences and cosine similarities use only the saved samples.
Only load results from a trusted diagnostic run, even with weights_only=True.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

# These describe execution, not the data/model being compared. All other
# metadata, including model and data paths, seeds, precision and shapes, must match.
IGNORED_METADATA_KEYS = frozenset(
    {
        "implementation",
        "implementation_name",
        "implementation_path",
        "gpus",
        "gpu",
        "gpu_ids",
        "gpu_names",
        "physical_gpus",
        "device",
        "devices",
        "cuda_visible_devices",
        "hostname",
        "pid",
        "timestamp",
        "created_at",
        "output_dir",
    }
)
REQUIRED_METADATA_KEYS = frozenset(
    {
        "rank",
        "model_path",
        "cache_path",
        "cache_sha256",
        "shape",
        "seed",
        "student_param_dtype",
        "fake_param_dtype",
        "running_dtype",
        "fake_cpu_offload",
        "scope",
    }
)
GRADIENT_NOTE = (
    "Full gradient norms come from saved sums of squares. Gradient differences "
    "and cosine similarities compare saved sketch samples only, not full gradients; "
    "matching sketches do not establish exact full-gradient equality. Optional "
    "matching SHA256 hashes provide separate evidence of full local-gradient byte "
    "identity, assuming the harness hashes the complete gradient bytes."
)


def safe_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return 0.0 if numerator == 0 else None
    return safe_number(numerator / denominator)


def metadata_value(value: Any) -> Any:
    """Make metadata equality/reporting independent of mapping order and tuples."""
    if isinstance(value, Mapping):
        return {str(key): metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metadata_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype), "value": metadata_value(value.tolist())}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def metadata_differences(old: Any, new: Any, path: str = "metadata") -> tuple[list[dict], list[str]]:
    differences, ignored = [], []
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        for key in sorted(set(old) | set(new)):
            item_path = f"{path}.{key}"
            if str(key).lower() in IGNORED_METADATA_KEYS:
                ignored.append(item_path)
            elif key not in old or key not in new:
                differences.append({"path": item_path, "old": metadata_value(old.get(key)), "new": metadata_value(new.get(key)), "missing_from": "old" if key not in old else "new"})
            else:
                child_differences, child_ignored = metadata_differences(old[key], new[key], item_path)
                differences.extend(child_differences)
                ignored.extend(child_ignored)
    elif metadata_value(old) != metadata_value(new):
        differences.append({"path": path, "old": metadata_value(old), "new": metadata_value(new)})
    return differences, ignored


def tensor_metrics(old: torch.Tensor, new: torch.Tensor) -> dict[str, Any]:
    """Metrics use float64/complex128; relative L2 is normalized by the old norm."""
    result: dict[str, Any] = {
        "old_shape": list(old.shape),
        "new_shape": list(new.shape),
        "old_dtype": str(old.dtype),
        "new_dtype": str(new.dtype),
        "shape_match": old.shape == new.shape,
    }
    if old.shape != new.shape:
        return result
    old_finite, new_finite = torch.isfinite(old), torch.isfinite(new)
    result.update(
        numel=old.numel(),
        exact_equal=bool(torch.equal(old, new)),
        old_nonfinite=int((~old_finite).sum().item()),
        new_nonfinite=int((~new_finite).sum().item()),
    )
    if not old.is_floating_point() and not old.is_complex() and not new.is_floating_point() and not new.is_complex():
        result["differing_elements"] = int((old != new).sum().item())
    if result["old_nonfinite"] or result["new_nonfinite"]:
        result.update(max_abs_diff=None, relative_l2=None, cosine=None, old_norm=None, new_norm=None, diff_norm=None)
        return result
    dtype = torch.complex128 if old.is_complex() or new.is_complex() else torch.float64
    old_flat, new_flat = old.detach().reshape(-1).to(dtype), new.detach().reshape(-1).to(dtype)
    old_norm = float(torch.linalg.vector_norm(old_flat).item())
    new_norm = float(torch.linalg.vector_norm(new_flat).item())
    delta = new_flat - old_flat
    diff_norm = float(torch.linalg.vector_norm(delta).item())
    if old_norm == 0 or new_norm == 0:
        cosine = 1.0 if old_norm == new_norm == 0 else None
    else:
        cosine = safe_number((torch.vdot(old_flat, new_flat).real / (old_norm * new_norm)).item())
        if cosine is not None:
            cosine = max(-1.0, min(1.0, cosine))
    result.update(
        max_abs_diff=safe_number(delta.abs().max().item()) if delta.numel() else 0.0,
        relative_l2=ratio(diff_norm, old_norm),
        cosine=cosine,
        old_norm=safe_number(old_norm),
        new_norm=safe_number(new_norm),
        diff_norm=safe_number(diff_norm),
    )
    return result


def read_gradient_stage(stage: Any, label: str) -> tuple[dict[str, dict], dict[str, torch.Tensor], dict[str, str]]:
    if not isinstance(stage, Mapping):
        raise ValueError(f"{label} must be a mapping")
    names, stats, sketches = stage.get("names"), stage.get("stats"), stage.get("sketch")
    hashes = stage.get("hashes", {})
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{label}.names must be a nonempty list of unique parameter names")
    if not isinstance(stats, torch.Tensor) or stats.shape != (len(names), 4) or stats.dtype != torch.float64:
        raise ValueError(f"{label}.stats must be a float64 tensor of shape [len(names), 4]")
    if not isinstance(sketches, Mapping) or not all(isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in sketches.items()):
        raise ValueError(f"{label}.sketch must map parameter names to tensors")
    if set(sketches) - set(names):
        raise ValueError(f"{label}.sketch contains parameter names absent from .names")
    if not isinstance(hashes, Mapping) or not all(isinstance(name, str) and isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) for name, value in hashes.items()):
        raise ValueError(f"{label}.hashes must map parameter names to SHA256 hex digests")
    if set(hashes) - set(names):
        raise ValueError(f"{label}.hashes contains parameter names absent from .names")
    rows = {}
    for name, row in zip(names, stats.tolist()):
        numel, l2sum, maxabs, finite = row
        if not math.isfinite(numel) or numel < 0 or numel != int(numel) or finite not in (0.0, 1.0):
            raise ValueError(f"{label}.stats contains invalid numel/finite for {name}")
        if (math.isfinite(l2sum) and l2sum < 0) or (math.isfinite(maxabs) and maxabs < 0):
            raise ValueError(f"{label}.stats contains a negative squared norm/maxabs for {name}")
        rows[name] = {"numel": int(numel), "l2sum": l2sum, "maxabs": maxabs, "finite": bool(finite) and math.isfinite(l2sum) and math.isfinite(maxabs)}
    return rows, dict(sketches), dict(hashes)


def compare_gradient_stage(old: Any, new: Any, label: str) -> dict[str, Any]:
    old_stats, old_sketch, old_hashes = read_gradient_stage(old, f"old.{label}")
    new_stats, new_sketch, new_hashes = read_gradient_stage(new, f"new.{label}")
    common = sorted(set(old_stats) & set(new_stats))
    numel_mismatches = [{"name": name, "old": old_stats[name]["numel"], "new": new_stats[name]["numel"]} for name in common if old_stats[name]["numel"] != new_stats[name]["numel"]]
    result: dict[str, Any] = {
        "old_parameter_count": len(old_stats),
        "new_parameter_count": len(new_stats),
        "missing_from_old": sorted(set(new_stats) - set(old_stats)),
        "missing_from_new": sorted(set(old_stats) - set(new_stats)),
        "numel_mismatches": numel_mismatches,
        "old_nonfinite_parameters": sorted(name for name, row in old_stats.items() if not row["finite"]),
        "new_nonfinite_parameters": sorted(name for name, row in new_stats.items() if not row["finite"]),
        "missing_sketch_from_old": sorted(set(old_stats) - set(old_sketch)),
        "missing_sketch_from_new": sorted(set(new_stats) - set(new_sketch)),
    }
    old_norm = math.sqrt(math.fsum(row["l2sum"] for row in old_stats.values())) if not result["old_nonfinite_parameters"] else None
    new_norm = math.sqrt(math.fsum(row["l2sum"] for row in new_stats.values())) if not result["new_nonfinite_parameters"] else None
    result["full_gradient_norm"] = {
        "old": safe_number(old_norm) if old_norm is not None else None,
        "new": safe_number(new_norm) if new_norm is not None else None,
        "new_over_old": (1.0 if old_norm == new_norm == 0 else ratio(new_norm, old_norm)) if old_norm is not None and new_norm is not None else None,
        "absolute_norm_difference": safe_number(abs(new_norm - old_norm)) if old_norm is not None and new_norm is not None else None,
    }
    per_parameter, old_samples, new_samples = [], [], []
    for name in sorted(set(old_sketch) & set(new_sketch)):
        metrics = tensor_metrics(old_sketch[name], new_sketch[name])
        per_parameter.append({"name": name, **metrics})
        if metrics["shape_match"]:
            old_samples.append(old_sketch[name].reshape(-1))
            new_samples.append(new_sketch[name].reshape(-1))
    result["sketch_shape_mismatches"] = [row["name"] for row in per_parameter if not row["shape_match"]]
    result["old_nonfinite_sketches"] = sorted(name for name, value in old_sketch.items() if not bool(torch.isfinite(value).all()))
    result["new_nonfinite_sketches"] = sorted(name for name, value in new_sketch.items() if not bool(torch.isfinite(value).all()))
    result["sketch_metrics"] = tensor_metrics(torch.cat(old_samples), torch.cat(new_samples)) if old_samples else None

    # Null relative L2 with a nonzero difference means the old norm was zero.
    def discrepancy(row: dict) -> tuple:
        relative = row.get("relative_l2")
        return (0 if row["shape_match"] and not row.get("old_nonfinite") and not row.get("new_nonfinite") else 1, float("inf") if relative is None else relative, row.get("max_abs_diff") or 0)

    result["largest_sketch_discrepancies"] = sorted(per_parameter, key=discrepancy, reverse=True)[:20]
    result["schema_match"] = not any(
        result[key] for key in ("missing_from_old", "missing_from_new", "numel_mismatches", "missing_sketch_from_old", "missing_sketch_from_new", "sketch_shape_mismatches")
    )
    result["finite"] = not any(result[key] for key in ("old_nonfinite_parameters", "new_nonfinite_parameters", "old_nonfinite_sketches", "new_nonfinite_sketches"))
    common_hashes = set(old_hashes) & set(new_hashes)
    mismatched_hashes = sorted(name for name in common_hashes if old_hashes[name].lower() != new_hashes[name].lower())
    hashes_complete = bool(old_stats) and set(old_hashes) == set(old_stats) and set(new_hashes) == set(new_stats)
    result["full_local_gradient_hashes"] = {
        "available_in_both": len(common_hashes),
        "matched": len(common_hashes) - len(mismatched_hashes),
        "mismatched_parameters": mismatched_hashes,
        "missing_from_old": sorted(set(old_stats) - set(old_hashes)),
        "missing_from_new": sorted(set(new_stats) - set(new_hashes)),
        "complete": hashes_complete,
        "all_bytes_identical_by_sha256": not mismatched_hashes if hashes_complete and result["schema_match"] else None,
    }
    return result


def load_result(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not all(isinstance(payload.get(key), Mapping) for key in ("metadata", "tensors", "gradients")):
        raise ValueError(f"{path}: expected mappings named metadata, tensors and gradients")
    if not all(isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in payload["tensors"].items()):
        raise ValueError(f"{path}: tensors must map names to tensors")
    return dict(payload)


def compare_rank(old: dict, new: dict, expected_rank: int | None = None) -> dict:
    differences, ignored = metadata_differences(old["metadata"], new["metadata"])
    missing_metadata_old = sorted(REQUIRED_METADATA_KEYS - set(old["metadata"]))
    missing_metadata_new = sorted(REQUIRED_METADATA_KEYS - set(new["metadata"]))
    rank_matches_filename = expected_rank is None or old["metadata"].get("rank") == new["metadata"].get("rank") == expected_rank
    old_tensors, new_tensors = old["tensors"], new["tensors"]
    missing_old = sorted(set(new_tensors) - set(old_tensors))
    missing_new = sorted(set(old_tensors) - set(new_tensors))
    tensors = {name: tensor_metrics(old_tensors[name], new_tensors[name]) for name in sorted(set(old_tensors) & set(new_tensors))}
    old_gradients, new_gradients = old["gradients"], new["gradients"]
    missing_stages_old = sorted(set(new_gradients) - set(old_gradients))
    missing_stages_new = sorted(set(old_gradients) - set(new_gradients))
    gradients = {stage: compare_gradient_stage(old_gradients[stage], new_gradients[stage], stage) for stage in sorted(set(old_gradients) & set(new_gradients))}
    schema_match = (
        bool(old_tensors and new_tensors and old_gradients and new_gradients)
        and not (missing_old or missing_new or missing_stages_old or missing_stages_new)
        and all(row["shape_match"] for row in tensors.values())
        and all(row["schema_match"] for row in gradients.values())
    )
    finite = all(not row.get("old_nonfinite", 0) and not row.get("new_nonfinite", 0) for row in tensors.values()) and all(row["finite"] for row in gradients.values())
    return {
        "comparison_valid": not (differences or missing_metadata_old or missing_metadata_new) and rank_matches_filename and schema_match,
        "finite": finite,
        "metadata": {
            "matches": not (differences or missing_metadata_old or missing_metadata_new) and rank_matches_filename,
            "differences": differences,
            "ignored_fields": ignored,
            "required_missing_from_old": missing_metadata_old,
            "required_missing_from_new": missing_metadata_new,
            "rank_matches_filename": rank_matches_filename,
        },
        "tensor_missing_from_old": missing_old,
        "tensor_missing_from_new": missing_new,
        "gradient_stage_missing_from_old": missing_stages_old,
        "gradient_stage_missing_from_new": missing_stages_new,
        "tensors": tensors,
        "gradients": gradients,
    }


def rank_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Result directory does not exist: {directory}")
    files = {path.stem: path for path in directory.glob("rank*.pt") if re.fullmatch(r"rank\d+", path.stem) and path.is_file()}
    if not files:
        raise ValueError(f"No rank<N>.pt result files in {directory}")
    return files


def summarize(report: dict) -> str:
    lines = [f"Comparison valid: {report['comparison_valid']}; all reported values finite: {report['finite']}."]
    if report["rank_missing_from_old"] or report["rank_missing_from_new"]:
        lines.append(f"Missing ranks: old={report['rank_missing_from_old']}; new={report['rank_missing_from_new']}.")
    for rank, row in report["ranks"].items():
        if "error" in row:
            lines.append(f"{rank}: ERROR: {row['error']}")
            continue
        if not row["comparison_valid"]:
            lines.append(f"{rank}: INVALID metadata/schema; inspect JSON diagnostics.")
        for name, metrics in row["tensors"].items():
            if not name.startswith("input.") or not metrics.get("exact_equal", False):
                lines.append(f"{rank} {name}: maxabs={metrics.get('max_abs_diff')} relL2={metrics.get('relative_l2')} cosine={metrics.get('cosine')}")
        for stage, metrics in row["gradients"].items():
            sketch = metrics["sketch_metrics"] or {}
            lines.append(
                f"{rank} {stage} gradients: full norm ratio={metrics['full_gradient_norm']['new_over_old']} sketch relL2={sketch.get('relative_l2')} cosine={sketch.get('cosine')} full local SHA256 identity={metrics['full_local_gradient_hashes']['all_bytes_identical_by_sha256']}"
            )
    lines.append(GRADIENT_NOTE)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True, help="Old implementation result directory")
    parser.add_argument("--new", type=Path, required=True, help="New implementation result directory")
    parser.add_argument("--output", type=Path, required=True, help="Write JSON report here (not inside an input directory)")
    args = parser.parse_args(argv)
    try:
        old_dir, new_dir, output = args.old.resolve(), args.new.resolve(), args.output.resolve()
        if output == old_dir or output == new_dir or old_dir in output.parents or new_dir in output.parents:
            raise ValueError("--output must be outside both read-only result directories")
        old_files, new_files = rank_files(old_dir), rank_files(new_dir)
    except ValueError as error:
        parser.error(str(error))
    report: dict[str, Any] = {
        "format_version": 1,
        "old": str(old_dir),
        "new": str(new_dir),
        "metric_definition": "relative_l2 = ||new-old|| / ||old||; null denotes undefined/nonfinite metrics; both-zero cosine = 1",
        "gradient_note": GRADIENT_NOTE,
        "verdict": "Metrics only; no numerical parity tolerance is imposed.",
        "required_ranks": ["rank0", "rank1"],
        "rank_missing_from_old": sorted((set(new_files) | {"rank0", "rank1"}) - set(old_files)),
        "rank_missing_from_new": sorted((set(old_files) | {"rank0", "rank1"}) - set(new_files)),
        "ranks": {},
    }
    for rank in sorted(set(old_files) & set(new_files), key=lambda name: int(name[4:])):
        try:
            report["ranks"][rank] = compare_rank(load_result(old_files[rank]), load_result(new_files[rank]), expected_rank=int(rank[4:]))
        except (ValueError, TypeError, RuntimeError, OSError, EOFError, pickle.UnpicklingError) as error:
            report["ranks"][rank] = {"comparison_valid": False, "finite": False, "error": str(error)}
    report["comparison_valid"] = not (report["rank_missing_from_old"] or report["rank_missing_from_new"]) and all(row["comparison_valid"] for row in report["ranks"].values())
    report["finite"] = all(row["finite"] for row in report["ranks"].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(summarize(report))
    print(f"JSON report: {output}")
    return 0 if report["comparison_valid"] and report["finite"] else 1


if __name__ == "__main__":
    sys.exit(main())
