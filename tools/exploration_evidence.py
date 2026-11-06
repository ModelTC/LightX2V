"""CPU readers for original source/module/operator evidence used by acceptance.

Readers recompute supported statistics, never execute model backends. Task
collectors verify each actual run, its raw association, resources and cleanup;
measurements remain valid=False until that separate verification is complete.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path


def _numbers(values):
    if not values or any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("missing/nonfinite/nonpositive normal measurements")
    return values


def source_reference(inputs, paths):
    sources = {}
    for value in paths:
        path = Path(value).resolve()
        inputs.read(path)
        sources[str(path)] = inputs.hashes[path]
    return {
        "kind": "source",
        "subject": {"scope": "source", "input_kind": "source", "files": sources},
        "sources": list(sources),
        "limits": ["File identity and locator only; structural applicability and deferral sufficiency require semantic review."],
    }


def module_observation(inputs, report_path, raw_path):
    report_path, raw_path = Path(report_path).resolve(), Path(raw_path).resolve()
    report = inputs.json(report_path)
    raw = inputs.jsonl(raw_path)
    if report.get("kind") != "module_benchmark_v1" or not raw:
        raise ValueError("expected native module_benchmark_v1 and per-rank raw")
    if not isinstance(report.get("factory"), dict) or not report["factory"]:
        raise ValueError("module report needs task-recorded factory context")
    identity = report["identity"]
    if identity["input"].get("kind") not in {"real", "synthetic"}:
        raise ValueError("module input kind unavailable")
    parameters = report["parameters"]
    count = parameters["iterations"] * parameters["repeats"]
    if count <= 0 or parameters["warmup"] < 0:
        raise ValueError("invalid module repetition parameters")
    ranks = [row["rank"] for row in raw]
    if len(set(ranks)) != len(ranks) or set(ranks) != {row["rank"] for row in report["ranks"]}:
        raise ValueError("module rank set mismatch")
    native_meta = {row["rank"]: row for row in report["ranks"]}
    indices = None
    for row in raw:
        if set(row["groups"]["measurement_ranks"]) != set(ranks):
            raise ValueError("module raw does not cover the complete measurement group")
        if row["identity"] != identity or row["parameters"] != parameters or len(row["samples"]) != count:
            raise ValueError("module raw identity/parameters/sample count mismatch")
        if row["groups"] != native_meta[row["rank"]]["groups"] or row["correctness"] != native_meta[row["rank"]]["correctness"]:
            raise ValueError("module report/raw rank metadata mismatch")
        if row["correctness"].get("input_unchanged") is False:
            raise ValueError("module producer input was modified")
        order = [(sample["repeat"], sample["iteration"]) for sample in row["samples"]]
        if len(set(order)) != count or (indices is not None and order != indices):
            raise ValueError("module per-rank sample alignment mismatch")
        indices = order
        _numbers([sample["latency_ms"] for sample in row["samples"]])
    maxima = [max(row["samples"][i]["latency_ms"] for row in raw) for i in range(count)]
    if maxima != report["max_rank_latency_ms"]:
        raise ValueError("module report is inconsistent with original per-rank samples")
    for name, value in [("median", statistics.median(maxima)), ("mean", statistics.mean(maxima))]:
        if not math.isclose(report["latency_ms"][name], value, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("module aggregate differs from native samples")
    statuses = {row["correctness"]["status"] for row in raw}
    if not statuses <= {"strict_equal", "pending_review", "unavailable"}:
        raise ValueError("invalid module numerical verification")
    quality = "strict_equal" if statuses == {"strict_equal"} else "pending_review" if "pending_review" in statuses else "unavailable"
    return {
        "identity": identity,
        "candidate": report["factory"],
        "median": statistics.median(maxima),
        "mean": statistics.mean(maxima),
        "samples": count,
        "quality": quality,
        "sources": [str(report_path), str(raw_path)],
        "limits": [
            "Candidate/factory is the original producer's post-call context; legacy rank raw alone does not independently identify candidate dispatch.",
            "Numerical status is the native preflight record, not a new payload-error or perceptual re-evaluation.",
        ],
    }


def module_comparison(inputs, parents, candidates, statistic="median"):
    """Read (report, per-rank raw) pairs; the task must verify run validity."""
    parent = [module_observation(inputs, *args) for args in parents]
    candidate = [module_observation(inputs, *args) for args in candidates]
    rows = parent + candidate
    if not parent or not candidate or statistic not in {"median", "mean"}:
        raise ValueError("unsupported/empty module comparison")
    seen_runs = set()
    for row in rows:
        original = Path(row["sources"][1]).stat()
        identity = (original.st_dev, original.st_ino)
        if identity in seen_runs:
            raise ValueError("duplicate module run original; file aliases are not independent groups")
        seen_runs.add(identity)
    if any(row["identity"] != rows[0]["identity"] for row in rows):
        raise ValueError("module comparison changes global target/input/source identity")
    if any(row["candidate"] != parent[0]["candidate"] for row in parent) or any(row["candidate"] != candidate[0]["candidate"] for row in candidate):
        raise ValueError("module repeats contain different actual candidates")
    qa = {row["quality"] for row in candidate}
    return {
        "kind": "measurement",
        "valid": False,  # Native sample consistency alone does not verify actual runs.
        "subject": {
            "scope": "module",
            "input_kind": rows[0]["identity"]["input"]["kind"],
            "identity": rows[0]["identity"],
            "parent": parent[0]["candidate"],
            "candidate": candidate[0]["candidate"],
        },
        "values": {"parent": statistics.median(row[statistic] for row in parent), "candidate": statistics.median(row[statistic] for row in candidate)},
        "unit": "ms",
        "statistic": f"{len(parent)}/{len(candidate)}组；组{statistic}再取中位；不同作业只作对应范围首筛",
        "quality": "strict_equal" if qa == {"strict_equal"} else "pending_review" if "pending_review" in qa else "unavailable",
        "sources": sorted({path for row in rows for path in row["sources"]}),
        "limits": rows[0]["limits"],
    }


def operator_comparison(inputs, path, parent_backend, candidate_backend, canonical_case):
    path = Path(path).resolve()
    raw = inputs.jsonl(path)
    groups = []
    for backend in (parent_backend, candidate_backend):
        selected = [row for row in raw if row.get("case", {}).get("source", {}).get("canonical_case_id") == canonical_case and row["case"]["backend"] == backend]
        if not selected or any(row.get("status") != "ok" for row in selected):
            raise ValueError("operator selection includes missing/failed native rows")
        groups.append(selected)
    selected = groups[0] + groups[1]
    first = selected[0]
    case = first["case"]
    # These are the native operator producer's executable-input fields, separate
    # from adapter precision and descriptive source labels.
    input_fields = ("operator_family", "shape", "routing", "replay", "sparse", "requested_input_dtype")
    identity = {key: case.get(key) for key in input_fields}
    env_keys = ("python", "torch", "cuda_runtime", "device", "cuda_visible_devices", "gpu", "git_commit")
    environment = {key: first["environment"].get(key) for key in env_keys}
    provenance = []
    for row in selected:
        if row["case"]["source"] not in provenance:
            provenance.append(row["case"]["source"])
        if {key: row["case"].get(key) for key in input_fields} != identity or {key: row["environment"].get(key) for key in env_keys} != environment:
            raise ValueError("operator input/environment changes across comparison")
        if row["case"]["run"]["timing_mode"] != "event_list_single_sync":
            raise ValueError("unsupported operator timing mode")
    ids = []
    for group in groups:
        run_ids = [row["run_id"] for row in group]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("duplicate operator repetition")
        ids.append(set(run_ids))
        if any(row["case"]["run"] != group[0]["case"]["run"] or row["case"]["precision"] != group[0]["case"]["precision"] for row in group):
            raise ValueError("operator repetition changes seed/timing/precision")
    if ids[0] != ids[1] or groups[0][0]["case"]["run"] != groups[1][0]["case"]["run"]:
        raise ValueError("operator parent/candidate repetitions are not paired")
    # The operator format owns the metric key; unknown versions are not inferred.
    means = [[row["metrics"]["latency_ms_mean"] for row in group] for group in groups]
    for group in means:
        _numbers(group)
    checked = [row.get("correctness", {}).get("checked") for row in selected]
    return {
        "kind": "measurement",
        "valid": False,  # Native sample consistency alone does not verify actual runs.
        "subject": {
            "scope": "operator",
            "input_kind": "real" if case.get("replay") else "synthetic",
            "input_identity": identity,
            "shape": case["shape"],
            "source": case["source"],
            "provenance": provenance,
            "environment": environment,
            "parent": parent_backend,
            "candidate": candidate_backend,
            "precision": [row["case"]["precision"] for row in (groups[0][0], groups[1][0])],
        },
        "values": {"parent": statistics.median(means[0]), "candidate": statistics.median(means[1])},
        "unit": "ms",
        "statistic": f"{len(groups[0])}/{len(groups[1])}组mean再取中位",
        "quality": "unavailable" if not all(checked) else "pending_review",
        "sources": [str(path)],
        "limits": [
            "Recorded operator input/replay and adapter precision only; descriptive source labels do not prove real operands, full-model state, block/request gain, or independent correctness."
        ],
    }
