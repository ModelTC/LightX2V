#!/usr/bin/env python3
"""Aggregate completed LIBERO suite summaries without launching inference."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EPISODES_PER_SUITE = 500


def _parse_suites(value: str) -> tuple[str, ...]:
    suites = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [suite for suite in suites if suite not in SUITES]
    if not suites or unknown or len(set(suites)) != len(suites):
        raise ValueError(f"invalid LIBERO suite selection: {value!r}")
    return suites


def _parse_status(values: list[str], suites: tuple[str, ...]) -> dict[str, int]:
    statuses: dict[str, int] = {}
    for value in values:
        suite, separator, return_code = value.partition("=")
        if not separator or suite not in suites or suite in statuses:
            raise ValueError(f"invalid worker status: {value!r}")
        statuses[suite] = int(return_code)
    missing = set(suites) - statuses.keys()
    if missing:
        raise ValueError(f"missing worker status for: {sorted(missing)}")
    return statuses


def aggregate(output_root: Path, suites: tuple[str, ...], statuses: dict[str, int], output_file: Path) -> tuple[dict, list[str]]:
    errors: list[str] = []
    shards: dict[str, dict] = {}
    successes = 0
    completed = 0
    rates: list[float] = []

    for suite in suites:
        summary_path = output_root / suite / "summary.json"
        if not summary_path.is_file():
            errors.append(f"{suite}: missing {summary_path}")
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{suite}: invalid summary: {exc}")
            continue

        try:
            shard_expected = int(summary["expected_episodes"])
            shard_completed = int(summary["completed_episodes"])
            shard_successes = int(summary["successes"])
            rate = float(summary["success_rate"])
            status = summary["status"]
            official_protocol = summary["official_protocol"]
            protocol_id = summary["protocol_id"]
            rollout_errors = int(summary["errors"])
            artifact_errors = int(summary["artifact_errors"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{suite}: incompatible summary schema: {exc}")
            continue
        if statuses[suite] != 0:
            errors.append(f"{suite}: worker exit code {statuses[suite]}")
        if status not in {"complete", "complete_with_errors"}:
            errors.append(f"{suite}: summary status is {status!r}")
        if official_protocol is not True:
            errors.append(f"{suite}: evaluator did not mark the shard as official protocol")
        if not isinstance(protocol_id, str) or not protocol_id:
            errors.append(f"{suite}: missing protocol_id")
        if rollout_errors or artifact_errors:
            errors.append(f"{suite}: rollout_errors={rollout_errors} artifact_errors={artifact_errors}")
        if shard_expected != EPISODES_PER_SUITE or shard_completed != EPISODES_PER_SUITE:
            errors.append(f"{suite}: expected official {EPISODES_PER_SUITE}/{EPISODES_PER_SUITE} episodes, got {shard_completed}/{shard_expected}")
        rates.append(rate)
        completed += shard_completed
        successes += shard_successes
        shards[suite] = {
            "output_dir": str(output_root / suite),
            "expected_episodes": shard_expected,
            "completed_episodes": shard_completed,
            "successes": shard_successes,
            "success_rate": rate,
            "protocol_id": protocol_id,
        }

    expected = EPISODES_PER_SUITE * len(suites)
    payload = {
        "schema_version": 1,
        "status": "complete" if not errors else "invalid",
        "expected_episodes": expected,
        "completed_episodes": completed,
        "successes": successes,
        "failures": completed - successes,
        "success_rate": successes / completed * 100.0 if completed else None,
        "mean_suite_success_rate": sum(rates) / len(rates) if len(rates) == len(suites) else None,
        "shards": shards,
        "validation_errors": errors,
        "updated_at_unix": time.time(),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(f".{output_file.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_file)
    return payload, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suites", required=True)
    parser.add_argument("--worker-status", action="append", default=[])
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    try:
        output_root = args.output_root.expanduser().resolve()
        suites = _parse_suites(args.suites)
        statuses = _parse_status(args.worker_status, suites)
        output_file = args.output_file.expanduser().resolve() if args.output_file else output_root / "parallel_summary.json"
        payload, errors = aggregate(output_root, suites, statuses, output_file)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print("\nLIBERO suite summary")
    for suite in suites:
        shard = payload["shards"].get(suite, {})
        rate = shard.get("success_rate")
        rate_text = "n/a" if rate is None else f"{float(rate):.2f}%"
        print(f"  {suite:<18} {shard.get('successes', 0):>3}/{shard.get('expected_episodes', 500):<3} {rate_text:>8}")
    overall = payload["success_rate"]
    overall_text = "n/a" if overall is None else f"{overall:.2f}%"
    print(f"  {'overall':<18} {payload['successes']:>3}/{payload['expected_episodes']:<3} {overall_text:>8}")
    print(f"  summary: {output_file}")
    for error in errors:
        print(f"error: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
