"""Quantitative, resumable local evaluation for the pi0.5-LIBERO policy."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import EpisodeKey, atomic_write_json, read_episode_records, save_evaluation_episode, write_aggregate_outputs
from .libero_protocol import (
    OFFICIAL_BENCHMARKS,
    OFFICIAL_RESULTS,
    EvaluationConfig,
    TaskSpec,
    build_task_inputs_manifest,
    build_task_specs,
    configure_libero,
    create_environment,
    ensure_task_inputs_manifest,
    expected_episode_keys,
    is_official_protocol,
    load_evaluation_config,
    load_policy_config,
    load_task_init_states,
    resolved_protocol,
    run_episode,
)
from .openpi_runner import OpenPIPolicy

LOGGER = logging.getLogger(__name__)


def _suite_keys(specs: list[TaskSpec], config: EvaluationConfig) -> tuple[EpisodeKey, ...]:
    return tuple((spec.benchmark, spec.task_id, init_state_id) for spec in specs for init_state_id in range(config.num_trials_per_task))


def _resume_prefixes(
    task_specs: dict[str, list[TaskSpec]],
    config: EvaluationConfig,
    records: Mapping[EpisodeKey, Mapping[str, Any]],
) -> dict[str, int]:
    """Require a committed prefix so one saved RNG state resumes the suite exactly."""
    prefixes: dict[str, int] = {}
    for benchmark in config.benchmarks:
        ordered = _suite_keys(task_specs[benchmark], config)
        present = [key in records for key in ordered]
        prefix = 0
        while prefix < len(present) and present[prefix]:
            prefix += 1
        if any(present[prefix:]):
            raise RuntimeError(f"Episode-level resume for {benchmark} requires a contiguous prefix; a later episode exists after the first missing metrics.json.")
        if 0 < prefix < len(ordered) and not records[ordered[prefix - 1]].get("policy_rng_state_after"):
            raise RuntimeError(f"Partial suite {benchmark} predates RNG-checkpoint resume. Keep it for analysis, but use a fresh output directory to continue evaluation.")
        prefixes[benchmark] = prefix
    return prefixes


def _task_metrics(records: list[Mapping[str, Any]], expected: int) -> dict[str, Any]:
    completed = len(records)
    successes = sum(bool(record.get("success")) for record in records)
    errors = sum(record.get("termination_reason") == "exception" for record in records)
    return {
        "expected": expected,
        "completed": completed,
        "successes": successes,
        "failures": completed - successes,
        "errors": errors,
        "success_rate": successes / expected * 100.0 if completed == expected and expected else None,
        "success_rate_completed": successes / completed * 100.0 if completed else None,
    }


def build_summary(
    records: Mapping[EpisodeKey, Mapping[str, Any]],
    task_specs: dict[str, list[TaskSpec]],
    config: EvaluationConfig,
    protocol_id: str,
) -> dict[str, Any]:
    official = is_official_protocol(config)
    per_benchmark: dict[str, Any] = {}
    per_task: dict[str, Any] = {}
    expected_total = 0

    for benchmark in config.benchmarks:
        specs = task_specs[benchmark]
        expected = len(specs) * config.num_trials_per_task
        expected_total += expected
        suite_records = [record for key, record in records.items() if key[0] == benchmark]
        suite_metrics = _task_metrics(suite_records, expected)
        reference = OFFICIAL_RESULTS.get(benchmark)
        rate = suite_metrics["success_rate"]
        suite_metrics.update(
            {
                "official_reference": reference,
                "official_comparable": official,
                "delta_percentage_points": rate - reference["success_rate"] if official and rate is not None and reference else None,
            }
        )
        per_benchmark[benchmark] = suite_metrics

        for spec in specs:
            task_records = [record for key, record in records.items() if key[0] == benchmark and key[1] == spec.task_id]
            metrics = _task_metrics(task_records, config.num_trials_per_task)
            metrics.update(
                {
                    "task_name": str(spec.task.name),
                    "task_description": str(spec.task.language),
                }
            )
            per_task[f"{benchmark}/task_{spec.task_id:02d}"] = metrics

    completed_total = len(records)
    successes = sum(bool(record.get("success")) for record in records.values())
    errors = sum(record.get("termination_reason") == "exception" for record in records.values())
    artifact_errors = sum(len(record.get("artifact_errors", [])) for record in records.values())
    complete = completed_total == expected_total
    suite_rates = [entry["success_rate"] for entry in per_benchmark.values()]
    mean_suite_rate = sum(suite_rates) / len(suite_rates) if complete and suite_rates and all(rate is not None for rate in suite_rates) else None
    return {
        "schema_version": 1,
        "status": "complete_with_errors" if complete and (errors or artifact_errors) else "complete" if complete else "in_progress",
        "protocol_id": protocol_id,
        "official_protocol": official,
        "resume_granularity": "episode_prefix",
        "policy_rng_scope": "per_suite_continuous",
        "full_official_table": official and set(config.benchmarks) == set(OFFICIAL_BENCHMARKS),
        "expected_episodes": expected_total,
        "completed_episodes": completed_total,
        "successes": successes,
        "failures": completed_total - successes,
        "errors": errors,
        "artifact_errors": artifact_errors,
        "success_rate": successes / expected_total * 100.0 if complete and expected_total else None,
        "success_rate_completed": successes / completed_total * 100.0 if completed_total else None,
        "mean_suite_success_rate": mean_suite_rate,
        "official_reference": {"comparable": official, "results": OFFICIAL_RESULTS},
        "per_benchmark": per_benchmark,
        "per_task": per_task,
        "updated_at_unix": time.time(),
    }


def _write_current_outputs(
    output_dir: Path,
    records: Mapping[EpisodeKey, Mapping[str, Any]],
    task_specs: dict[str, list[TaskSpec]],
    config: EvaluationConfig,
    protocol_id: str,
) -> dict[str, Any]:
    summary = build_summary(records, task_specs, config, protocol_id)
    write_aggregate_outputs(output_dir, records.values(), summary)
    return summary


def _prepare_output_directory(output_dir: Path, resolved: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_eval_config.json"
    if resolved_path.is_file():
        try:
            existing = json.loads(resolved_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read existing resolved evaluation config: {resolved_path}") from exc
        if existing.get("protocol_id") != resolved["protocol_id"]:
            committed = next((output_dir / "episodes").rglob("metrics.json"), None) if (output_dir / "episodes").is_dir() else None
            if committed is not None:
                raise RuntimeError(f"Evaluation output contains records for a different protocol: {committed}")
            LOGGER.warning("Replacing unused resolved evaluation config: %s", resolved_path)
    atomic_write_json(resolved_path, resolved)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    config = load_evaluation_config(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    np.random.seed(config.env_seed)
    runtime = configure_libero(config.libero_root, config.libero_config_dir)
    task_specs = build_task_specs(runtime, config)
    task_inputs_manifest = build_task_inputs_manifest(task_specs, config)
    resolved, protocol_id = resolved_protocol(
        config,
        model_path=Path(args.model_path),
        config_json=Path(args.config_json),
    )
    _prepare_output_directory(output_dir, resolved)

    all_keys = expected_episode_keys(task_specs, config)
    records = read_episode_records(output_dir, all_keys, protocol_id)
    if records and not config.resume:
        raise FileExistsError(f"Evaluation output already contains {len(records)} committed episodes; use --resume or a new output directory.")
    ensure_task_inputs_manifest(output_dir, task_inputs_manifest, records, task_specs)
    prefixes = _resume_prefixes(task_specs, config, records)
    summary = _write_current_outputs(output_dir, records, task_specs, config, protocol_id)
    if len(records) == len(all_keys):
        LOGGER.info("All requested LIBERO episodes are already complete: %s", output_dir)
        return summary

    policy = OpenPIPolicy(
        load_policy_config(
            Path(args.config_json),
            Path(args.model_path),
            seed=config.policy_seed,
        )
    )
    global_indices = {key: index for index, key in enumerate(all_keys)}
    try:
        for benchmark_index, benchmark in enumerate(config.benchmarks):
            ordered_suite_keys = _suite_keys(task_specs[benchmark], config)
            prefix = prefixes[benchmark]
            if prefix == len(ordered_suite_keys):
                LOGGER.info("Episode resume: skipping complete suite %s", benchmark)
                continue

            policy.reset()
            # Official evaluation starts each suite in its own process. Mirror
            # that process-level NumPy seed when several suites share a worker.
            np.random.seed(config.env_seed)
            if prefix:
                policy.import_rng_state(str(records[ordered_suite_keys[prefix - 1]]["policy_rng_state_after"]))
                LOGGER.info("Episode resume: %s continues after %d/%d episodes", benchmark, prefix, len(ordered_suite_keys))
            else:
                LOGGER.info("Starting suite %s", benchmark)

            for spec in task_specs[benchmark]:
                task_keys = [(benchmark, spec.task_id, init_state_id) for init_state_id in range(config.num_trials_per_task)]
                if all(key in records for key in task_keys):
                    continue

                initial_states, init_states_loader = load_task_init_states(spec)
                if config.num_trials_per_task > len(initial_states):
                    raise ValueError(f"{benchmark} task {spec.task_id} has {len(initial_states)} init states, but {config.num_trials_per_task} trials were requested")
                env = create_environment(runtime, spec, config.render_size, config.env_seed)
                try:
                    for init_state_id, key in enumerate(task_keys):
                        if key in records:
                            # Reproduce the official per-task reset count before a
                            # partially completed task's first pending episode.
                            env.reset()
                            env.set_init_state(initial_states[init_state_id])
                            continue

                        episode, frames, actions = run_episode(
                            policy=policy,
                            env=env,
                            initial_state=initial_states[init_state_id],
                            task_description=str(spec.task.language),
                            max_steps=config.max_steps[benchmark],
                            num_steps_wait=config.num_steps_wait,
                            actions_per_plan=config.actions_per_plan,
                            collect_frames=config.video_policy != "none",
                        )
                        record = {
                            "schema_version": 1,
                            "protocol_id": protocol_id,
                            "global_episode_index": global_indices[key],
                            "benchmark_index": benchmark_index,
                            "benchmark": benchmark,
                            "task_id": spec.task_id,
                            "task_name": str(spec.task.name),
                            "task_description": str(spec.task.language),
                            "bddl_file": str(spec.bddl_path),
                            "init_states_file": str(spec.init_states_path),
                            "init_states_loader": init_states_loader,
                            "init_state_id": init_state_id,
                            "env_seed": config.env_seed,
                            "policy_seed": config.policy_seed,
                            "policy_rng_scope": "per_suite_continuous",
                            "policy_rng_state_after": policy.export_rng_state(),
                            "max_policy_steps": config.max_steps[benchmark],
                            "actions_per_plan": config.actions_per_plan,
                            **episode,
                        }
                        saved = save_evaluation_episode(
                            output_dir,
                            record,
                            frames,
                            actions,
                            video_policy=config.video_policy,
                            video_fps=config.video_fps,
                            save_actions=config.save_actions,
                        )
                        records[key] = saved
                        summary = _write_current_outputs(output_dir, records, task_specs, config, protocol_id)
                        LOGGER.info(
                            "suite=%s task=%02d init=%02d success=%s completed=%d/%d",
                            benchmark,
                            spec.task_id,
                            init_state_id,
                            saved["success"],
                            summary["completed_episodes"],
                            summary["expected_episodes"],
                        )
                        if saved["termination_reason"] == "exception" and config.fail_fast:
                            raise RuntimeError(f"Fail-fast: {benchmark} task {spec.task_id} init {init_state_id} failed: {saved['exception_type']}: {saved['exception_message']}")
                finally:
                    env.close()
    finally:
        summary = _write_current_outputs(output_dir, records, task_specs, config, protocol_id)

    LOGGER.info(
        "LIBERO evaluation finished: successes=%d/%d success_rate=%s mean_suite_success_rate=%s",
        summary["successes"],
        summary["expected_episodes"],
        summary["success_rate"],
        summary["mean_suite_success_rate"],
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local quantitative pi0.5-LIBERO evaluation")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--libero-config-dir", type=Path, required=True)
    parser.add_argument("--benchmarks")
    parser.add_argument("--task-ids")
    parser.add_argument("--num-trials-per-task", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--video-policy", choices=("all", "failures", "none"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-actions", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    summary = run_evaluation(build_parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
