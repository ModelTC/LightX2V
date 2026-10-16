"""FewStep-style scalar summaries, with full metrics retained locally.

Only presentation is changed: rewards, advantage computation and optimization
receive the original values. No W&B connection is made on module import.
"""

import json
import logging
import math
import re
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from uuid import uuid4

from lightx2v_rl.paths import repository_path

logger = logging.getLogger(__name__)

TRAIN_METRICS = {
    "actor/loss": "train/loss",
    "actor/policy_loss": "train/policy_loss",
    "actor/kl_loss": "train/kl_loss",
    "actor/approx_kl": "train/approx_kl",
    "actor/clip_frac": "train/clip_frac",
    "actor/grad_norm": "optim/grad_norm",
    "actor/lr": "optim/lr",
    "training/epoch": "train/epoch",
    "critic/rewards/zero_std_ratio": "sampling/zero_std_ratio",
    "critic/rewards/group_size": "sampling/group_size",
    "perf/total_num_images": "sampling/images",
    "train/nonfinite_grad_steps": "health/nonfinite_grad_steps",
    "training/rollout_failure/evicted_groups": "health/failed_groups",
    "training/rollout_failure/refill_rounds": "health/refill_rounds",
    "timing_s/gen": "time/rollout_and_reward_s",
    "timing_s/update_actor": "time/update_s",
    "timing_s/step": "time/step_s",
    "timing_s/testing": "time/eval_s",
    "timing_s/save_checkpoint": "time/checkpoint_s",
}

# One unweighted value per reward, rather than raw/calibrated/weighted copies.
# The raw per-axis scores and weighted contributions remain in metrics JSONL.
REWARD_METRICS = {
    "unifiedreward_normalized": "unifiedreward",
    "hpsv3_general_raw": "hpsv3_raw",
    "hpsv3_percentile_raw": "hpsv3_percentile",
    "videoalign_ta_raw": "videoalign_ta",
    "videoalign_mq_raw": "videoalign_mq",
    "videoalign_vq_raw": "videoalign_vq",
    "unifiedreward_range_clipped": "ur_clipped_fraction",
}
VAL_METRIC = re.compile(r"^val-(?:core|aux)/(.+)/([^/]+)/mean@(\d+)$")


def compact_metrics(data):
    """Select readable scalars; never infer T2I/I2I from iteration parity."""
    result = {target: data[source] for source, target in TRAIN_METRICS.items() if source in data}
    task_flag = data.get("critic/task_is_i2i/mean")
    task = {0: "t2i", 1: "i2i"}.get(task_flag, "mixed" if task_flag is not None else None)
    prefix = f"train_{task}" if task else "train"
    for source, target in {"rewards": "reward", **REWARD_METRICS}.items():
        key = f"critic/{source}/mean"
        if key in data:
            result[f"{prefix}/{target}"] = data[key]
    images = result.get("sampling/images")
    group_size = result.get("sampling/group_size")
    if images is not None and group_size is not None and math.isfinite(group_size) and group_size > 0:
        result["sampling/prompts"] = images / group_size

    # Keep the actual mean over all validation samples, not best-of/majority or
    # redundant std/min/max. Preserve dataset identity if several share a task.
    for key, value in data.items():
        match = VAL_METRIC.fullmatch(key)
        if not match:
            continue
        source, metric, _ = match.groups()
        target = {"reward": "reward", **REWARD_METRICS}.get(metric)
        if target is None:
            continue
        if source in ("opengpt4o/t2i", "opengpt4o/i2i"):
            group = "eval_" + source.rsplit("/", 1)[1]
        else:
            group = "eval/" + source
        result[f"{group}/{target}"] = value

    if result:
        # NaN values alone may disappear from charts; this finite counter keeps
        # numerical failures visible. Raw diagnostics retain the exact values.
        bad = [key for key, value in result.items() if isinstance(value, Real) and not math.isfinite(value)]
        result["health/nonfinite_core_metrics"] = len(bad)
        if bad:
            logger.warning("Nonfinite core training/eval metrics: %s", ", ".join(bad))
    return result


def _json_value(value):
    """Standard JSON, preserving NaN/Inf as strings instead of silently dropping them."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Real) and not math.isfinite(value):
        return str(float(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _create_backend(**kwargs):
    from verl.utils.tracking import Tracking

    return Tracking(**kwargs)


class FewStepTracking:
    """Drop-in v1 tracking adapter; media logging stays with verl-omni."""

    def __init__(self, project_name, experiment_name, default_backend="console", config=None):
        config = config or {}
        options = config.get("trainer", {}).get("metric_logging", {})
        self.mode = options.get("mode", "compact")
        if self.mode not in ("compact", "full"):
            raise ValueError("trainer.metric_logging.mode must be compact or full")
        self.raw_path = None
        if options.get("save_raw", True):
            directory = Path(repository_path(options.get("raw_dir", "lightx2v_rl/outputs/metrics")))
            directory.mkdir(parents=True, exist_ok=True)
            name = re.sub(r"[^a-zA-Z0-9_.-]", "_", experiment_name)[:100]
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.raw_path = directory / f"{name}-{timestamp}-{uuid4().hex[:8]}.jsonl"
            self.raw_path.touch(exist_ok=False)
            print(f"Full training metrics: {self.raw_path}", flush=True)
        self._backend = _create_backend(
            project_name=project_name,
            experiment_name=experiment_name,
            default_backend=default_backend,
            config=config,
        )

    def log(self, data, step, backend=None):
        if self.raw_path is not None:
            record = {"step": step, "metrics": data}
            with self.raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_json_value(record), ensure_ascii=False, allow_nan=False) + "\n")
        payload = compact_metrics(data) if self.mode == "compact" else data
        self._backend.log(data=payload, step=step, backend=backend)

    def finish(self, exit_code=0):
        backend = getattr(self, "_backend", None)
        if backend is not None:
            backend.finish(exit_code=exit_code)

    def __del__(self):
        self.finish()
