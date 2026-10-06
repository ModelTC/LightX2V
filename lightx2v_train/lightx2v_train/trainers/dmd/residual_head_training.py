"""Frozen-fake residual fitting and independent held-out correction gates.

This strategy deliberately adds head fitting and held-out work; it is not a
matched-compute baseline. Only the existing fake-fit batches train the head.
The student's next batch and the gate's held-out batch are both independent.
"""

import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from statistics import mean, stdev

import torch
import torch.distributed as dist
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.runtime.distributed import get_sequence_parallel_world_size
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value

from .head_diagnostics import residual_risk_statistics, student_direction_statistics
from .residual_head import NoiseBinGate, TokenResidualHead


def _detach_condition(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_condition(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_detach_condition(item) for item in value)
    return value


@dataclass
class _FitBatch:
    generated: torch.Tensor
    renoised: torch.Tensor
    sigma: torch.Tensor
    condition: object


def _distributed_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _gather_records(records):
    """Gather tiny detached diagnostics, not latent tensors, at fixed call sites."""
    if dist.is_available() and dist.is_initialized():
        rank_records = [None] * dist.get_world_size()
        dist.all_gather_object(rank_records, records)
        return [record for batch in rank_records for record in batch]
    return records


def _paired_delta_summary(records, risk_name):
    # A rank's batched query is the conservative independent unit. Never use
    # latent pixels (or multiple items in one query) as uncertainty replicates.
    queries = {}
    for record in records:
        if record["fake_mse"] is None or record[risk_name] is None:
            continue
        key = (record["rank"], record["query_index"])
        queries.setdefault(key, []).append(record["fake_mse"] - record[risk_name])
    deltas = [mean(values) for values in queries.values()]
    valid = len(deltas) > 1
    finite_records = [record for record in records if record["fake_mse"] is not None and record[risk_name] is not None]
    return {
        "mean": mean(record["fake_mse"] - record[risk_name] for record in finite_records) if finite_records else None,
        "query_mean": mean(deltas) if deltas else None,
        "std": stdev(deltas) if valid else None,
        "stderr": stdev(deltas) / math.sqrt(len(deltas)) if valid else None,
        "uncertainty_defined": valid,
        "independent_query_count": len(deltas),
        "uncertainty_unit": "rank_query_mean",
    }


def _finite_mean(values):
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else None


def _risk_summary_by_bin(records, noise_bins):
    summaries = []
    risk_names = (
        "fake_mse",
        "full_corrected_mse",
        "selected_lambda_mse",
        "residual_head_dot",
        "head_energy",
        "optimal_lambda",
        "optimal_lambda_mse",
        "candidate_mse_0",
        "candidate_mse_025",
        "candidate_mse_05",
        "candidate_mse_075",
        "candidate_mse_1",
    )
    for index in range(noise_bins):
        selected = [record for record in records if record["bin"] == index]
        summary = {"bin": index, "sample_count": len(selected)}
        if selected:
            summary.update({name: _finite_mean(record[name] for record in selected) for name in risk_names})
            # Ratio of pooled sufficient statistics is not the average of the
            # per-sample clipped optima. Both are counterfactual diagnostics.
            pooled_valid = all(record[name] is not None for record in selected for name in ("fake_mse", "residual_head_dot", "head_energy")) and summary["head_energy"] > 0
            pooled_lambda = max(0.0, min(1.0, summary["residual_head_dot"] / summary["head_energy"])) if pooled_valid else None
            summary.update(
                {
                    "lambda_applied_mean": mean(record["lambda_actual"] for record in selected),
                    "positive_lambda_sample_count": sum(record["lambda_actual"] > 0 for record in selected),
                    "optimal_lambda_valid_sample_count": sum(record["optimal_lambda_valid"] for record in selected),
                    "pooled_optimal_lambda": pooled_lambda,
                    "pooled_optimal_lambda_valid": pooled_valid,
                    "pooled_optimal_lambda_mse": max(0.0, summary["fake_mse"] - 2 * pooled_lambda * summary["residual_head_dot"] + pooled_lambda**2 * summary["head_energy"]) if pooled_valid else None,
                    "full_correction_delta": _paired_delta_summary(selected, "full_corrected_mse"),
                    "applied_correction_delta": _paired_delta_summary(selected, "selected_lambda_mse"),
                }
            )
        summaries.append(summary)
    return summaries


def _student_scalar_metrics(records):
    metrics = {}
    for name in (
        "correction_rms",
        "applied_correction_rms",
        "correction_direction_norm_ratio",
        "direction_cosine",
        "direction_angle_degrees",
        "direction_raw_rms",
        "direction_corrected_rms",
    ):
        valid_name = (
            "correction_direction_norm_ratio_valid" if name == "correction_direction_norm_ratio" else "direction_cosine_valid" if name in {"direction_cosine", "direction_angle_degrees"} else None
        )
        value = _finite_mean(record[name] for record in records if valid_name is None or record[valid_name])
        # Validity fractions distinguish undefined quantities from finite zero
        # placeholders required by the trainer's scalar metric interface.
        metrics[f"head_student_{name}"] = value if value is not None else 0.0
    metrics["head_student_direction_valid_fraction"] = mean(record["direction_cosine_valid"] for record in records) if records else 0.0
    metrics["head_student_norm_ratio_valid_fraction"] = mean(record["correction_direction_norm_ratio_valid"] for record in records) if records else 0.0
    metrics["head_student_lambda_applied_mean"] = mean(record["lambda_actual"] for record in records) if records else 0.0
    return metrics


class ResidualHeadTraining:
    """Auxiliary head outside FSDP, with DDP-synchronized head gradients."""

    def __init__(self, trainer, config):
        if trainer.trainer_name != "dmd":
            raise ValueError("training.dmd.residual_head currently supports only the standard dmd trainer.")
        if getattr(trainer.student, "projected_dmd", False):
            raise ValueError("Residual-head DMD uses F-lambda*C-T directly; disable projected_dmd.")
        if get_sequence_parallel_world_size() != 1:
            raise ValueError("Residual-head DMD currently requires sequence_parallel.size=1.")
        self.trainer = trainer
        self.config = config
        self.device = torch.device(trainer.student.device)
        self.feature_dim, self.latent_channels, self.patch_size, head_type = self._head_spec(trainer)
        # CPU-only initialization restores the training RNG and gives every
        # rank exactly the same initial weights without seeding other GPUs.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.module = head_type(
                self.feature_dim,
                self.latent_channels,
                self.patch_size,
                config.hidden_dim,
            ).to(device=self.device, dtype=torch.float32)
        self.head = self.module
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            device_ids = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else None
            self.head = DistributedDataParallel(self.module, device_ids=device_ids, broadcast_buffers=False)
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.gate = NoiseBinGate(config, self.device)
        self.collecting_fit = False
        self.fit_batches = []
        self._student_query = None
        self._student_records = []
        self._student_query_index = 0
        # Globally pooled sample counters are identical on every rank and are
        # auxiliary checkpoint state, separate from the gate's round counts.
        self._usage_counts = torch.zeros(3, config.noise_bins, device=self.device, dtype=torch.int64)
        self._student_updates = 0
        self._positive_lambda_updates = 0
        self._nonzero_correction_updates = 0
        self._usage_history_complete = True
        self._head_optimizer_updates = 0
        self._head_fit_microbatches = 0
        self._fit_history_complete = True
        self._last_fit_metrics = {}

    def _head_spec(self, trainer):
        if not callable(getattr(trainer.fake_model, "predict_velocity_with_features", None)):
            raise ValueError("Residual-head DMD requires a native Wan predict_velocity_with_features model.")
        return int(trainer.fake_model.transformer.dim), int(trainer.fake_model._latent_channels()), tuple(trainer.fake_model.patch_size), TokenResidualHead

    def _report(self, event, report):
        if _distributed_rank() == 0:
            modality = getattr(self, "modality", None)
            if modality is not None:
                report = {"modality": modality, **report}
            logger.info(f"[head][{event}] {{}}", json.dumps(report, separators=(",", ":"), allow_nan=False))

    def checkpoint_metadata(self):
        return {
            **self.config.checkpoint_metadata(),
            "feature_dim": self.feature_dim,
            "latent_channels": self.latent_channels,
            "patch_size": list(self.patch_size),
        }

    def state_dict(self):
        return {
            "head": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gate": self.gate.state_dict(),
            "fit_progress": {
                "optimizer_updates": self._head_optimizer_updates,
                "microbatches": self._head_fit_microbatches,
                "history_complete": self._fit_history_complete,
            },
            "usage": {
                "counts": self._usage_counts.clone(),
                "student_updates": self._student_updates,
                "positive_lambda_updates": self._positive_lambda_updates,
                "nonzero_correction_updates": self._nonzero_correction_updates,
                "history_complete": self._usage_history_complete,
            },
        }

    def load_state_dict(self, state):
        required = {"head", "optimizer", "gate"}
        if not isinstance(state, dict) or not required.issubset(state):
            raise RuntimeError("Residual-head checkpoint must contain head, optimizer, and gate state.")
        self.module.load_state_dict(state["head"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.gate.load_state_dict(state["gate"], strict=True)
        fit_progress = state.get("fit_progress")
        self._head_optimizer_updates = self._head_fit_microbatches = 0
        self._fit_history_complete = fit_progress is not None
        self._last_fit_metrics = {}
        if fit_progress is not None:
            if not isinstance(fit_progress, dict) or not {"optimizer_updates", "microbatches"}.issubset(fit_progress):
                raise RuntimeError("Residual-head fit progress must contain optimizer_updates and microbatches.")
            values = [fit_progress[name] for name in ("optimizer_updates", "microbatches")]
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                raise RuntimeError("Residual-head fit progress counters must be non-negative integers.")
            self._head_optimizer_updates, self._head_fit_microbatches = values
            self._fit_history_complete = bool(fit_progress.get("history_complete", True))
        usage = state.get("usage")
        self._usage_counts.zero_()
        self._student_updates = self._positive_lambda_updates = self._nonzero_correction_updates = 0
        self._usage_history_complete = usage is not None
        if usage is not None:
            if not isinstance(usage, dict) or not {"counts", "student_updates", "positive_lambda_updates", "nonzero_correction_updates"}.issubset(usage):
                raise RuntimeError("Residual-head usage state must contain counts and all update counters.")
            counts = torch.as_tensor(usage["counts"], device=self.device)
            if counts.shape != self._usage_counts.shape or counts.dtype != torch.int64 or (counts < 0).any():
                raise RuntimeError("Residual-head usage counts must be non-negative int64 [3, noise_bins].")
            update_counts = [usage[name] for name in ("student_updates", "positive_lambda_updates", "nonzero_correction_updates")]
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in update_counts):
                raise RuntimeError("Residual-head usage update counters must be non-negative integers.")
            self._usage_counts.copy_(counts)
            self._student_updates, self._positive_lambda_updates, self._nonzero_correction_updates = update_counts
            self._usage_history_complete = bool(usage.get("history_complete", True))
        elif _distributed_rank() == 0:
            logger.warning("[head] checkpoint has no usage counters; historic usage is unknown and resumed usage starts at zero")
        # Legacy checkpoints had no usage counters; resume explicitly starts
        # them at zero rather than inventing historic use from gate lambdas.
        self.clear_student_query()
        self._student_records.clear()
        self._student_query_index = 0

    def remember_fit(self, generated, renoised, sigma, condition):
        if self.collecting_fit:
            if not torch.is_tensor(generated) or generated.ndim != 5:
                raise ValueError("Residual-head DMD requires [B,C,T,H,W] tensor latents.")
            # fit() only consumes the first fit_steps queries when enough are
            # available. Do not retain or rescore the unused tail when critic
            # gradient accumulation produces many more queries (e.g. 16*5).
            # With fewer queries, fit() keeps its existing replay behavior.
            if len(self.fit_batches) >= self.config.fit_steps:
                return
            self.fit_batches.append(
                _FitBatch(
                    generated.detach(),
                    renoised.detach(),
                    sigma.detach(),
                    _detach_condition(condition),
                )
            )

    @torch.no_grad()
    def _fake_x0_features(self, renoised, sigma, condition):
        self.trainer.fake.set_training(False)
        velocity, features = self.trainer.fake_model.predict_velocity_with_features(
            renoised,
            sigma,
            condition,
        )
        fake_x0 = self.trainer.student.x0_from_velocity(
            renoised.float(),
            velocity.float(),
            sigma.float(),
        ).float()
        return fake_x0.detach(), features.detach()

    @torch.no_grad()
    def predict_corrected_fake(self, renoised, sigma, condition):
        fake_x0, features = self._fake_x0_features(renoised, sigma, condition)
        # Inference uses the underlying module: DDP collectives belong only to
        # head fitting, not the frozen student/held-out scoring paths.
        correction = self.module(features, sigma, tuple(renoised.shape))
        lam = self.gate.lambda_for(sigma).float()
        if lam.ndim == 0:
            lam = lam.reshape(1)
        lam = lam.reshape(-1, *([1] * (fake_x0.ndim - 1)))
        self._student_query = (fake_x0.detach(), correction.detach(), sigma.detach(), lam.detach())
        return fake_x0 - lam * correction.float()

    def clear_student_query(self):
        self._student_query = None

    def _sample_records(self, statistics, sigma, query_index):
        batch_size = statistics["fake_mse"].numel()
        sigma = torch.as_tensor(sigma, device=self.device, dtype=torch.float32).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(batch_size)
        bins = self.gate._bin_indices(sigma)
        names = list(statistics)
        boolean_names = {name for name, value in statistics.items() if value.dtype == torch.bool or name.endswith("_valid")}
        # One small device-to-host copy per query, never a copy of the latents.
        values = torch.stack([sigma, bins.float(), *(statistics[name].float() for name in names)], dim=1).cpu().tolist()
        records = []
        for sample_index, row in enumerate(values):
            record = {"rank": _distributed_rank(), "query_index": query_index, "sample_index": sample_index, "sigma": row[0], "bin": int(row[1])}
            for name, value in zip(names, row[2:]):
                record[name] = bool(value) if name in boolean_names else value if math.isfinite(value) else None
            records.append(record)
        return records

    @torch.no_grad()
    def log_student_query(self, generated, teacher_x0):
        """Observe the exact prediction already used by DMD; no extra forwards.

        The fresh student query is independent of the earlier gate decision.
        Direction angles/ratios are in x0 output space, not parameter gradients.
        """
        if self._student_query is None:
            raise RuntimeError("Residual-head student diagnostics require the current corrected-fake query.")
        fake_x0, correction, sigma, lam = self._student_query
        self.clear_student_query()
        risk = residual_risk_statistics(fake_x0, generated, correction, lam)
        directions = student_direction_statistics(fake_x0, teacher_x0, correction, lam)
        records = self._sample_records({**risk, **directions}, sigma, self._student_query_index)
        self._student_records.extend(records)
        self._student_query_index += 1
        return _student_scalar_metrics(records)

    @torch.no_grad()
    def _finish_student_iteration(self, outer_iteration):
        local_counts = [[0] * self.config.noise_bins for _ in range(3)]
        for record in self._student_records:
            index = record["bin"]
            local_counts[0][index] += 1
            local_counts[1][index] += int(record["lambda_actual"] > 0)
            local_counts[2][index] += int((record["applied_correction_rms"] or 0) > 0)
        counts = torch.tensor(local_counts, device=self.device, dtype=torch.int64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        iteration_counts = counts.cpu().tolist()
        self._usage_counts.add_(counts)
        self._student_updates += 1
        self._positive_lambda_updates += int(sum(iteration_counts[1]) > 0)
        self._nonzero_correction_updates += int(sum(iteration_counts[2]) > 0)
        records = _gather_records(self._student_records)
        cumulative_counts = self._usage_counts.cpu().tolist()
        per_bin_usage = []
        for index in range(self.config.noise_bins):
            selected = [record for record in records if record["bin"] == index]
            sample_count = iteration_counts[0][index]
            per_bin_usage.append(
                {
                    "bin": index,
                    "sample_count": sample_count,
                    "positive_lambda_sample_count": iteration_counts[1][index],
                    "nonzero_correction_sample_count": iteration_counts[2][index],
                    "positive_lambda_fraction": iteration_counts[1][index] / sample_count if sample_count else None,
                    "lambda_applied_mean": mean(record["lambda_actual"] for record in selected) if selected else None,
                    "cumulative_sample_count": cumulative_counts[0][index],
                    "cumulative_positive_lambda_sample_count": cumulative_counts[1][index],
                    "cumulative_nonzero_correction_sample_count": cumulative_counts[2][index],
                }
            )
        summary = {
            "iteration": outer_iteration + 1,
            "evaluation": "student_fresh_query",
            "direction_space": "x0_output_not_parameter_gradient",
            "candidate_and_optimal_lambda_role": "diagnostic_only_never_selected",
            "gate_mode": self.config.gate_mode,
            "lambda_policy_source": "independent_calibration_then_validation" if self.config.gate_mode == "calibrated" else "full_correction_validation_then_ramp",
            "samples": records,
            "usage_by_bin": per_bin_usage,
            "risk_by_bin": _risk_summary_by_bin(records, self.config.noise_bins),
            "sample_count": sum(iteration_counts[0]),
            "positive_lambda_sample_count": sum(iteration_counts[1]),
            "nonzero_correction_sample_count": sum(iteration_counts[2]),
            "cumulative_sample_count": sum(cumulative_counts[0]),
            "cumulative_positive_lambda_sample_count": sum(cumulative_counts[1]),
            "cumulative_nonzero_correction_sample_count": sum(cumulative_counts[2]),
            "cumulative_student_updates": self._student_updates,
            "cumulative_any_rank_positive_lambda_updates": self._positive_lambda_updates,
            "cumulative_any_rank_nonzero_correction_updates": self._nonzero_correction_updates,
            "usage_history_complete": self._usage_history_complete,
        }
        self._report("student", summary)
        self._student_records.clear()
        sample_count = summary["sample_count"]
        return {
            # Override microbatch/rank-local means with pooled, validity-aware
            # values; every rank returns the same global diagnostics.
            **_student_scalar_metrics(records),
            "head_student_positive_lambda_fraction": summary["positive_lambda_sample_count"] / sample_count if sample_count else 0.0,
            "head_student_positive_lambda_samples_cumulative": summary["cumulative_positive_lambda_sample_count"],
            "head_student_nonzero_correction_samples_cumulative": summary["cumulative_nonzero_correction_sample_count"],
            "head_student_any_rank_used_updates_cumulative": self._positive_lambda_updates,
        }

    def fit(self):
        if not self.fit_batches:
            raise RuntimeError("Residual head cannot fit without the current student's fake-fit batches.")
        cache = []
        # Fake is now a fixed snapshot: no fake/student optimizer step occurs
        # between these targets, head fitting, held-out scoring, and student loss.
        for batch in self.fit_batches:
            fake_x0, features = self._fake_x0_features(batch.renoised, batch.sigma, batch.condition)
            cache.append((features, batch.sigma, (fake_x0 - batch.generated.float()).detach()))
        self.fit_batches.clear()
        loss_sum = 0.0
        groups = []
        # Never count replay of one cached query as independent effective
        # batch. If fit_steps exceeds the cache size it still denotes replay,
        # as before, but one accumulation group visits distinct cache entries.
        step = 0
        while step < self.config.fit_steps:
            group_size = min(self.config.fit_grad_accum_steps, len(cache), self.config.fit_steps - step)
            groups.append([cache[index % len(cache)] for index in range(step, step + group_size)])
            step += group_size
        group_samples = [sum(target.shape[0] for _, _, target in group) for group in groups]
        self.head.train()
        for group, sample_count in zip(groups, group_samples):
            self.optimizer.zero_grad(set_to_none=True)
            for index, (features, sigma, target) in enumerate(group):
                # DDP must wrap both forward and backward in no_sync. The
                # final microbatch synchronizes the whole accumulated bucket.
                no_sync = getattr(self.head, "no_sync", None)
                context = no_sync() if no_sync is not None and index + 1 < len(group) else nullcontext()
                with context:
                    correction = self.head(features, sigma, tuple(target.shape))
                    loss = (correction.float() - target).square().mean()
                    # Normalize by this actual group's sample count, including
                    # a short final group, not always by accumulation_steps.
                    (loss * (target.shape[0] / sample_count)).backward()
                loss_sum += loss.detach().item()
                self._head_fit_microbatches += 1
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self._head_optimizer_updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.head.eval()
        unique_count = min(len(cache), self.config.fit_steps)
        local_counts = [sum(target.shape[0] for _, _, target in cache[:unique_count]), *group_samples]
        pooled_counts = torch.tensor(local_counts, device=self.device, dtype=torch.int64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(pooled_counts, op=dist.ReduceOp.SUM)
        pooled_counts = pooled_counts.cpu().tolist()
        report = {
            "source": "existing_fake_fit_cache_rescored_under_one_fixed_fake_snapshot",
            "grad_accum_steps": self.config.fit_grad_accum_steps,
            "microbatch_passes": self.config.fit_steps,
            "unique_cached_microbatches_per_rank": unique_count,
            "replayed_microbatch_passes_per_rank": self.config.fit_steps - unique_count,
            "unique_fit_samples_global": pooled_counts[0],
            "optimizer_updates": len(groups),
            "group_microbatches": [len(group) for group in groups],
            "effective_batch_samples_global": pooled_counts[1:],
            "fake_optimizer_updates_during_accumulation": 0,
            "student_optimizer_updates_during_accumulation": 0,
            "cross_outer_iteration_lag": 0,
            "cumulative_optimizer_updates": self._head_optimizer_updates,
            "cumulative_microbatch_passes": self._head_fit_microbatches,
            "history_complete": self._fit_history_complete,
        }
        self._report("fit", report)
        self._last_fit_metrics = {
            "head_fit_optimizer_updates": len(groups),
            "head_fit_optimizer_updates_cumulative": self._head_optimizer_updates,
            "head_fit_grad_accum_steps": self.config.fit_grad_accum_steps,
            "head_fit_unique_samples_global": pooled_counts[0],
            "head_fit_effective_batch_samples_global_mean": mean(pooled_counts[1:]),
            "head_fit_effective_batch_samples_global_min": min(pooled_counts[1:]),
            "head_fit_effective_batch_samples_global_max": max(pooled_counts[1:]),
            "head_fit_cross_outer_iteration_lag": 0,
        }
        return loss_sum / self.config.fit_steps

    @torch.no_grad()
    def check(self, samples, outer_iteration):
        if self.config.gate_mode == "calibrated":
            return self._check_calibrated(samples, outer_iteration)
        # A dedicated deterministic RNG stream makes gate probes independent of
        # fit/student noise, and restores the main stream after this extra work.
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        seed = int(self.trainer.config.get("seed", 42)) + 1000003 * (outer_iteration + 1) + 1000033 * rank + 7919
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            sample = next(samples)
            condition = self.trainer._encode_conditions(sample)[0]
            latent_shape = self.trainer._latent_shape(sample)
            initial = self.trainer.sample_initial_latents(latent_shape)
            generated, start, end = self.trainer.run_back_simulation(
                condition,
                latent_shape,
                grad_enabled=False,
                xt=initial,
                student_query=True,
            )
            sigma = self.trainer._sample_score_sigma(
                denoised_timestep_from=start,
                denoised_timestep_to=end,
                device=self.device,
                dtype=self.trainer.latent_dtype,
                latent_hw=self.trainer.student.latent_hw(latent_shape),
            )
            noise = self.trainer.student.random_noise_like(
                generated,
                torch.float32,
                broadcast_sequence_parallel_value,
            )
            renoised = self.trainer.student.add_noise(self.trainer.scheduler, generated, noise, sigma)
            fake_x0, features = self._fake_x0_features(renoised, sigma, condition)
            correction = self.module(features, sigma, tuple(generated.shape))
            # This query trains the gate, so evaluate the already chosen
            # policy *before* letting its full-correction evidence update it.
            # Post-update lambdas are logged as decisions, not as unbiased
            # performance estimates on the query that selected them.
            pre_gate_lambda = self.gate.lambda_for(sigma)
            risk = residual_risk_statistics(fake_x0, generated, correction, pre_gate_lambda)
            records = self._sample_records(risk, sigma, 0)
            residual = fake_x0 - generated.float()
            fake_mse = residual.flatten(1).square().mean(dim=1)
            corrected_mse = (residual - correction.float()).flatten(1).square().mean(dim=1)
            self.gate.update([(sigma, fake_mse, corrected_mse)])
        records = _gather_records(records)
        if _distributed_rank() == 0:
            report = {
                "iteration": outer_iteration + 1,
                "evaluation": "gate_train_check",
                "applied_policy": "lambda_snapshot_before_this_gate_update",
                "gate_decision_target": "full_F_minus_C_only",
                "candidate_and_optimal_lambda_role": "diagnostic_only_never_selected",
                "samples": records,
                "risk_by_bin": _risk_summary_by_bin(records, self.config.noise_bins),
                "lambda_after_gate_update": self.gate.lambdas.cpu().tolist(),
                "check_round_counts_after_update": self.gate.counts.cpu().tolist(),
            }
            logger.info("[head][check] {}", json.dumps(report, separators=(",", ":"), allow_nan=False))
        return {
            "head_heldout_fake_mse": fake_mse.mean().item(),
            "head_heldout_corrected_mse": corrected_mse.mean().item(),
            "head_heldout_delta": (fake_mse - corrected_mse).mean().item(),
            "head_heldout_pre_gate_selected_lambda_mse": risk["selected_lambda_mse"].mean().item(),
            **{f"head_{name}": value for name, value in self.gate.metrics().items()},
        }

    @torch.no_grad()
    def _fresh_gate_query(self, samples, outer_iteration, stream):
        """Generate one new query without advancing the student RNG stream.

        Calibration and validation consume different data batches and use
        separate deterministic streams for rollout, exit, score sigma and
        re-noising. Neither query is cached for fitting or student updates.
        """
        stream_offsets = {"calibration": 104729, "validation": 130363}
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        seed = int(self.trainer.config.get("seed", 42)) + 1000003 * (outer_iteration + 1) + 1000033 * _distributed_rank() + stream_offsets[stream]
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            sample = next(samples)
            condition = self.trainer._encode_conditions(sample)[0]
            latent_shape = self.trainer._latent_shape(sample)
            initial = self.trainer.sample_initial_latents(latent_shape)
            generated, start, end = self.trainer.run_back_simulation(
                condition,
                latent_shape,
                grad_enabled=False,
                xt=initial,
                student_query=True,
            )
            sigma = self.trainer._sample_score_sigma(
                denoised_timestep_from=start,
                denoised_timestep_to=end,
                device=self.device,
                dtype=self.trainer.latent_dtype,
                latent_hw=self.trainer.student.latent_hw(latent_shape),
            )
            noise = self.trainer.student.random_noise_like(
                generated,
                torch.float32,
                broadcast_sequence_parallel_value,
            )
            renoised = self.trainer.student.add_noise(self.trainer.scheduler, generated, noise, sigma)
            fake_x0, features = self._fake_x0_features(renoised, sigma, condition)
            correction = self.module(features, sigma, tuple(generated.shape))
        return sigma, generated.detach(), fake_x0.detach(), correction.detach()

    @torch.no_grad()
    def _check_calibrated(self, samples, outer_iteration):
        # The fake, student and head all stay fixed throughout calibration,
        # independent validation and the subsequent student's fresh query.
        sigma, generated, fake_x0, correction = self._fresh_gate_query(samples, outer_iteration, "calibration")
        calibration = residual_risk_statistics(fake_x0, generated, correction, 0.0)
        candidates = (
            self.gate.calibrate(
                [
                    (
                        sigma,
                        calibration["residual_head_dot"],
                        calibration["head_energy"],
                    )
                ]
            )
            .detach()
            .clone()
        )
        selected = candidates[self.gate._bin_indices(sigma)]
        calibration = residual_risk_statistics(fake_x0, generated, correction, selected)
        calibration_records = _gather_records(self._sample_records(calibration, sigma, 0))
        # Drop large query tensors before making a second independent query.
        del generated, fake_x0, correction

        sigma, generated, fake_x0, correction = self._fresh_gate_query(samples, outer_iteration, "validation")
        selected = candidates[self.gate._bin_indices(sigma)]
        validation = residual_risk_statistics(fake_x0, generated, correction, selected)
        validation_records = _gather_records(self._sample_records(validation, sigma, 0))
        self.gate.update_calibrated(
            [
                (
                    sigma,
                    validation["fake_mse"],
                    validation["residual_head_dot"],
                    validation["head_energy"],
                )
            ],
            candidates,
        )
        if _distributed_rank() == 0:
            for name, records, role in (
                ("calibration", calibration_records, "scale_selected_from_calibration_ema_not_unbiased_evaluation"),
                ("validation", validation_records, "frozen_calibration_scale_evaluated_on_independent_query"),
            ):
                report = {
                    "iteration": outer_iteration + 1,
                    "evaluation": f"gate_{name}",
                    "gate_mode": "calibrated",
                    "applied_policy": "calibration_candidate_snapshot_before_validation",
                    "evaluation_role": role,
                    "gate_decision_target": "exact_calibration_selected_F_minus_lambda_C",
                    "candidate_and_optimal_lambda_role": "this_query_counterfactual_optima_are_diagnostic_only",
                    "selected_candidate_lambdas": candidates.cpu().tolist(),
                    "samples": records,
                    "risk_by_bin": _risk_summary_by_bin(records, self.config.noise_bins),
                    "lambda_after_gate_update": self.gate.lambdas.cpu().tolist(),
                    "check_round_counts_after_update": self.gate.counts.cpu().tolist(),
                    "fresh_student_query_is_separate": True,
                }
                self._report(name, report)
        return {
            "head_heldout_fake_mse": validation["fake_mse"].mean().item(),
            "head_heldout_corrected_mse": validation["selected_lambda_mse"].mean().item(),
            "head_heldout_full_corrected_mse": validation["full_corrected_mse"].mean().item(),
            "head_heldout_delta": (validation["fake_mse"] - validation["selected_lambda_mse"]).mean().item(),
            "head_calibration_candidate_lambda_mean": candidates.mean().item(),
            "head_calibration_in_sample_selected_mse": calibration["selected_lambda_mse"].mean().item(),
            **{f"head_{name}": value for name, value in self.gate.metrics().items()},
        }

    def train_iteration(self, samples, grad_accum_iters, outer_iteration):
        self.fit_batches.clear()
        self.clear_student_query()
        self._student_records.clear()
        self._student_query_index = 0
        self.collecting_fit = True
        fake_loss, fake_real_loss = 0.0, 0.0
        try:
            for index in range(self.trainer.fake_update_ratio):
                result = self.trainer._train_one_stage(
                    samples,
                    stage="fake",
                    grad_accum_iters=grad_accum_iters,
                    outer_iteration=outer_iteration,
                    fake_update_index=index,
                )
                fake_loss += result["loss"]
                fake_real_loss += result["fake_real"]
        finally:
            self.collecting_fit = False
        diagnostics = {"head_fit_loss": self.fit()}
        diagnostics.update(self._last_fit_metrics)
        diagnostics.update(self.check(samples, outer_iteration))
        try:
            student_result = self.trainer._train_one_stage(
                samples,
                stage="student",
                grad_accum_iters=grad_accum_iters,
                outer_iteration=outer_iteration,
            )
        finally:
            self.clear_student_query()
        student_result.update(self._finish_student_iteration(outer_iteration))
        student_result.update(diagnostics)
        return student_result, fake_loss / self.trainer.fake_update_ratio, fake_real_loss / self.trainer.fake_update_ratio
