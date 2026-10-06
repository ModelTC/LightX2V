"""Residual-head DMD for native H3 joint video/audio latent tokens.

Each modality has its own FP32 token head, optimizer, and noise-bin gate. A
joint query shares one frozen-fake forward, but never pools audio/video risk
by token count. Fitting, calibration, validation, and student batches remain
independent exactly as in the Wan residual-head schedule.
"""

from dataclasses import dataclass

import torch

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import MiniMaxH3JointLatents
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value

from .head_diagnostics import residual_risk_statistics
from .residual_head import PackedTokenResidualHead
from .residual_head_training import ResidualHeadTraining, _FitBatch, _detach_condition, _distributed_rank, _gather_records, _risk_summary_by_bin


def _joint_map(value, function):
    return MiniMaxH3JointLatents(function(value.video), function(value.audio), value.shape)


def _prefix_metrics(modality, metrics):
    return {f"head_{modality}_{key.removeprefix('head_')}": value for key, value in metrics.items()}


@dataclass
class _JointFitBatch:
    generated: MiniMaxH3JointLatents
    renoised: MiniMaxH3JointLatents
    sigma: torch.Tensor
    condition: object


class _H3ModalityTraining(ResidualHeadTraining):
    """Reuse tensor fitting, calibration, diagnostics, and checkpoint logic."""

    def __init__(self, owner, modality):
        self.owner = owner
        self.modality = modality
        super().__init__(owner.trainer, owner.config)

    def _head_spec(self, trainer):
        model = trainer.fake_model
        return int(model.residual_head_feature_dims[self.modality]), int(model.residual_head_output_dims[self.modality]), (1, 1, 1), PackedTokenResidualHead

    @torch.no_grad()
    def _fake_x0_features(self, renoised, sigma, condition):
        # The owner performs the native joint forward/x0 conversion once;
        # these are detached snapshots, not targets evaluated under two fakes.
        del renoised, sigma
        return condition["fake_x0"], condition["features"]

    @torch.no_grad()
    def _fresh_gate_query(self, samples, outer_iteration, stream):
        return self.owner._modality_gate_query(self.modality, samples, outer_iteration, stream)

    @torch.no_grad()
    def check(self, samples, outer_iteration):
        if self.config.gate_mode == "calibrated":
            return self._check_calibrated(samples, outer_iteration)
        sigma, generated, fake_x0, correction = self._fresh_gate_query(samples, outer_iteration, "check")
        risk = residual_risk_statistics(fake_x0, generated, correction, self.gate.lambda_for(sigma))
        self.gate.update([(sigma, risk["fake_mse"], risk["full_corrected_mse"])])
        records = _gather_records(self._sample_records(risk, sigma, 0))
        self._report(
            "check",
            {
                "iteration": outer_iteration + 1,
                "evaluation": "gate_train_check",
                "applied_policy": "lambda_snapshot_before_this_gate_update",
                "gate_decision_target": "full_F_minus_C_only",
                "samples": records,
                "risk_by_bin": _risk_summary_by_bin(records, self.config.noise_bins),
                "lambda_after_gate_update": self.gate.lambdas.cpu().tolist(),
                "check_round_counts_after_update": self.gate.counts.cpu().tolist(),
            },
        )
        return {
            "head_heldout_fake_mse": risk["fake_mse"].mean().item(),
            "head_heldout_corrected_mse": risk["full_corrected_mse"].mean().item(),
            "head_heldout_delta": (risk["fake_mse"] - risk["full_corrected_mse"]).mean().item(),
            "head_heldout_pre_gate_selected_lambda_mse": risk["selected_lambda_mse"].mean().item(),
            **{f"head_{name}": value for name, value in self.gate.metrics().items()},
        }


class H3ResidualHeadTraining(ResidualHeadTraining):
    """Joint H3 orchestration with independent video/audio residual policies."""

    def __init__(self, trainer, config):
        self.trainer = trainer
        self.config = config
        self.device = torch.device(trainer.student.device)
        if getattr(getattr(trainer.student, "options", None), "projected_dmd", False):
            raise ValueError("Residual-head DMD uses F-lambda*C-T directly; disable projected_dmd.")
        required = ("predict_velocity_with_features", "residual_head_sigmas")
        if any(not callable(getattr(trainer.fake_model, name, None)) for name in required):
            raise ValueError("H3 residual heads require native joint features and modality sigma conversion.")
        for name in ("residual_head_feature_dims", "residual_head_output_dims"):
            if set(getattr(trainer.fake_model, name, {})) != {"video", "audio"}:
                raise ValueError(f"H3 {name} must define both video and audio.")
        # Each child also validates standard DMD, projected=false, and SP=1.
        self.modalities = {name: _H3ModalityTraining(self, name) for name in ("video", "audio")}
        self.collecting_fit = False
        self.fit_batches = []
        self._gate_queries = {}
        self._student_records = []
        self._student_query_index = 0
        self._last_fit_metrics = {}

    def checkpoint_metadata(self):
        return {
            **self.config.checkpoint_metadata(),
            "architecture": "h3_joint_tokens_v1",
            "sigma_space": "per_modality_shifted",
            "gate_policy": "independent_video_audio",
            "modalities": {name: runtime.checkpoint_metadata() for name, runtime in self.modalities.items()},
        }

    def state_dict(self):
        return {"format_version": 1, "modalities": {name: runtime.state_dict() for name, runtime in self.modalities.items()}}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("format_version") != 1 or set(state.get("modalities", {})) != set(self.modalities):
            raise RuntimeError("H3 residual-head checkpoint requires version 1 video and audio states.")
        for name, runtime in self.modalities.items():
            runtime.load_state_dict(state["modalities"][name])
        self.fit_batches.clear()
        self._gate_queries.clear()
        self._last_fit_metrics = {}

    @staticmethod
    def _validate_joint(value):
        if not isinstance(value, MiniMaxH3JointLatents):
            raise ValueError("H3 residual-head DMD requires native joint video/audio latents.")
        if any(not torch.is_tensor(tensor) or tensor.ndim != 3 for tensor in (value.video, value.audio)):
            raise ValueError("H3 residual-head latents must have shape [B, N, C] for each modality.")
        if tuple(value.video.shape) != value.shape.video_tokens or tuple(value.audio.shape) != value.shape.audio_tokens:
            raise ValueError("H3 residual-head latent shapes must match generated token geometry.")

    def remember_fit(self, generated, renoised, sigma, condition):
        if not self.collecting_fit:
            return
        self._validate_joint(generated)
        self._validate_joint(renoised)
        if len(self.fit_batches) < self.config.fit_steps:
            self.fit_batches.append(_JointFitBatch(_joint_map(generated, torch.Tensor.detach), _joint_map(renoised, torch.Tensor.detach), sigma.detach(), _detach_condition(condition)))

    @torch.no_grad()
    def _joint_x0_features(self, renoised, sigma, condition):
        self._validate_joint(renoised)
        self.trainer.fake.set_training(False)
        velocity, features = self.trainer.fake_model.predict_velocity_with_features(renoised, sigma, condition)
        self._validate_joint(velocity)
        # H3's velocity points toward clean data, with distinct video/audio
        # shifts. Only its native capability may convert velocity into x0.
        fake_x0 = self.trainer.student.x0_from_velocity(_joint_map(renoised, torch.Tensor.float), _joint_map(velocity, torch.Tensor.float), sigma.float())
        fake_x0 = _joint_map(fake_x0, lambda tensor: tensor.detach().float())
        if not isinstance(features, dict) or set(features) != set(self.modalities):
            raise ValueError("H3 residual features must contain video and audio tensors.")
        for name, runtime in self.modalities.items():
            tensor = features[name]
            target = getattr(fake_x0, name)
            if not torch.is_tensor(tensor) or tuple(tensor.shape) != (*target.shape[:2], runtime.feature_dim):
                raise ValueError(f"H3 {name} residual features must contain only generated rows, aligned with its latent tokens.")
        return fake_x0, {name: tensor.detach() for name, tensor in features.items()}

    def fit(self):
        if not self.fit_batches:
            raise RuntimeError("H3 residual heads cannot fit without the current fake-fit batches.")
        for runtime in self.modalities.values():
            runtime.fit_batches.clear()
        try:
            # All targets come from the final frozen fake snapshot. Nothing
            # updates fake/student between this rescore and the student query.
            for batch in self.fit_batches:
                fake_x0, features = self._joint_x0_features(batch.renoised, batch.sigma, batch.condition)
                sigmas = self.trainer.fake_model.residual_head_sigmas(batch.sigma)
                for name, runtime in self.modalities.items():
                    runtime.fit_batches.append(_FitBatch(getattr(batch.generated, name), getattr(batch.renoised, name), sigmas[name], {"fake_x0": getattr(fake_x0, name), "features": features[name]}))
            self.fit_batches.clear()
            losses = {name: runtime.fit() for name, runtime in self.modalities.items()}
            self._last_fit_metrics = {
                key: value for name, runtime in self.modalities.items() for key, value in _prefix_metrics(name, {"head_fit_loss": losses[name], **runtime._last_fit_metrics}).items()
            }
            return sum(losses.values())
        finally:
            self.fit_batches.clear()
            for runtime in self.modalities.values():
                runtime.fit_batches.clear()

    @torch.no_grad()
    def predict_corrected_fake(self, renoised, sigma, condition):
        fake_x0, features = self._joint_x0_features(renoised, sigma, condition)
        sigmas = self.trainer.fake_model.residual_head_sigmas(sigma)
        corrected = {
            name: runtime.predict_corrected_fake(getattr(renoised, name), sigmas[name], {"fake_x0": getattr(fake_x0, name), "features": features[name]}) for name, runtime in self.modalities.items()
        }
        return MiniMaxH3JointLatents(corrected["video"], corrected["audio"], renoised.shape)

    def clear_student_query(self):
        for runtime in self.modalities.values():
            runtime.clear_student_query()

    @torch.no_grad()
    def log_student_query(self, generated, teacher_x0):
        return {key: value for name, runtime in self.modalities.items() for key, value in _prefix_metrics(name, runtime.log_student_query(getattr(generated, name), getattr(teacher_x0, name))).items()}

    def _finish_student_iteration(self, outer_iteration):
        return {key: value for name, runtime in self.modalities.items() for key, value in _prefix_metrics(name, runtime._finish_student_iteration(outer_iteration)).items()}

    @torch.no_grad()
    def _modality_gate_query(self, modality, samples, outer_iteration, stream):
        if stream not in self._gate_queries:
            # Keep native joint rollout/noise inside an isolated stream. A
            # calibration batch is never replayed as validation or student data.
            offsets = {"check": 7919, "calibration": 104729, "validation": 130363}
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
            seed = int(self.trainer.config.get("seed", 42)) + 1000003 * (outer_iteration + 1) + 1000033 * _distributed_rank() + offsets[stream]
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    torch.cuda.manual_seed(seed)
                sample = next(samples)
                condition = self.trainer._encode_conditions(sample)[0]
                shape = self.trainer._latent_shape(sample)
                initial = self.trainer.sample_initial_latents(shape)
                generated, start, end = self.trainer.run_back_simulation(condition, shape, grad_enabled=False, xt=initial, student_query=True)
                sigma = self.trainer._sample_score_sigma(
                    denoised_timestep_from=start, denoised_timestep_to=end, device=self.device, dtype=self.trainer.latent_dtype, latent_hw=self.trainer.student.latent_hw(shape)
                )
                noise = self.trainer.student.random_noise_like(generated, torch.float32, broadcast_sequence_parallel_value)
                renoised = self.trainer.student.add_noise(self.trainer.scheduler, generated, noise, sigma)
                fake_x0, features = self._joint_x0_features(renoised, sigma, condition)
                sigmas = self.trainer.fake_model.residual_head_sigmas(sigma)
                self._gate_queries[stream] = {
                    name: (
                        sigmas[name].detach(),
                        getattr(generated, name).detach(),
                        getattr(fake_x0, name),
                        runtime.module(features[name], sigmas[name], tuple(getattr(generated, name).shape)).detach(),
                    )
                    for name, runtime in self.modalities.items()
                }
        # Only compact latent predictions survive the shared forward; feature
        # tensors are released before the independent validation query.
        query = self._gate_queries[stream].pop(modality)
        if not self._gate_queries[stream]:
            del self._gate_queries[stream]
        return query

    def check(self, samples, outer_iteration):
        self._gate_queries.clear()
        try:
            return {key: value for name, runtime in self.modalities.items() for key, value in _prefix_metrics(name, runtime.check(samples, outer_iteration)).items()}
        finally:
            self._gate_queries.clear()

    def train_iteration(self, samples, grad_accum_iters, outer_iteration):
        for runtime in self.modalities.values():
            runtime._student_records.clear()
            runtime._student_query_index = 0
        return super().train_iteration(samples, grad_accum_iters, outer_iteration)
