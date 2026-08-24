from __future__ import annotations

import copy
import os

import torch
from loguru import logger

from lightx2v_train.model_capabilities import (
    ConsistencyModelCapability,
    LossResult,
    ParallelCapability,
    TrainableModelCapability,
)
from lightx2v_train.model_zoo import build_loaded_model
from lightx2v_train.runtime.distributed import get_data_parallel_world_size
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.utils.registry import TRAINER_REGISTER

from ..base import BaseTrainer
from ..flow_matching import FlowMatchingTrainer
from .base import (
    CapabilityDenoiser,
    ConsistencyBatch,
    ConsistencyStepContext,
    RectifiedFlowPath,
)
from .objective_factory import build_consistency_objective


@TRAINER_REGISTER("consistency")
class ConsistencyTrainer(FlowMatchingTrainer):
    """Shared trainer shell for consistency-model objectives.

    The trainer owns data encoding, distributed synchronization, optimizer and
    checkpoint state, and the optional frozen teacher.  Algorithm-specific
    mathematics lives behind ``ConsistencyObjective``.
    """

    trainer_name = "consistency"
    required_capabilities = (
        *BaseTrainer.required_capabilities,
        ConsistencyModelCapability,
    )

    def __init__(self, config):
        super().__init__(config)
        self.path = RectifiedFlowPath()
        self.objective = build_consistency_objective(config, self.path)
        self.teacher_model = None
        self.teacher_denoiser = None
        self.reference_models = {}
        self.reference_denoisers = {}
        self._setup_resume_checkpoint = None

    def set_model(self, model):
        BaseTrainer.set_model(self, model)
        self.consistency_model = model.capabilities.require(ConsistencyModelCapability)

    def setup(self, resume_ckpt_path=None):
        self._setup_resume_checkpoint = resume_ckpt_path
        super().setup(resume_ckpt_path=resume_ckpt_path)
        self.student_denoiser = CapabilityDenoiser(
            self.consistency_model,
            self.path,
        )

        if self.objective.requires_teacher:
            self.teacher_model = self._build_frozen_teacher()
            self.teacher_denoiser = CapabilityDenoiser(
                self.teacher_consistency_model,
                self.path,
            )

        for spec in self.objective.reference_model_specs:
            reference_model = self._build_frozen_reference(spec)
            self.reference_models[spec.role] = reference_model
            self.reference_denoisers[spec.role] = CapabilityDenoiser(
                self.reference_capabilities[spec.role],
                self.path,
            )

        logger.info(
            "[train] consistency algorithm={} mode={} teacher={}",
            self.objective.algorithm_name,
            getattr(getattr(self.objective, "config", None), "mode", "custom"),
            self.objective.requires_teacher,
        )

    def _setup_trainable_model(self, model):
        capability = model.ensure_capabilities().require(ConsistencyModelCapability)
        capability.configure(self.objective.model_capabilities)
        super()._setup_trainable_model(model)
        capability.restore_trainable_auxiliary()

        initialization = self.objective.student_initialization_checkpoint
        if initialization is not None and self._setup_resume_checkpoint is None:
            self._load_initial_model_weights(model, initialization)
            # Loading weights does not change requires_grad, but keeping this
            # call here makes that lifecycle guarantee explicit for new models.
            capability.restore_trainable_auxiliary()

    def _restore_trainable_model(self, model):
        super()._restore_trainable_model(model)
        model.ensure_capabilities().require(ConsistencyModelCapability).restore_trainable_auxiliary()

    def _load_initial_model_weights(self, model, checkpoint, *, role="student"):
        if not os.path.isdir(checkpoint):
            raise RuntimeError(f"Consistency initialization checkpoint does not exist: {checkpoint}")
        self._load_model_weights(model, checkpoint)
        logger.info("[train] initialized consistency {} from {}", role, checkpoint)

    def _build_frozen_teacher(self):
        role_keys = {
            "fake",
            "fake_2",
            "fake_low_high",
            "fake_real",
            "fake_real_high",
            "fake_real_low",
            "student",
            "student_2",
            "stage1",
            "teacher",
            "teacher_2",
        }
        base_model_config = {key: copy.deepcopy(value) for key, value in self.model_config.items() if key not in role_keys}
        teacher_override = self.model_config.get("teacher", {})
        if not isinstance(teacher_override, dict):
            raise ValueError("model.teacher must be a mapping when provided.")

        teacher_config = copy.deepcopy(self.config)
        teacher_config["model"] = base_model_config
        teacher_config["model"].update(copy.deepcopy(teacher_override))
        teacher_model = build_loaded_model(
            teacher_config,
            load_transformer=True,
            load_vae=False,
            load_condition_encoder=False,
        )
        teacher_model.reuse_frozen_components_from(self.model)
        self.teacher_consistency_model = teacher_model.capabilities.require(ConsistencyModelCapability)
        self.teacher_consistency_model.set_frozen()
        teacher_model.capabilities.require(ParallelCapability).apply(self.config)
        self.teacher_consistency_model.set_frozen()
        logger.info(
            "[train] consistency teacher model={} path={}",
            teacher_config["model"]["name"],
            teacher_config["model"]["pretrained_model_name_or_path"],
        )
        return teacher_model

    def _build_frozen_reference(self, spec):
        role_keys = {
            "fake",
            "fake_2",
            "fake_low_high",
            "fake_real",
            "fake_real_high",
            "fake_real_low",
            "student",
            "student_2",
            "stage1",
            "teacher",
            "teacher_2",
        }
        model_config = {key: copy.deepcopy(value) for key, value in self.model_config.items() if key not in role_keys}
        override = self.model_config.get(spec.role, {})
        if not isinstance(override, dict):
            raise ValueError(f"model.{spec.role} must be a mapping when provided.")
        model_config.update(copy.deepcopy(override))

        reference_config = copy.deepcopy(self.config)
        reference_config["model"] = model_config
        reference_model = build_loaded_model(
            reference_config,
            load_transformer=True,
            load_vae=False,
            load_condition_encoder=False,
        )
        reference_model.reuse_frozen_components_from(self.model)
        capability = reference_model.capabilities.require(ConsistencyModelCapability)
        if self.train_type == "lora":
            reference_model.capabilities.require(TrainableModelCapability).configure(
                "lora",
                {
                    "rank": self.lora_rank,
                    "alpha": self.lora_alpha,
                    "target_modules": self.lora_target_modules,
                },
            )
        self._load_initial_model_weights(reference_model, spec.checkpoint, role=spec.role)
        capability.set_frozen(training=spec.training_mode)
        reference_model.capabilities.require(ParallelCapability).apply(self.config)
        capability.set_frozen(training=spec.training_mode)
        if not hasattr(self, "reference_capabilities"):
            self.reference_capabilities = {}
        self.reference_capabilities[spec.role] = capability
        logger.info(
            "[train] consistency reference role={} model={} checkpoint={}",
            spec.role,
            reference_config["model"]["name"],
            spec.checkpoint,
        )
        return reference_model

    def compute_loss_on_sample(self, sample):
        with torch.no_grad():
            clean = self.consistency_model.encode_latent(sample)
            clean = broadcast_sequence_parallel_value(clean)
            condition = self.consistency_model.encode_condition(sample)
            condition = broadcast_sequence_parallel_value(condition)

            negative_condition = None
            if self.objective.requires_negative_condition:
                negative_condition = self._encode_negative_condition(sample)
                negative_condition = broadcast_sequence_parallel_value(negative_condition)

            context = ConsistencyStepContext(
                iteration=int(getattr(self, "current_train_iteration", 0)),
                global_batch_size=(clean.shape[0] * get_data_parallel_world_size() * self.gradient_accumulation_iters),
                latent_hw=self.consistency_model.sampling_latent_hw(sample, clean),
            )
            training_state = self.objective.sample_training_state(
                clean,
                self.noise_scheduler,
                context,
            )
            training_state = broadcast_sequence_parallel_value(training_state)

        output = self.objective.compute(
            ConsistencyBatch(
                clean=clean,
                condition=condition,
                negative_condition=negative_condition,
            ),
            training_state,
            self.student_denoiser,
            self.teacher_denoiser,
            self.reference_denoisers,
        )
        return LossResult(loss=output.loss, metrics=output.metrics)

    def _encode_negative_condition(self, sample):
        conditioning = sample.get("conditioning", {})
        prompt = conditioning.get("prompt", "")
        negative_prompt = conditioning.get("negative_prompt")

        if negative_prompt is None:
            values = [self.objective.negative_prompt]
        elif isinstance(negative_prompt, str):
            values = [negative_prompt]
        else:
            values = list(negative_prompt)
            if len(values) != 1:
                raise ValueError(f"Expected exactly one negative prompt, got {len(values)}.")

        fallback = self.objective.negative_prompt
        if not isinstance(fallback, str):
            fallback = self.model.unconditional_prompt
        values = [value if isinstance(value, str) and value.strip() else fallback for value in values]
        encoded_prompt = values[0] if isinstance(prompt, str) else values

        negative_sample = dict(sample)
        negative_sample["conditioning"] = dict(conditioning)
        cached_negative = negative_sample["conditioning"].get("negative")
        if cached_negative is None:
            cache_path = sample.get("meta", {}).get("training_cache_path")
            if cache_path is not None:
                raise KeyError(f"Training cache {cache_path} has no conditioning.negative entry required by this consistency objective.")
            negative_sample["conditioning"].pop("positive", None)
        else:
            negative_sample["conditioning"]["active"] = "negative"
        negative_sample["conditioning"]["prompt"] = encoded_prompt
        return self.consistency_model.encode_condition(negative_sample)
