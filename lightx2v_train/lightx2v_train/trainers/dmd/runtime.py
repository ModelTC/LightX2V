import copy

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_train.model_capabilities import (
    CheckpointCapability,
    DistributionMatchingCapability,
    ParallelCapability,
    TrainableModelCapability,
)
from lightx2v_train.model_zoo import build_loaded_model
from lightx2v_train.runtime.distributed import is_distributed
from lightx2v_train.runtime.sequence_parallel import (
    broadcast_sequence_parallel_value,
    sync_sequence_parallel_gradients,
)
from lightx2v_train.schedulers import DMDFlowMatchingScheduler
from lightx2v_train.tricks import (
    IdaModelPair,
    IdaSetupContext,
    IdaStepContext,
    ImplicitDistributionAlignmentTrick,
)
from lightx2v_train.utils.generation_shapes import parse_generation_shapes

from ..base import BaseTrainer
from .checkpoint import DmdCheckpointManager
from .config import DmdConfig
from .math import (
    dmd_loss,
    do_cfg,
)
from .roles import DmdRoleRegistry


class _DmdRuntime(BaseTrainer):
    trainer_name = "dmd"
    required_capabilities = (
        *BaseTrainer.required_capabilities,
        DistributionMatchingCapability,
    )
    default_negative_prompt = None
    supports_ida = True
    defer_ida_setup = False

    def _resolve_train_type(self):
        if "train_type" in self.training_config:
            raise ValueError("DMD trainers use training.student.train_type and training.fake.train_type; remove training.train_type.")
        return None

    def __init__(self, config):
        super().__init__(config)
        self.role_registry = DmdRoleRegistry(self)
        self.checkpoint_manager = DmdCheckpointManager(self)
        parsed = DmdConfig.from_mapping(
            config,
            default_negative_prompt=self.default_negative_prompt,
        )
        self.parsed_dmd_config = parsed
        self.student_config = parsed.student
        self.fake_config = parsed.fake
        self.student_train_type = parsed.student_train_type
        self.fake_train_type = parsed.fake_train_type
        self.student_lora_config = parsed.student_lora
        self.fake_lora_config = parsed.fake_lora

        self.fake_optimizer_config = self.fake_config["optimizer"]
        self.fake_optimizer_learning_rate = self.fake_optimizer_config.get("learning_rate", self.optimizer_learning_rate)
        self.fake_optimizer_adam_beta1 = self.fake_optimizer_config.get("adam_beta1", self.optimizer_adam_beta1)
        self.fake_optimizer_adam_beta2 = self.fake_optimizer_config.get("adam_beta2", self.optimizer_adam_beta2)
        self.fake_optimizer_weight_decay = self.fake_optimizer_config.get("weight_decay", self.optimizer_weight_decay)
        self.fake_optimizer_adam_epsilon = self.fake_optimizer_config.get("adam_epsilon", self.optimizer_adam_epsilon)

        self.dmd_config = parsed.dmd
        self.ida_trick = ImplicitDistributionAlignmentTrick.from_mapping(self.dmd_config.get("ida", {}))
        if self.ida_trick.enabled and not self.supports_ida:
            raise ValueError(f"{self.trainer_name} does not support training.dmd.ida.")
        self.num_inference_steps = parsed.num_inference_steps
        self.fake_update_ratio = parsed.fake_update_ratio
        self.guidance_scale = parsed.guidance_scale
        self.negative_prompt = parsed.negative_prompt
        self.cfg_norm = parsed.cfg_norm
        self.generation_shapes = parsed.generation_shapes
        self.random_schedule_enabled = parsed.random_schedule_enabled
        self.random_schedule_num_steps_min = parsed.random_schedule_num_steps_min
        self.random_schedule_num_steps_max = parsed.random_schedule_num_steps_max
        self.random_schedule_sigma_min = parsed.random_schedule_sigma_min
        self.random_schedule_sigma_max = parsed.random_schedule_sigma_max
        self.random_schedule_sampling_method = parsed.random_schedule_sampling_method
        self._configured_latent_dtype = parsed.latent_dtype
        self.latent_dtype = parsed.latent_dtype or self.running_dtype

    def set_model(self, model):
        super().set_model(model)
        self.student = model.capabilities.require(DistributionMatchingCapability)
        parse_generation_shapes(
            self.generation_shapes,
            expected_dimensions=self.student.generation_shape_dimensions,
        )
        profile = self.student.profile
        if self._configured_latent_dtype is None and profile.default_latent_dtype is not None:
            self.latent_dtype = profile.default_latent_dtype
        self._validate_distribution_matching_profile(profile)
        if self.negative_prompt is None:
            self.negative_prompt = self.student.default_negative_prompt
        for lora_config in (
            self.student_lora_config,
            self.fake_lora_config,
        ):
            if lora_config is not None and "target_modules" not in lora_config and self.student.default_lora_target_modules is not None:
                lora_config["target_modules"] = list(self.student.default_lora_target_modules)

        logger.info("[train] dmd latent_dtype={}", self.latent_dtype)

    def _validate_distribution_matching_profile(self, profile):
        model_name = self.model_config.get("name", type(self.model).__name__)
        supported_methods = profile.supported_training_methods
        if supported_methods is not None and self.trainer_name not in supported_methods:
            methods = ", ".join(sorted(supported_methods))
            raise ValueError(f"model={model_name!r} supports only these distribution-matching training methods: {methods}; got {self.trainer_name!r}.")
        requested_features = (
            (self.ida_trick.enabled, profile.supports_ida, "IDA"),
            (getattr(getattr(self, "diversity_trick", None), "enabled", False), profile.supports_diversity, "diversity loss"),
            (getattr(getattr(self, "real_data_fake_trick", None), "enabled", False), profile.supports_real_data_fake, "real-data fake loss"),
        )
        unsupported = [name for enabled, supported, name in requested_features if enabled and not supported]
        if unsupported:
            raise ValueError(f"model={model_name!r} does not support DMD features: {', '.join(unsupported)}.")
        if not profile.supports_guidance and self.guidance_scale != 1.0:
            raise ValueError(f"model={model_name!r} has no unconditional branch; training.teacher.guidance_scale must be 1.0.")
        if getattr(self, "warp_denoising_step", False) and not profile.supports_warped_denoising_schedule:
            raise ValueError(f"model={model_name!r} requires an unwarped base denoising schedule; set training.dmd.warp_denoising_step=false.")

    def _get_optimizer_config(self):
        return self.training_config["student"]["optimizer"]

    def _setup_trainable_model(self, model, role="student"):
        if role == "student":
            train_type = self.student_train_type
            lora_config = self.student_lora_config
        elif role in {"fake", "fake_real"}:
            train_type = self.fake_train_type
            lora_config = self.fake_lora_config
        else:
            raise ValueError(f"Unsupported DMD model role: {role}")
        model.ensure_capabilities().require(TrainableModelCapability).configure(train_type, lora_config or {})

    def _restore_trainable_model(self, model, role="student"):
        if role == "student":
            train_type = self.student_train_type
        elif role in {"fake", "fake_real"}:
            train_type = self.fake_train_type
        else:
            raise ValueError(f"Unsupported DMD model role: {role}")
        model.ensure_capabilities().require(TrainableModelCapability).restore(train_type)

    def _save_model_weights(self, model, save_dir, role="student"):
        train_type = self.student_train_type if role == "student" else self.fake_train_type
        model.ensure_capabilities().require(CheckpointCapability).save_weights(save_dir, train_type)

    def _load_model_weights(self, model, save_dir, role="student"):
        train_type = self.student_train_type if role == "student" else self.fake_train_type
        model.ensure_capabilities().require(CheckpointCapability).load_weights(save_dir, train_type)

    def setup(self, resume_ckpt_path=None):
        super().setup(resume_ckpt_path=None)
        base_model_config = {
            key: copy.deepcopy(value)
            for key, value in self.model_config.items()
            if key
            not in {
                "fake",
                "teacher",
                "student",
                "student_2",
                "fake_2",
                "fake_low_high",
                "fake_real",
                "fake_real_high",
                "fake_real_low",
                "teacher_2",
            }
        }

        fake_model_config = copy.deepcopy(self.config)
        fake_model_config["model"] = copy.deepcopy(base_model_config)
        if "fake" in self.model_config:
            if not isinstance(self.model_config["fake"], dict):
                raise ValueError("model.fake must be a mapping.")
            fake_model_config["model"].update(copy.deepcopy(self.model_config["fake"]))
        self.fake_model_config = copy.deepcopy(fake_model_config)
        self.fake_model = build_loaded_model(
            fake_model_config,
            load_transformer=True,
            load_vae=False,
            load_condition_encoder=False,
        )
        self.fake_model.reuse_frozen_components_from(self.model)
        self.fake = self.fake_model.capabilities.require(DistributionMatchingCapability)
        self._setup_trainable_model(self.fake_model, role="fake")
        self.fake_model.capabilities.require(ParallelCapability).apply(self.config)
        if self.gradient_checkpointing:
            self.fake_model.capabilities.require(TrainableModelCapability).enable_gradient_checkpointing()

        teacher_model_config = copy.deepcopy(self.config)
        teacher_model_config["model"] = copy.deepcopy(base_model_config)
        if "teacher" in self.model_config:
            if not isinstance(self.model_config["teacher"], dict):
                raise ValueError("model.teacher must be a mapping.")
            teacher_model_config["model"].update(copy.deepcopy(self.model_config["teacher"]))
        self.teacher_model = build_loaded_model(
            teacher_model_config,
            load_transformer=True,
            load_vae=False,
            load_condition_encoder=False,
        )
        self.teacher_model.reuse_frozen_components_from(self.model)
        self.teacher = self.teacher_model.capabilities.require(DistributionMatchingCapability)
        self.teacher.denoiser().requires_grad_(False)
        self.teacher.set_training(False)
        self.teacher_model.capabilities.require(ParallelCapability).apply(self.config)
        self.teacher.set_training(False)

        self.fake_trainable_params = list(self.fake_model.capabilities.require(TrainableModelCapability).parameters())
        self.fake_optimizer = self._build_optimizer(
            self.fake_trainable_params,
            {
                "learning_rate": self.fake_optimizer_learning_rate,
                "adam_beta1": self.fake_optimizer_adam_beta1,
                "adam_beta2": self.fake_optimizer_adam_beta2,
                "weight_decay": self.fake_optimizer_weight_decay,
                "adam_epsilon": self.fake_optimizer_adam_epsilon,
            },
        )
        self.fake_lr_scheduler = self._build_lr_scheduler(
            self.fake_optimizer,
            num_warmup_steps=0,
            num_training_steps=max(1, self.max_train_iters * self.fake_update_ratio),
        )

        self.scheduler = DMDFlowMatchingScheduler(self.config, self.dmd_config)

        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)

        if not self.defer_ida_setup:
            self._setup_ida_trick()

        logger.info("[train] dmd student model={} path={}", self.model_config["name"], self.model_config["pretrained_model_name_or_path"])
        logger.info("[train] dmd fake model={} path={}", fake_model_config["model"]["name"], fake_model_config["model"]["pretrained_model_name_or_path"])
        logger.info("[train] dmd teacher model={} path={}", teacher_model_config["model"]["name"], teacher_model_config["model"]["pretrained_model_name_or_path"])
        logger.info("[train] dmd train_types student={} fake={}", self.student_train_type, self.fake_train_type)
        logger.info("[train] dmd student trainable params={}", self._count_trainable(self.student.denoiser()))
        logger.info("[train] dmd fake trainable params={}", self._count_trainable(self.fake.denoiser()))
        if self.random_schedule_enabled:
            logger.info(
                "[train] dmd random sigma schedule enabled: steps=[{}, {}], sigma=[{}, {}], sampling_method={}",
                self.random_schedule_num_steps_min,
                self.random_schedule_num_steps_max,
                self.random_schedule_sigma_min,
                self.random_schedule_sigma_max,
                self.random_schedule_sampling_method,
            )

    @staticmethod
    def _count_trainable(module):
        return sum(1 for param in module.parameters() if param.requires_grad)

    def _ida_model_pairs(self):
        return {
            "main": IdaModelPair(
                student=self.student.denoiser(),
                fake=self.fake.denoiser(),
            )
        }

    def _setup_ida_trick(self):
        self.ida_trick.setup(IdaSetupContext(model_pairs=self._ida_model_pairs()))
        logger.info(
            "[train] {} SenseFlow IDA enabled={} decay={}",
            self.trainer_name,
            self.ida_trick.enabled,
            self.ida_trick.config.decay,
        )

    def _after_student_optimizer_step(self, role):
        self.ida_trick.after_student_step(IdaStepContext(role=role))

    @staticmethod
    def _do_cfg(cond_pred, uncond_pred, cfg_scale, cfg_norm):
        return do_cfg(
            cond_pred,
            uncond_pred,
            cfg_scale,
            cfg_norm,
        )

    @staticmethod
    def _dmd_loss(latents, x_pred_fake_flow, x_pred_teacher, norm_clip_min=None):
        return dmd_loss(
            latents,
            x_pred_fake_flow,
            x_pred_teacher,
            norm_clip_min=norm_clip_min,
        )

    def _prepare_sampling_schedule(self, latent_shape):
        latent_hw = self.student.latent_hw(latent_shape)
        if self.random_schedule_enabled:
            num_steps = self._sample_synced_int(self.random_schedule_num_steps_min, self.random_schedule_num_steps_max + 1)
            self.scheduler.set_random_timesteps(
                self.random_schedule_num_steps_min,
                self.random_schedule_num_steps_max,
                sigma_min=self.random_schedule_sigma_min,
                sigma_max=self.random_schedule_sigma_max,
                sampling_method=self.random_schedule_sampling_method,
                latent_hw=latent_hw,
                device=self.student.device,
                num_steps=num_steps,
            )
            return
        sigma_values = self.dmd_config.get("denoising_sigma_values")
        if sigma_values is not None:
            sigma_values = tuple(float(value) for value in sigma_values)
            expected = self.num_inference_steps + 1
            if len(sigma_values) != expected:
                raise ValueError(f"training.dmd.denoising_sigma_values must contain {expected} values, got {len(sigma_values)}.")
            sigma_values = sigma_values[:-1]
        self.scheduler.set_timesteps(
            self.num_inference_steps,
            sigmas=sigma_values,
            latent_hw=latent_hw,
            device=self.student.device,
        )

    def _latent_shape(self, sample):
        return self.student.latent_shape(
            sample,
            self.generation_shapes,
            broadcast_sequence_parallel_value,
        )

    def _encode_conditions(self, sample):
        return self.student.encode_conditions(
            sample,
            self.negative_prompt,
            self.guidance_scale,
            broadcast_sequence_parallel_value,
        )

    @staticmethod
    def _predict_velocity(capability, latents, sigma, condition):
        return capability.predict_velocity(latents, sigma, condition)

    def _predict_teacher_velocity(self, latents, sigma, condition, negative_condition):
        return self.teacher.predict_guided_velocity(
            latents,
            sigma,
            condition,
            negative_condition,
            self.guidance_scale,
            self.cfg_norm,
        )

    def sample_initial_latents(self, latent_shape):
        return self.student.initial_latents(
            latent_shape,
            self.latent_dtype,
            broadcast_sequence_parallel_value,
        )

    def _sample_synced_int(self, low, high):
        value = torch.randint(int(low), int(high), (1,), device=self.student.device, dtype=torch.int64)
        if is_distributed():
            dist.broadcast(value, src=0)
        return int(value.item())

    def _iter_train_samples(self):
        epoch = 0
        while True:
            sampler = getattr(self.dataloader_train, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            for sample in self.dataloader_train:
                yield sample
            epoch += 1

    def _set_student_gradient_sync(self, enabled):
        self.model.capabilities.require(ParallelCapability).set_gradient_sync(enabled)

    def _set_fake_gradient_sync(self, enabled):
        self.fake_model.capabilities.require(ParallelCapability).set_gradient_sync(enabled)

    def _set_gradient_sync(self, enabled):
        self._set_student_gradient_sync(enabled)
        self._set_fake_gradient_sync(enabled)

    def _sync_sequence_parallel_grads(self, params):
        sync_sequence_parallel_gradients(params)

    def _fake_weights_dir(self, root_dir):
        return self.checkpoint_manager._fake_weights_dir(root_dir)

    def _trick_checkpoint_metadata(self):
        return self.checkpoint_manager._trick_checkpoint_metadata()

    def _validate_optional_trick_metadata(self, state, state_path):
        return self.checkpoint_manager._validate_optional_trick_metadata(state, state_path)

    def _load_resume_state(self, resume_ckpt_path):
        return self.checkpoint_manager._load_resume_state(resume_ckpt_path)

    def _validate_dmd_checkpoint_metadata(self, state, state_path, resume_ckpt_path):
        return self.checkpoint_manager._validate_dmd_checkpoint_metadata(state, state_path, resume_ckpt_path)

    def _load_single_process_state(self, resume_ckpt_path):
        return self.checkpoint_manager._load_single_process_state(resume_ckpt_path)

    def _load_distributed_state(self, resume_ckpt_path):
        return self.checkpoint_manager._load_distributed_state(resume_ckpt_path)

    def save_checkpoint(self, iteration, save_total_limit):
        return self.checkpoint_manager.save_checkpoint(iteration, save_total_limit)

    def _should_save_consolidated_student(self):
        return self.checkpoint_manager._should_save_consolidated_student()

    def _save_consolidated_student_weights(self, save_dir):
        return self.checkpoint_manager._save_consolidated_student_weights(save_dir)

    def _save_distributed_state(self, save_dir, iteration):
        return self.checkpoint_manager._save_distributed_state(save_dir, iteration)
