import copy
import os
from functools import partial

import torch
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from lightx2v_train.model_capabilities import (
    DistributionMatchingCapability,
    ParallelCapability,
    TrainableModelCapability,
)
from lightx2v_train.model_zoo import build_loaded_model
from lightx2v_train.runtime.distributed import (
    barrier,
    get_world_size,
    is_main_process,
    reduce_mean,
)
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.schedulers import DMDFlowMatchingScheduler
from lightx2v_train.schedulers.flow_matching import CausalForcingFlowMatchScheduler
from lightx2v_train.tricks import (
    DiversitySetupContext,
    DiversityStepContext,
    DiversityTrainerConstraints,
    DiversityTrick,
    RealDataFakeSetupContext,
    RealDataFakeStepContext,
    RealDataFakeTrick,
)
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .config import DmdScheduleConfig
from .runtime import _DmdRuntime
from .score_sampling import ScoreSigmaContext, build_score_sigma_sampler


@TRAINER_REGISTER("dmd")
class DmdTrainer(_DmdRuntime):
    """Distribution matching for image, video, and joint latent models."""

    trainer_name = "dmd"
    supports_diversity_loss = True
    supports_real_data_fake = True

    def __init__(self, config):
        super().__init__(config)
        parsed = DmdScheduleConfig.from_mapping(
            config,
            dmd_config=self.dmd_config,
            student_config=self.student_config,
            model_config=self.model_config,
            num_inference_steps=self.num_inference_steps,
        )
        self.parsed_dmd_schedule_config = parsed
        self.num_train_timestep = parsed.num_train_timestep
        self.denoising_step_list = parsed.denoising_step_list
        self.num_inference_steps = len(self.denoising_step_list)
        self.warp_denoising_step = parsed.warp_denoising_step
        self.min_step = parsed.min_step
        self.max_step = parsed.max_step
        self.score_timestep_shift = parsed.score_timestep_shift
        self.ts_schedule = parsed.ts_schedule
        self.ts_schedule_max = parsed.ts_schedule_max
        self.min_score_timestep = parsed.min_score_timestep
        self.student_checkpoint_path = parsed.student_checkpoint_path
        self.student_checkpoint_strict = parsed.student_checkpoint_strict
        self.score_sigma_sampler = build_score_sigma_sampler(
            self.dmd_config.get("score_sampling"),
            sample_min_timestep=self.min_score_timestep,
            clamp_min_timestep=self.min_step,
            clamp_max_timestep=self.max_step,
            timestep_shift=self.score_timestep_shift,
            use_rollout_min=self.ts_schedule,
            use_rollout_max=self.ts_schedule_max,
        )

        self.diversity_trick = DiversityTrick.from_mapping(self.dmd_config.get("div_loss", {}))
        self.diversity_trick.validate_trainer(
            DiversityTrainerConstraints(
                supported=self.supports_diversity_loss,
                rollout_steps=self.num_inference_steps,
            ),
            self.trainer_name,
        )
        diversity_scheduler = (
            DMDFlowMatchingScheduler(
                self.config,
                self.dmd_config,
            )
            if self.diversity_trick.enabled
            else None
        )
        self.diversity_trick.setup(DiversitySetupContext(scheduler=diversity_scheduler))
        self.real_data_fake_trick = RealDataFakeTrick.from_mappings(
            {
                "main": (
                    "fake_real",
                    self.dmd_config.get("fake_real", {}),
                )
            }
        )
        if self.real_data_fake_trick.enabled and not self.supports_real_data_fake:
            raise ValueError(f"{self.trainer_name} does not support training.dmd.fake_real.")
        self.real_data_fake_trick.validate_schedule(
            mode="standard",
            num_train_timestep=self.num_train_timestep,
        )
        self.fake_real_train_type = self.fake_train_type
        self.fake_real_lora_config = copy.deepcopy(self.fake_lora_config)
        self.fake_real_optimizer_config = copy.deepcopy(self.fake_optimizer_config)

    def setup(self, resume_ckpt_path=None):
        if resume_ckpt_path is None and self.student_checkpoint_path:
            self._load_student_checkpoint(self.student_checkpoint_path, strict=self.student_checkpoint_strict)
        super().setup(resume_ckpt_path=None)
        self._setup_fake_real_resources()
        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)
        if resume_ckpt_path is None:
            self.lr_scheduler = self._build_lr_scheduler(
                self.optimizer,
                num_training_steps=max(1, self.max_train_iters),
            )
            self.fake_lr_scheduler = self._build_lr_scheduler(
                self.fake_optimizer,
                num_warmup_steps=0,
                num_training_steps=max(
                    1,
                    self.max_train_iters * self.fake_update_ratio,
                ),
            )

        time_shift_settings = self.config["scheduler"].get("time_shift_settings", {})
        self.denoising_scheduler = CausalForcingFlowMatchScheduler(
            num_train_timesteps=self.config["scheduler"].get("num_train_timesteps", 1000),
            time_shift_settings=time_shift_settings,
        )
        self.denoising_steps = self._build_denoising_steps(self.student.device)
        self.denoising_sigmas = (self.denoising_steps / self.num_train_timestep).to(dtype=torch.float32)
        self.real_data_fake_trick.setup(self._real_data_fake_setup_context())
        logger.info(
            "[train] {} denoising_steps={} warped={}",
            self.trainer_name,
            [round(float(step), 4) for step in self.denoising_steps.detach().cpu()],
            self.warp_denoising_step,
        )

    def _setup_fake_real_resources(self):
        self.fake_real_model = None
        self.fake_real_optimizer = None
        self.fake_real_lr_scheduler = None
        self.fake_real_trainable_params = []
        if not self.real_data_fake_trick.enabled_for("main"):
            return
        model_config = copy.deepcopy(self.fake_model_config)
        self.fake_real_model = build_loaded_model(
            model_config,
            load_transformer=True,
            load_vae=False,
            load_condition_encoder=False,
        )
        self.fake_real_model.reuse_frozen_components_from(self.model)
        self.fake_real = self.fake_real_model.capabilities.require(DistributionMatchingCapability)
        self._setup_trainable_model(
            self.fake_real_model,
            role="fake_real",
        )
        self.fake_real_model.capabilities.require(ParallelCapability).apply(self.config)
        if self.gradient_checkpointing:
            self.fake_real_model.capabilities.require(TrainableModelCapability).enable_gradient_checkpointing()
        self.fake_real_trainable_params = list(self.fake_real_model.capabilities.require(TrainableModelCapability).parameters())
        self.fake_real_optimizer = self._build_optimizer(
            self.fake_real_trainable_params,
            self.fake_real_optimizer_config,
        )
        self.fake_real_lr_scheduler = self._build_lr_scheduler(
            self.fake_real_optimizer,
            num_warmup_steps=0,
            num_training_steps=max(
                1,
                self.max_train_iters * self.fake_update_ratio,
            ),
        )
        logger.info(
            "[train] {} independent fake_real model={} path={} train_type={} trainable_params={}",
            self.trainer_name,
            model_config["model"]["name"],
            model_config["model"]["pretrained_model_name_or_path"],
            self.fake_real_train_type,
            len(self.fake_real_trainable_params),
        )

    def _load_student_checkpoint(self, checkpoint_path, strict=True):
        model_state_path = checkpoint_path
        if os.path.isdir(model_state_path):
            dist_state_path = os.path.join(model_state_path, "dist_state")
            if os.path.isdir(dist_state_path):
                module = self.parallel.state_module()
                options = StateDictOptions(ignore_frozen_params=False, strict=strict)
                model_state = get_model_state_dict(module, options=options)
                state = {"model": model_state}
                dcp.load(state, checkpoint_id=dist_state_path)
                set_model_state_dict(
                    module,
                    model_state_dict=state["model"],
                    options=options,
                )
                logger.info("[train] loaded {} student checkpoint from distributed state {}", self.trainer_name, dist_state_path)
                return
            model_state_path = os.path.join(model_state_path, "model_state.pt")
        if not os.path.exists(model_state_path):
            raise RuntimeError(f"{self.trainer_name} student checkpoint not found: {checkpoint_path}")

        state = torch.load(model_state_path, map_location="cpu", weights_only=False)
        for key in ("generator_ema", "generator", "model", "state_dict"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break

        fixed = {}
        for key, value in state.items():
            for prefix in (
                "model._fsdp_wrapped_module.",
                "model._checkpoint_wrapped_module.",
                "model._orig_mod.",
                "model.",
                "_fsdp_wrapped_module.",
                "_checkpoint_wrapped_module.",
                "_orig_mod.",
            ):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            fixed[key] = value

        incompatible = self.parallel.state_module().load_state_dict(
            fixed,
            strict=strict,
        )
        if not strict:
            if incompatible.missing_keys:
                logger.warning("Missing keys when loading {} student checkpoint: {}", self.trainer_name, incompatible.missing_keys)
            if incompatible.unexpected_keys:
                logger.warning("Unexpected keys when loading {} student checkpoint: {}", self.trainer_name, incompatible.unexpected_keys)
        logger.info("[train] loaded {} student checkpoint path={}", self.trainer_name, model_state_path)

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        max_train_iters = self.max_train_iters
        grad_accum_iters = max(1, int(self.gradient_accumulation_iters))
        save_every_iters = self.save_every_iters
        save_total_limit = self.save_total_limit

        logger.info(
            "[train] start method={} student_train_type={} fake_train_type={} iter={}/{} world_size={} grad_accum={} train_log_every_iters={} fake_update_ratio={}",
            self.training_config.get("method", self.trainer_name),
            self.student_train_type,
            self.fake_train_type,
            current_iter,
            max_train_iters,
            get_world_size(),
            grad_accum_iters,
            self.train_log_every_iters,
            self.fake_update_ratio,
        )
        logger.info(
            "[train] {} diversity enabled={} weight={} teacher_steps={} anchor_step={}",
            self.trainer_name,
            self.diversity_trick.enabled,
            self.diversity_trick.config.weight,
            self.diversity_trick.config.teacher_inference_steps,
            self.diversity_trick.config.anchor_step,
        )
        real_data_config = self.real_data_fake_trick.config.regions["main"]
        logger.info(
            "[train] {} real_data_fake enabled={} weight={} timesteps={}",
            self.trainer_name,
            real_data_config.enabled,
            real_data_config.weight,
            list(real_data_config.timestep_list),
        )
        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_val)
            if current_iter == 0:
                self.run_inference(current_iter)

        samples = self._iter_train_samples()
        while current_iter < max_train_iters:
            student_result = self._train_one_stage(
                samples,
                stage="student",
                grad_accum_iters=grad_accum_iters,
            )
            loss_dmd_value = student_result["dmd"]
            loss_div_value = student_result["div_loss"]
            loss_real_dmd_value = student_result["real_dmd"]
            loss_fake_value = 0.0
            loss_fake_real_value = 0.0
            for _ in range(self.fake_update_ratio):
                fake_result = self._train_one_stage(
                    samples,
                    stage="fake",
                    grad_accum_iters=grad_accum_iters,
                )
                loss_fake_value += fake_result["loss"]
                loss_fake_real_value += fake_result["fake_real"]
            loss_fake_value /= self.fake_update_ratio
            loss_fake_real_value /= self.fake_update_ratio

            current_iter += 1
            display_fake = reduce_mean(loss_fake_value)
            display_dmd = reduce_mean(loss_dmd_value) if loss_dmd_value is not None else None
            display_div = reduce_mean(loss_div_value)
            display_real_dmd = reduce_mean(loss_real_dmd_value)
            display_fake_real = reduce_mean(loss_fake_real_value)
            current_lr = self.lr_scheduler.get_last_lr()[0]
            if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= max_train_iters:
                dmd_text = "nan" if display_dmd is None else f"{display_dmd:.6f}"
                logger.info(
                    "[train] iter={}/{} dmd={} div_loss={:.6f} real_dmd={:.6f} fake={:.6f} fake_real={:.6f} lr={:.8f}",
                    current_iter,
                    max_train_iters,
                    dmd_text,
                    display_div,
                    display_real_dmd,
                    display_fake,
                    display_fake_real,
                    current_lr,
                )
                metrics = {
                    "train/div_loss": display_div,
                    "train/real_dmd": display_real_dmd,
                    "train/fake": display_fake,
                    "train/fake_real": display_fake_real,
                    "train/lr": current_lr,
                }
                if display_dmd is not None:
                    metrics["train/dmd"] = display_dmd
                self.log_metrics(metrics, step=current_iter)

            if save_every_iters and current_iter % save_every_iters == 0:
                self.save_checkpoint(current_iter, save_total_limit)

            if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                self.run_inference(current_iter)

        logger.info("[train] finished iter={}/{}", current_iter, max_train_iters)

    def _train_one_stage(
        self,
        samples,
        stage,
        grad_accum_iters,
    ):
        if stage == "student":
            optimizer = self.optimizer
            scheduler = self.lr_scheduler
            params = self.trainable_params
            set_sync = self._set_student_gradient_sync
        elif stage == "fake":
            optimizer = self.fake_optimizer
            scheduler = self.fake_lr_scheduler
            params = self.fake_trainable_params
            set_sync = self._set_fake_gradient_sync
        else:
            raise ValueError(f"Unsupported {self.trainer_name} training stage: {stage}")

        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        use_real_data_fake = self.real_data_fake_trick.enabled_for("main")
        if stage == "fake" and use_real_data_fake:
            self.fake_real_optimizer.zero_grad(set_to_none=True)
        if stage == "student":
            metric_values = {
                "div_loss": 0.0,
                "real_dmd": 0.0,
            }
        else:
            metric_values = {"fake_real": 0.0}
        for micro_idx in range(grad_accum_iters):
            sample = next(samples)
            conditions = self._encode_conditions(sample)
            latent_shape = self._latent_shape(sample)
            initial_noise = self.sample_initial_latents(latent_shape)
            set_sync((stage == "student" and use_real_data_fake) or micro_idx == grad_accum_iters - 1)
            result = self.forward_loss(
                latent_shape,
                conditions,
                stage=stage,
                initial_noise=initial_noise,
            )
            if isinstance(result, dict):
                loss = result["loss"]
                for name, value in result.items():
                    if name == "loss":
                        continue
                    scalar = value.item() if torch.is_tensor(value) else float(value)
                    metric_values[name] = metric_values.get(name, 0.0) + scalar / grad_accum_iters
                del result
            else:
                loss = result
            (loss / grad_accum_iters).backward()
            loss_value += loss.item() / grad_accum_iters
            del loss
            if stage == "student" and self.diversity_trick.enabled:
                div_raw, _ = self._backward_diversity_loss(
                    initial_noise,
                    conditions,
                    grad_accum_iters,
                )
                metric_values["div_loss"] += div_raw
            del initial_noise
            if stage == "student" and use_real_data_fake:
                real_result = self.real_data_fake_trick.student_loss(
                    self._real_data_fake_context(
                        sample,
                        conditions,
                    )
                )
                (real_result.loss / grad_accum_iters).backward()
                metric_values["real_dmd"] += real_result.metrics["real_dmd"].item() / grad_accum_iters
                del real_result
            elif stage == "fake" and use_real_data_fake:
                self._set_fake_real_gradient_sync(micro_idx == grad_accum_iters - 1)
                fake_real_result = self.real_data_fake_trick.fake_loss(
                    self._real_data_fake_context(
                        sample,
                        conditions,
                    )
                )
                (fake_real_result.loss / grad_accum_iters).backward()
                metric_values["fake_real"] += fake_real_result.metrics["fake_real"].item() / grad_accum_iters
                del fake_real_result

        self._sync_sequence_parallel_grads(params)
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()
        if stage == "student":
            self._after_student_optimizer_step("main")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if stage == "fake" and use_real_data_fake:
            self._sync_sequence_parallel_grads(self.fake_real_trainable_params)
            torch.nn.utils.clip_grad_norm_(
                self.fake_real_trainable_params,
                self.max_grad_norm,
            )
            self.fake_real_optimizer.step()
            self.fake_real_lr_scheduler.step()
            self.fake_real_optimizer.zero_grad(set_to_none=True)
        if metric_values:
            return {
                "loss": loss_value,
                **metric_values,
            }
        return loss_value

    def _build_denoising_steps(self, device):
        raw_steps = torch.tensor(self.denoising_step_list, dtype=torch.long, device=device)
        if not self.warp_denoising_step:
            return raw_steps.to(dtype=torch.float32)

        timesteps = torch.cat(
            [
                self.denoising_scheduler.timesteps.to(device=device, dtype=torch.float32),
                torch.zeros(1, device=device, dtype=torch.float32),
            ]
        )
        indices = self.denoising_scheduler.num_train_timesteps - raw_steps
        return timesteps[indices]

    def _extract_real_latents(self, sample):
        return self.student.extract_real_latents(
            sample,
            self.latent_dtype,
            broadcast_sequence_parallel_value,
        )

    def _predict_real_student_velocity(
        self,
        region,
        xt,
        sigma,
        condition,
        grad_enabled,
    ):
        del region
        context = torch.enable_grad if grad_enabled else torch.no_grad
        self.student.set_training(grad_enabled)
        with context():
            return self._predict_velocity(
                self.student,
                xt,
                sigma,
                condition,
            )

    def _predict_real_teacher_velocity(
        self,
        region,
        xt,
        sigma,
        condition,
        negative_condition,
    ):
        del region
        self.teacher.set_training(False)
        return self._predict_teacher_velocity(
            xt,
            sigma,
            condition,
            negative_condition,
        )

    def _diversity_step_context(self, initial_noise, conditions):
        return DiversityStepContext(
            initial_noise=initial_noise.detach(),
            condition=conditions[0],
            negative_condition=conditions[1],
            latent_hw=initial_noise.shape[-2:],
            device=self.student.device,
            dtype=self.latent_dtype,
            predict_teacher_velocity=self._predict_teacher_velocity,
            predict_student_velocity=partial(
                self._predict_velocity,
                self.student,
            ),
            student_scheduler=self.scheduler,
            expand_to_ndim=self.scheduler._expand_to_ndim,
        )

    def _backward_diversity_loss(
        self,
        initial_noise,
        conditions,
        grad_accum_iters,
    ):
        self.diversity_trick.prepare_models(
            self.teacher.denoiser(),
            self.student.denoiser(),
        )
        result = self.diversity_trick.student_loss(self._diversity_step_context(initial_noise, conditions))
        (result.loss / grad_accum_iters).backward()
        raw_value = result.metrics["div_loss"].item() / grad_accum_iters
        weighted_value = result.loss.detach().item() / grad_accum_iters
        return raw_value, weighted_value

    def _real_data_fake_setup_context(self):
        return RealDataFakeSetupContext(
            mode="standard",
            scheduler=self.scheduler,
            denoising_scheduler=self.denoising_scheduler,
            num_train_timestep=self.num_train_timestep,
            warp_denoising_step=self.warp_denoising_step,
            score_timestep_shift=self.score_timestep_shift,
            min_step=self.min_step,
            max_step=self.max_step,
            min_score_timestep=self.min_score_timestep,
        )

    def _real_data_fake_context(self, sample, conditions, region="main"):
        return RealDataFakeStepContext(
            region=region,
            fake_model=self.fake_real,
            sample=sample,
            condition=conditions[0],
            negative_condition=conditions[1],
            device=self.student.device,
            dtype=self.latent_dtype,
            extract_real_latents=self._extract_real_latents,
            sample_synced_int=self._sample_synced_int,
            broadcast_noise=broadcast_sequence_parallel_value,
            predict_student_velocity=self._predict_real_student_velocity,
            predict_fake_velocity=self._predict_velocity,
            predict_teacher_velocity=(self._predict_real_teacher_velocity),
            dmd_loss=self._dmd_loss,
        )

    def _set_fake_real_gradient_sync(self, enabled):
        if self.fake_real_model is not None:
            self.fake_real_model.capabilities.require(ParallelCapability).set_gradient_sync(enabled)

    def _set_gradient_sync(self, enabled):
        super()._set_gradient_sync(enabled)
        self._set_fake_real_gradient_sync(enabled)

    def forward_loss(
        self,
        latent_shape,
        conditions,
        stage,
        initial_noise=None,
    ):
        condition, negative_condition = conditions
        (
            generated,
            denoised_timestep_from,
            denoised_timestep_to,
        ) = self.run_back_simulation(
            condition,
            latent_shape,
            grad_enabled=(stage != "fake"),
            xt=initial_noise,
        )

        sigma = self._sample_score_sigma(
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            device=self.student.device,
            dtype=self.latent_dtype,
        )
        noise = self.student.random_noise_like(
            generated,
            torch.float32,
            broadcast_sequence_parallel_value,
        )

        with torch.no_grad():
            renoised_xt = self.student.add_noise(
                self.scheduler,
                generated,
                noise,
                sigma,
            )

        if stage == "fake":
            self.fake.set_training(False)
            velocity_fake = self._predict_velocity(
                self.fake,
                renoised_xt,
                sigma,
                condition,
            )
            velocity_gt = self.student.training_target(generated, noise)
            return self.student.regression_loss(
                velocity_fake,
                velocity_gt,
            )

        with torch.no_grad():
            self.fake.set_training(False)
            self.teacher.set_training(False)
            velocity_fake = self._predict_velocity(
                self.fake,
                renoised_xt,
                sigma,
                condition,
            )
            velocity_teacher = self.teacher.predict_guided_velocity(
                renoised_xt,
                sigma,
                condition,
                negative_condition if self.guidance_scale != 0 else None,
                self.guidance_scale,
                self.cfg_norm,
            )

            x_pred_fake = self.student.x0_from_velocity(
                renoised_xt,
                velocity_fake,
                sigma,
            )
            x_pred_teacher = self.student.x0_from_velocity(
                renoised_xt,
                velocity_teacher,
                sigma,
            )

        loss_dmd = self.student.dmd_loss(
            generated,
            x_pred_fake,
            x_pred_teacher,
        )
        return {
            "loss": loss_dmd,
            "dmd": loss_dmd.detach(),
        }

    def sample_end_step(self):
        return self._sample_synced_int(
            self.diversity_trick.minimum_dmd_step_index,
            self.scheduler.num_inference_steps,
        )

    def run_back_simulation(self, condition, latent_shape, grad_enabled, xt=None):
        if self.random_schedule_enabled:
            self._prepare_sampling_schedule(latent_shape)
            self._active_denoising_steps = self.scheduler.timesteps
        else:
            self.scheduler.set_timesteps(
                self.num_inference_steps,
                sigmas=[float(sigma) for sigma in self.denoising_sigmas.detach().cpu()],
                latent_hw=self.student.latent_hw(latent_shape),
                device=self.student.device,
            )
            self._active_denoising_steps = self.denoising_steps
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)

        end_step_idx = (
            self.sample_end_step()
            if grad_enabled
            else self._sample_synced_int(
                0,
                self.scheduler.num_inference_steps,
            )
        )
        x0 = None
        # Zoe runs every DMD role in eval mode. Gradients are controlled by
        # autograd contexts, not by module.train(), which keeps rollout
        # dropout and other training-only behavior deterministic.
        self.student.set_training(False)
        for idx in range(end_step_idx + 1):
            sigma = self.scheduler.sigma_at(
                idx,
                device=self.student.device,
                dtype=self.latent_dtype,
            )
            context = torch.enable_grad if (grad_enabled and idx == end_step_idx) else torch.no_grad
            with context():
                velocity = self._predict_velocity(
                    self.student,
                    xt,
                    sigma,
                    condition,
                )
            xt, x0 = self.student.step(
                self.scheduler,
                velocity,
                idx,
                xt,
            )
        return self.student.to_dtype(
            x0,
            self.latent_dtype,
        ), *self._denoised_timestep_window(end_step_idx)

    def _sample_score_sigma(self, denoised_timestep_from, denoised_timestep_to, device, dtype):
        sigma = self.score_sigma_sampler.sample(
            ScoreSigmaContext(
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to,
                num_train_timesteps=self.num_train_timestep,
                device=device,
            )
        )
        return broadcast_sequence_parallel_value(sigma).to(dtype=dtype)

    def _denoised_timestep_window(self, exit_idx):
        exit_idx = int(exit_idx)
        steps = getattr(
            self,
            "_active_denoising_steps",
            self.denoising_steps,
        )
        denoised_timestep_from = self._raw_timestep_from_warped_step(steps[exit_idx])
        if exit_idx == len(steps) - 1:
            denoised_timestep_to = 0
        else:
            denoised_timestep_to = self._raw_timestep_from_warped_step(steps[exit_idx + 1])
        return denoised_timestep_from, denoised_timestep_to

    def _raw_timestep_from_warped_step(self, warped_step):
        if not self.warp_denoising_step:
            return int(round(float(warped_step)))
        timesteps = self.denoising_scheduler.timesteps.to(device=warped_step.device, dtype=torch.float32)
        index = torch.argmin((timesteps - warped_step.float()).abs(), dim=0).item()
        return self.denoising_scheduler.num_train_timesteps - int(index)
