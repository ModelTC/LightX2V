import copy
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime.distributed import (
    barrier,
    get_world_size,
    is_distributed,
    is_main_process,
    reduce_mean,
)
from lightx2v_train.runtime.parallel import (
    apply_parallel,
    set_parallel_gradient_sync,
)
from lightx2v_train.runtime.sequence_parallel import (
    broadcast_sequence_parallel_value,
    sync_sequence_parallel_gradients,
)
from lightx2v_train.schedulers import DMDFlowMatchingScheduler
from lightx2v_train.tricks import (
    CdmStepContext,
    CdmTrainerConstraints,
    CdmTrick,
    IdaModelPair,
    IdaSetupContext,
    IdaStepContext,
    ImplicitDistributionAlignmentTrick,
)
from lightx2v_train.utils.registry import TRAINER_REGISTER

from ..base import BaseTrainer
from .checkpoint import DmdCheckpointManager
from .config import DmdConfig
from .math import (
    dmd_loss,
    do_cfg,
)
from .roles import DmdRoleRegistry


@TRAINER_REGISTER("dmd")
class DmdTrainer(BaseTrainer):
    trainer_name = "dmd"
    default_negative_prompt = None
    supports_cdm = True
    supported_cdm_model_names = frozenset(
        {
            "flux2_dev",
            "longcat_image",
            "qwen_image",
        }
    )
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
        self.cdm_trick = CdmTrick.from_mapping(self.dmd_config.get("cdm", {}))
        self.cdm_trick.validate_trainer(
            CdmTrainerConstraints(
                supported=self.supports_cdm,
                model_name=self.model_config["name"],
                supported_model_names=self.supported_cdm_model_names,
            ),
            self.trainer_name,
        )
        self.ida_trick = ImplicitDistributionAlignmentTrick.from_mapping(self.dmd_config.get("ida", {}))
        if self.ida_trick.enabled and not self.supports_ida:
            raise ValueError(f"{self.trainer_name} does not support training.dmd.ida.")
        self.num_inference_steps = parsed.num_inference_steps
        self.fake_update_ratio = parsed.fake_update_ratio
        self.guidance_scale = parsed.guidance_scale
        self.negative_prompt = parsed.negative_prompt
        self.cfg_norm = parsed.cfg_norm
        self.image_sizes = parsed.image_sizes
        self.random_schedule_enabled = parsed.random_schedule_enabled
        self.random_schedule_num_steps_min = parsed.random_schedule_num_steps_min
        self.random_schedule_num_steps_max = parsed.random_schedule_num_steps_max
        self.random_schedule_sigma_min = parsed.random_schedule_sigma_min
        self.random_schedule_sigma_max = parsed.random_schedule_sigma_max
        self.random_schedule_sampling_method = parsed.random_schedule_sampling_method

    @property
    def cdm_enabled(self):
        return self.cdm_trick.enabled

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
        if train_type == "lora":
            model.add_lora(lora_config["rank"], lora_config["alpha"], lora_config.get("target_modules"))
            model.set_lora_trainable()
            return
        model.set_full_trainable()

    def _restore_trainable_model(self, model, role="student"):
        if role == "student":
            train_type = self.student_train_type
        elif role in {"fake", "fake_real"}:
            train_type = self.fake_train_type
        else:
            raise ValueError(f"Unsupported DMD model role: {role}")
        if train_type == "lora":
            model.set_lora_trainable()
            return
        model.set_full_trainable()

    def _save_model_weights(self, model, save_dir, role="student"):
        train_type = self.student_train_type if role == "student" else self.fake_train_type
        if train_type == "lora":
            model.save_lora_weights(save_dir)
            return
        if is_main_process():
            torch.save(model.denoiser_module().state_dict(), os.path.join(save_dir, "model_state.pt"))

    def _load_model_weights(self, model, save_dir, role="student"):
        train_type = self.student_train_type if role == "student" else self.fake_train_type
        if train_type == "lora":
            model.load_lora_weights_for_resume(save_dir)
            return
        model_state_path = os.path.join(save_dir, "model_state.pt")
        if not os.path.exists(model_state_path):
            raise RuntimeError(f"model_state.pt not found in {save_dir}")
        state_dict = torch.load(model_state_path, map_location="cpu", weights_only=False)
        model.denoiser_module().load_state_dict(state_dict)

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
        self.fake_model = build_model(fake_model_config)
        self.fake_model.load_components(transformer_only=True, reference_model=self.model)
        self._setup_trainable_model(self.fake_model, role="fake")
        apply_parallel(self.fake_model, self.config)
        if self.gradient_checkpointing:
            self.fake_model.enable_gradient_checkpointing()

        teacher_model_config = copy.deepcopy(self.config)
        teacher_model_config["model"] = copy.deepcopy(base_model_config)
        if "teacher" in self.model_config:
            if not isinstance(self.model_config["teacher"], dict):
                raise ValueError("model.teacher must be a mapping.")
            teacher_model_config["model"].update(copy.deepcopy(self.model_config["teacher"]))
        self.teacher_model = build_model(teacher_model_config)
        self.teacher_model.load_components(transformer_only=True, reference_model=self.model)
        self.teacher_model.transformer.requires_grad_(False)
        self.teacher_model.transformer.eval()
        apply_parallel(self.teacher_model, self.config)
        self.teacher_model.transformer.eval()

        self.fake_trainable_params = list(self.fake_model.trainable_parameters())
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
        logger.info("[train] dmd student trainable params={}", self._count_trainable(self.model.transformer))
        logger.info("[train] dmd fake trainable params={}", self._count_trainable(self.fake_model.transformer))
        if self.random_schedule_enabled:
            logger.info(
                "[train] dmd random sigma schedule enabled: steps=[{}, {}], sigma=[{}, {}], sampling_method={}",
                self.random_schedule_num_steps_min,
                self.random_schedule_num_steps_max,
                self.random_schedule_sigma_min,
                self.random_schedule_sigma_max,
                self.random_schedule_sampling_method,
            )
        if self.cdm_enabled:
            logger.info(
                "[train] dmd CDM enabled: weight={} warmup_iters={}",
                self.cdm_trick.config.weight,
                self.cdm_trick.config.warmup_iters,
            )

    @staticmethod
    def _count_trainable(module):
        return sum(1 for param in module.parameters() if param.requires_grad)

    def _ida_model_pairs(self):
        return {
            "main": IdaModelPair(
                student=self.model.denoiser_module(),
                fake=self.fake_model.denoiser_module(),
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
        latent_hw = latent_shape[-2:]
        if self.random_schedule_enabled:
            num_steps = self._sample_synced_int(self.random_schedule_num_steps_min, self.random_schedule_num_steps_max + 1)
            self.scheduler.set_random_timesteps(
                self.random_schedule_num_steps_min,
                self.random_schedule_num_steps_max,
                sigma_min=self.random_schedule_sigma_min,
                sigma_max=self.random_schedule_sigma_max,
                sampling_method=self.random_schedule_sampling_method,
                latent_hw=latent_hw,
                device=self.model.device,
                num_steps=num_steps,
            )
            return
        self.scheduler.set_timesteps(self.num_inference_steps, latent_hw=latent_hw, device=self.model.device)

    def _latent_shape(self, sample):
        image = sample["inputs"].get("target_image")
        if image is None:
            raise KeyError("DMD image latent shape expects inputs.target_image.")
        batch_size = image.shape[0]
        if self.image_sizes:
            image_size_index = int(torch.randint(0, len(self.image_sizes), (1,), device=self.model.device).item())
            image_size_index = broadcast_sequence_parallel_value(image_size_index)
            height, width = self.image_sizes[image_size_index]
        else:
            height, width = image.shape[-2], image.shape[-1]

        latent_channels = getattr(self.model.vae.config, "z_dim", None)
        if latent_channels is None:
            latent_channels = self.model.transformer.config.in_channels // 4
        return (
            batch_size,
            int(latent_channels),
            1,
            height // self.model.vae_scale_factor,
            width // self.model.vae_scale_factor,
        )

    def _encode_conditions(self, sample):
        conditioning = sample["conditioning"]
        prompt = conditioning.get("prompt", "")
        with torch.no_grad():
            condition = self.model.encode_prompt_condition(prompt)
            if self.guidance_scale > 1:
                is_scalar_prompt = isinstance(prompt, str)
                batch_size = 1 if is_scalar_prompt else len(prompt)
                negative_prompt = self._negative_prompt_for_conditioning(
                    conditioning,
                    batch_size,
                    return_scalar=is_scalar_prompt,
                )
                negative_condition = self.model.encode_prompt_condition(negative_prompt)
            else:
                negative_condition = None
        condition = broadcast_sequence_parallel_value(condition)
        negative_condition = broadcast_sequence_parallel_value(negative_condition) if negative_condition is not None else None
        return condition, negative_condition

    def _negative_prompt_for_conditioning(self, conditioning, batch_size, return_scalar=False):
        negative_prompt = conditioning.get("negative_prompt")
        if negative_prompt is None:
            values = []
        elif isinstance(negative_prompt, str):
            values = [negative_prompt]
        else:
            values = list(negative_prompt)

        if not values:
            values = [self.negative_prompt] * batch_size
        elif len(values) == 1 and batch_size > 1:
            values *= batch_size
        elif len(values) != batch_size:
            raise ValueError(f"Expected {batch_size} negative prompts, got {len(values)}.")

        values = [value if isinstance(value, str) and value.strip() else self.negative_prompt for value in values]
        return values[0] if return_scalar else values

    def _predict_velocity(self, model, latents, sigma, condition):
        denoiser_input = model.prepare_denoiser_input(latents)
        prediction = model.denoise(denoiser_input, sigma, condition)
        prediction = model.postprocess_denoiser_output(prediction, denoiser_input)
        return prediction

    def _predict_teacher_velocity(self, latents, sigma, condition, negative_condition):
        if negative_condition is None:
            return self._predict_velocity(self.teacher_model, latents, sigma, condition)

        if self.teacher_model.cfg_on_denoiser_output():
            denoiser_input = self.teacher_model.prepare_denoiser_input(latents)
            cond_prediction = self.teacher_model.denoise(denoiser_input, sigma, condition)
            uncond_prediction = self.teacher_model.denoise(denoiser_input, sigma, negative_condition)
            prediction = self._do_cfg(cond_prediction, uncond_prediction, self.guidance_scale, self.cfg_norm)
            return self.teacher_model.postprocess_denoiser_output(prediction, denoiser_input)

        velocity_teacher_cond = self._predict_velocity(self.teacher_model, latents, sigma, condition)
        velocity_teacher_uncond = self._predict_velocity(self.teacher_model, latents, sigma, negative_condition)
        return self._do_cfg(velocity_teacher_cond, velocity_teacher_uncond, self.guidance_scale, self.cfg_norm)

    def sample_initial_latents(self, latent_shape):
        return broadcast_sequence_parallel_value(torch.randn(latent_shape, device=self.model.device, dtype=self.running_dtype))

    def _sample_synced_int(self, low, high):
        value = torch.randint(int(low), int(high), (1,), device=self.model.device, dtype=torch.int64)
        if is_distributed():
            dist.broadcast(value, src=0)
        return int(value.item())

    def sample_end_step(self):
        return self._sample_synced_int(0, self.scheduler.num_inference_steps)

    def run_back_simulation(self, condition, latent_shape, end_step_idx, grad_enabled, xt=None):
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        x0 = None
        xt_end = None
        vt_end = None
        self.model.transformer.train()
        for idx in range(end_step_idx + 1):
            sigma = self.scheduler.sigma_at(idx, latent_shape[0], device=self.model.device, dtype=self.running_dtype)
            context = torch.enable_grad if (grad_enabled and idx == end_step_idx) else torch.no_grad
            with context():
                velocity = self._predict_velocity(self.model, xt, sigma, condition)
            if idx == end_step_idx:
                xt_end = xt.detach()
                vt_end = velocity.detach()
            xt, x0 = self.scheduler.step_by_index(velocity, idx, xt)
        return x0, xt_end, vt_end

    def _cdm_step_context(
        self,
        xt,
        vt,
        end_step_idx,
        condition,
        current_iter,
    ):
        return CdmStepContext(
            trajectory_latent=xt,
            trajectory_velocity=vt,
            end_step_index=end_step_idx,
            condition=condition,
            current_iteration=current_iter,
            device=self.model.device,
            dtype=self.running_dtype,
            scheduler=self.scheduler,
            predict_student_velocity=lambda latent, sigma, cond: (
                self._predict_velocity(
                    self.model,
                    latent,
                    sigma,
                    cond,
                )
            ),
            predict_fake_velocity=lambda latent, sigma, cond: (
                self._predict_velocity(
                    self.fake_model,
                    latent,
                    sigma,
                    cond,
                )
            ),
            predict_teacher_velocity=lambda latent, sigma, cond: (
                self._predict_teacher_velocity(
                    latent,
                    sigma,
                    cond,
                    None,
                )
            ),
            prepare_fake_model=self.fake_model.transformer.eval,
            dmd_loss=self._dmd_loss,
        )

    def forward_loss(self, latent_shape, conditions, stage, current_iter=None):
        condition, negative_condition = conditions
        self._prepare_sampling_schedule(latent_shape)
        end_step_idx = self.sample_end_step()
        xt_start = self.sample_initial_latents(latent_shape)
        x0, xt_end, vt_end = self.run_back_simulation(condition, latent_shape, end_step_idx, grad_enabled=(stage != "fake"), xt=xt_start)

        sigma = self.scheduler.sample_renoise_sigma(latent_shape[0], device=self.model.device, dtype=self.running_dtype)
        sigma = broadcast_sequence_parallel_value(sigma)
        noise = broadcast_sequence_parallel_value(torch.randn(latent_shape, device=self.model.device, dtype=torch.float32))
        renoised_xt = self.scheduler.add_noise(x0.detach(), noise, sigma)

        if stage == "fake":
            self.fake_model.transformer.train()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            velocity_gt = self.scheduler.build_train_gt(x0.float(), noise)
            loss_fake = F.mse_loss(velocity_fake.float(), velocity_gt.float(), reduction="mean")
            return {"fake": loss_fake}

        with torch.no_grad():
            self.fake_model.transformer.eval()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            velocity_teacher = self._predict_teacher_velocity(renoised_xt, sigma, condition, negative_condition)

        expanded_sigma = self.scheduler._expand_to_ndim(sigma, renoised_xt.ndim)
        x_pred_fake = renoised_xt - expanded_sigma * velocity_fake
        x_pred_teacher = renoised_xt - expanded_sigma * velocity_teacher
        loss_dmd = self._dmd_loss(x0, x_pred_fake, x_pred_teacher)
        cdm_result = self.cdm_trick.student_loss(
            self._cdm_step_context(
                xt_end,
                vt_end,
                end_step_idx,
                condition,
                current_iter,
            )
        )
        return {
            "student": loss_dmd + cdm_result.loss,
            "dmd": loss_dmd.detach(),
            "cdm": cdm_result.metrics["cdm"],
            "cdm_weight": cdm_result.metrics["cdm_weight"],
        }

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        max_train_iters = self.max_train_iters
        grad_accum_iters = max(1, int(self.gradient_accumulation_iters))
        fake_update_ratio = self.fake_update_ratio
        max_grad_norm = self.max_grad_norm
        save_every_iters = self.save_every_iters
        save_total_limit = self.save_total_limit

        logger.info(
            "[train] start method={} student_train_type={} fake_train_type={} iter={}/{} world_size={} grad_accum={} train_log_every_iters={} fake_update_ratio={}",
            self.training_config.get("method", "dmd"),
            self.student_train_type,
            self.fake_train_type,
            current_iter,
            max_train_iters,
            get_world_size(),
            grad_accum_iters,
            self.train_log_every_iters,
            fake_update_ratio,
        )
        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_eval)
            if current_iter == 0:
                self.run_inference(current_iter)

        samples = self._iter_train_samples()
        while current_iter < max_train_iters:
            self.optimizer.zero_grad(set_to_none=True)
            running_dmd = 0.0
            running_cdm = 0.0
            running_cdm_weight = 0.0
            for micro_idx in range(grad_accum_iters):
                sample = next(samples)
                conditions = self._encode_conditions(sample)
                latent_shape = self._latent_shape(sample)
                self._set_student_gradient_sync(micro_idx == grad_accum_iters - 1)
                res_student = self.forward_loss(latent_shape, conditions, stage="student", current_iter=current_iter)
                loss_student = res_student["student"]
                (loss_student / grad_accum_iters).backward()
                running_dmd += res_student["dmd"].item() / grad_accum_iters
                if self.cdm_enabled:
                    running_cdm += res_student["cdm"].item() / grad_accum_iters
                    running_cdm_weight = res_student["cdm_weight"]

            self._sync_sequence_parallel_grads(self.trainable_params)
            torch.nn.utils.clip_grad_norm_(self.trainable_params, max_grad_norm)
            self.optimizer.step()
            self._after_student_optimizer_step("main")
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            running_fake = 0.0
            for _ in range(fake_update_ratio):
                self.fake_optimizer.zero_grad(set_to_none=True)
                fake_loss = 0.0
                for micro_idx in range(grad_accum_iters):
                    sample = next(samples)
                    conditions = self._encode_conditions(sample)
                    latent_shape = self._latent_shape(sample)
                    self._set_fake_gradient_sync(micro_idx == grad_accum_iters - 1)
                    res_fake = self.forward_loss(
                        latent_shape,
                        conditions,
                        stage="fake",
                    )
                    loss_fake = res_fake["fake"]
                    (loss_fake / grad_accum_iters).backward()
                    fake_loss += loss_fake.item() / grad_accum_iters
                self._sync_sequence_parallel_grads(self.fake_trainable_params)
                torch.nn.utils.clip_grad_norm_(
                    self.fake_trainable_params,
                    max_grad_norm,
                )
                self.fake_optimizer.step()
                self.fake_lr_scheduler.step()
                self.fake_optimizer.zero_grad(set_to_none=True)
                running_fake += fake_loss / fake_update_ratio

            current_iter += 1
            display_dmd = reduce_mean(running_dmd)
            display_fake = reduce_mean(running_fake)
            current_lr = self.lr_scheduler.get_last_lr()[0]
            if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= max_train_iters:
                if self.cdm_enabled:
                    display_cdm = reduce_mean(running_cdm)
                    logger.info(
                        "[train] iter={}/{} dmd={:.6f} cdm={:.6f} cdm_w={:.6f} fake={:.6f} lr={:.8f}",
                        current_iter,
                        max_train_iters,
                        display_dmd,
                        display_cdm,
                        running_cdm_weight,
                        display_fake,
                        current_lr,
                    )
                    self.log_metrics(
                        {
                            "train/dmd": display_dmd,
                            "train/cdm": display_cdm,
                            "train/cdm_weight": running_cdm_weight,
                            "train/fake": display_fake,
                            "train/lr": current_lr,
                        },
                        step=current_iter,
                    )
                else:
                    logger.info(
                        "[train] iter={}/{} dmd={:.6f} fake={:.6f} lr={:.8f}",
                        current_iter,
                        max_train_iters,
                        display_dmd,
                        display_fake,
                        current_lr,
                    )
                    self.log_metrics(
                        {
                            "train/dmd": display_dmd,
                            "train/fake": display_fake,
                            "train/lr": current_lr,
                        },
                        step=current_iter,
                    )

            if save_every_iters and current_iter % save_every_iters == 0:
                self.save_checkpoint(current_iter, save_total_limit)

            if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                self.run_inference(current_iter)

        logger.info("[train] finished iter={}/{}", current_iter, max_train_iters)

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
        set_parallel_gradient_sync(self.model, enabled)

    def _set_fake_gradient_sync(self, enabled):
        set_parallel_gradient_sync(self.fake_model, enabled)

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
