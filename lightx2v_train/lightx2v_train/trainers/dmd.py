import copy
import math
import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime.checkpoint import prune_checkpoints
from lightx2v_train.runtime.distributed import (
    barrier,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_world_size,
    is_distributed,
    is_main_process,
    is_sequence_parallel_enabled,
    reduce_mean,
)
from lightx2v_train.runtime.parallel import apply_parallel, set_parallel_gradient_sync
from lightx2v_train.runtime.sequence_parallel import all_gather_sequence, broadcast_sequence_parallel_value, sync_sequence_parallel_gradients
from lightx2v_train.schedulers import DMDFlowMatchingScheduler
from lightx2v_train.schedulers.flow_matching import CausalForcingFlowMatchScheduler
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .base import BaseTrainer


@TRAINER_REGISTER("dmd")
class DmdTrainer(BaseTrainer):
    def _resolve_train_type(self):
        if "train_type" in self.training_config:
            raise ValueError("DMD trainers use training.student.train_type and training.fake.train_type; remove training.train_type.")
        return None

    def __init__(self, config):
        super().__init__(config)
        self.student_config = self.training_config["student"]
        self.fake_config = self.training_config["fake"]
        self.student_train_type = self.student_config["train_type"]
        self.fake_train_type = self.fake_config["train_type"]

        if "lora" in self.training_config:
            raise ValueError("DMD trainers do not read training.lora. Use training.student.lora and training.fake.lora.")

        self.student_lora_config = None
        if self.student_train_type == "lora":
            self.student_lora_config = copy.deepcopy(self.student_config["lora"])
            self.student_lora_config["rank"] = int(self.student_lora_config["rank"])
            self.student_lora_config["alpha"] = int(self.student_lora_config["alpha"])

        self.fake_lora_config = None
        if self.fake_train_type == "lora":
            self.fake_lora_config = copy.deepcopy(self.fake_config["lora"])
            self.fake_lora_config["rank"] = int(self.fake_lora_config["rank"])
            self.fake_lora_config["alpha"] = int(self.fake_lora_config["alpha"])

        self.fake_optimizer_config = self.fake_config["optimizer"]
        self.fake_optimizer_learning_rate = self.fake_optimizer_config.get("learning_rate", self.optimizer_learning_rate)
        self.fake_optimizer_adam_beta1 = self.fake_optimizer_config.get("adam_beta1", self.optimizer_adam_beta1)
        self.fake_optimizer_adam_beta2 = self.fake_optimizer_config.get("adam_beta2", self.optimizer_adam_beta2)
        self.fake_optimizer_weight_decay = self.fake_optimizer_config.get("weight_decay", self.optimizer_weight_decay)
        self.fake_optimizer_adam_epsilon = self.fake_optimizer_config.get("adam_epsilon", self.optimizer_adam_epsilon)

        self.dmd_config = self.training_config["dmd"]
        teacher_config = self.training_config["teacher"] if "teacher" in self.training_config else {}
        self.num_inference_steps = int(self.dmd_config.get("num_inference_steps", 4))
        self.fake_update_ratio = max(1, int(self.dmd_config.get("fake_update_ratio", 1)))
        self.guidance_scale = float(teacher_config["guidance_scale"] if "guidance_scale" in teacher_config else self.dmd_config.get("guidance_scale", 3.0))
        self.negative_prompt = teacher_config["negative_prompt"] if "negative_prompt" in teacher_config else self.dmd_config.get("negative_prompt", " ")
        self.cfg_norm = teacher_config["cfg_norm"] if "cfg_norm" in teacher_config else self.dmd_config.get("cfg_norm", "layer_norm")
        self.image_sizes = self.dmd_config.get("image_sizes", [])

        random_schedule_config = self.dmd_config.get("random_schedule", {})
        self.random_schedule_enabled = bool(random_schedule_config.get("enabled", False))
        self.random_schedule_num_steps_min = int(random_schedule_config.get("num_steps_min", 1))
        self.random_schedule_num_steps_max = int(random_schedule_config.get("num_steps_max", self.num_inference_steps))
        self.random_schedule_sigma_min = float(random_schedule_config.get("sigma_min", 0.02))
        self.random_schedule_sigma_max = float(random_schedule_config.get("sigma_max", 0.98))
        self.random_schedule_sampling_method = random_schedule_config.get("sampling_method", "stratified")

        self.cdm_config = self.dmd_config.get("cdm", {})
        self.cdm_enabled = bool(self.cdm_config.get("enabled", False))
        self.cdm_weight = float(self.cdm_config.get("weight", 1.0))
        self.cdm_warmup_iters = int(self.cdm_config.get("warmup_iters", 0))
        self.cdm_norm_clip_min = float(self.cdm_config.get("norm_clip_min", 0.1))

    def _get_optimizer_config(self):
        return self.training_config["student"]["optimizer"]

    def _setup_trainable_model(self, model, role="student"):
        if role == "student":
            train_type = self.student_train_type
            lora_config = self.student_lora_config
        elif role == "fake":
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
        elif role == "fake":
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
        base_model_config = {key: copy.deepcopy(value) for key, value in self.model_config.items() if key not in {"fake", "teacher", "student"}}

        fake_model_config = copy.deepcopy(self.config)
        fake_model_config["model"] = copy.deepcopy(base_model_config)
        if "fake" in self.model_config:
            if not isinstance(self.model_config["fake"], dict):
                raise ValueError("model.fake must be a mapping.")
            fake_model_config["model"].update(copy.deepcopy(self.model_config["fake"]))
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
            logger.info("[train] dmd CDM enabled: weight={} warmup_iters={}", self.cdm_weight, self.cdm_warmup_iters)

    @staticmethod
    def _count_trainable(module):
        return sum(1 for param in module.parameters() if param.requires_grad)

    @staticmethod
    def _do_cfg(cond_pred, uncond_pred, cfg_scale, cfg_norm):
        pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        if cfg_norm in (None, "none"):
            return pred
        if cfg_norm == "layer_norm":
            cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True)
            pred_norm = torch.norm(pred, dim=-1, keepdim=True)
            return pred * (cond_norm / torch.clamp(pred_norm, min=1e-12))
        if cfg_norm == "scalar":
            cond_norm = torch.norm(cond_pred)
            pred_norm = torch.norm(pred)
            return pred * min(1.0, (cond_norm / torch.clamp(pred_norm, min=1e-12)).item())
        raise ValueError(f"Unsupported cfg_norm: {cfg_norm}")

    @staticmethod
    def _dmd_loss(latents, x_pred_fake_flow, x_pred_teacher, norm_clip_min=None):
        with torch.no_grad():
            grad = x_pred_fake_flow - x_pred_teacher
            dims = tuple(range(1, latents.ndim))
            normalizer = torch.abs(latents - x_pred_teacher).mean(dim=dims, keepdim=True)
            if norm_clip_min is not None:
                normalizer = normalizer.clamp(min=float(norm_clip_min))
            grad = torch.nan_to_num(grad / normalizer)
        return 0.5 * F.mse_loss(latents.float(), (latents.float() - grad.float()).detach(), reduction="mean")

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

    def _effective_cdm_weight(self, current_iter=None):
        if self.cdm_warmup_iters <= 0 or current_iter is None:
            return self.cdm_weight
        progress = min(1.0, max(0.0, float(current_iter) / float(self.cdm_warmup_iters)))
        return progress * self.cdm_weight

    def _latent_shape(self, sample):
        image = sample["target_image"]
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
        prompt = sample["prompt"]
        with torch.no_grad():
            condition = self.model.encode_prompt_condition(prompt)
            if self.guidance_scale > 1:
                if isinstance(prompt, str):
                    negative_prompt = self.negative_prompt
                else:
                    negative_prompt = [self.negative_prompt] * len(prompt)
                negative_condition = self.model.encode_prompt_condition(negative_prompt)
            else:
                negative_condition = None
        condition = broadcast_sequence_parallel_value(condition)
        negative_condition = broadcast_sequence_parallel_value(negative_condition) if negative_condition is not None else None
        return condition, negative_condition

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

    def _compute_cdm_loss(self, xt, vt, end_step_idx, condition):
        batch_size = xt.shape[0]

        traj_sigma = self.scheduler.sigma_at(end_step_idx, batch_size, device=self.model.device, dtype=torch.float32)
        student_sigma = self.scheduler.sample_renoise_sigma(batch_size, device=self.model.device, dtype=torch.float32)

        traj_sigma_expanded = self.scheduler._expand_to_ndim(traj_sigma, vt.ndim)
        student_sigma_expanded = self.scheduler._expand_to_ndim(student_sigma, vt.ndim)
        student_xt = xt + (student_sigma_expanded - traj_sigma_expanded) * vt

        student_prediction = self._predict_velocity(self.model, student_xt.to(self.running_dtype), student_sigma, condition)
        student_x0 = student_xt - student_sigma_expanded * student_prediction
        student_x0 = student_x0.to(self.running_dtype)

        teacher_sigma = self.scheduler.sample_renoise_sigma(batch_size, device=self.model.device, dtype=self.running_dtype)
        teacher_noise = torch.randn_like(student_x0)
        teacher_xt = self.scheduler.add_noise(student_x0, teacher_noise, teacher_sigma)

        with torch.no_grad():
            self.fake_model.transformer.eval()
            velocity_fake = self._predict_velocity(self.fake_model, teacher_xt, teacher_sigma, condition)
            velocity_teacher = self._predict_teacher_velocity(teacher_xt, teacher_sigma, condition, None)

        teacher_sigma_expanded = self.scheduler._expand_to_ndim(teacher_sigma, teacher_xt.ndim)
        x_pred_fake = teacher_xt - teacher_sigma_expanded * velocity_fake
        x_pred_teacher = teacher_xt - teacher_sigma_expanded * velocity_teacher
        return self._dmd_loss(student_x0, x_pred_fake, x_pred_teacher, norm_clip_min=self.cdm_norm_clip_min)

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
        total_loss = loss_dmd

        cdm_weight = self._effective_cdm_weight(current_iter)
        if self.cdm_enabled and cdm_weight != 0:
            loss_cdm = self._compute_cdm_loss(xt_end, vt_end, end_step_idx, condition)
            total_loss = total_loss + cdm_weight * loss_cdm
        else:
            loss_cdm = total_loss.new_zeros(())
        return {"student": total_loss, "dmd": loss_dmd.detach(), "cdm": loss_cdm.detach(), "cdm_weight": cdm_weight}

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
        grad_accum_counter = 0
        running_dmd = 0.0
        running_cdm = 0.0
        running_fake = 0.0
        running_cdm_weight = 0.0
        microbatches = []

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

        epoch = 0
        while current_iter < max_train_iters:
            sampler = getattr(self.dataloader_train, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for sample in self.dataloader_train:
                conditions = self._encode_conditions(sample)
                latent_shape = self._latent_shape(sample)
                sync_grad = (grad_accum_counter + 1) % grad_accum_iters == 0

                self._set_student_gradient_sync(sync_grad)
                res_student = self.forward_loss(latent_shape, conditions, stage="student", current_iter=current_iter)
                loss_student = res_student["student"]
                (loss_student / grad_accum_iters).backward()
                running_dmd += res_student["dmd"].item() / grad_accum_iters
                if self.cdm_enabled:
                    running_cdm += res_student["cdm"].item() / grad_accum_iters
                    running_cdm_weight = res_student["cdm_weight"]
                microbatches.append((latent_shape, conditions))

                grad_accum_counter += 1
                if grad_accum_counter % grad_accum_iters != 0:
                    continue

                self._sync_sequence_parallel_grads(self.trainable_params)
                torch.nn.utils.clip_grad_norm_(self.trainable_params, max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                fake_loss = 0.0
                for _ in range(fake_update_ratio):
                    for microbatch_idx, (micro_latent_shape, micro_conditions) in enumerate(microbatches):
                        sync_fake_grad = microbatch_idx == len(microbatches) - 1
                        self._set_fake_gradient_sync(sync_fake_grad)
                        res_fake = self.forward_loss(micro_latent_shape, micro_conditions, stage="fake")
                        loss_fake = res_fake["fake"]
                        (loss_fake / len(microbatches)).backward()
                        fake_loss += loss_fake.item() / len(microbatches)
                    self._sync_sequence_parallel_grads(self.fake_trainable_params)
                    torch.nn.utils.clip_grad_norm_(self.fake_trainable_params, max_grad_norm)
                    self.fake_optimizer.step()
                    self.fake_lr_scheduler.step()
                    self.fake_optimizer.zero_grad(set_to_none=True)
                running_fake += fake_loss / fake_update_ratio
                microbatches = []

                current_iter += 1
                display_dmd = reduce_mean(running_dmd)
                display_fake = reduce_mean(running_fake)
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
                            self.lr_scheduler.get_last_lr()[0],
                        )
                    else:
                        logger.info(
                            "[train] iter={}/{} dmd={:.6f} fake={:.6f} lr={:.8f}",
                            current_iter,
                            max_train_iters,
                            display_dmd,
                            display_fake,
                            self.lr_scheduler.get_last_lr()[0],
                        )
                running_dmd = 0.0
                running_cdm = 0.0
                running_fake = 0.0

                if save_every_iters and current_iter % save_every_iters == 0:
                    self.save_checkpoint(current_iter, save_total_limit)

                if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                    self.run_inference(current_iter)

                if current_iter >= max_train_iters:
                    break

            epoch += 1

        logger.info("[train] finished iter={}/{}", current_iter, max_train_iters)

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
        return os.path.join(root_dir, "fake_lora" if self.fake_train_type == "lora" else "fake_model")

    def _load_resume_state(self, resume_ckpt_path):
        if self.model.is_fsdp2_wrapped() or self.fake_model.is_fsdp2_wrapped():
            self._load_distributed_state(resume_ckpt_path)
            return

        self._load_single_process_state(resume_ckpt_path)

    def _validate_dmd_checkpoint_metadata(self, state, state_path, resume_ckpt_path):
        if "world_size" not in state:
            if get_world_size() != 1:
                raise RuntimeError(f"Cannot resume checkpoint without world_size metadata using world_size={get_world_size()}: {state_path}")
            logger.warning("Checkpoint {} has no world_size metadata. Assuming world_size=1 for backward compatibility.", state_path)
            state["world_size"] = 1
        self._validate_checkpoint_metadata(state, state_path, resume_ckpt_path)
        checkpoint_student_train_type = state.get("student_train_type")
        if checkpoint_student_train_type is not None and checkpoint_student_train_type != self.student_train_type:
            raise RuntimeError(f"Cannot resume checkpoint saved with student_train_type={checkpoint_student_train_type!r} using training.student.train_type={self.student_train_type!r}: {state_path}")
        checkpoint_fake_train_type = state.get("fake_train_type")
        if checkpoint_fake_train_type is not None and checkpoint_fake_train_type != self.fake_train_type:
            raise RuntimeError(f"Cannot resume checkpoint saved with fake_train_type={checkpoint_fake_train_type!r} using training.fake.train_type={self.fake_train_type!r}: {state_path}")

    def _load_single_process_state(self, resume_ckpt_path):
        training_state_path = os.path.join(resume_ckpt_path, "training_state.pt")
        fake_weights_dir = self._fake_weights_dir(resume_ckpt_path)

        if not os.path.exists(training_state_path):
            raise RuntimeError(f"training_state.pt not found in {resume_ckpt_path}")

        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        self._validate_dmd_checkpoint_metadata(state, training_state_path, resume_ckpt_path)
        self._load_model_weights(self.model, resume_ckpt_path, role="student")
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])

        if os.path.exists(fake_weights_dir):
            self._load_model_weights(self.fake_model, fake_weights_dir, role="fake")
        else:
            logger.warning("Fake model weights not found in {}. Fake model not restored.", fake_weights_dir)

        if "fake_optimizer" in state:
            self.fake_optimizer.load_state_dict(state["fake_optimizer"])
        else:
            logger.warning("fake_optimizer state not found in {}.", training_state_path)

        if "fake_lr_scheduler" in state:
            self.fake_lr_scheduler.load_state_dict(state["fake_lr_scheduler"])
        else:
            logger.warning("fake_lr_scheduler state not found in {}.", training_state_path)
        logger.info("Restored DMD training state from {}", training_state_path)

    def _load_distributed_state(self, resume_ckpt_path):
        dist_state_path = os.path.join(resume_ckpt_path, "dist_state")
        if not os.path.exists(dist_state_path):
            raise RuntimeError(f"FSDP2 resume requires dist_state/, but it was not found in {resume_ckpt_path}")

        trainer_state_path = os.path.join(resume_ckpt_path, "trainer_state.pt")
        if not os.path.exists(trainer_state_path):
            raise RuntimeError(f"trainer_state.pt not found in {resume_ckpt_path}")
        trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        self._validate_dmd_checkpoint_metadata(trainer_state, trainer_state_path, resume_ckpt_path)

        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        student_model_state, student_optim_state = get_state_dict(self.model.fsdp2_state_module(), self.optimizer, options=options)
        fake_model_state, fake_optim_state = get_state_dict(self.fake_model.fsdp2_state_module(), self.fake_optimizer, options=options)
        state = {
            "student_model": student_model_state,
            "student_optimizer": student_optim_state,
            "fake_model": fake_model_state,
            "fake_optimizer": fake_optim_state,
        }
        dcp.load(state, checkpoint_id=dist_state_path)
        set_state_dict(
            self.model.fsdp2_state_module(),
            self.optimizer,
            model_state_dict=state["student_model"],
            optim_state_dict=state["student_optimizer"],
            options=options,
        )
        set_state_dict(
            self.fake_model.fsdp2_state_module(),
            self.fake_optimizer,
            model_state_dict=state["fake_model"],
            optim_state_dict=state["fake_optimizer"],
            options=options,
        )

        self.lr_scheduler.load_state_dict(trainer_state["lr_scheduler"])
        self.fake_lr_scheduler.load_state_dict(trainer_state["fake_lr_scheduler"])
        logger.info("Restored distributed DMD training state from {}", resume_ckpt_path)

    def save_checkpoint(self, iteration, save_total_limit):
        if is_main_process():
            prune_checkpoints(self.output_train_dir, save_total_limit)

        save_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        logger.info("[train] saving checkpoint iter={} path={}", iteration, save_dir)
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        barrier()

        save_student_weights = self.student_train_type == "lora" or not self.model.is_fsdp2_wrapped()
        if save_student_weights:
            self._save_model_weights(self.model, save_dir, role="student")
        barrier()

        fake_save_dir = self._fake_weights_dir(save_dir)
        save_fake_weights = self.fake_train_type == "lora" or not self.fake_model.is_fsdp2_wrapped()
        if save_fake_weights and is_main_process():
            os.makedirs(fake_save_dir, exist_ok=True)
        barrier()
        if save_fake_weights:
            self._save_model_weights(self.fake_model, fake_save_dir, role="fake")
        barrier()

        config_path = self.config.get("config_path")
        if is_main_process() and config_path is not None:
            shutil.copy2(config_path, os.path.join(save_dir, "config.yaml"))

        if self.model.is_fsdp2_wrapped() or self.fake_model.is_fsdp2_wrapped():
            self._save_distributed_state(save_dir, iteration)
            if self._should_save_consolidated_student():
                self._save_consolidated_student_weights(save_dir)
            barrier()
            logger.info("[train] saved checkpoint iter={} path={}", iteration, save_dir)
            return

        training_state = {
            "iteration": iteration,
            "world_size": get_world_size(),
            "student_train_type": self.student_train_type,
            "fake_train_type": self.fake_train_type,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "fake_optimizer": self.fake_optimizer.state_dict(),
            "fake_lr_scheduler": self.fake_lr_scheduler.state_dict(),
        }
        if is_main_process():
            torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
        barrier()
        logger.info("[train] saved checkpoint iter={} path={}", iteration, save_dir)

    def _should_save_consolidated_student(self):
        enabled = bool(self.training_config.get("save_consolidated_student", False))
        if not enabled:
            return False
        if self.student_train_type != "full":
            logger.warning("save_consolidated_student=true is ignored because training.student.train_type='{}'.", self.student_train_type)
            return False
        return True

    def _save_consolidated_student_weights(self, save_dir):
        output_dir = os.path.join(save_dir, "student_consolidated")
        logger.info("[train] saving consolidated student weights to {}", output_dir)
        self.model.save_full_model(output_dir)
        barrier()

    def _save_distributed_state(self, save_dir, iteration):
        dist_state_path = os.path.join(save_dir, "dist_state")
        if is_main_process():
            os.makedirs(dist_state_path, exist_ok=True)
            torch.save(
                {
                    "iteration": iteration,
                    "world_size": get_world_size(),
                    "student_train_type": self.student_train_type,
                    "fake_train_type": self.fake_train_type,
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "fake_lr_scheduler": self.fake_lr_scheduler.state_dict(),
                },
                os.path.join(save_dir, "trainer_state.pt"),
            )
        barrier()

        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        student_model_state, student_optim_state = get_state_dict(self.model.fsdp2_state_module(), self.optimizer, options=options)
        fake_model_state, fake_optim_state = get_state_dict(self.fake_model.fsdp2_state_module(), self.fake_optimizer, options=options)
        dcp.save(
            {
                "student_model": student_model_state,
                "student_optimizer": student_optim_state,
                "fake_model": fake_model_state,
                "fake_optimizer": fake_optim_state,
            },
            checkpoint_id=dist_state_path,
        )


@TRAINER_REGISTER("video_dmd")
class VideoDmdTrainer(DmdTrainer):
    trainer_name = "video_dmd"
    allowed_model_names = {"wan_t2v", "wan_t2v_14b", "wan_ti2v_5b"}

    def __init__(self, config):
        super().__init__(config)
        model_name = self.model_config["name"]
        if model_name not in self.allowed_model_names:
            allowed = ", ".join(repr(name) for name in sorted(self.allowed_model_names))
            raise ValueError(f"{self.trainer_name} trainer currently requires model.name in {{{allowed}}}.")
        default_lora_target_modules = ["q", "k", "v", "o", "ffn.0", "ffn.2"]
        if self.student_lora_config is not None and "target_modules" not in self.student_lora_config:
            self.student_lora_config["target_modules"] = list(default_lora_target_modules)
        if self.fake_lora_config is not None and "target_modules" not in self.fake_lora_config:
            self.fake_lora_config["target_modules"] = list(default_lora_target_modules)

        scheduler_config = self.config["scheduler"]
        self.num_train_timestep = int(self.dmd_config["num_train_timestep"] if "num_train_timestep" in self.dmd_config else scheduler_config.get("num_train_timesteps", 1000))
        default_denoising_steps = [int(round(self.num_train_timestep * (1.0 - step_idx / self.num_inference_steps))) for step_idx in range(self.num_inference_steps)]
        self.denoising_step_list = list(self.dmd_config.get("denoising_step_list", default_denoising_steps))
        self.num_inference_steps = len(self.denoising_step_list)
        self.num_training_frames = int(self.dmd_config.get("num_training_frames", 21))
        self.warp_denoising_step = bool(self.dmd_config.get("warp_denoising_step", True))
        self.min_step = int(float(self.dmd_config.get("min_step_ratio", 0.02)) * self.num_train_timestep)
        self.max_step = int(float(self.dmd_config.get("max_step_ratio", 0.98)) * self.num_train_timestep)
        time_shift_settings = scheduler_config["time_shift_settings"] if "time_shift_settings" in scheduler_config else {}
        if not isinstance(time_shift_settings, dict):
            raise ValueError("scheduler.time_shift_settings must be a mapping.")
        self.score_timestep_shift = float(self.dmd_config["timestep_shift"] if "timestep_shift" in self.dmd_config else time_shift_settings.get("time_shift_mu", 5.0))
        self.ts_schedule = bool(self.dmd_config.get("ts_schedule", False))
        self.ts_schedule_max = bool(self.dmd_config.get("ts_schedule_max", False))
        self.min_score_timestep = int(self.dmd_config.get("min_score_timestep", 0))

        self.student_checkpoint_path = None
        self.student_checkpoint_strict = True
        if "checkpoint_strict" in self.student_config:
            self.student_checkpoint_strict = bool(self.student_config["checkpoint_strict"])
        if "student" in self.model_config:
            if not isinstance(self.model_config["student"], dict):
                raise ValueError("model.student must be a mapping.")
            model_student_config = self.model_config["student"]
            if "checkpoint_path" in model_student_config:
                self.student_checkpoint_path = model_student_config["checkpoint_path"]
            if "checkpoint_strict" in model_student_config:
                self.student_checkpoint_strict = bool(model_student_config["checkpoint_strict"])
        if self.student_checkpoint_path is None and "checkpoint_path" in self.student_config:
            self.student_checkpoint_path = self.student_config["checkpoint_path"]
        if self.student_checkpoint_path is None and "generator_ckpt" in self.training_config:
            self.student_checkpoint_path = self.training_config["generator_ckpt"]
        if self.student_checkpoint_path is None and "generator_ckpt" in self.dmd_config:
            self.student_checkpoint_path = self.dmd_config["generator_ckpt"]

    def setup(self, resume_ckpt_path=None):
        if resume_ckpt_path is None and self.student_checkpoint_path:
            self._load_student_checkpoint(self.student_checkpoint_path, strict=self.student_checkpoint_strict)
        super().setup(resume_ckpt_path=resume_ckpt_path)
        if resume_ckpt_path is None:
            student_steps = max(1, math.ceil(self.max_train_iters / self.fake_update_ratio))
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer, num_training_steps=student_steps)
            self.fake_lr_scheduler = self._build_lr_scheduler(self.fake_optimizer, num_warmup_steps=0, num_training_steps=max(1, self.max_train_iters))

        time_shift_settings = self.config["scheduler"].get("time_shift_settings", {})
        self.denoising_scheduler = CausalForcingFlowMatchScheduler(
            num_train_timesteps=self.config["scheduler"].get("num_train_timesteps", 1000),
            time_shift_settings=time_shift_settings,
        )
        self.denoising_steps = self._build_denoising_steps(self.model.device)
        self.denoising_sigmas = (self.denoising_steps / self.num_train_timestep).to(dtype=torch.float32)
        logger.info(
            "[train] {} denoising_steps={} warped={}",
            self.trainer_name,
            [round(float(step), 4) for step in self.denoising_steps.detach().cpu()],
            self.warp_denoising_step,
        )

    def _load_student_checkpoint(self, checkpoint_path, strict=True):
        model_state_path = checkpoint_path
        if os.path.isdir(model_state_path):
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

        incompatible = self.model.denoiser_module().load_state_dict(fixed, strict=strict)
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
        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_eval)
            if current_iter == 0:
                self.run_inference(current_iter)

        samples = self._iter_train_samples()
        last_dmd = None
        while current_iter < max_train_iters:
            train_student = current_iter % self.fake_update_ratio == 0

            if train_student:
                loss_dmd_value = self._train_one_stage(samples, stage="student", grad_accum_iters=grad_accum_iters)
                last_dmd = loss_dmd_value
            else:
                loss_dmd_value = last_dmd

            loss_fake_value = self._train_one_stage(samples, stage="fake", grad_accum_iters=grad_accum_iters)

            current_iter += 1
            display_fake = reduce_mean(loss_fake_value)
            display_dmd = reduce_mean(loss_dmd_value) if loss_dmd_value is not None else None
            if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= max_train_iters:
                dmd_text = "nan" if display_dmd is None else f"{display_dmd:.6f}"
                logger.info(
                    "[train] iter={}/{} dmd={} fake={:.6f} lr={:.8f}",
                    current_iter,
                    max_train_iters,
                    dmd_text,
                    display_fake,
                    self.lr_scheduler.get_last_lr()[0],
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

    def _train_one_stage(self, samples, stage, grad_accum_iters):
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
        for micro_idx in range(grad_accum_iters):
            sample = next(samples)
            conditions = self._encode_conditions(sample)
            latent_shape = self._latent_shape(sample)
            set_sync(micro_idx == grad_accum_iters - 1)
            loss = self.forward_loss(latent_shape, conditions, stage=stage)
            (loss / grad_accum_iters).backward()
            loss_value += loss.item() / grad_accum_iters

        self._sync_sequence_parallel_grads(params)
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
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

    def _latent_shape(self, sample):
        prompt = sample["prompt"]
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        if "image_or_video_shape" in self.dmd_config:
            configured_shape = self.dmd_config["image_or_video_shape"]
        elif "image_or_video_shape" in self.training_config:
            configured_shape = self.training_config["image_or_video_shape"]
        else:
            configured_shape = None
        if configured_shape is not None:
            shape = list(configured_shape)
            shape[0] = batch_size
            return tuple(int(dim) for dim in shape)

        infer_config = self.config["inference"] if "inference" in self.config else {}
        height = infer_config["default_height"] if "default_height" in infer_config else infer_config.get("height", 480)
        width = infer_config["default_width"] if "default_width" in infer_config else infer_config.get("width", 832)
        num_frames = infer_config.get("num_frames", 81)
        num_latent_frames = (int(num_frames) - 1) // self.model.vae_scale_factor_temporal + 1
        return (
            batch_size,
            int(self.model._latent_channels()),
            num_latent_frames,
            int(height) // self.model.vae_scale_factor_spatial,
            int(width) // self.model.vae_scale_factor_spatial,
        )

    def forward_loss(self, latent_shape, conditions, stage):
        condition, negative_condition = conditions
        generated, denoised_timestep_from, denoised_timestep_to = self.run_back_simulation(condition, latent_shape, grad_enabled=(stage != "fake"))

        sigma = self._sample_score_sigma(
            latent_shape[0],
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            device=self.model.device,
            dtype=self.running_dtype,
        )
        noise = torch.randn(latent_shape, device=self.model.device, dtype=torch.float32)
        noise = broadcast_sequence_parallel_value(noise)

        with torch.no_grad():
            renoised_xt = self.scheduler.add_noise(generated, noise, sigma)

        if stage == "fake":
            self.fake_model.transformer.train()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            velocity_gt = self.scheduler.build_train_gt(generated.float(), noise)
            return F.mse_loss(velocity_fake.float(), velocity_gt.float(), reduction="mean")

        with torch.no_grad():
            self.fake_model.transformer.eval()
            self.teacher_model.transformer.eval()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            velocity_teacher_cond = self._predict_velocity(self.teacher_model, renoised_xt, sigma, condition)
            if self.guidance_scale == 0:
                velocity_teacher = velocity_teacher_cond
            else:
                velocity_teacher_uncond = self._predict_velocity(self.teacher_model, renoised_xt, sigma, negative_condition)
                velocity_teacher = self._do_cfg(velocity_teacher_cond, velocity_teacher_uncond, self.guidance_scale, self.cfg_norm)

            expanded_sigma = self.scheduler._expand_to_ndim(sigma, renoised_xt.ndim)
            x_pred_fake = renoised_xt - expanded_sigma * velocity_fake
            x_pred_teacher = renoised_xt - expanded_sigma * velocity_teacher

        return self._dmd_loss(generated, x_pred_fake, x_pred_teacher)

    def run_back_simulation(self, condition, latent_shape, grad_enabled, xt=None):
        transformer = self.model.denoiser_module()
        if hasattr(transformer, "_forward_inference"):
            raise RuntimeError("video_dmd requires the bidirectional Wan transformer. Use ar_dmd for causal student models.")

        self.scheduler.set_timesteps(
            self.num_inference_steps,
            sigmas=[float(sigma) for sigma in self.denoising_sigmas.detach().cpu()],
            latent_hw=latent_shape[-2:],
            device=self.model.device,
        )
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)

        end_step_idx = self.sample_end_step()
        x0 = None
        self.model.transformer.train()
        for idx in range(end_step_idx + 1):
            sigma = self.scheduler.sigma_at(idx, latent_shape[0], device=self.model.device, dtype=self.running_dtype)
            context = torch.enable_grad if (grad_enabled and idx == end_step_idx) else torch.no_grad
            with context():
                velocity = self._predict_velocity(self.model, xt, sigma, condition)
            xt, x0 = self.scheduler.step_by_index(velocity, idx, xt)

        return x0.to(dtype=self.running_dtype), *self._denoised_timestep_window(end_step_idx)

    def _sample_score_sigma(self, batch_size, denoised_timestep_from, denoised_timestep_to, device, dtype):
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        min_timestep = max(0, int(min_timestep))
        max_timestep = min(self.num_train_timestep, int(max_timestep))
        if max_timestep <= min_timestep:
            max_timestep = min(self.num_train_timestep, min_timestep + 1)

        timestep = torch.randint(min_timestep, max_timestep, (int(batch_size),), device=device, dtype=torch.long).float()
        if self.score_timestep_shift > 1:
            t = timestep / self.num_train_timestep
            timestep = self.score_timestep_shift * t / (1 + (self.score_timestep_shift - 1) * t) * self.num_train_timestep
        timestep = timestep.clamp(self.min_step, self.max_step)
        return broadcast_sequence_parallel_value((timestep / self.num_train_timestep).to(dtype=dtype))

    def _denoised_timestep_window(self, exit_idx):
        exit_idx = int(exit_idx)
        denoised_timestep_from = self._raw_timestep_from_warped_step(self.denoising_steps[exit_idx])
        if exit_idx == len(self.denoising_steps) - 1:
            denoised_timestep_to = 0
        else:
            denoised_timestep_to = self._raw_timestep_from_warped_step(self.denoising_steps[exit_idx + 1])
        return denoised_timestep_from, denoised_timestep_to

    def _raw_timestep_from_warped_step(self, warped_step):
        if not self.warp_denoising_step:
            return int(round(float(warped_step)))
        timesteps = self.denoising_scheduler.timesteps.to(device=warped_step.device, dtype=torch.float32)
        index = torch.argmin((timesteps - warped_step.float()).abs(), dim=0).item()
        return self.denoising_scheduler.num_train_timesteps - int(index)


@TRAINER_REGISTER("video_ar_dmd")
class VideoArDmdTrainer(VideoDmdTrainer):
    trainer_name = "video_ar_dmd"
    allowed_model_names = {"wan_t2v_ar", "wan_t2v_14b_ar", "wan_ti2v_5b", "wan_ti2v_5b_ar"}

    def __init__(self, config):
        super().__init__(config)
        self.num_frame_per_chunk = int(self.dmd_config["num_frame_per_chunk"] if "num_frame_per_chunk" in self.dmd_config else self.model_config.get("num_frame_per_chunk", 3))
        self.same_step_across_blocks = bool(self.dmd_config.get("same_step_across_blocks", True))
        self.context_noise = float(self.dmd_config.get("context_noise", 0.0))
        self.sequence_parallel_cache = bool(self.dmd_config["sp_cache"] if "sp_cache" in self.dmd_config else False)

    def run_back_simulation(self, condition, latent_shape, grad_enabled, xt=None):
        transformer = self.model.denoiser_module()
        if not hasattr(transformer, "_forward_inference"):
            raise RuntimeError("ar_dmd requires the causal Wan transformer.")

        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        batch_size, _, num_frames, _, _ = xt.shape
        if num_frames % self.num_frame_per_chunk != 0:
            raise ValueError(f"ar_dmd latent frames={num_frames} must be divisible by num_frame_per_chunk={self.num_frame_per_chunk}.")

        use_sp_cache = self._use_sequence_parallel_cache()
        sp_size = get_sequence_parallel_world_size() if use_sp_cache else 1
        sp_rank = get_sequence_parallel_rank() if use_sp_cache else 0
        if use_sp_cache and self.num_frame_per_chunk % sp_size != 0:
            raise ValueError(f"training.dmd.num_frame_per_chunk={self.num_frame_per_chunk} must be divisible by sequence_parallel.size={sp_size} when training.dmd.sp_cache=true.")

        self.model.transformer.train()
        output_chunks = []
        context = self.model._condition_to_context_tensor(condition, batch_size=batch_size)
        frame_seq_length = self._frame_seq_length(xt)
        kv_cache, crossattn_cache = self._new_caches(batch_size, xt.dtype, xt.device, num_frames, frame_seq_length, sequence_parallel_cache=use_sp_cache)
        num_blocks = num_frames // self.num_frame_per_chunk
        exit_indices = self._sample_exit_indices(num_blocks, len(self.denoising_steps), xt.device)

        current_start_frame = 0
        for block_idx in range(num_blocks):
            current_num_frames = self.num_frame_per_chunk
            if use_sp_cache:
                local_num_frames = current_num_frames // sp_size
                local_start_frame = current_start_frame + sp_rank * local_num_frames
            else:
                local_num_frames = current_num_frames
                local_start_frame = current_start_frame
            latents = xt[:, :, local_start_frame : local_start_frame + local_num_frames]
            exit_idx = int(exit_indices[0] if self.same_step_across_blocks else exit_indices[block_idx])

            x0 = None
            for step_idx, current_timestep in enumerate(self.denoising_steps):
                timestep = torch.full(
                    (batch_size, local_num_frames),
                    float(current_timestep),
                    device=xt.device,
                    dtype=torch.float32,
                )
                enable_step_grad = grad_enabled and step_idx == exit_idx
                context_mgr = torch.enable_grad if enable_step_grad else torch.no_grad
                with context_mgr():
                    flow_pred = self._forward_causal_chunk(
                        self.model,
                        latents,
                        timestep,
                        context,
                        kv_cache,
                        crossattn_cache,
                        current_start=current_start_frame * frame_seq_length,
                        cache_start=current_start_frame * frame_seq_length,
                        local_frame_offset=local_start_frame,
                        balanced_sequence_parallel=use_sp_cache,
                    )
                    x0 = self._flow_to_x0(latents, flow_pred, timestep)

                if step_idx == exit_idx:
                    break

                next_timestep = torch.full(
                    (batch_size, local_num_frames),
                    float(self.denoising_steps[step_idx + 1]),
                    device=xt.device,
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    latents = self._add_noise_by_timestep(x0, torch.randn_like(x0), next_timestep)

            output_chunks.append(all_gather_sequence(x0, dim=2) if use_sp_cache else x0)

            cache_latents = x0.detach()
            cache_timestep = torch.full(
                (batch_size, local_num_frames),
                self.context_noise,
                device=xt.device,
                dtype=torch.float32,
            )
            if self.context_noise > 0:
                cache_latents = self._add_noise_by_timestep(cache_latents, torch.randn_like(cache_latents), cache_timestep)
            with torch.no_grad():
                self._forward_causal_chunk(
                    self.model,
                    cache_latents,
                    cache_timestep,
                    context,
                    kv_cache,
                    crossattn_cache,
                    current_start=current_start_frame * frame_seq_length,
                    cache_start=current_start_frame * frame_seq_length,
                    local_frame_offset=local_start_frame,
                    balanced_sequence_parallel=use_sp_cache,
                )

            current_start_frame += current_num_frames

        return (
            torch.cat(output_chunks, dim=2).to(dtype=self.running_dtype),
            *self._denoised_timestep_window(exit_indices),
        )

    def _use_sequence_parallel_cache(self):
        enabled = bool(self.sequence_parallel_cache and is_sequence_parallel_enabled())
        if enabled and bool(getattr(self.model.denoiser_module(), "defer_kv_cache_updates", False)):
            raise ValueError("training.dmd.sp_cache=true does not support model.defer_kv_cache_updates=true. Set model.defer_kv_cache_updates=false or disable training.dmd.sp_cache.")
        return enabled

    def _denoised_timestep_window(self, exit_indices):
        if not self.same_step_across_blocks:
            return None, None

        exit_idx = int(exit_indices[0])
        denoised_timestep_from = self._raw_timestep_from_warped_step(self.denoising_steps[exit_idx])
        if exit_idx == len(self.denoising_steps) - 1:
            denoised_timestep_to = 0
        else:
            denoised_timestep_to = self._raw_timestep_from_warped_step(self.denoising_steps[exit_idx + 1])
        return denoised_timestep_from, denoised_timestep_to

    def _sample_exit_indices(self, num_blocks, num_steps, device):
        count = 1 if self.same_step_across_blocks else num_blocks
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                indices = torch.randint(0, num_steps, (count,), device=device)
            else:
                indices = torch.empty(count, dtype=torch.long, device=device)
            dist.broadcast(indices, src=0)
            return indices.tolist()
        return torch.randint(0, num_steps, (count,), device=device).tolist()

    def _forward_causal_chunk(
        self,
        model,
        latents,
        timestep,
        context,
        kv_cache,
        crossattn_cache,
        current_start,
        cache_start,
        local_frame_offset=0,
        balanced_sequence_parallel=False,
    ):
        transformer = model.denoiser_module()
        seq_len = model._sequence_length(latents)
        forward_context = model.transformer_forward_context() if hasattr(model, "transformer_forward_context") else torch.no_grad()
        with forward_context:
            return transformer(
                latents,
                t=timestep,
                context=context,
                seq_len=seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                local_frame_offset=local_frame_offset,
                balanced_sequence_parallel=balanced_sequence_parallel,
            )

    def _new_caches(self, batch_size, dtype, device, num_frames, frame_seq_length, sequence_parallel_cache=False):
        transformer = self.model.denoiser_module()
        num_layers = int(getattr(transformer, "num_layers", len(transformer.blocks)))
        num_heads = int(transformer.num_heads)
        head_dim = int(transformer.dim // transformer.num_heads)
        kv_num_heads = num_heads
        if sequence_parallel_cache:
            sp_size = get_sequence_parallel_world_size()
            if num_heads % sp_size != 0:
                raise ValueError(f"transformer.num_heads={num_heads} must be divisible by sequence_parallel.size={sp_size}.")
            kv_num_heads = num_heads // sp_size
        local_attn_size = int(getattr(transformer, "local_attn_size", -1))
        kv_cache_size = num_frames * frame_seq_length if local_attn_size == -1 else local_attn_size * frame_seq_length

        kv_cache = []
        crossattn_cache = []
        for _ in range(num_layers):
            kv_cache.append(
                {
                    "k": torch.zeros((batch_size, kv_cache_size, kv_num_heads, head_dim), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size, kv_num_heads, head_dim), dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
            crossattn_cache.append(
                {
                    "k": torch.zeros((batch_size, self.model.max_sequence_length, num_heads, head_dim), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, self.model.max_sequence_length, num_heads, head_dim), dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        return kv_cache, crossattn_cache

    def _frame_seq_length(self, latent):
        _, _, _, latent_height, latent_width = latent.shape
        patch_t, patch_h, patch_w = self.model.patch_size
        if patch_t != 1:
            raise ValueError(f"ar_dmd expects temporal patch size 1, got {patch_t}.")
        return latent_height * latent_width // (patch_h * patch_w)

    def _sigma_from_timestep(self, timestep, dtype):
        timesteps = self.denoising_scheduler.timesteps.to(device=timestep.device, dtype=torch.float32)
        sigmas = self.denoising_scheduler.sigmas.to(device=timestep.device, dtype=dtype)
        flat_timestep = timestep.flatten().float()
        index = torch.argmin((timesteps.unsqueeze(0) - flat_timestep.unsqueeze(1)).abs(), dim=1)
        return sigmas[index].reshape(timestep.shape)

    def _expand_frame_sigma(self, sigma, ndim):
        return sigma.reshape(sigma.shape[0], 1, sigma.shape[1], *([1] * (ndim - 3)))

    def _flow_to_x0(self, xt, flow_pred, timestep):
        sigma = self._sigma_from_timestep(timestep, xt.dtype)
        sigma = self._expand_frame_sigma(sigma, xt.ndim)
        return (xt - sigma * flow_pred).to(dtype=xt.dtype)

    def _add_noise_by_timestep(self, x0, noise, timestep):
        sigma = self._sigma_from_timestep(timestep, x0.dtype)
        sigma = self._expand_frame_sigma(sigma, x0.ndim)
        return ((1.0 - sigma) * x0 + sigma * noise).to(dtype=x0.dtype)
