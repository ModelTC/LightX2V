import os
import shutil

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime.checkpoint import prune_checkpoints
from lightx2v_train.schedulers import DMDFlowMatchingScheduler
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .lora import LoraTrainer


@TRAINER_REGISTER("dmd_lora")
class DmdLoraTrainer(LoraTrainer):
    def __init__(self, config):
        super().__init__(config)
        fake_config = self.training_config.get("fake", {})
        fake_optimizer_config = fake_config.get("optimizer", {})
        self.fake_optimizer_learning_rate = fake_optimizer_config.get("learning_rate", self.optimizer_learning_rate)
        self.fake_optimizer_adam_beta1 = fake_optimizer_config.get("adam_beta1", self.optimizer_adam_beta1)
        self.fake_optimizer_adam_beta2 = fake_optimizer_config.get("adam_beta2", self.optimizer_adam_beta2)
        self.fake_optimizer_weight_decay = fake_optimizer_config.get("weight_decay", self.optimizer_weight_decay)
        self.fake_optimizer_adam_epsilon = fake_optimizer_config.get("adam_epsilon", self.optimizer_adam_epsilon)

        self.dmd_config = self.training_config.get("dmd", {})
        self.num_inference_steps = int(self.dmd_config.get("num_inference_steps", 4))
        self.fake_update_ratio = int(self.dmd_config.get("fake_update_ratio", 1))
        self.guidance_scale = float(self.dmd_config.get("guidance_scale", 3.0))
        self.negative_prompt = self.dmd_config.get("negative_prompt", " ")
        self.cfg_norm = self.dmd_config.get("cfg_norm", "layer_norm")
        self.image_sizes = self.dmd_config.get("image_sizes", [])

        random_schedule_config = self.dmd_config.get("random_schedule", {})
        self.random_schedule_enabled = bool(
            random_schedule_config.get("enabled", False)
        )
        self.random_schedule_num_steps_min = int(
            random_schedule_config.get("num_steps_min", 1)
        )
        self.random_schedule_num_steps_max = int(
            random_schedule_config.get("num_steps_max", self.num_inference_steps)
        )
        self.random_schedule_sigma_min = float(
            random_schedule_config.get("sigma_min", 0.02)
        )
        self.random_schedule_sigma_max = float(
            random_schedule_config.get("sigma_max", 0.98)
        )
        self.random_schedule_sampling_method = random_schedule_config.get(
            "sampling_method", "stratified"
        )

        self.cdm_config = self.dmd_config.get("cdm", {})
        self.cdm_enabled = bool(self.cdm_config.get("enabled", False))
        self.cdm_weight = float(self.cdm_config.get("weight", 1.0))
        self.cdm_warmup_iters = int(self.cdm_config.get("warmup_iters", 0))
        self.cdm_norm_clip_min = float(self.cdm_config.get("norm_clip_min", 0.1))

    def setup(self, resume_ckpt_path=None):
        super().setup(resume_ckpt_path=resume_ckpt_path)
        self.fake_model = build_model(self.config)
        self.fake_model.load_components(transformer_only=True, reference_model=self.model)
        self.fake_model.add_lora(self.lora_rank, self.lora_alpha, self.lora_target_modules)
        self.fake_model.set_lora_trainable()
        if self.gradient_checkpointing:
            self.fake_model.enable_gradient_checkpointing()

        self.teacher_model = build_model(self.config)
        self.teacher_model.load_components(transformer_only=True, reference_model=self.model)
        self.teacher_model.transformer.requires_grad_(False)
        self.teacher_model.transformer.eval()

        self.fake_optimizer = torch.optim.AdamW(
            self.fake_model.trainable_parameters(),
            lr=self.fake_optimizer_learning_rate,
            betas=(self.fake_optimizer_adam_beta1, self.fake_optimizer_adam_beta2),
            weight_decay=self.fake_optimizer_weight_decay,
            eps=self.fake_optimizer_adam_epsilon,
        )
        self.fake_lr_scheduler = get_scheduler(
            self.lr_scheduler_name,
            optimizer=self.fake_optimizer,
            num_warmup_steps=0,
            num_training_steps=max(1, self.max_train_iters * self.fake_update_ratio),
        )

        self.scheduler = DMDFlowMatchingScheduler(self.config, self.dmd_config)

        if resume_ckpt_path is not None:
            self.load_resume_ckpt(resume_ckpt_path)

        print(f"[dmd_lora] student trainable params={self._count_trainable(self.model.transformer)}")
        print(f"[dmd_lora] fake trainable params={self._count_trainable(self.fake_model.transformer)}")
        if self.random_schedule_enabled:
            print(
                "[dmd_lora] random sigma schedule enabled: "
                f"steps=[{self.random_schedule_num_steps_min}, {self.random_schedule_num_steps_max}], "
                f"sigma=[{self.random_schedule_sigma_min}, {self.random_schedule_sigma_max}], "
                f"sampling_method={self.random_schedule_sampling_method}"
            )
        if self.cdm_enabled:
            print(
                "[dmd_lora] CDM enabled: "
                f"weight={self.cdm_weight}, warmup_iters={self.cdm_warmup_iters}"
            )

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

    def _prepare_sampling_schedule(self):
        if self.random_schedule_enabled:
            self.scheduler.set_random_timesteps(
                self.random_schedule_num_steps_min,
                self.random_schedule_num_steps_max,
                sigma_min=self.random_schedule_sigma_min,
                sigma_max=self.random_schedule_sigma_max,
                sampling_method=self.random_schedule_sampling_method,
                device=self.model.device,
            )
        else:
            self.scheduler.set_timesteps(self.num_inference_steps, device=self.model.device)

    def _effective_cdm_weight(self, current_iter=None):
        if self.cdm_warmup_iters <= 0 or current_iter is None:
            return self.cdm_weight
        progress = min(1.0, max(0.0, float(current_iter) / float(self.cdm_warmup_iters)))
        return progress * self.cdm_weight

    def _latent_shape(self, sample):
        image = sample["target_image"]
        batch_size = image.shape[0]
        if self.image_sizes:
            height, width = self.image_sizes[torch.randint(0, len(self.image_sizes), (1,), device=self.model.device).item()]
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
        if isinstance(prompt, str):
            negative_prompt = self.negative_prompt
        else:
            negative_prompt = [self.negative_prompt] * len(prompt)
        with torch.no_grad():
            condition = self.model.encode_prompt_condition(prompt)
            negative_condition = self.model.encode_prompt_condition(negative_prompt)
        return condition, negative_condition

    def _predict_velocity(self, model, latents, sigma, condition):
        denoiser_input = model.prepare_denoiser_input(latents)
        prediction = model.denoise(denoiser_input, sigma, condition)
        prediction = model.postprocess_denoiser_output(prediction, denoiser_input)
        return prediction

    def sample_initial_latents(self, latent_shape):
        return torch.randn(latent_shape, device=self.model.device, dtype=self.running_dtype)

    def sample_end_step(self):
        return int(torch.randint(0, self.scheduler.num_inference_steps, (1,), device=self.model.device).item())

    def run_back_simulation(self, condition, latent_shape, end_step_idx, xt=None):
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        x0 = None
        xt_end = None
        vt_end = None
        self.model.transformer.train()
        for idx in range(end_step_idx + 1):
            sigma = self.scheduler.sigma_at(idx, latent_shape[0], device=self.model.device, dtype=self.running_dtype)
            with torch.no_grad():
                velocity = self._predict_velocity(self.model, xt, sigma, condition)
            if idx == end_step_idx:
                xt_end = xt.detach()
                vt_end = velocity.detach()
            xt, x0 = self.scheduler.step_by_index(velocity, idx, xt)
        return x0.detach(), xt_end, vt_end

    def _compute_cdm_loss(self, xt, vt, end_step_idx, condition):
        batch_size = xt.shape[0]

        traj_sigma = self.scheduler.sigma_at(end_step_idx, batch_size, device=self.model.device, dtype=torch.float32)
        student_sigma = self.scheduler.sample_renoise_sigma(batch_size, device=self.model.device, dtype=torch.float32)
        student_xt = xt + (student_sigma - traj_sigma) * vt

        student_prediction = self._predict_velocity(self.model, student_xt.to(self.running_dtype), student_sigma, condition)
        student_x0 = student_xt - student_sigma * student_prediction
        student_x0 = student_x0.to(self.running_dtype)

        teacher_sigma = self.scheduler.sample_renoise_sigma(batch_size, device=self.model.device, dtype=self.running_dtype)
        teacher_noise = torch.randn_like(student_x0)
        teacher_xt = self.scheduler.add_noise(student_x0, teacher_noise, teacher_sigma)

        with torch.no_grad():
            self.fake_model.transformer.eval()
            velocity_fake = self._predict_velocity(self.fake_model, teacher_xt, teacher_sigma, condition)
            velocity_teacher = self._predict_velocity(self.teacher_model, teacher_xt, teacher_sigma, condition)

        x_pred_fake = teacher_xt - teacher_sigma * velocity_fake
        x_pred_teacher = teacher_xt - teacher_sigma * velocity_teacher
        loss = self._dmd_loss(student_x0, x_pred_fake, x_pred_teacher, norm_clip_min=self.cdm_norm_clip_min)
        return loss

    def forward_loss(self, latent_shape, conditions, stage, current_iter=None):
        condition, negative_condition = conditions
        self._prepare_sampling_schedule()
        end_step_idx = self.sample_end_step()
        xt_start = self.sample_initial_latents(latent_shape)
        x0_ref, xt_end, vt_end = self.run_back_simulation(condition, latent_shape, end_step_idx, xt=xt_start)

        sigma = self.scheduler.sample_renoise_sigma(latent_shape[0], device=self.model.device, dtype=self.running_dtype)
        noise = torch.randn(latent_shape, device=self.model.device, dtype=torch.float32)
        renoised_xt = self.scheduler.add_noise(x0_ref, noise, sigma)

        if stage == "fake":
            self.fake_model.transformer.train()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            velocity_gt = self.scheduler.build_train_gt(x0_ref.float(), noise)
            loss_fake = F.mse_loss(velocity_fake.float(), velocity_gt.float(), reduction="mean")
            return {"fake": loss_fake}

        with torch.no_grad():
            self.fake_model.transformer.eval()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            velocity_teacher_cond = self._predict_velocity(self.teacher_model, renoised_xt, sigma, condition)
            velocity_teacher_uncond = self._predict_velocity(self.teacher_model, renoised_xt, sigma, negative_condition)
            velocity_teacher = self._do_cfg(velocity_teacher_cond, velocity_teacher_uncond, self.guidance_scale, self.cfg_norm)

        x_pred_fake = renoised_xt - sigma * velocity_fake
        x_pred_teacher = renoised_xt - sigma * velocity_teacher
        sigma_end = self.scheduler.sigma_at(end_step_idx, latent_shape[0], device=self.model.device, dtype=self.running_dtype)
        xt_velocity = self._predict_velocity(self.model, xt_end, sigma_end, condition)
        x0 = xt_end - sigma_end * xt_velocity

        loss_dmd = self._dmd_loss(x0, x_pred_fake, x_pred_teacher)
        total_loss = loss_dmd

        cdm_weight = self._effective_cdm_weight(current_iter)
        if self.cdm_enabled and cdm_weight != 0:
            loss_cdm = self._compute_cdm_loss(xt_end, vt_end, end_step_idx, condition)
            total_loss = total_loss + cdm_weight * loss_cdm
        else:
            loss_cdm = torch.zeros_like(total_loss)
        return {"student": total_loss, "dmd": loss_dmd, "cdm": loss_cdm, "cdm_weight": cdm_weight}

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        os.makedirs(self.output_train_dir, exist_ok=True)

        max_train_iters = self.max_train_iters
        fake_update_ratio = self.fake_update_ratio
        max_grad_norm = self.max_grad_norm
        save_every_iters = self.save_every_iters
        save_total_limit = self.save_total_limit
        running_dmd = 0.0
        running_cdm = 0.0
        running_fake = 0.0

        progress = tqdm(total=max_train_iters, desc="DMD-LoRA iterations", initial=current_iter)
        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_eval)
            if current_iter == 0:
                self.run_inference(current_iter)

        while current_iter < max_train_iters:
            for sample in self.dataloader_train:
                conditions = self._encode_conditions(sample)
                latent_shape = self._latent_shape(sample)

                res_student = self.forward_loss(latent_shape, conditions, stage="student", current_iter=current_iter)
                loss_student = res_student["student"]
                loss_student.backward()
                torch.nn.utils.clip_grad_norm_(self.model.transformer.parameters(), max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                running_dmd += res_student["dmd"].item()
                running_cdm += res_student["cdm"].item()
                running_cdm_weight = res_student["cdm_weight"]

                fake_loss = 0.0
                for _ in range(fake_update_ratio):
                    res_fake = self.forward_loss(latent_shape, conditions, stage="fake")
                    loss_fake = res_fake["fake"]
                    loss_fake.backward()
                    torch.nn.utils.clip_grad_norm_(self.fake_model.transformer.parameters(), max_grad_norm)
                    self.fake_optimizer.step()
                    self.fake_lr_scheduler.step()
                    self.fake_optimizer.zero_grad(set_to_none=True)
                    fake_loss += loss_fake.item()
                running_fake += fake_loss / fake_update_ratio

                current_iter += 1
                progress.update(1)
                progress.set_postfix(
                    dmd=running_dmd,
                    cdm=running_cdm,
                    cdm_w=running_cdm_weight,
                    fake=running_fake,
                    lr=self.lr_scheduler.get_last_lr()[0],
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

        progress.close()

    def load_resume_ckpt(self, resume_ckpt_path):
        training_state_path = os.path.join(resume_ckpt_path, "training_state.pt")
        fake_lora_path = os.path.join(resume_ckpt_path, "fake_lora")
        fake_lora_weights_path = os.path.join(fake_lora_path, "pytorch_lora_weights.safetensors")

        if os.path.exists(fake_lora_weights_path):
            self.fake_model.load_lora_weights_for_resume(fake_lora_path)
        else:
            print(f"Warning: fake LoRA weights not found in {fake_lora_path}. Fake model not restored.")

        if not os.path.exists(training_state_path):
            return

        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        if "fake_optimizer" in state:
            self.fake_optimizer.load_state_dict(state["fake_optimizer"])
        else:
            print(f"Warning: fake optimizer state not found in {training_state_path}.")

        if "fake_lr_scheduler" in state:
            self.fake_lr_scheduler.load_state_dict(state["fake_lr_scheduler"])
        else:
            print(f"Warning: fake lr scheduler state not found in {training_state_path}.")

    def save_checkpoint(self, iteration, save_total_limit):
        prune_checkpoints(self.output_train_dir, save_total_limit)

        save_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_lora_weights(save_dir)

        fake_save_dir = os.path.join(save_dir, "fake_lora")
        os.makedirs(fake_save_dir, exist_ok=True)
        self.fake_model.save_lora_weights(fake_save_dir)

        config_path = self.config.get("config_path")
        if config_path is not None:
            shutil.copy2(config_path, os.path.join(save_dir, "config.yaml"))

        training_state = {
            "iteration": iteration,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "fake_optimizer": self.fake_optimizer.state_dict(),
            "fake_lr_scheduler": self.fake_lr_scheduler.state_dict(),
        }
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
