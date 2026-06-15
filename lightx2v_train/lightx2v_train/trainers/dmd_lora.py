import os
import shutil

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime.checkpoint import prune_checkpoints
from lightx2v_train.runtime.distributed import barrier, get_world_size, is_main_process, reduce_mean
from lightx2v_train.runtime.fsdp import apply_fsdp2
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
        teacher_config = self.training_config.get("teacher", {})
        self.num_inference_steps = int(self.dmd_config.get("num_inference_steps", 4))
        self.fake_update_ratio = max(1, int(self.dmd_config.get("fake_update_ratio", 1)))
        self.guidance_scale = float(teacher_config.get("guidance_scale", self.dmd_config.get("guidance_scale", 3.0)))
        self.negative_prompt = teacher_config.get("negative_prompt", self.dmd_config.get("negative_prompt", " "))
        self.cfg_norm = teacher_config.get("cfg_norm", self.dmd_config.get("cfg_norm", "layer_norm"))
        self.image_sizes = self.dmd_config.get("image_sizes", [])

    def _get_optimizer_config(self):
        student_config = self.training_config.get("student", {})
        return student_config.get("optimizer", self.training_config.get("optimizer", {}))

    def setup(self, resume_ckpt_path=None):
        super().setup(resume_ckpt_path=None)
        self.fake_model = build_model(self.config)
        self.fake_model.load_components(transformer_only=True, reference_model=self.model)
        self.fake_model.add_lora(self.lora_rank, self.lora_alpha, self.lora_target_modules)
        self.fake_model.set_lora_trainable()
        apply_fsdp2(self.fake_model, self.config)
        if self.gradient_checkpointing:
            self.fake_model.enable_gradient_checkpointing()

        self.teacher_model = build_model(self.config)
        self.teacher_model.load_components(transformer_only=True, reference_model=self.model)
        self.teacher_model.transformer.requires_grad_(False)
        self.teacher_model.transformer.eval()
        apply_fsdp2(self.teacher_model, self.config)
        self.teacher_model.transformer.eval()

        self.fake_trainable_params = list(self.fake_model.trainable_parameters())

        self.fake_optimizer = torch.optim.AdamW(
            self.fake_trainable_params,
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
            self._load_resume_state(resume_ckpt_path)

        logger.info("[train] dmd_lora student trainable params={}", self._count_trainable(self.model.transformer))
        logger.info("[train] dmd_lora fake trainable params={}", self._count_trainable(self.fake_model.transformer))

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
    def _dmd_loss(latents, x_pred_fake_flow, x_pred_teacher):
        with torch.no_grad():
            grad = x_pred_fake_flow - x_pred_teacher
            dims = tuple(range(1, latents.ndim))
            normalizer = torch.abs(latents - x_pred_teacher).mean(dim=dims, keepdim=True)
            grad = torch.nan_to_num(grad / normalizer)
        return 0.5 * F.mse_loss(latents.float(), (latents.float() - grad.float()).detach(), reduction="mean")

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
        return int(torch.randint(0, self.num_inference_steps, (1,), device=self.model.device).item())

    def run_back_simulation(self, condition, latent_shape, end_step_idx, grad_enabled, xt=None):
        self.scheduler.set_timesteps(self.num_inference_steps, latent_hw=latent_shape[-2:])
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        x0 = None
        self.model.transformer.train()
        for idx in range(end_step_idx + 1):
            sigma = self.scheduler.sigma_at(idx, latent_shape[0], device=self.model.device, dtype=self.running_dtype)
            context = torch.enable_grad if (grad_enabled and idx == end_step_idx) else torch.no_grad
            with context():
                velocity = self._predict_velocity(self.model, xt, sigma, condition)
            xt, x0 = self.scheduler.step_by_index(velocity, idx, xt)
        return x0

    def forward_loss(self, latent_shape, conditions, stage):
        condition, negative_condition = conditions
        end_step_idx = self.sample_end_step()
        xt_start = self.sample_initial_latents(latent_shape)
        x0_ref = self.run_back_simulation(condition, latent_shape, end_step_idx, grad_enabled=False, xt=xt_start)

        sigma = self.scheduler.sample_renoise_sigma(latent_shape[0], device=self.model.device, dtype=self.running_dtype)
        noise = torch.randn(latent_shape, device=self.model.device, dtype=torch.float32)
        renoised_xt = self.scheduler.add_noise(x0_ref, noise, sigma)

        if stage == "fake":
            self.fake_model.transformer.train()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            velocity_gt = self.scheduler.build_train_gt(x0_ref.float(), noise)
            return F.mse_loss(velocity_fake.float(), velocity_gt.float(), reduction="mean")

        with torch.no_grad():
            self.fake_model.transformer.eval()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            velocity_teacher_cond = self._predict_velocity(self.teacher_model, renoised_xt, sigma, condition)
            velocity_teacher_uncond = self._predict_velocity(self.teacher_model, renoised_xt, sigma, negative_condition)
            velocity_teacher = self._do_cfg(velocity_teacher_cond, velocity_teacher_uncond, self.guidance_scale, self.cfg_norm)

        x_pred_fake = renoised_xt - sigma * velocity_fake
        x_pred_teacher = renoised_xt - sigma * velocity_teacher
        x0 = self.run_back_simulation(condition, latent_shape, end_step_idx, grad_enabled=True, xt=xt_start)
        return self._dmd_loss(x0, x_pred_fake, x_pred_teacher)

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
        running_fake = 0.0
        microbatches = []

        logger.info(
            "[train] start iter={}/{} world_size={} grad_accum={} train_log_every_iters={} fake_update_ratio={}",
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
                loss_dmd = self.forward_loss(latent_shape, conditions, stage="student")
                (loss_dmd / grad_accum_iters).backward()
                running_dmd += loss_dmd.item() / grad_accum_iters
                microbatches.append((latent_shape, conditions))

                grad_accum_counter += 1
                if grad_accum_counter % grad_accum_iters != 0:
                    continue

                torch.nn.utils.clip_grad_norm_(self.trainable_params, max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                fake_loss = 0.0
                for _ in range(fake_update_ratio):
                    for microbatch_idx, (micro_latent_shape, micro_conditions) in enumerate(microbatches):
                        sync_fake_grad = microbatch_idx == len(microbatches) - 1
                        self._set_fake_gradient_sync(sync_fake_grad)
                        loss_fake = self.forward_loss(micro_latent_shape, micro_conditions, stage="fake")
                        (loss_fake / len(microbatches)).backward()
                        fake_loss += loss_fake.item() / len(microbatches)
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
                    logger.info(
                        "[train] iter={}/{} dmd={:.6f} fake={:.6f} lr={:.8f}",
                        current_iter,
                        max_train_iters,
                        display_dmd,
                        display_fake,
                        self.lr_scheduler.get_last_lr()[0],
                    )
                running_dmd = 0.0
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
        self.model.set_fsdp2_gradient_sync(enabled)

    def _set_fake_gradient_sync(self, enabled):
        self.fake_model.set_fsdp2_gradient_sync(enabled)

    def _set_gradient_sync(self, enabled):
        self._set_student_gradient_sync(enabled)
        self._set_fake_gradient_sync(enabled)

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

    def _load_single_process_state(self, resume_ckpt_path):
        training_state_path = os.path.join(resume_ckpt_path, "training_state.pt")
        fake_lora_path = os.path.join(resume_ckpt_path, "fake_lora")
        fake_lora_weights_path = os.path.join(fake_lora_path, "pytorch_lora_weights.safetensors")

        if not os.path.exists(training_state_path):
            raise RuntimeError(f"training_state.pt not found in {resume_ckpt_path}")

        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        self._validate_dmd_checkpoint_metadata(state, training_state_path, resume_ckpt_path)
        self.model.load_lora_weights_for_resume(resume_ckpt_path)
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])

        if os.path.exists(fake_lora_weights_path):
            self.fake_model.load_lora_weights_for_resume(fake_lora_path)
        else:
            print(f"Warning: fake LoRA weights not found in {fake_lora_path}. Fake model not restored.")

        if "fake_optimizer" in state:
            self.fake_optimizer.load_state_dict(state["fake_optimizer"])
        else:
            print(f"Warning: fake optimizer state not found in {training_state_path}.")

        if "fake_lr_scheduler" in state:
            self.fake_lr_scheduler.load_state_dict(state["fake_lr_scheduler"])
        else:
            print(f"Warning: fake lr scheduler state not found in {training_state_path}.")
        logger.info("Restored DMD-LoRA training state from {}", training_state_path)

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
        logger.info("Restored distributed DMD-LoRA training state from {}", resume_ckpt_path)

    def save_checkpoint(self, iteration, save_total_limit):
        if is_main_process():
            prune_checkpoints(self.output_train_dir, save_total_limit)

        save_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        logger.info("[train] saving checkpoint iter={} path={}", iteration, save_dir)
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        barrier()

        self.model.save_lora_weights(save_dir)
        barrier()

        fake_save_dir = os.path.join(save_dir, "fake_lora")
        if is_main_process():
            os.makedirs(fake_save_dir, exist_ok=True)
        barrier()
        self.fake_model.save_lora_weights(fake_save_dir)
        barrier()

        config_path = self.config.get("config_path")
        if is_main_process() and config_path is not None:
            shutil.copy2(config_path, os.path.join(save_dir, "config.yaml"))

        if self.model.is_fsdp2_wrapped() or self.fake_model.is_fsdp2_wrapped():
            self._save_distributed_state(save_dir, iteration)
            barrier()
            logger.info("[train] saved checkpoint iter={} path={}", iteration, save_dir)
            return

        training_state = {
            "iteration": iteration,
            "world_size": get_world_size(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "fake_optimizer": self.fake_optimizer.state_dict(),
            "fake_lr_scheduler": self.fake_lr_scheduler.state_dict(),
        }
        if is_main_process():
            torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
        barrier()
        logger.info("[train] saved checkpoint iter={} path={}", iteration, save_dir)

    def _save_distributed_state(self, save_dir, iteration):
        dist_state_path = os.path.join(save_dir, "dist_state")
        if is_main_process():
            os.makedirs(dist_state_path, exist_ok=True)
            torch.save(
                {
                    "iteration": iteration,
                    "world_size": get_world_size(),
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
