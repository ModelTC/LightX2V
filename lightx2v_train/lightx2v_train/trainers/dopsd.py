###
### Source idea: D-OPSD Trainer (https://github.com/vvvvvjdy/D-OPSD)
###
import os
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from lightx2v_train.infer import build_inferencer
from lightx2v_train.infer.dopsd_trajectory_viz import save_student_teacher_trajectory_grid
from lightx2v_train.runtime.checkpoint import find_latest_checkpoint, parse_checkpoint_iteration, prune_checkpoints
from lightx2v_train.runtime.distributed import barrier, get_rank, get_world_size, is_distributed, is_main_process, reduce_mean
from lightx2v_train.runtime.fsdp import apply_fsdp2
from lightx2v_train.utils.registry import TRAINER_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .base import BaseTrainer


@TRAINER_REGISTER("dopsd")
class DopsdTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.running_dtype = get_running_dtype(self.model_config["running_dtype"])

        lora_config = self.training_config.get("lora", {})
        self.lora_rank = lora_config.get("rank", 16)
        self.lora_alpha = lora_config.get("alpha", self.lora_rank)
        self.lora_target_modules = lora_config.get("target_modules")

        dopsd_config = self.training_config.get("dopsd", {})
        self.num_training_steps = dopsd_config.get("num_training_steps", 4)
        self.ema_decay = dopsd_config.get("ema_decay", 0.999)
        self.ema_decay_warmup = dopsd_config.get("ema_decay_warmup")
        self.ema_decay_warmup_iters = int(dopsd_config.get("ema_decay_warmup_iters", 0))
        self.step_loss_weights = dopsd_config.get("step_loss_weights")
        self.student_adapter = dopsd_config.get("student_adapter", "student")
        self.teacher_adapter = dopsd_config.get("teacher_adapter", "teacher")
        self.edit_sys_prompt = dopsd_config.get(
            "edit_sys_prompt",
            "The output must be exactly the same as the reference image.",
        )
        self.teacher_use_dataset_prompt = dopsd_config.get("teacher_use_dataset_prompt", False)
        self._train_reference_map = None
        self.trajectory_every_iters = dopsd_config.get("trajectory_every_iters", None)
        if self.trajectory_every_iters is not None:
            self.trajectory_every_iters = int(self.trajectory_every_iters)

        self.gradient_checkpointing = self.training_config.get("gradient_checkpointing", True)

        optimizer_config = self.training_config.get("optimizer", {})
        self.optimizer_learning_rate = optimizer_config.get("learning_rate", 1e-4)
        self.optimizer_adam_beta1 = optimizer_config.get("adam_beta1", 0.9)
        self.optimizer_adam_beta2 = optimizer_config.get("adam_beta2", 0.999)
        self.optimizer_weight_decay = optimizer_config.get("weight_decay", 0.01)
        self.optimizer_adam_epsilon = optimizer_config.get("adam_epsilon", 1e-8)

        self.lr_scheduler_name = self.training_config.get("lr_scheduler", "constant")
        self.lr_warmup_iters = self.training_config["lr_warmup_iters"]
        self.max_train_iters = self.training_config["max_train_iters"]

        self.output_train_dir = self.training_config["output_dir"]
        self.gradient_accumulation_iters = self.training_config["gradient_accumulation_iters"]
        self.max_grad_norm = self.training_config.get("max_grad_norm", 1.0)
        self.save_every_iters = self.training_config["save_every_iters"]
        self.save_total_limit = self.training_config["save_total_limit"]

        self.infer_every_iters = self.infer_config.get("infer_every_iters", None)
        logging_config = self.config.get("logging", {})
        self.train_log_every_iters = max(1, int(logging_config.get("train_log_every_iters", 10)))

        resume_config = self.config.get("resume", {})
        self.auto_resume = resume_config.get("auto_resume", False)

    def _resolve_ema_decay(self, current_iter):
        if self.ema_decay_warmup is not None and self.ema_decay_warmup_iters > 0 and current_iter <= self.ema_decay_warmup_iters:
            return float(self.ema_decay_warmup)
        return float(self.ema_decay)

    def _step_loss_weights_for(self, num_steps):
        if not self.step_loss_weights:
            return [1.0] * num_steps
        weights = [float(w) for w in self.step_loss_weights]
        if len(weights) < num_steps:
            weights.extend([weights[-1]] * (num_steps - len(weights)))
        return weights[:num_steps]

    def setup(self, resume_ckpt_path=None):
        self.model.add_dual_lora(
            self.lora_rank,
            self.lora_alpha,
            self.lora_target_modules,
            student_adapter=self.student_adapter,
            teacher_adapter=self.teacher_adapter,
            init_teacher_from_student=resume_ckpt_path is None,
        )
        self.model.set_dual_lora_trainable(self.student_adapter, self.teacher_adapter)

        apply_fsdp2(self.model, self.config)

        if self.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        if self.infer_every_iters:
            self.inferencer = build_inferencer(self.config)
            self.inferencer.set_model(self.model)

        self.model.log_model_structure()

        self.trainable_params = list(self.model.trainable_parameters())
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=self.optimizer_learning_rate,
            betas=(self.optimizer_adam_beta1, self.optimizer_adam_beta2),
            weight_decay=self.optimizer_weight_decay,
            eps=self.optimizer_adam_epsilon,
        )
        self.lr_scheduler = get_scheduler(
            self.lr_scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=self.lr_warmup_iters,
            num_training_steps=self.max_train_iters,
        )

        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)

    def _teacher_lora_checkpoint_path(self, resume_ckpt_path):
        return os.path.join(resume_ckpt_path, "teacher", "pytorch_lora_weights.safetensors")

    def _load_resume_state(self, resume_ckpt_path):
        if self.model.is_fsdp2_wrapped():
            self._load_distributed_state(resume_ckpt_path)
        else:
            self._load_single_process_state(resume_ckpt_path)

        teacher_weights_path = self._teacher_lora_checkpoint_path(resume_ckpt_path)
        if os.path.exists(teacher_weights_path):
            self.model.load_lora_weights_for_resume(
                resume_ckpt_path,
                adapter_name=self.teacher_adapter,
                weights_subdir="teacher",
            )
            logger.info("Restored teacher EMA LoRA from {}", teacher_weights_path)
        else:
            self.model.copy_lora_adapter_weights(self.student_adapter, self.teacher_adapter)
            logger.warning(
                "Teacher LoRA not found in checkpoint {}; initialized teacher from student",
                resume_ckpt_path,
            )

    def _load_single_process_state(self, resume_ckpt_path):
        training_state_path = os.path.join(resume_ckpt_path, "training_state.pt")
        if not os.path.exists(training_state_path):
            raise RuntimeError(f"training_state.pt not found in {resume_ckpt_path}")

        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_metadata(state, training_state_path, resume_ckpt_path)
        self.model.load_lora_weights_for_resume(resume_ckpt_path, adapter_name=self.student_adapter)
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        logger.info("Restored training state from {}", training_state_path)

    def _load_distributed_state(self, resume_ckpt_path):
        dist_state_path = os.path.join(resume_ckpt_path, "dist_state")
        if not os.path.exists(dist_state_path):
            raise RuntimeError(f"FSDP2 resume requires dist_state/, but it was not found in {resume_ckpt_path}")

        trainer_state_path = os.path.join(resume_ckpt_path, "trainer_state.pt")
        if not os.path.exists(trainer_state_path):
            raise RuntimeError(f"trainer_state.pt not found in {resume_ckpt_path}")
        trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_metadata(trainer_state, trainer_state_path, resume_ckpt_path)

        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        state_module = self.model.fsdp2_state_module()
        model_state, optim_state = get_state_dict(state_module, self.optimizer, options=options)
        state = {"model": model_state, "optimizer": optim_state}
        dcp.load(state, checkpoint_id=dist_state_path)
        set_state_dict(
            state_module,
            self.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=options,
        )

        self.lr_scheduler.load_state_dict(trainer_state["lr_scheduler"])
        logger.info("Restored distributed training state from {}", resume_ckpt_path)

    def _teacher_edit_prompts(self, prompts):
        if isinstance(prompts, str):
            base_prompts = [prompts]
        else:
            base_prompts = list(prompts)
        suffix = self.edit_sys_prompt.strip()
        if not suffix:
            return base_prompts
        if not self.teacher_use_dataset_prompt:
            return [suffix] * len(base_prompts)
        return [f"{prompt} {suffix}".strip() for prompt in base_prompts]

    def _build_train_reference_map(self):
        if self._train_reference_map is not None:
            return
        self._train_reference_map = {}
        train_dataset = self.dataloader_train.dataset
        for sample in train_dataset.samples:
            target_image = sample.get("target_image")
            prompt = sample.get("prompt")
            if target_image is None or prompt is None:
                continue
            self._train_reference_map[str(prompt).strip()] = target_image

    def _resolve_teacher_reference_image(self, sample, dataset):
        target_image = sample.get("target_image")
        if target_image is not None:
            if isinstance(target_image, str):
                return dataset.load_image(target_image)
            return target_image

        self._build_train_reference_map()
        fallback_path = self._train_reference_map.get(str(sample.get("prompt", "")).strip())
        if fallback_path is None:
            return None
        return dataset.load_image(fallback_path)

    def _validate_checkpoint_metadata(self, state, state_path, resume_ckpt_path):
        checkpoint_world_size = state.get("world_size")
        current_world_size = get_world_size()
        if checkpoint_world_size != current_world_size:
            raise RuntimeError(f"Cannot resume checkpoint saved with world_size={checkpoint_world_size} using world_size={current_world_size}: {state_path}")

        expected_iteration = parse_checkpoint_iteration(resume_ckpt_path)
        checkpoint_iteration = state.get("iteration")
        if checkpoint_iteration != expected_iteration:
            raise RuntimeError(f"Cannot resume checkpoint with iteration={checkpoint_iteration} in {state_path}, expected iteration={expected_iteration} from {resume_ckpt_path}")

    def compute_loss_on_sample(self, sample, collect_trajectory=False):
        if sample.get("target_image") is None:
            raise ValueError("D-OPSD training requires target_image in each sample.")

        image = sample["target_image"].to(device=self.model.device, dtype=self.running_dtype)
        bsz = image.shape[0]
        height, width = image.shape[2], image.shape[3]
        latent_hw = (height // 16, width // 16)
        t_scale = float(self.noise_scheduler.num_train_timesteps)

        with torch.no_grad():
            student_condition = self.model.encode_condition(sample)
            teacher_condition = self.model.encode_prompt_text(self._teacher_edit_prompts(sample["prompt"]))
            teacher_image_latents, teacher_image_latent_ids = self.model.prepare_reference_image_latents(image)
            latents_begin, latent_ids = self.model.prepare_dopsd_initial_latents(height, width, bsz)

            self.noise_scheduler.set_timesteps(self.num_training_steps, latent_hw=latent_hw)
            timesteps = self.noise_scheduler.infer_timesteps

        latents_student = latents_begin
        total_loss = 0.0
        num_steps = len(timesteps)
        step_loss_weights = self._step_loss_weights_for(num_steps)
        weight_sum = 0.0
        student_x0_traj = []
        teacher_x0_traj = []

        for back_step in range(num_steps):
            t = timesteps[back_step].expand(bsz) / t_scale
            t = t.to(device=self.model.device, dtype=self.running_dtype)

            if back_step < num_steps - 1:
                next_t = timesteps[back_step + 1].expand(bsz) / t_scale
            else:
                next_t = torch.zeros_like(t)
            next_t = next_t.to(device=self.model.device, dtype=self.running_dtype)
            dt = next_t - t

            latents_student = latents_student.detach().requires_grad_(True)

            with torch.no_grad():
                v_pred_teacher = self.model.predict_velocity(
                    latents_student,
                    t,
                    teacher_condition,
                    latent_ids,
                    self.teacher_adapter,
                    teacher_image_latents=teacher_image_latents,
                    teacher_image_latent_ids=teacher_image_latent_ids,
                )
                latents_teacher_cur = latents_student
                x_0_teacher = latents_teacher_cur + (0 - t).reshape(bsz, 1, 1) * v_pred_teacher
                # latents_teacher = latents_teacher_cur + v_pred_teacher * dt.reshape(bsz, 1, 1)

            v_pred_student = self.model.predict_velocity(
                latents_student,
                t,
                student_condition,
                latent_ids,
                self.student_adapter,
            )
            latents_student_cur = latents_student
            x_0_student = latents_student_cur + (0 - t).reshape(bsz, 1, 1) * v_pred_student
            latents_student = latents_student_cur + v_pred_student * dt.reshape(bsz, 1, 1)

            loss_dopsd = F.mse_loss(x_0_student, x_0_teacher.detach(), reduction="mean")
            step_weight = step_loss_weights[back_step]
            total_loss = total_loss + loss_dopsd * step_weight
            weight_sum += step_weight
            if collect_trajectory:
                student_x0_traj.append(x_0_student.detach())
                teacher_x0_traj.append(x_0_teacher.detach())

        avg_loss = total_loss / weight_sum
        if collect_trajectory:
            return avg_loss, student_x0_traj, teacher_x0_traj, latent_ids, height, width
        return avg_loss

    @torch.no_grad()
    def _save_training_trajectory(self, current_iter, student_x0_traj, teacher_x0_traj, latent_ids):
        traj_dir = Path(self.output_train_dir) / "trajectory" / f"iter-{current_iter:09d}"
        traj_dir.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        student_step_images = []
        teacher_step_images = []
        num_steps = len(student_x0_traj)
        for step_idx, (x_0_student, x_0_teacher) in enumerate(zip(student_x0_traj, teacher_x0_traj)):
            logger.info("[train] trajectory decode iter={} step={}/{} student", current_iter, step_idx + 1, num_steps)
            student_step_images.extend(self.model.decode_packed_x0_to_images(x_0_student, latent_ids))
            logger.info("[train] trajectory decode iter={} step={}/{} teacher", current_iter, step_idx + 1, num_steps)
            teacher_step_images.extend(self.model.decode_packed_x0_to_images(x_0_teacher, latent_ids))

        save_path = traj_dir / "student_teacher_x0_traj.png"
        save_student_teacher_trajectory_grid(student_step_images, teacher_step_images, save_path)
        logger.info("[train] saved trajectory iter={} path={}", current_iter, save_path)

    def _resolve_resume(self):
        if not self.auto_resume:
            return None, 0
        ckpt_path, current_iter = find_latest_checkpoint(self.output_train_dir)
        if ckpt_path is None:
            logger.info("Auto-resume enabled but no checkpoint found in '{}'. Starting from scratch.", self.output_train_dir)
        else:
            logger.info("Auto-resuming from checkpoint: {} (iteration {})", ckpt_path, current_iter)
        return ckpt_path, current_iter

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        max_train_iters = self.max_train_iters
        grad_accum_iters = self.gradient_accumulation_iters
        max_grad_norm = self.max_grad_norm
        save_every_iters = self.save_every_iters
        save_total_limit = self.save_total_limit
        grad_accum_counter = 0
        running_loss = 0.0

        logger.info(
            "[train] dopsd start iter={}/{} world_size={} grad_accum={} num_training_steps={} "
            "ema_decay={} ema_decay_warmup={} ema_decay_warmup_iters={} step_loss_weights={} "
            "edit_sys_prompt={!r} teacher_use_dataset_prompt={} trajectory_every_iters={}",
            current_iter,
            max_train_iters,
            get_world_size(),
            grad_accum_iters,
            self.num_training_steps,
            self.ema_decay,
            self.ema_decay_warmup,
            self.ema_decay_warmup_iters,
            self._step_loss_weights_for(self.num_training_steps),
            self.edit_sys_prompt,
            self.teacher_use_dataset_prompt,
            self.trajectory_every_iters,
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
                sync_grad = (grad_accum_counter + 1) % grad_accum_iters == 0
                self._set_gradient_sync(sync_grad)

                should_save_trajectory = self.trajectory_every_iters and (current_iter + 1) % self.trajectory_every_iters == 0
                if should_save_trajectory:
                    loss, student_x0_traj, teacher_x0_traj, latent_ids, _height, _width = self.compute_loss_on_sample(
                        sample,
                        collect_trajectory=True,
                    )
                else:
                    loss = self.compute_loss_on_sample(sample)
                (loss / grad_accum_iters).backward()
                running_loss += loss.item() / grad_accum_iters

                grad_accum_counter += 1
                if grad_accum_counter % grad_accum_iters != 0:
                    continue

                torch.nn.utils.clip_grad_norm_(self.trainable_params, max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                current_ema_decay = self._resolve_ema_decay(current_iter + 1)
                self.model.ema_update_lora_adapter(
                    self.student_adapter,
                    self.teacher_adapter,
                    current_ema_decay,
                )

                current_iter += 1
                display_loss = reduce_mean(running_loss)
                current_lr = self.lr_scheduler.get_last_lr()[0]
                if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= max_train_iters:
                    logger.info(
                        "[train] iter={}/{} loss_dopsd={:.6f} lr={:.8f} ema_decay={:.4f}",
                        current_iter,
                        max_train_iters,
                        display_loss,
                        current_lr,
                        current_ema_decay,
                    )
                running_loss = 0.0

                if should_save_trajectory:
                    barrier()
                    if is_main_process():
                        logger.info("[train] saving trajectory iter={} (decoding {} x0 pairs)...", current_iter, len(student_x0_traj))
                        self.model.set_denoiser_eval()
                        self._save_training_trajectory(
                            current_iter,
                            student_x0_traj,
                            teacher_x0_traj,
                            latent_ids,
                        )
                    self.model.set_dual_lora_trainable(self.student_adapter, self.teacher_adapter)
                    barrier()

                if save_every_iters and current_iter % save_every_iters == 0:
                    self.save_checkpoint(current_iter, save_total_limit)

                if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                    self.run_inference(current_iter)

                if current_iter >= max_train_iters:
                    break

            epoch += 1

        logger.info("[train] finished iter={}/{}", current_iter, max_train_iters)

    def _set_gradient_sync(self, enabled):
        self.model.set_fsdp2_gradient_sync(enabled)

    @torch.no_grad()
    def _run_teacher_inference(self, current_iter, iter_output_dir):
        dataset = self.inferencer.dataloader_eval.dataset
        samples = dataset.samples
        rank = get_rank()
        world_size = get_world_size()

        num_inference_steps = self.infer_config.get("num_inference_steps", self.num_training_steps)
        base_seed = self.infer_config.get("seed", 42)
        t_scale = float(self.noise_scheduler.num_train_timesteps)

        teacher_output_dir = os.path.join(iter_output_dir, "teacher")
        os.makedirs(teacher_output_dir, exist_ok=True)

        self.model.set_active_adapter(self.teacher_adapter)
        self.model.set_denoiser_eval()

        num_slots = (len(samples) + world_size - 1) // world_size if is_distributed() else len(samples)
        saved_count = 0
        skipped_count = 0
        logger.info(
            "[train] running teacher inference iter={} output_dir={} steps={}",
            current_iter,
            teacher_output_dir,
            num_inference_steps,
        )
        for slot in range(num_slots):
            i = slot * world_size + rank if is_distributed() else slot
            if i >= len(samples):
                continue

            sample = samples[i]
            reference_image = self._resolve_teacher_reference_image(sample, dataset)
            if reference_image is None:
                skipped_count += 1
                logger.warning(
                    "[train] teacher infer skip sample={}/{}: missing target_image in eval metadata",
                    i + 1,
                    len(samples),
                )
                continue

            image = reference_image.unsqueeze(0).to(device=self.model.device, dtype=self.running_dtype)
            height, width = image.shape[2], image.shape[3]
            latent_hw = (height // 16, width // 16)

            teacher_prompts = self._teacher_edit_prompts(sample["prompt"])
            teacher_condition = self.model.encode_prompt_text(teacher_prompts)
            teacher_image_latents, teacher_image_latent_ids = self.model.prepare_reference_image_latents(image)

            seed = base_seed + i
            generator = torch.Generator(device=self.model.device).manual_seed(seed)
            latents, latent_ids = self.model.prepare_dopsd_initial_latents(height, width, 1, generator=generator)

            self.noise_scheduler.set_timesteps(num_inference_steps, latent_hw=latent_hw)
            timesteps = self.noise_scheduler.infer_timesteps
            num_steps = len(timesteps)

            logger.info(
                "[train] teacher infer sample={}/{} seed={} size={}x{} prompt={!r}",
                i + 1,
                len(samples),
                seed,
                height,
                width,
                teacher_prompts[0],
            )
            for back_step in range(num_steps):
                t = timesteps[back_step].expand(1) / t_scale
                t = t.to(device=self.model.device, dtype=self.running_dtype)
                if back_step < num_steps - 1:
                    next_t = timesteps[back_step + 1].expand(1) / t_scale
                else:
                    next_t = torch.zeros_like(t)
                next_t = next_t.to(device=self.model.device, dtype=self.running_dtype)
                dt = next_t - t

                v_pred = self.model.predict_velocity(
                    latents,
                    t,
                    teacher_condition,
                    latent_ids,
                    self.teacher_adapter,
                    teacher_image_latents=teacher_image_latents,
                    teacher_image_latent_ids=teacher_image_latent_ids,
                )
                latents = latents + v_pred * dt.reshape(1, 1, 1)

            images = self.model.decode_packed_x0_to_images(latents, latent_ids)
            save_path = Path(teacher_output_dir) / f"{i:05d}.png"
            images[0].save(save_path)
            saved_count += 1
            logger.info("[train] teacher infer sample={}/{} saved path={}", i + 1, len(samples), save_path)

        barrier()
        if is_distributed():
            saved_tensor = torch.tensor(saved_count, device=self.model.device, dtype=torch.int64)
            skipped_tensor = torch.tensor(skipped_count, device=self.model.device, dtype=torch.int64)
            torch.distributed.all_reduce(saved_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(skipped_tensor, op=torch.distributed.ReduceOp.SUM)
            saved_count = saved_tensor.item()
            skipped_count = skipped_tensor.item()
        logger.info(
            "[train] finished teacher inference iter={} saved={} skipped={}",
            current_iter,
            saved_count,
            skipped_count,
        )

    def run_inference(self, current_iter):
        base_output_dir = self.infer_config.get("output_dir", "./output_infer")
        iter_output_dir = os.path.join(base_output_dir, f"iter-{current_iter:09d}")

        self.model.set_active_adapter(self.student_adapter)
        self.inferencer.output_infer_dir = iter_output_dir
        os.makedirs(iter_output_dir, exist_ok=True)
        logger.info("[train] running student inference iter={} output_dir={}", current_iter, iter_output_dir)
        self.inferencer.infer()
        barrier()
        self._run_teacher_inference(current_iter, iter_output_dir)
        logger.info("[train] finished inference iter={}", current_iter)

        self.model.set_dual_lora_trainable(self.student_adapter, self.teacher_adapter)

    def save_checkpoint(self, iteration, save_total_limit):
        if is_main_process():
            prune_checkpoints(self.output_train_dir, save_total_limit)

        save_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        logger.info("[train] saving checkpoint iter={} path={}", iteration, save_dir)
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        barrier()

        self.model.save_lora_weights(save_dir, adapter_name=self.student_adapter)
        barrier()
        self.model.save_lora_weights(save_dir, adapter_name=self.teacher_adapter, weights_subdir="teacher")
        barrier()

        config_path = self.config.get("config_path")
        if is_main_process() and config_path is not None:
            shutil.copy2(config_path, os.path.join(save_dir, "config.yaml"))

        if self.model.is_fsdp2_wrapped():
            self._save_distributed_state(save_dir, iteration)
            barrier()
            logger.info("[train] saved checkpoint iter={} path={} (student + teacher EMA LoRA)", iteration, save_dir)
            return

        training_state = {
            "iteration": iteration,
            "world_size": get_world_size(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        if is_main_process():
            torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
        barrier()
        logger.info("[train] saved checkpoint iter={} path={} (student + teacher EMA LoRA)", iteration, save_dir)

    def _save_distributed_state(self, save_dir, iteration):
        dist_state_path = os.path.join(save_dir, "dist_state")
        if is_main_process():
            os.makedirs(dist_state_path, exist_ok=True)
            torch.save(
                {
                    "iteration": iteration,
                    "world_size": get_world_size(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                },
                os.path.join(save_dir, "trainer_state.pt"),
            )
        barrier()

        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        model_state, optim_state = get_state_dict(self.model.fsdp2_state_module(), self.optimizer, options=options)
        dcp.save(
            {"model": model_state, "optimizer": optim_state},
            checkpoint_id=dist_state_path,
        )
