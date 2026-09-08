"""Shared optimizer loop for trainers with one trainable model."""

import os

import torch
from loguru import logger

from lightx2v_train.runtime.distributed import barrier, get_world_size, is_main_process, reduce_mean

from .base import BaseTrainer


class OptimizerTrainer(BaseTrainer):
    trainer_name = "optimizer"

    def compute_loss_on_sample(self, sample):
        raise NotImplementedError

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.current_train_iteration = current_iter
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        grad_accum_counter = 0
        running_loss = 0.0
        running_metrics = {}

        logger.info(
            "[train] start method={} train_type={} iter={}/{} world_size={} grad_accum={} train_log_every_iters={}",
            self.trainer_name,
            self.train_type,
            current_iter,
            self.max_train_iters,
            get_world_size(),
            self.gradient_accumulation_iters,
            self.train_log_every_iters,
        )
        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_eval)
            if current_iter == 0:
                self.run_inference(current_iter)

        epoch = 0
        while current_iter < self.max_train_iters:
            sampler = getattr(self.dataloader_train, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for sample in self.dataloader_train:
                sync_grad = (grad_accum_counter + 1) % self.gradient_accumulation_iters == 0
                self._set_gradient_sync(sync_grad)

                loss_result = self.compute_loss_on_sample(sample)
                loss = loss_result.loss
                (loss / self.gradient_accumulation_iters).backward()
                running_loss += loss.item() / self.gradient_accumulation_iters
                for name, value in loss_result.metrics.items():
                    scalar = value.detach().item() if torch.is_tensor(value) else float(value)
                    running_metrics[name] = running_metrics.get(name, 0.0) + scalar / self.gradient_accumulation_iters

                grad_accum_counter += 1
                if not sync_grad:
                    continue

                self._after_backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                current_iter += 1
                self.current_train_iteration = current_iter
                display_loss = reduce_mean(running_loss)
                current_lr = self.lr_scheduler.get_last_lr()[0]
                if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= self.max_train_iters:
                    display_metrics = {name: reduce_mean(value) for name, value in running_metrics.items()}
                    metric_text = " ".join(f"{name}={value:.6f}" for name, value in sorted(display_metrics.items()))
                    logger.info(
                        "[train] iter={}/{} loss={:.6f} {}lr={:.8f}",
                        current_iter,
                        self.max_train_iters,
                        display_loss,
                        f"{metric_text} " if metric_text else "",
                        current_lr,
                    )
                    logged_metrics = {"train/loss": display_loss, "train/lr": current_lr}
                    logged_metrics.update({f"train/{name}": value for name, value in display_metrics.items()})
                    self.log_metrics(logged_metrics, step=current_iter)
                running_loss = 0.0
                running_metrics = {}

                if self.save_every_iters and current_iter % self.save_every_iters == 0:
                    self.save_checkpoint(current_iter, self.save_total_limit)
                if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                    self.run_inference(current_iter)
                if current_iter >= self.max_train_iters:
                    break

            epoch += 1

        logger.info("[train] finished iter={}/{}", current_iter, self.max_train_iters)
