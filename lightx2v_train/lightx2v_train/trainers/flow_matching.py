import os

import torch
from loguru import logger

from lightx2v_train.model_capabilities import (
    FlowMatchingSFTCapability,
    SFTStepContext,
)
from lightx2v_train.runtime.distributed import barrier, get_world_size, is_main_process, reduce_mean
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value, sync_sequence_parallel_gradients
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .base import BaseTrainer


@TRAINER_REGISTER("flow_matching")
class FlowMatchingTrainer(BaseTrainer):
    trainer_name = "flow_matching"
    required_capabilities = (
        *BaseTrainer.required_capabilities,
        FlowMatchingSFTCapability,
    )

    def set_model(self, model):
        super().set_model(model)
        self.sft = model.capabilities.require(FlowMatchingSFTCapability)

    def compute_loss_on_sample(self, sample):
        return self.sft.compute_loss(
            sample,
            SFTStepContext(
                noise_scheduler=self.noise_scheduler,
                running_dtype=self.running_dtype,
                broadcast=broadcast_sequence_parallel_value,
            ),
        )

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        # Objectives with a curriculum (for example CM's shrinking time
        # interval) need the optimizer-step index, including after resume.
        self.current_train_iteration = current_iter
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
        running_metrics = {}

        logger.info(
            "[train] start method={} train_type={} iter={}/{} world_size={} grad_accum={} train_log_every_iters={}",
            self.trainer_name,
            self.train_type,
            current_iter,
            max_train_iters,
            get_world_size(),
            grad_accum_iters,
            self.train_log_every_iters,
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

                loss_result = self.compute_loss_on_sample(sample)
                loss = loss_result.loss
                sample_metrics = loss_result.metrics
                (loss / grad_accum_iters).backward()
                running_loss += loss.item() / grad_accum_iters
                for name, value in sample_metrics.items():
                    scalar = value.detach().item() if torch.is_tensor(value) else float(value)
                    running_metrics[name] = running_metrics.get(name, 0.0) + scalar / grad_accum_iters

                grad_accum_counter += 1
                if grad_accum_counter % grad_accum_iters != 0:
                    continue

                self._after_backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_params, max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                current_iter += 1
                self.current_train_iteration = current_iter
                display_loss = reduce_mean(running_loss)
                current_lr = self.lr_scheduler.get_last_lr()[0]
                if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= max_train_iters:
                    display_metrics = {name: reduce_mean(value) for name, value in running_metrics.items()}
                    metric_text = " ".join(f"{name}={value:.6f}" for name, value in sorted(display_metrics.items()))
                    if metric_text:
                        logger.info("[train] iter={}/{} loss={:.6f} {} lr={:.8f}", current_iter, max_train_iters, display_loss, metric_text, current_lr)
                    else:
                        logger.info("[train] iter={}/{} loss={:.6f} lr={:.8f}", current_iter, max_train_iters, display_loss, current_lr)
                    logged_metrics = {
                        "train/loss": display_loss,
                        "train/lr": current_lr,
                    }
                    logged_metrics.update({f"train/{name}": value for name, value in display_metrics.items()})
                    self.log_metrics(logged_metrics, step=current_iter)
                running_loss = 0.0
                running_metrics = {}

                if save_every_iters and current_iter % save_every_iters == 0:
                    self.save_checkpoint(current_iter, save_total_limit)

                if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                    self.run_inference(current_iter)

                if current_iter >= max_train_iters:
                    break

            epoch += 1

        logger.info("[train] finished iter={}/{}", current_iter, max_train_iters)

    def _after_backward(self):
        sync_sequence_parallel_gradients(self.trainable_params)
