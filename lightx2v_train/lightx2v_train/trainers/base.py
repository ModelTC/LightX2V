import os
import shutil

import torch
import torch.distributed.checkpoint as dcp
from diffusers.optimization import get_scheduler
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from lightx2v_train.infer import build_inferencer
from lightx2v_train.runtime.checkpoint import find_latest_checkpoint, parse_checkpoint_iteration, prune_checkpoints
from lightx2v_train.runtime.distributed import barrier, get_world_size, is_main_process
from lightx2v_train.runtime.parallel import apply_parallel, set_parallel_gradient_sync
from lightx2v_train.schedulers.flow_matching import RectifiedFlowMatchingScheduler
from lightx2v_train.utils.utils import get_running_dtype


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.model_config = self.config["model"]
        self.training_config = self.config["training"]
        self.infer_config = self.config["inference"]

        self.noise_scheduler = RectifiedFlowMatchingScheduler(config)
        self.running_dtype = get_running_dtype(self.model_config["running_dtype"])
        self.train_type = self._resolve_train_type()
        logger.info("[train] train_type={}", self.train_type)

        lora_config = self._get_lora_config()
        self.lora_rank = lora_config.get("rank", 16)
        self.lora_alpha = lora_config.get("alpha", self.lora_rank)
        self.lora_target_modules = lora_config.get("target_modules")

        self.gradient_checkpointing = self.training_config.get("gradient_checkpointing", True)

        optimizer_config = self._get_optimizer_config()
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

    def _resolve_train_type(self):
        train_type = self.training_config.get("train_type", None)
        if train_type is None:
            raise ValueError("training.train_type must be set explicitly to 'lora' or 'full'.")
        if train_type not in {"lora", "full"}:
            raise ValueError(f"Unsupported training.train_type={train_type!r}; expected 'lora' or 'full'.")
        return train_type

    def _get_lora_config(self):
        return self.training_config.get("lora", {})

    def _get_optimizer_config(self):
        return self.training_config.get("optimizer", {})

    def set_model(self, model):
        self.model = model

    def set_data(self, dataloader_train, dataloader_eval=None):
        self.dataloader_train = dataloader_train
        self.dataloader_eval = dataloader_eval

    def _setup_trainable_model(self, model):
        if self.train_type == "lora":
            model.add_lora(self.lora_rank, self.lora_alpha, self.lora_target_modules)
            model.set_lora_trainable()
            return
        model.set_full_trainable()

    def _restore_trainable_model(self, model):
        if self.train_type == "lora":
            model.set_lora_trainable()
            return
        model.set_full_trainable()

    def _build_optimizer(self, params, optimizer_config=None):
        if optimizer_config is None:
            optimizer_config = {
                "learning_rate": self.optimizer_learning_rate,
                "adam_beta1": self.optimizer_adam_beta1,
                "adam_beta2": self.optimizer_adam_beta2,
                "weight_decay": self.optimizer_weight_decay,
                "adam_epsilon": self.optimizer_adam_epsilon,
            }
        return torch.optim.AdamW(
            params,
            lr=optimizer_config.get("learning_rate", 1e-4),
            betas=(optimizer_config.get("adam_beta1", 0.9), optimizer_config.get("adam_beta2", 0.999)),
            weight_decay=optimizer_config.get("weight_decay", 0.01),
            eps=optimizer_config.get("adam_epsilon", 1e-8),
        )

    def _build_lr_scheduler(self, optimizer, num_training_steps=None, num_warmup_steps=None):
        return get_scheduler(
            self.lr_scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_iters if num_warmup_steps is None else num_warmup_steps,
            num_training_steps=self.max_train_iters if num_training_steps is None else num_training_steps,
        )

    def setup(self, resume_ckpt_path=None):
        self._setup_trainable_model(self.model)

        apply_parallel(self.model, self.config)

        if self.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        if self.infer_every_iters:
            self.inferencer = build_inferencer(self.config)
            self.inferencer.set_model(self.model)

        self.model.log_model_structure()

        self.trainable_params = list(self.model.trainable_parameters())
        self.optimizer = self._build_optimizer(self.trainable_params)
        self.lr_scheduler = self._build_lr_scheduler(self.optimizer)

        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)

    def _save_model_weights(self, model, save_dir):
        if self.train_type == "lora":
            model.save_lora_weights(save_dir)
            return
        if is_main_process():
            torch.save(model.denoiser_module().state_dict(), os.path.join(save_dir, "model_state.pt"))

    def _load_model_weights(self, model, save_dir):
        if self.train_type == "lora":
            model.load_lora_weights_for_resume(save_dir)
            return
        model_state_path = os.path.join(save_dir, "model_state.pt")
        if not os.path.exists(model_state_path):
            raise RuntimeError(f"model_state.pt not found in {save_dir}")
        state_dict = torch.load(model_state_path, map_location="cpu", weights_only=False)
        model.denoiser_module().load_state_dict(state_dict)

    def _load_resume_state(self, resume_ckpt_path):
        if self.model.is_fsdp2_wrapped():
            self._load_distributed_state(resume_ckpt_path)
            return

        self._load_single_process_state(resume_ckpt_path)

    def _load_single_process_state(self, resume_ckpt_path):
        training_state_path = os.path.join(resume_ckpt_path, "training_state.pt")
        if not os.path.exists(training_state_path):
            raise RuntimeError(f"training_state.pt not found in {resume_ckpt_path}")

        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_metadata(state, training_state_path, resume_ckpt_path)
        self._load_model_weights(self.model, resume_ckpt_path)
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
        model_state, optim_state = get_state_dict(self.model.fsdp2_state_module(), self.optimizer, options=options)
        state = {"model": model_state, "optimizer": optim_state}
        dcp.load(state, checkpoint_id=dist_state_path)
        set_state_dict(
            self.model.fsdp2_state_module(),
            self.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=options,
        )

        self.lr_scheduler.load_state_dict(trainer_state["lr_scheduler"])
        logger.info("Restored distributed training state from {}", resume_ckpt_path)

    def _validate_checkpoint_metadata(self, state, state_path, resume_ckpt_path):
        checkpoint_world_size = state.get("world_size")
        current_world_size = get_world_size()
        if checkpoint_world_size != current_world_size:
            raise RuntimeError(f"Cannot resume checkpoint saved with world_size={checkpoint_world_size} using world_size={current_world_size}: {state_path}")

        expected_iteration = parse_checkpoint_iteration(resume_ckpt_path)
        checkpoint_iteration = state.get("iteration")
        if checkpoint_iteration != expected_iteration:
            raise RuntimeError(f"Cannot resume checkpoint with iteration={checkpoint_iteration} in {state_path}, expected iteration={expected_iteration} from {resume_ckpt_path}")

    def _resolve_resume(self):
        if not self.auto_resume:
            return None, 0
        ckpt_path, current_iter = find_latest_checkpoint(self.output_train_dir)
        if ckpt_path is None:
            logger.info("Auto-resume enabled but no checkpoint found in '{}'. Starting from scratch.", self.output_train_dir)
        else:
            logger.info("Auto-resuming from checkpoint: {} (iteration {})", ckpt_path, current_iter)
        return ckpt_path, current_iter

    def _set_gradient_sync(self, enabled):
        set_parallel_gradient_sync(self.model, enabled)

    def run_inference(self, current_iter):
        base_output_dir = self.infer_config.get("output_dir", "./output_infer")
        iter_output_dir = os.path.join(base_output_dir, f"iter-{current_iter:09d}")

        self.inferencer.output_infer_dir = iter_output_dir
        os.makedirs(iter_output_dir, exist_ok=True)
        logger.info("[train] running inference iter={} output_dir={}", current_iter, iter_output_dir)
        self.inferencer.infer()
        barrier()
        logger.info("[train] finished inference iter={}", current_iter)

        self._restore_trainable_model(self.model)

    def save_checkpoint(self, iteration, save_total_limit):
        if is_main_process():
            prune_checkpoints(self.output_train_dir, save_total_limit)

        save_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        logger.info("[train] saving checkpoint iter={} path={}", iteration, save_dir)
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        barrier()

        save_standalone_weights = self.train_type == "lora" or not self.model.is_fsdp2_wrapped()
        if save_standalone_weights:
            self._save_model_weights(self.model, save_dir)
        barrier()

        config_path = self.config.get("config_path")
        if is_main_process() and config_path is not None:
            shutil.copy2(config_path, os.path.join(save_dir, "config.yaml"))

        if self.model.is_fsdp2_wrapped():
            self._save_distributed_state(save_dir, iteration)
            barrier()
            logger.info("[train] saved checkpoint iter={} path={}", iteration, save_dir)
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

    def train(self):
        raise NotImplementedError
