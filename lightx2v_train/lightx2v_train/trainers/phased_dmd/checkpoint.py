import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

from lightx2v_train.runtime.checkpoint import prune_checkpoints
from lightx2v_train.runtime.distributed import (
    barrier,
    get_world_size,
    is_main_process,
)

from ..dmd.checkpoint import DmdCheckpointManager


class PhasedCheckpointManager(DmdCheckpointManager):
    """Extend DMD checkpoint I/O for phased role layouts."""

    checkpoint_version_key = "phased_checkpoint_version"
    checkpoint_version = 3

    def _role_train_type(self, role):
        return self.role_registry.runtime(role).train_type

    def _trainable_role_models(self):
        return self.role_registry.trainable_models()

    def _trainable_role_states(self):
        return self.role_registry.trainable_states()

    def _save_model_weights(self, model, save_dir, role="student"):
        train_type = self._role_train_type(role)
        self._checkpoint(model).save_weights(save_dir, train_type)

    def _load_model_weights(self, model, save_dir, role="student"):
        train_type = self._role_train_type(role)
        self._checkpoint(model).load_weights(save_dir, train_type)

    def _validate_checkpoint_state(self, state, state_path, resume_ckpt_path):
        super()._validate_checkpoint_state(state, state_path, resume_ckpt_path)
        expected = {"fake_low_high_enabled": self.enable_fake_low_high, "phased_match_timestep": self.match_timestep}
        self._require_checkpoint_keys(state, expected, state_path)
        for key, value in expected.items():
            if state[key] != value:
                raise RuntimeError(f"Checkpoint {key}={state[key]!r} does not match the current value {value!r}: {state_path}")

    def _load_resume_state(self, resume_ckpt_path):
        models = tuple(model for _, model in self._trainable_role_models())
        if any(self._parallel(model).is_fsdp() for model in models):
            self._load_distributed_state(resume_ckpt_path)
            return
        self._load_single_process_state(resume_ckpt_path)

    def _get_checkpoint_process_group(self):
        if self._checkpoint_process_group is None:
            self._checkpoint_process_group = dist.new_group(
                backend="gloo",
            )
        return self._checkpoint_process_group

    def _load_distributed_state(self, resume_ckpt_path):
        dist_state_path = os.path.join(resume_ckpt_path, "dist_state")
        trainer_state_path = os.path.join(resume_ckpt_path, "trainer_state.pt")
        trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_state(trainer_state, trainer_state_path, resume_ckpt_path)
        self._require_role_state(trainer_state, trainer_state_path, distributed=True)
        roles = self._active_role_runtimes()
        for role, _ in roles:
            role_path = os.path.join(dist_state_path, role)
            if not os.path.isdir(role_path):
                raise RuntimeError(f"Checkpoint is missing {role} distributed state: {role_path}")
        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        checkpoint_group = self._get_checkpoint_process_group()
        for role, runtime in roles:
            module = self._parallel(runtime.model).state_module()
            model_state, optimizer_state = get_state_dict(module, runtime.optimizer, options=options)
            role_state = {"model": model_state, "optimizer": optimizer_state}
            dcp.load(role_state, checkpoint_id=os.path.join(dist_state_path, role), process_group=checkpoint_group)
            set_state_dict(module, runtime.optimizer, model_state_dict=role_state["model"], optim_state_dict=role_state["optimizer"], options=options)
            runtime.scheduler.load_state_dict(trainer_state[runtime.spec.scheduler_attribute])
            logger.info("[checkpoint][resume][role] role={} model=restored optimizer=restored scheduler=restored", role)
            del role_state, model_state, optimizer_state
        logger.info("Restored distributed phased DMD training state from {}", resume_ckpt_path)

    def _finalize_checkpoint(
        self,
        temporary_dir,
        final_dir,
        iteration,
        save_total_limit,
    ):
        barrier()
        if is_main_process():
            with open(
                os.path.join(temporary_dir, "_SUCCESS"),
                "w",
                encoding="utf-8",
            ):
                pass
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            os.replace(temporary_dir, final_dir)
            prune_checkpoints(
                self.output_train_dir,
                save_total_limit,
            )
        barrier()
        logger.info(
            "[train] saved checkpoint iter={} path={}",
            iteration,
            final_dir,
        )
        logger.info(
            "[checkpoint][save][done] iteration={} path={} roles={}",
            iteration,
            final_dir,
            list(self.role_registry.trainable_names()),
        )

    def save_checkpoint(self, iteration, save_total_limit):
        final_dir = os.path.join(
            self.output_train_dir,
            f"checkpoint-{iteration:09d}",
        )
        save_dir = os.path.join(
            self.output_train_dir,
            f".checkpoint-{iteration:09d}.tmp",
        )
        active_roles = list(self.role_registry.trainable_names())
        logger.info(
            "[checkpoint][save][start] iteration={} path={} roles={}",
            iteration,
            final_dir,
            active_roles,
        )
        if is_main_process():
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)
        barrier()

        role_models = self._trainable_role_models()
        for role, model in role_models:
            weights_dir = self._role_weights_dir(save_dir, role)
            save_weights = self._role_train_type(role) == "lora" or not self._parallel(model).is_fsdp()
            if save_weights and role != "student" and is_main_process():
                os.makedirs(weights_dir, exist_ok=True)
            barrier()
            if save_weights:
                self._save_model_weights(
                    model,
                    weights_dir,
                    role=role,
                )
            barrier()
            logger.info(
                "[checkpoint][save][role] role={} path={} weights={}",
                role,
                weights_dir,
                save_weights,
            )

        config_path = self.config.get("config_path")
        if is_main_process() and config_path is not None:
            shutil.copy2(
                config_path,
                os.path.join(save_dir, "config.yaml"),
            )

        if any(self._parallel(model).is_fsdp() for _, model in role_models):
            self._save_distributed_state(save_dir, iteration)
            self._finalize_checkpoint(
                save_dir,
                final_dir,
                iteration,
                save_total_limit,
            )
            return

        training_state = {
            "iteration": iteration,
            "world_size": get_world_size(),
            "phased_checkpoint_version": 3,
            "student_train_type": self.student_train_type,
            "fake_train_type": self.fake_train_type,
            "student_2_train_type": self.student_2_train_type,
            "fake_2_train_type": self.fake_2_train_type,
            "fake_low_high_enabled": self.enable_fake_low_high,
            "fake_low_high_train_type": (self.fake_low_high_train_type),
            "phased_match_timestep": self.match_timestep,
            "optimizer": self.optimizer.state_dict(),
            "fake_optimizer": self.fake_optimizer.state_dict(),
            "student_2_optimizer": (self.student_2_optimizer.state_dict()),
            "fake_2_optimizer": self.fake_2_optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "fake_lr_scheduler": self.fake_lr_scheduler.state_dict(),
            "student_2_lr_scheduler": (self.student_2_lr_scheduler.state_dict()),
            "fake_2_lr_scheduler": (self.fake_2_lr_scheduler.state_dict()),
        }
        if self.fake_low_high_optimizer is not None:
            training_state["fake_low_high_optimizer"] = self.fake_low_high_optimizer.state_dict()
            training_state["fake_low_high_lr_scheduler"] = self.fake_low_high_lr_scheduler.state_dict()
        for role in ("fake_real_high", "fake_real_low"):
            runtime = self.role_registry.runtime(role)
            if runtime.model is None:
                continue
            training_state[f"{role}_train_type"] = runtime.train_type
            training_state[f"{role}_optimizer"] = runtime.optimizer.state_dict()
            training_state[f"{role}_lr_scheduler"] = runtime.scheduler.state_dict()
        training_state.update(self._trick_checkpoint_metadata())
        if is_main_process():
            torch.save(
                training_state,
                os.path.join(save_dir, "training_state.pt"),
            )
        barrier()
        self._finalize_checkpoint(
            save_dir,
            final_dir,
            iteration,
            save_total_limit,
        )

    def _save_distributed_state(self, save_dir, iteration):
        dist_state_path = os.path.join(save_dir, "dist_state")
        if is_main_process():
            os.makedirs(dist_state_path, exist_ok=True)
            torch.save(
                {
                    "iteration": iteration,
                    "world_size": get_world_size(),
                    "phased_checkpoint_version": 3,
                    "student_train_type": self.student_train_type,
                    "fake_train_type": self.fake_train_type,
                    "student_2_train_type": (self.student_2_train_type),
                    "fake_2_train_type": self.fake_2_train_type,
                    "fake_low_high_enabled": (self.enable_fake_low_high),
                    "fake_low_high_train_type": (self.fake_low_high_train_type),
                    "phased_match_timestep": self.match_timestep,
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "fake_lr_scheduler": (self.fake_lr_scheduler.state_dict()),
                    "student_2_lr_scheduler": (self.student_2_lr_scheduler.state_dict()),
                    "fake_2_lr_scheduler": (self.fake_2_lr_scheduler.state_dict()),
                    "fake_low_high_lr_scheduler": (self.fake_low_high_lr_scheduler.state_dict() if self.fake_low_high_lr_scheduler is not None else None),
                    "fake_real_high_train_type": (self.fake_real_high_train_type if self.fake_real_high_model is not None else None),
                    "fake_real_low_train_type": (self.fake_real_low_train_type if self.fake_real_low_model is not None else None),
                    "fake_real_high_lr_scheduler": (self.fake_real_high_lr_scheduler.state_dict() if self.fake_real_high_lr_scheduler is not None else None),
                    "fake_real_low_lr_scheduler": (self.fake_real_low_lr_scheduler.state_dict() if self.fake_real_low_lr_scheduler is not None else None),
                    **self._trick_checkpoint_metadata(),
                },
                os.path.join(save_dir, "trainer_state.pt"),
            )
        barrier()

        options = StateDictOptions(
            ignore_frozen_params=True,
            strict=False,
        )
        checkpoint_group = self._get_checkpoint_process_group()
        for role, model, optimizer in self._trainable_role_states():
            logger.info(
                "[train] collecting checkpoint role={}",
                role,
            )
            model_state, optimizer_state = get_state_dict(
                self._parallel(model).state_module(),
                optimizer,
                options=options,
            )
            role_state = {
                "model": model_state,
                "optimizer": optimizer_state,
            }
            role_checkpoint_path = os.path.join(
                dist_state_path,
                role,
            )
            logger.info(
                "[train] writing checkpoint role={} path={}",
                role,
                role_checkpoint_path,
            )
            dcp.save(
                role_state,
                checkpoint_id=role_checkpoint_path,
                process_group=checkpoint_group,
            )
            logger.info(
                "[train] wrote checkpoint role={}",
                role,
            )
            del role_state, model_state, optimizer_state
