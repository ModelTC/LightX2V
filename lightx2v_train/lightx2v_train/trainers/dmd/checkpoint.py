import os
import shutil

import torch
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

from lightx2v_train.model_capabilities import (
    CheckpointCapability,
    ParallelCapability,
)
from lightx2v_train.runtime.checkpoint import prune_checkpoints
from lightx2v_train.runtime.distributed import (
    barrier,
    get_world_size,
    is_main_process,
)


class DmdCheckpointManager:
    """Coordinate DMD checkpoint I/O without owning trainer resources."""

    checkpoint_version_key = "dmd_checkpoint_version"
    checkpoint_version = 2

    def __init__(self, owner):
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value):
        setattr(self.owner, name, value)

    def _fake_weights_dir(self, root_dir):
        directory_name = self.role_registry.weight_directory_name("fake")
        return os.path.join(root_dir, directory_name)

    @staticmethod
    def _parallel(model):
        return model.ensure_capabilities().require(ParallelCapability)

    @staticmethod
    def _checkpoint(model):
        return model.ensure_capabilities().require(CheckpointCapability)

    def _trick_checkpoint_metadata(self):
        metadata = {}
        for name in (
            "ida_trick",
            "diversity_trick",
            "real_data_fake_trick",
        ):
            trick = getattr(self, name, None)
            if trick is not None:
                metadata.update(trick.checkpoint_metadata())
        return metadata

    @staticmethod
    def _require_checkpoint_keys(state, keys, state_path):
        missing = sorted(set(keys) - state.keys())
        if missing:
            raise RuntimeError(f"Checkpoint is missing required state {missing}: {state_path}")

    def _role_weights_dir(self, root_dir, role):
        directory = self.role_registry.weight_directory_name(role)
        return os.path.join(root_dir, directory) if directory is not None else root_dir

    def _active_role_runtimes(self):
        return [(role, runtime) for role, runtime in self.role_registry.runtimes().items() if runtime.model is not None]

    def _require_role_state(self, state, state_path, *, distributed):
        keys = []
        for _, runtime in self._active_role_runtimes():
            keys.append(runtime.spec.scheduler_attribute)
            if not distributed:
                keys.append(runtime.spec.optimizer_attribute)
        self._require_checkpoint_keys(state, keys, state_path)

    def _load_resume_state(self, resume_ckpt_path):
        if self.parallel.is_fsdp() or self._parallel(self.fake_model).is_fsdp():
            self._load_distributed_state(resume_ckpt_path)
            return

        self._load_single_process_state(resume_ckpt_path)

    def _validate_checkpoint_state(self, state, state_path, resume_ckpt_path):
        self._validate_checkpoint_metadata(state, state_path, resume_ckpt_path)
        expected = {self.checkpoint_version_key: self.checkpoint_version}
        for role, runtime in self._active_role_runtimes():
            expected[f"{role}_train_type"] = runtime.train_type
        expected.update(self._trick_checkpoint_metadata())
        self._require_checkpoint_keys(state, expected, state_path)
        for key, value in expected.items():
            if state[key] != value:
                raise RuntimeError(f"Checkpoint {key}={state[key]!r} does not match the current value {value!r}: {state_path}")

    def _load_single_process_state(self, resume_ckpt_path):
        state_path = os.path.join(resume_ckpt_path, "training_state.pt")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_state(state, state_path, resume_ckpt_path)
        self._require_role_state(state, state_path, distributed=False)
        roles = self._active_role_runtimes()
        for role, _ in roles:
            weights_dir = self._role_weights_dir(resume_ckpt_path, role)
            if not os.path.isdir(weights_dir):
                raise RuntimeError(f"Checkpoint is missing {role} weights: {weights_dir}")
        for role, runtime in roles:
            self._load_model_weights(runtime.model, self._role_weights_dir(resume_ckpt_path, role), role=role)
            runtime.optimizer.load_state_dict(state[runtime.spec.optimizer_attribute])
            runtime.scheduler.load_state_dict(state[runtime.spec.scheduler_attribute])
            logger.info("[checkpoint][resume][role] role={} model=restored optimizer=restored scheduler=restored", role)
        logger.info("Restored training state from {} at iteration {}", state_path, state["iteration"])

    def _load_distributed_state(self, resume_ckpt_path):
        dist_state_path = os.path.join(resume_ckpt_path, "dist_state")
        trainer_state_path = os.path.join(resume_ckpt_path, "trainer_state.pt")
        trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_state(trainer_state, trainer_state_path, resume_ckpt_path)
        self._require_role_state(trainer_state, trainer_state_path, distributed=True)
        if not os.path.isdir(dist_state_path):
            raise RuntimeError(f"Checkpoint is missing distributed state: {dist_state_path}")
        roles = dict(self._active_role_runtimes())
        for role in roles.keys() - {"student", "fake"}:
            role_path = os.path.join(dist_state_path, role)
            if not os.path.isdir(role_path):
                raise RuntimeError(f"Checkpoint is missing {role} distributed state: {role_path}")

        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        state = {}
        for role in ("student", "fake"):
            runtime = roles[role]
            model_state, optimizer_state = get_state_dict(self._parallel(runtime.model).state_module(), runtime.optimizer, options=options)
            state[f"{role}_model"] = model_state
            state[f"{role}_optimizer"] = optimizer_state
        dcp.load(state, checkpoint_id=dist_state_path)
        for role in ("student", "fake"):
            runtime = roles[role]
            set_state_dict(self._parallel(runtime.model).state_module(), runtime.optimizer, model_state_dict=state[f"{role}_model"], optim_state_dict=state[f"{role}_optimizer"], options=options)
        for role in roles.keys() - {"student", "fake"}:
            runtime = roles[role]
            module = self._parallel(runtime.model).state_module()
            model_state, optimizer_state = get_state_dict(module, runtime.optimizer, options=options)
            role_state = {"model": model_state, "optimizer": optimizer_state}
            dcp.load(role_state, checkpoint_id=os.path.join(dist_state_path, role))
            set_state_dict(module, runtime.optimizer, model_state_dict=role_state["model"], optim_state_dict=role_state["optimizer"], options=options)
        for _, runtime in roles.items():
            runtime.scheduler.load_state_dict(trainer_state[runtime.spec.scheduler_attribute])
        logger.info("Restored distributed DMD training state from {}", resume_ckpt_path)

    def save_checkpoint(self, iteration, save_total_limit):
        if is_main_process():
            prune_checkpoints(self.output_train_dir, save_total_limit)

        save_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        active_roles = ["student", "fake"]
        if getattr(self, "fake_real_model", None) is not None:
            active_roles.append("fake_real")
        logger.info(
            "[checkpoint][save][start] iteration={} path={} roles={}",
            iteration,
            save_dir,
            active_roles,
        )
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        barrier()

        save_student_weights = self.student_train_type == "lora" or not self.parallel.is_fsdp()
        if save_student_weights:
            self._save_model_weights(self.model, save_dir, role="student")
        barrier()

        fake_save_dir = self._fake_weights_dir(save_dir)
        fake_parallel = self._parallel(self.fake_model)
        save_fake_weights = self.fake_train_type == "lora" or not fake_parallel.is_fsdp()
        if save_fake_weights and is_main_process():
            os.makedirs(fake_save_dir, exist_ok=True)
        barrier()
        if save_fake_weights:
            self._save_model_weights(self.fake_model, fake_save_dir, role="fake")
        barrier()
        if getattr(self, "fake_real_model", None) is not None:
            role = "fake_real"
            fake_real_save_dir = os.path.join(
                save_dir,
                self.role_registry.weight_directory_name(role),
            )
            save_fake_real_weights = self.fake_real_train_type == "lora" or not self._parallel(self.fake_real_model).is_fsdp()
            if save_fake_real_weights and is_main_process():
                os.makedirs(fake_real_save_dir, exist_ok=True)
            barrier()
            if save_fake_real_weights:
                self._save_model_weights(
                    self.fake_real_model,
                    fake_real_save_dir,
                    role=role,
                )
            barrier()
            logger.info(
                "[checkpoint][save][role] role={} path={} weights={}",
                role,
                fake_real_save_dir,
                save_fake_real_weights,
            )

        config_path = self.config.get("config_path")
        if is_main_process() and config_path is not None:
            shutil.copy2(config_path, os.path.join(save_dir, "config.yaml"))

        if self.parallel.is_fsdp() or fake_parallel.is_fsdp():
            self._save_distributed_state(save_dir, iteration)
            if self._should_save_consolidated_student():
                self._save_consolidated_student_weights(save_dir)
            barrier()
            logger.info("[train] saved checkpoint iter={} path={}", iteration, save_dir)
            logger.info(
                "[checkpoint][save][done] iteration={} path={} roles={}",
                iteration,
                save_dir,
                active_roles,
            )
            return

        training_state = {
            "iteration": iteration,
            "world_size": get_world_size(),
            "dmd_checkpoint_version": 2,
            "student_train_type": self.student_train_type,
            "fake_train_type": self.fake_train_type,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "fake_optimizer": self.fake_optimizer.state_dict(),
            "fake_lr_scheduler": self.fake_lr_scheduler.state_dict(),
        }
        if getattr(self, "fake_real_optimizer", None) is not None:
            training_state["fake_real_train_type"] = self.fake_real_train_type
            training_state["fake_real_optimizer"] = self.fake_real_optimizer.state_dict()
            training_state["fake_real_lr_scheduler"] = self.fake_real_lr_scheduler.state_dict()
        training_state.update(self._trick_checkpoint_metadata())
        if is_main_process():
            torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
        barrier()
        logger.info("[train] saved checkpoint iter={} path={}", iteration, save_dir)
        logger.info(
            "[checkpoint][save][done] iteration={} path={} roles={}",
            iteration,
            save_dir,
            active_roles,
        )

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
        self._checkpoint(self.model).save_full_model(output_dir)
        barrier()

    def _save_distributed_state(self, save_dir, iteration):
        dist_state_path = os.path.join(save_dir, "dist_state")
        trainer_state = {
            "iteration": iteration,
            "world_size": get_world_size(),
            "dmd_checkpoint_version": 2,
            "student_train_type": self.student_train_type,
            "fake_train_type": self.fake_train_type,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "fake_lr_scheduler": self.fake_lr_scheduler.state_dict(),
        }
        if getattr(self, "fake_real_lr_scheduler", None) is not None:
            trainer_state["fake_real_train_type"] = self.fake_real_train_type
            trainer_state["fake_real_lr_scheduler"] = self.fake_real_lr_scheduler.state_dict()
        trainer_state.update(self._trick_checkpoint_metadata())
        if is_main_process():
            os.makedirs(dist_state_path, exist_ok=True)
            torch.save(
                trainer_state,
                os.path.join(save_dir, "trainer_state.pt"),
            )
        barrier()

        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        student_model_state, student_optim_state = get_state_dict(
            self.parallel.state_module(),
            self.optimizer,
            options=options,
        )
        fake_model_state, fake_optim_state = get_state_dict(
            self._parallel(self.fake_model).state_module(),
            self.fake_optimizer,
            options=options,
        )
        state = {
            "student_model": student_model_state,
            "student_optimizer": student_optim_state,
            "fake_model": fake_model_state,
            "fake_optimizer": fake_optim_state,
        }
        dcp.save(state, checkpoint_id=dist_state_path)
        if getattr(self, "fake_real_model", None) is not None:
            role_path = os.path.join(dist_state_path, "fake_real")
            model_state, optimizer_state = get_state_dict(
                self._parallel(self.fake_real_model).state_module(),
                self.fake_real_optimizer,
                options=options,
            )
            logger.debug(
                "[checkpoint][save][role] role=fake_real path={} status=writing",
                role_path,
            )
            dcp.save(
                {
                    "model": model_state,
                    "optimizer": optimizer_state,
                },
                checkpoint_id=role_path,
            )
            logger.info(
                "[checkpoint][save][role] role=fake_real path={} status=restorable",
                role_path,
            )
