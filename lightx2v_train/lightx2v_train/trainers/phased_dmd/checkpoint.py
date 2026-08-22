import os
import shutil
import warnings

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

    def _role_weights_dir(self, root_dir, role):
        directory_name = self.role_registry.weight_directory_name(role)
        if directory_name is None:
            return root_dir
        return os.path.join(root_dir, directory_name)

    def _copy_fake_low_high_from_fake(self):
        if self.fake_low_high_model is None:
            return
        self._copy_role_model("fake", "fake_low_high")

    def _fast_forward_fake_low_high_scheduler(self, iteration):
        if self.fake_low_high_lr_scheduler is None:
            return
        completed_steps = int(iteration) * self.fake_update_ratio
        if completed_steps <= 0:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fake_low_high_lr_scheduler.step(completed_steps)
        logger.warning(
            "Checkpoint has no fake_low_high LR scheduler state; advanced the fresh scheduler to step {}.",
            completed_steps,
        )

    def _fast_forward_fake_real_scheduler(
        self,
        role,
        iteration,
    ):
        runtime = self.role_registry.runtime(role)
        region_iterations = (int(iteration) + 1) // 2 if role == "fake_real_high" else int(iteration) // 2
        completed_steps = region_iterations * self.fake_update_ratio
        if completed_steps <= 0:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runtime.scheduler.step(completed_steps)
        logger.warning(
            "[checkpoint][resume][fallback] role={} scheduler=fast_forward step={} reason=missing_in_checkpoint",
            role,
            completed_steps,
        )

    def _validate_phased_checkpoint_metadata(
        self,
        state,
        state_path,
        resume_ckpt_path,
    ):
        self._validate_dmd_checkpoint_metadata(
            state,
            state_path,
            resume_ckpt_path,
        )
        for role, expected in (
            ("student_2", self.student_2_train_type),
            ("fake_2", self.fake_2_train_type),
        ):
            key = f"{role}_train_type"
            if state.get(key) != expected:
                raise RuntimeError(f"Cannot resume checkpoint with {key}={state.get(key)!r}, expected {expected!r}: {state_path}")
        checkpoint_version = int(state.get("phased_checkpoint_version", 1))
        checkpoint_fake_low_high_enabled = bool(state.get("fake_low_high_enabled", False))
        if checkpoint_fake_low_high_enabled and not self.enable_fake_low_high:
            logger.warning(
                "Checkpoint contains fake_low_high, but it is disabled in the current configuration and will be ignored: {}",
                state_path,
            )
        if checkpoint_version >= 2 and checkpoint_fake_low_high_enabled and self.enable_fake_low_high and state.get("fake_low_high_train_type") != self.fake_low_high_train_type:
            raise RuntimeError(f"Cannot resume checkpoint with fake_low_high_train_type={state.get('fake_low_high_train_type')!r}, expected {self.fake_low_high_train_type!r}: {state_path}")
        for region, role in (
            ("high", "fake_real_high"),
            ("low", "fake_real_low"),
        ):
            checkpoint_enabled = bool(state.get(f"{role}_enabled", False))
            config = self.real_data_fake_trick.config.regions[region]
            current_enabled = config.enabled
            train_type_key = f"{role}_train_type"
            expected_train_type = getattr(
                self,
                train_type_key,
            )
            if checkpoint_enabled and state.get(train_type_key) is not None and state[train_type_key] != expected_train_type:
                raise RuntimeError(f"Cannot resume checkpoint with {train_type_key}={state[train_type_key]!r}, expected {expected_train_type!r}: {state_path}")
            if checkpoint_enabled and not current_enabled:
                raise RuntimeError(f"Cannot resume a {role}-enabled checkpoint with {role} disabled: {state_path}")
            if current_enabled and not checkpoint_enabled:
                logger.warning(
                    "Checkpoint has {} disabled; enabling it from the current configuration: {}",
                    role,
                    state_path,
                )
            if checkpoint_enabled and list(state.get(f"{role}_timestep_list", [])) != list(config.timestep_list):
                raise RuntimeError(f"Checkpoint {role}_timestep_list does not match {list(config.timestep_list)!r}: {state_path}")
        if int(state.get("phased_match_timestep", -1)) != (self.match_timestep):
            raise RuntimeError(f"Checkpoint phased_match_timestep does not match {self.match_timestep}: {state_path}")

    def _load_resume_state(self, resume_ckpt_path):
        models = tuple(model for _, model in self._trainable_role_models())
        if any(self._parallel(model).is_fsdp() for model in models):
            self._load_distributed_state(resume_ckpt_path)
            return
        self._load_single_process_state(resume_ckpt_path)

    def _load_single_process_state(self, resume_ckpt_path):
        state_path = os.path.join(
            resume_ckpt_path,
            "training_state.pt",
        )
        if not os.path.exists(state_path):
            raise RuntimeError(f"training_state.pt not found in {resume_ckpt_path}")
        state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )
        logger.info(
            "[checkpoint][resume][start] path={} mode=single layout=per_role version={} iteration={} roles={}",
            resume_ckpt_path,
            state.get("phased_checkpoint_version", 1),
            state.get("iteration"),
            list(self.role_registry.trainable_names()),
        )
        self._log_disabled_roles(
            (
                "fake_low_high",
                "fake_real_high",
                "fake_real_low",
            )
        )
        self._validate_phased_checkpoint_metadata(
            state,
            state_path,
            resume_ckpt_path,
        )

        role_models = self._trainable_role_models()
        for role, model in role_models:
            if role in {
                "fake_low_high",
                "fake_real_high",
                "fake_real_low",
            }:
                continue
            self._load_model_weights(
                model,
                self._role_weights_dir(resume_ckpt_path, role),
                role=role,
            )
            logger.info(
                "[checkpoint][resume][role] role={} model=restored path={}",
                role,
                self._role_weights_dir(resume_ckpt_path, role),
            )
        restored_fake_low_high = False
        if self.fake_low_high_model is not None:
            checkpoint_requires_fake_low_high = int(state.get("phased_checkpoint_version", 1)) >= 2 and bool(state.get("fake_low_high_enabled", False))
            weights_dir = self._role_weights_dir(
                resume_ckpt_path,
                "fake_low_high",
            )
            if os.path.exists(weights_dir):
                self._load_model_weights(
                    self.fake_low_high_model,
                    weights_dir,
                    role="fake_low_high",
                )
                restored_fake_low_high = True
            elif checkpoint_requires_fake_low_high:
                raise RuntimeError(f"Checkpoint metadata enables fake_low_high, but its weights are missing: {weights_dir}")
            else:
                self._copy_fake_low_high_from_fake()
        restored_fake_real = {role: False for role in ("fake_real_high", "fake_real_low") if self.role_registry.runtime(role).model is not None}
        for role, source_role in (
            ("fake_real_high", "fake"),
            ("fake_real_low", "fake_2"),
        ):
            runtime = self.role_registry.runtime(role)
            if runtime.model is None:
                continue
            weights_dir = self._role_weights_dir(
                resume_ckpt_path,
                role,
            )
            restored = os.path.exists(weights_dir)
            if not restored and int(state.get("phased_checkpoint_version", 1)) >= 3 and bool(state.get(f"{role}_enabled", False)):
                logger.error(
                    "[checkpoint][resume][error] role={} component=model path={} reason=missing",
                    role,
                    weights_dir,
                )
                raise RuntimeError(f"Checkpoint enables {role}, but its weights are missing: {weights_dir}")
            if restored:
                self._load_model_weights(
                    runtime.model,
                    weights_dir,
                    role=role,
                )
            else:
                self._copy_role_model(source_role, role)
            restored_fake_real[role] = restored
        state_targets = [
            ("optimizer", self.optimizer),
            ("fake_optimizer", self.fake_optimizer),
            ("student_2_optimizer", self.student_2_optimizer),
            ("fake_2_optimizer", self.fake_2_optimizer),
            ("lr_scheduler", self.lr_scheduler),
            ("fake_lr_scheduler", self.fake_lr_scheduler),
            ("student_2_lr_scheduler", self.student_2_lr_scheduler),
            ("fake_2_lr_scheduler", self.fake_2_lr_scheduler),
        ]
        for key, target in state_targets:
            target.load_state_dict(state[key])
        if self.fake_low_high_optimizer is not None:
            optimizer_key = "fake_low_high_optimizer"
            scheduler_key = "fake_low_high_lr_scheduler"
            if restored_fake_low_high and optimizer_key in state:
                self.fake_low_high_optimizer.load_state_dict(state[optimizer_key])
            elif restored_fake_low_high:
                raise RuntimeError(f"Checkpoint has fake_low_high weights but no {optimizer_key}: {state_path}")
            else:
                logger.warning(
                    "Using a fresh fake_low_high optimizer while resuming from {}.",
                    resume_ckpt_path,
                )
            if restored_fake_low_high and scheduler_key in state:
                self.fake_low_high_lr_scheduler.load_state_dict(state[scheduler_key])
            elif restored_fake_low_high:
                raise RuntimeError(f"Checkpoint has fake_low_high weights but no {scheduler_key}: {state_path}")
            else:
                self._fast_forward_fake_low_high_scheduler(state["iteration"])
        for role, restored in restored_fake_real.items():
            runtime = self.role_registry.runtime(role)
            optimizer_key = f"{role}_optimizer"
            scheduler_key = f"{role}_lr_scheduler"
            if restored and optimizer_key in state:
                runtime.optimizer.load_state_dict(state[optimizer_key])
                optimizer_status = "restored"
            else:
                optimizer_status = "fresh"
                logger.warning(
                    "[checkpoint][resume][fallback] role={} optimizer=fresh reason=missing_in_checkpoint",
                    role,
                )
            if restored and scheduler_key in state:
                runtime.scheduler.load_state_dict(state[scheduler_key])
                scheduler_status = "restored"
            else:
                self._fast_forward_fake_real_scheduler(
                    role,
                    state["iteration"],
                )
                scheduler_status = "fast_forward"
            logger.info(
                "[checkpoint][resume][role] role={} model={} optimizer={} scheduler={} path={}",
                role,
                ("restored" if restored else "copied_from:" + ("fake" if role == "fake_real_high" else "fake_2")),
                optimizer_status,
                scheduler_status,
                self._role_weights_dir(resume_ckpt_path, role),
            )
        logger.info(
            "Restored single-process phased DMD state from {}",
            resume_ckpt_path,
        )
        logger.info(
            "[checkpoint][resume][done] path={} iteration={} roles={}",
            resume_ckpt_path,
            state["iteration"],
            list(self.role_registry.trainable_names()),
        )

    def _get_checkpoint_process_group(self):
        if self._checkpoint_process_group is None:
            self._checkpoint_process_group = dist.new_group(
                backend="gloo",
            )
        return self._checkpoint_process_group

    @staticmethod
    def _checkpoint_role_layout(dist_state_path, required_roles):
        existing_roles = tuple(role for role in required_roles if os.path.isdir(os.path.join(dist_state_path, role)))
        if not existing_roles:
            return "flat"
        missing_roles = [role for role in required_roles if role not in existing_roles]
        if missing_roles:
            raise RuntimeError(f"Distributed phased DMD checkpoint is missing required role directories {missing_roles}: {dist_state_path}")
        return "per_role"

    def _load_distributed_state(self, resume_ckpt_path):
        dist_state_path = os.path.join(resume_ckpt_path, "dist_state")
        trainer_state_path = os.path.join(
            resume_ckpt_path,
            "trainer_state.pt",
        )
        if not os.path.exists(dist_state_path):
            raise RuntimeError(f"dist_state not found in {resume_ckpt_path}")
        if not os.path.exists(trainer_state_path):
            raise RuntimeError(f"trainer_state.pt not found in {resume_ckpt_path}")
        trainer_state = torch.load(
            trainer_state_path,
            map_location="cpu",
            weights_only=False,
        )
        self._validate_phased_checkpoint_metadata(
            trainer_state,
            trainer_state_path,
            resume_ckpt_path,
        )

        options = StateDictOptions(
            ignore_frozen_params=True,
            strict=False,
        )
        role_states = self._trainable_role_states()
        role_state_by_name = {role: (model, optimizer) for role, model, optimizer in role_states}
        required_roles = (
            "student",
            "fake",
            "student_2",
            "fake_2",
        )
        checkpoint_group = self._get_checkpoint_process_group()
        role_checkpoint_dirs = {role: os.path.join(dist_state_path, role) for role, _, _ in role_states}
        checkpoint_layout = self._checkpoint_role_layout(
            dist_state_path,
            required_roles,
        )
        logger.info(
            "[checkpoint][resume][start] path={} mode=fsdp layout={} version={} iteration={} roles={}",
            resume_ckpt_path,
            checkpoint_layout,
            trainer_state.get("phased_checkpoint_version", 1),
            trainer_state.get("iteration"),
            list(self.role_registry.trainable_names()),
        )
        self._log_disabled_roles(
            (
                "fake_low_high",
                "fake_real_high",
                "fake_real_low",
            )
        )
        if (
            checkpoint_layout == "flat"
            and self.fake_low_high_model is not None
            and int(trainer_state.get("phased_checkpoint_version", 1)) >= 2
            and bool(trainer_state.get("fake_low_high_enabled", False))
        ):
            raise RuntimeError(f"Checkpoint metadata enables fake_low_high, but the distributed checkpoint uses no per-role state: {dist_state_path}")
        if (
            checkpoint_layout == "flat"
            and int(trainer_state.get("phased_checkpoint_version", 1)) >= 3
            and any(
                bool(trainer_state.get(f"{role}_enabled", False))
                for role in (
                    "fake_real_high",
                    "fake_real_low",
                )
            )
        ):
            raise RuntimeError(f"Checkpoint metadata enables independent Fake-Real roles, but the distributed checkpoint uses no per-role state: {dist_state_path}")
        restored_fake_low_high = False
        restored_fake_real = {role: False for role in ("fake_real_high", "fake_real_low") if self.role_registry.runtime(role).model is not None}
        if checkpoint_layout == "per_role":
            roles_to_load = list(required_roles)
            fake_low_high_dir = role_checkpoint_dirs.get("fake_low_high")
            if fake_low_high_dir is not None and os.path.isdir(fake_low_high_dir):
                roles_to_load.append("fake_low_high")
                restored_fake_low_high = True
            elif (
                self.fake_low_high_model is not None
                and int(
                    trainer_state.get(
                        "phased_checkpoint_version",
                        1,
                    )
                )
                >= 2
                and bool(
                    trainer_state.get(
                        "fake_low_high_enabled",
                        False,
                    )
                )
            ):
                raise RuntimeError(f"Checkpoint metadata enables fake_low_high, but its distributed role directory is missing: {fake_low_high_dir}")
            for role in ("fake_real_high", "fake_real_low"):
                runtime = self.role_registry.runtime(role)
                if runtime.model is None:
                    continue
                role_dir = role_checkpoint_dirs.get(role)
                restored = bool(role_dir is not None and os.path.isdir(role_dir))
                if restored:
                    roles_to_load.append(role)
                elif int(
                    trainer_state.get(
                        "phased_checkpoint_version",
                        1,
                    )
                ) >= 3 and bool(
                    trainer_state.get(
                        f"{role}_enabled",
                        False,
                    )
                ):
                    logger.error(
                        "[checkpoint][resume][error] role={} component=model path={} reason=missing",
                        role,
                        role_dir,
                    )
                    raise RuntimeError(f"Checkpoint enables {role}, but its distributed role directory is missing: {role_dir}")
                restored_fake_real[role] = restored
            for role in roles_to_load:
                model, optimizer = role_state_by_name[role]
                logger.info(
                    "[train] loading checkpoint role={}",
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
                dcp.load(
                    role_state,
                    checkpoint_id=role_checkpoint_dirs[role],
                    process_group=checkpoint_group,
                )
                set_state_dict(
                    self._parallel(model).state_module(),
                    optimizer,
                    model_state_dict=role_state["model"],
                    optim_state_dict=role_state["optimizer"],
                    options=options,
                )
                logger.info(
                    "[checkpoint][resume][role] role={} model=restored optimizer=restored path={}",
                    role,
                    role_checkpoint_dirs[role],
                )
                del role_state, model_state, optimizer_state
        else:
            state = {}
            for role in required_roles:
                model, optimizer = role_state_by_name[role]
                model_state, optimizer_state = get_state_dict(
                    self._parallel(model).state_module(),
                    optimizer,
                    options=options,
                )
                state[f"{role}_model"] = model_state
                state[f"{role}_optimizer"] = optimizer_state
            dcp.load(
                state,
                checkpoint_id=dist_state_path,
                process_group=checkpoint_group,
            )
            for role in required_roles:
                model, optimizer = role_state_by_name[role]
                set_state_dict(
                    self._parallel(model).state_module(),
                    optimizer,
                    model_state_dict=state[f"{role}_model"],
                    optim_state_dict=state[f"{role}_optimizer"],
                    options=options,
                )
                logger.info(
                    "[checkpoint][resume][role] role={} model=restored optimizer=restored path={} layout=flat",
                    role,
                    dist_state_path,
                )
            del state
        if self.fake_low_high_model is not None and not restored_fake_low_high:
            self._copy_fake_low_high_from_fake()
        for role, restored in restored_fake_real.items():
            if restored:
                continue
            source_role = "fake" if role == "fake_real_high" else "fake_2"
            self._copy_role_model(source_role, role)
        scheduler_targets = [
            ("lr_scheduler", self.lr_scheduler),
            ("fake_lr_scheduler", self.fake_lr_scheduler),
            ("student_2_lr_scheduler", self.student_2_lr_scheduler),
            ("fake_2_lr_scheduler", self.fake_2_lr_scheduler),
        ]
        for key, target in scheduler_targets:
            target.load_state_dict(trainer_state[key])
        if self.fake_low_high_lr_scheduler is not None:
            scheduler_key = "fake_low_high_lr_scheduler"
            if restored_fake_low_high and scheduler_key in trainer_state:
                self.fake_low_high_lr_scheduler.load_state_dict(trainer_state[scheduler_key])
            elif restored_fake_low_high:
                raise RuntimeError(f"Checkpoint has fake_low_high state but no {scheduler_key}: {trainer_state_path}")
            else:
                self._fast_forward_fake_low_high_scheduler(trainer_state["iteration"])
        if self.fake_low_high_optimizer is not None and not restored_fake_low_high:
            logger.warning(
                "Using a fresh fake_low_high optimizer while resuming from {}.",
                resume_ckpt_path,
            )
        for role, restored in restored_fake_real.items():
            runtime = self.role_registry.runtime(role)
            scheduler_key = f"{role}_lr_scheduler"
            if restored and trainer_state.get(scheduler_key) is not None:
                runtime.scheduler.load_state_dict(trainer_state[scheduler_key])
                scheduler_status = "restored"
            else:
                self._fast_forward_fake_real_scheduler(
                    role,
                    trainer_state["iteration"],
                )
                scheduler_status = "fast_forward"
            logger.info(
                "[checkpoint][resume][role] role={} model={} optimizer={} scheduler={} path={}",
                role,
                ("restored" if restored else "copied_from:" + ("fake" if role == "fake_real_high" else "fake_2")),
                "restored" if restored else "fresh",
                scheduler_status,
                role_checkpoint_dirs.get(role),
            )
        logger.info(
            "Restored distributed phased DMD state from {}",
            resume_ckpt_path,
        )
        logger.info(
            "[checkpoint][resume][done] path={} iteration={} roles={}",
            resume_ckpt_path,
            trainer_state["iteration"],
            list(self.role_registry.trainable_names()),
        )

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
