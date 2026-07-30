import os
import shutil
import warnings

import torch
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)

from lightx2v_train.runtime.checkpoint import prune_checkpoints
from lightx2v_train.runtime.distributed import (
    barrier,
    get_world_size,
    is_main_process,
)


class DmdCheckpointManager:
    """Coordinate DMD checkpoint I/O without owning trainer resources."""

    def __init__(self, owner):
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value):
        setattr(self.owner, name, value)

    def _fake_weights_dir(self, root_dir):
        directory_name = self.role_registry.weight_directory_name("fake")
        return os.path.join(root_dir, directory_name)

    def _copy_role_model(self, source_role, target_role):
        source = self.role_registry.runtime(source_role).model
        target = self.role_registry.runtime(target_role).model
        source_module = source.fsdp2_state_module()
        target_module = target.fsdp2_state_module()
        if source.is_fsdp2_wrapped():
            options = StateDictOptions(
                ignore_frozen_params=False,
                strict=True,
            )
            state = get_model_state_dict(
                source_module,
                options=options,
            )
            set_model_state_dict(
                target_module,
                model_state_dict=state,
                options=options,
            )
            del state
        else:
            target_module.load_state_dict(
                source_module.state_dict(),
                strict=True,
            )
        logger.warning(
            "[checkpoint][resume][fallback] role={} model=copied source={} reason=missing_in_checkpoint",
            target_role,
            source_role,
        )

    def _fast_forward_scheduler(self, role, iteration):
        scheduler = self.role_registry.runtime(role).scheduler
        completed_steps = int(iteration) * self.fake_update_ratio
        if scheduler is None or completed_steps <= 0:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scheduler.step(completed_steps)
        logger.warning(
            "[checkpoint][resume][fallback] role={} scheduler=fast_forward step={} reason=missing_in_checkpoint",
            role,
            completed_steps,
        )

    def _log_disabled_roles(self, roles):
        for role in roles:
            runtime = self.role_registry.runtime(role)
            if runtime.model is None:
                logger.info(
                    "[checkpoint][resume][role] role={} status=disabled",
                    role,
                )

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

    def _validate_optional_trick_metadata(self, state, state_path):
        for name, enabled_key in (
            ("ida_trick", "ida_enabled"),
            ("diversity_trick", "div_loss_enabled"),
        ):
            trick = getattr(self, name, None)
            if trick is None:
                continue
            expected = trick.checkpoint_metadata()
            if enabled_key not in state:
                if trick.enabled:
                    logger.warning(
                        "Checkpoint has no {} metadata; assuming the current configuration is compatible: {}",
                        trick.name,
                        state_path,
                    )
                continue
            checkpoint_enabled = bool(state[enabled_key])
            if checkpoint_enabled != trick.enabled:
                raise RuntimeError(f"Checkpoint {enabled_key}={checkpoint_enabled!r} does not match the current value {trick.enabled!r}: {state_path}")
            if not trick.enabled:
                continue
            for key, value in expected.items():
                if key == enabled_key:
                    continue
                if key in state and state[key] != value:
                    raise RuntimeError(f"Checkpoint {key}={state[key]!r} does not match the current value {value!r}: {state_path}")

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
        self._validate_optional_trick_metadata(state, state_path)
        trick = getattr(self, "real_data_fake_trick", None)
        if trick is not None:
            checkpoint_enabled = bool(state.get("fake_real_enabled", False))
            current_enabled = trick.enabled_for("main")
            checkpoint_fake_real_train_type = state.get("fake_real_train_type")
            if checkpoint_fake_real_train_type is not None and checkpoint_fake_real_train_type != self.fake_real_train_type:
                raise RuntimeError(f"Cannot resume checkpoint with fake_real_train_type={checkpoint_fake_real_train_type!r}, expected {self.fake_real_train_type!r}: {state_path}")
            if checkpoint_enabled and not current_enabled:
                raise RuntimeError(f"Cannot resume a fake_real-enabled checkpoint with fake_real disabled: {state_path}")
            if current_enabled and not checkpoint_enabled:
                logger.warning(
                    "Checkpoint has fake_real disabled; enabling it from the current configuration: {}",
                    state_path,
                )
            if checkpoint_enabled and current_enabled:
                checkpoint_timesteps = state.get("fake_real_timestep_list")
                current_timesteps = list(trick.config.regions["main"].timestep_list)
                if checkpoint_timesteps is not None and list(checkpoint_timesteps) != current_timesteps:
                    raise RuntimeError(f"Cannot resume fake_real with a different timestep_list: {state_path}")

    def _load_single_process_state(self, resume_ckpt_path):
        training_state_path = os.path.join(resume_ckpt_path, "training_state.pt")
        fake_weights_dir = self._fake_weights_dir(resume_ckpt_path)

        if not os.path.exists(training_state_path):
            raise RuntimeError(f"training_state.pt not found in {resume_ckpt_path}")

        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        logger.info(
            "[checkpoint][resume][start] path={} mode=single iteration={} roles={}",
            resume_ckpt_path,
            state.get("iteration"),
            list(self.role_registry.names()),
        )
        self._log_disabled_roles(("fake_real",))
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
        if getattr(self, "fake_real_model", None) is not None:
            role = "fake_real"
            weights_dir = os.path.join(
                resume_ckpt_path,
                self.role_registry.weight_directory_name(role),
            )
            restored = os.path.exists(weights_dir)
            if not restored and int(state.get("dmd_checkpoint_version", 1)) >= 2 and bool(state.get("fake_real_enabled", False)):
                logger.error(
                    "[checkpoint][resume][error] role={} component=model path={} reason=missing",
                    role,
                    weights_dir,
                )
                raise RuntimeError(f"Checkpoint declares an independent fake_real role, but its weights are missing: {weights_dir}")
            if restored:
                self._load_model_weights(
                    self.fake_real_model,
                    weights_dir,
                    role=role,
                )
            else:
                self._copy_role_model("fake", role)
            if restored and "fake_real_optimizer" in state:
                self.fake_real_optimizer.load_state_dict(state["fake_real_optimizer"])
                optimizer_status = "restored"
            else:
                optimizer_status = "fresh"
                logger.warning(
                    "[checkpoint][resume][fallback] role={} optimizer=fresh reason=missing_in_checkpoint",
                    role,
                )
            if restored and "fake_real_lr_scheduler" in state:
                self.fake_real_lr_scheduler.load_state_dict(state["fake_real_lr_scheduler"])
                scheduler_status = "restored"
            else:
                self._fast_forward_scheduler(
                    role,
                    state["iteration"],
                )
                scheduler_status = "fast_forward"
            logger.info(
                "[checkpoint][resume][role] role={} model={} optimizer={} scheduler={} path={}",
                role,
                "restored" if restored else "copied_from:fake",
                optimizer_status,
                scheduler_status,
                weights_dir,
            )
        logger.info("Restored DMD training state from {}", training_state_path)
        logger.info(
            "[checkpoint][resume][done] path={} iteration={}",
            resume_ckpt_path,
            state["iteration"],
        )

    def _load_distributed_state(self, resume_ckpt_path):
        dist_state_path = os.path.join(resume_ckpt_path, "dist_state")
        if not os.path.exists(dist_state_path):
            raise RuntimeError(f"FSDP2 resume requires dist_state/, but it was not found in {resume_ckpt_path}")

        trainer_state_path = os.path.join(resume_ckpt_path, "trainer_state.pt")
        if not os.path.exists(trainer_state_path):
            raise RuntimeError(f"trainer_state.pt not found in {resume_ckpt_path}")
        trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        logger.info(
            "[checkpoint][resume][start] path={} mode=fsdp layout=flat_with_optional_roles iteration={} roles={}",
            resume_ckpt_path,
            trainer_state.get("iteration"),
            list(self.role_registry.names()),
        )
        self._log_disabled_roles(("fake_real",))
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
        if getattr(self, "fake_real_model", None) is not None:
            role = "fake_real"
            role_path = os.path.join(dist_state_path, role)
            restored = os.path.isdir(role_path)
            if not restored and int(trainer_state.get("dmd_checkpoint_version", 1)) >= 2 and bool(trainer_state.get("fake_real_enabled", False)):
                logger.error(
                    "[checkpoint][resume][error] role={} component=model path={} reason=missing",
                    role,
                    role_path,
                )
                raise RuntimeError(f"Checkpoint declares an independent fake_real role, but its distributed state is missing: {role_path}")
            if restored:
                model_state, optimizer_state = get_state_dict(
                    self.fake_real_model.fsdp2_state_module(),
                    self.fake_real_optimizer,
                    options=options,
                )
                role_state = {
                    "model": model_state,
                    "optimizer": optimizer_state,
                }
                dcp.load(role_state, checkpoint_id=role_path)
                set_state_dict(
                    self.fake_real_model.fsdp2_state_module(),
                    self.fake_real_optimizer,
                    model_state_dict=role_state["model"],
                    optim_state_dict=role_state["optimizer"],
                    options=options,
                )
                optimizer_status = "restored"
            else:
                self._copy_role_model("fake", role)
                optimizer_status = "fresh"
            if restored and "fake_real_lr_scheduler" in trainer_state:
                self.fake_real_lr_scheduler.load_state_dict(trainer_state["fake_real_lr_scheduler"])
                scheduler_status = "restored"
            else:
                self._fast_forward_scheduler(
                    role,
                    trainer_state["iteration"],
                )
                scheduler_status = "fast_forward"
            logger.info(
                "[checkpoint][resume][role] role={} model={} optimizer={} scheduler={} path={}",
                role,
                "restored" if restored else "copied_from:fake",
                optimizer_status,
                scheduler_status,
                role_path,
            )
        logger.info("Restored distributed DMD training state from {}", resume_ckpt_path)
        logger.info(
            "[checkpoint][resume][done] path={} iteration={}",
            resume_ckpt_path,
            trainer_state["iteration"],
        )

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
        if getattr(self, "fake_real_model", None) is not None:
            role = "fake_real"
            fake_real_save_dir = os.path.join(
                save_dir,
                self.role_registry.weight_directory_name(role),
            )
            save_fake_real_weights = self.fake_real_train_type == "lora" or not self.fake_real_model.is_fsdp2_wrapped()
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

        if self.model.is_fsdp2_wrapped() or self.fake_model.is_fsdp2_wrapped():
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
        self.model.save_full_model(output_dir)
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
        student_model_state, student_optim_state = get_state_dict(self.model.fsdp2_state_module(), self.optimizer, options=options)
        fake_model_state, fake_optim_state = get_state_dict(self.fake_model.fsdp2_state_module(), self.fake_optimizer, options=options)
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
                self.fake_real_model.fsdp2_state_module(),
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
