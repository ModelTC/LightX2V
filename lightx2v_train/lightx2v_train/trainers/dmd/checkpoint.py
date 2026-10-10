import os
import shutil
from collections.abc import Mapping
from datetime import timedelta

import torch
import torch.distributed as dist
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

    def _get_checkpoint_process_group(self):
        """Use CPU collectives for DCP plans, without changing tensor shards.

        All ranks must enter this method in the same order on first save/load.
        Keep the group on the trainer and reuse it, as phased DMD does; global
        distributed cleanup also destroys this subgroup. Training/FSDP still
        uses its original NCCL groups.
        """
        if not dist.is_available() or not dist.is_initialized():
            return None
        group = getattr(self, "_checkpoint_process_group", None)
        if group is None:
            if not dist.is_gloo_available():
                raise RuntimeError("DMD distributed checkpoints require a PyTorch build with Gloo support.")
            timeout_minutes = self.config.get("distributed", {}).get("timeout_minutes", 10)
            group = dist.new_group(backend="gloo", timeout=timedelta(minutes=timeout_minutes))
            self._checkpoint_process_group = group
            logger.info("[checkpoint] DCP save/load communication uses Gloo (timeout={} minutes); training groups unchanged", timeout_minutes)
        return group

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
        metadata.update(self._extra_checkpoint_metadata())
        return metadata

    def _extra_checkpoint_metadata(self):
        metadata = dict(self.student.extra_checkpoint_metadata())
        metadata["student_ema_config"] = getattr(self, "student_ema_config", {"enabled": False})
        residual_head = getattr(self, "residual_head", None)
        metadata["residual_head_enabled"] = residual_head is not None
        if getattr(self, "trainer_name", None) == "dmd":
            metadata["dmd_update_order"] = getattr(self, "dmd_update_order", "student_first")
            metadata["dmd_model_mode"] = getattr(self, "dmd_model_mode", "eval")
        if residual_head is not None:
            metadata["residual_head_config"] = residual_head.checkpoint_metadata()
        sampler = getattr(self.dataloader_train, "sampler", None)
        if getattr(sampler, "is_minimax_h3_ref_cost_sampler", False) or getattr(sampler, "is_minimax_h3_task_cycle_sampler", False):
            metadata["minimax_h3_route_sampling"] = sampler.checkpoint_metadata(
                gradient_accumulation_iters=self.gradient_accumulation_iters,
                fake_update_ratio=self.fake_update_ratio,
            )
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
        self._validate_residual_head_state(state, state_path)
        self._validate_student_ema_state(state, state_path)
        expected = {self.checkpoint_version_key: self.checkpoint_version}
        for role, runtime in self._active_role_runtimes():
            expected[f"{role}_train_type"] = runtime.train_type
        expected.update(self._trick_checkpoint_metadata())
        extra_metadata = self._extra_checkpoint_metadata()
        legacy_metadata = self.student.legacy_extra_checkpoint_metadata()
        if getattr(self, "trainer_name", None) == "dmd":
            legacy_metadata = {**legacy_metadata, "dmd_update_order": "student_first", "dmd_model_mode": "eval"}
        missing = set(expected) - state.keys() - legacy_metadata.keys() - extra_metadata.keys()
        self._require_checkpoint_keys(state, missing, state_path)
        allow_transition = bool(self.config.get("resume", {}).get("allow_distribution_matching_transition", False))
        self._validate_h3_checkpoint_metadata(state, extra_metadata, state_path, allow_transition)
        for key, value in expected.items():
            if key in state:
                saved = state[key]
            elif key in legacy_metadata:
                saved = legacy_metadata[key]
                logger.warning("[checkpoint][resume] legacy checkpoint assumes {}={}: {}", key, saved, state_path)
            else:
                logger.warning("[checkpoint][resume] legacy checkpoint has no {} metadata; current setting={} will be saved in the next checkpoint: {}", key, value, state_path)
                continue
            if key == "residual_head_config":
                saved = self._normalized_residual_head_config(saved)
                value = self._normalized_residual_head_config(value)
            if saved != value:
                if key in extra_metadata and allow_transition and key != "residual_head_config":
                    logger.warning("[checkpoint][resume] explicitly changing {} from {} to {}: {}", key, saved, value, state_path)
                    continue
                message = f"Checkpoint {key}={saved!r} does not match the current value {value!r}: {state_path}"
                if key == "residual_head_config":
                    message += "; use a fresh output directory when changing the residual-head recipe."
                elif key in extra_metadata:
                    message += "; use a fresh output directory or set resume.allow_distribution_matching_transition=true for an intentional change."
                raise RuntimeError(message)

    @staticmethod
    def _normalized_residual_head_config(metadata):
        """Recognize only the two defaults absent from pre-calibration runs.

        This is not a general configuration migration: all other missing,
        extra, or mismatching fields remain strict. In particular, a legacy
        full/accumulation=1 checkpoint must not resume as calibrated/accum=4,
        even with the distribution-matching transition escape hatch enabled.
        Never modify the supplied checkpoint or runtime metadata mapping.
        """
        if not isinstance(metadata, Mapping):
            return metadata
        return {"fit_grad_accum_steps": 1, "gate_mode": "full", **metadata}

    def _validate_residual_head_state(self, state, state_path):
        residual_head = getattr(self, "residual_head", None)
        enabled = residual_head is not None
        saved_enabled = state.get("residual_head_enabled", False)
        if saved_enabled != enabled:
            raise RuntimeError(f"Checkpoint residual_head_enabled={saved_enabled!r} does not match current {enabled!r}: {state_path}; head/non-head experiments require separate output directories.")
        if enabled:
            self._require_checkpoint_keys(state, ["residual_head_config", "residual_head_state"], state_path)
            current = residual_head.checkpoint_metadata()
            if self._normalized_residual_head_config(state["residual_head_config"]) != self._normalized_residual_head_config(current):
                raise RuntimeError(f"Checkpoint residual_head_config={state['residual_head_config']!r} does not match current {current!r}: {state_path}")

    def _validate_student_ema_state(self, state, state_path):
        current = getattr(self, "student_ema_config", {"enabled": False})
        saved = state.get("student_ema_config", {"enabled": False})
        if saved != current:
            raise RuntimeError(f"Checkpoint student_ema_config={saved!r} does not match current {current!r}: {state_path}; use a fresh output directory when changing the EMA recipe.")
        if current["enabled"]:
            self._require_checkpoint_keys(state, ["student_ema"], state_path)

    def _extra_residual_head_training_state(self):
        residual_head = getattr(self, "residual_head", None)
        return {} if residual_head is None else {"residual_head_state": residual_head.state_dict()}

    def _load_residual_head_training_state(self, state):
        residual_head = getattr(self, "residual_head", None)
        if residual_head is not None:
            residual_head.load_state_dict(state["residual_head_state"])

    @staticmethod
    def _validate_h3_checkpoint_metadata(state, current, state_path, allow_transition):
        """Keep H3 topology strict and objective transitions explicit."""
        topology = current.get("minimax_h3_parallel_topology")
        if topology is None:
            return
        saved_topology = state.get("minimax_h3_parallel_topology")
        if saved_topology is None:
            if topology["sequence_parallel_size"] > 1:
                raise RuntimeError(f"Cannot resume a checkpoint without MiniMax-H3 parallel-topology metadata using sequence parallelism: {state_path}")
        elif saved_topology != topology:
            raise RuntimeError(f"Checkpoint minimax_h3_parallel_topology={saved_topology!r} does not match the current value {topology!r}: {state_path}")

        geometry = current["minimax_h3_target_geometry"]
        requires_transition = []
        if "minimax_h3_target_geometry" not in state and geometry["fixed_num_frames"] is not None:
            requires_transition.append("minimax_h3_target_geometry")
        sampler = current.get("minimax_h3_route_sampling")
        if sampler is not None and "minimax_h3_route_sampling" not in state and sampler["route_mode"] != "homogeneous":
            requires_transition.append("minimax_h3_route_sampling")
        if sampler is None and "minimax_h3_route_sampling" in state:
            requires_transition.append("minimax_h3_route_sampling")
        saved_sla = state.get("student_sparse_attention")
        current_sla = current.get("student_sparse_attention")
        if saved_sla is None and current_sla is not None and current_sla.get("enabled", False):
            requires_transition.append("student_sparse_attention")
        if saved_sla is not None and saved_sla.get("enabled", False) and "student_sparse_attention" not in current:
            requires_transition.append("student_sparse_attention")
        for key in requires_transition:
            saved, value = state.get(key), current.get(key)
            if not allow_transition:
                raise RuntimeError(
                    f"Checkpoint {key}={saved!r} cannot silently transition to {value!r}: {state_path}; "
                    "use a fresh output directory or set resume.allow_distribution_matching_transition=true for an intentional change."
                )
            logger.warning("[checkpoint][resume] explicitly changing {} from {} to {}: {}", key, saved, value, state_path)

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
        self.student.load_extra_training_state(state)
        self._load_residual_head_training_state(state)
        if getattr(self, "student_ema", None) is not None:
            self.student_ema.load_state_dict(state["student_ema"])
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

        checkpoint_group = self._get_checkpoint_process_group()
        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        state = {}
        for role in ("student", "fake"):
            runtime = roles[role]
            model_state, optimizer_state = get_state_dict(self._parallel(runtime.model).state_module(), runtime.optimizer, options=options)
            state[f"{role}_model"] = model_state
            state[f"{role}_optimizer"] = optimizer_state
        if getattr(self, "student_ema", None) is not None:
            state["student_ema"] = self.student_ema.shadow
        dcp.load(state, checkpoint_id=dist_state_path, process_group=checkpoint_group)
        for role in ("student", "fake"):
            runtime = roles[role]
            set_state_dict(self._parallel(runtime.model).state_module(), runtime.optimizer, model_state_dict=state[f"{role}_model"], optim_state_dict=state[f"{role}_optimizer"], options=options)
        for role in roles.keys() - {"student", "fake"}:
            runtime = roles[role]
            module = self._parallel(runtime.model).state_module()
            model_state, optimizer_state = get_state_dict(module, runtime.optimizer, options=options)
            role_state = {"model": model_state, "optimizer": optimizer_state}
            dcp.load(role_state, checkpoint_id=os.path.join(dist_state_path, role), process_group=checkpoint_group)
            set_state_dict(module, runtime.optimizer, model_state_dict=role_state["model"], optim_state_dict=role_state["optimizer"], options=options)
        for _, runtime in roles.items():
            runtime.scheduler.load_state_dict(trainer_state[runtime.spec.scheduler_attribute])
        self.student.load_extra_training_state(trainer_state)
        self._load_residual_head_training_state(trainer_state)
        if getattr(self, "student_ema", None) is not None:
            self.student_ema.load_state_dict(
                {
                    **trainer_state["student_ema"],
                    "shadow": state["student_ema"],
                }
            )
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
            if getattr(self, "student_ema", None) is not None:
                ema_dir = os.path.join(save_dir, "student_ema")
                with self.student_ema.average_parameters():
                    self._save_model_weights(self.model, ema_dir, role="student")
                logger.info("[checkpoint][save] student EMA weights path={}", ema_dir)
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
        training_state.update(self.student.extra_training_state())
        training_state.update(self._extra_residual_head_training_state())
        if getattr(self, "student_ema", None) is not None:
            training_state["student_ema"] = self.student_ema.state_dict()
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
        trainer_state.update(self.student.extra_training_state())
        trainer_state.update(self._extra_residual_head_training_state())
        if getattr(self, "student_ema", None) is not None:
            trainer_state["student_ema"] = {
                "decay": self.student_ema.decay,
                "num_updates": self.student_ema.num_updates,
            }
        if is_main_process():
            os.makedirs(dist_state_path, exist_ok=True)
            torch.save(
                trainer_state,
                os.path.join(save_dir, "trainer_state.pt"),
            )
        barrier()

        checkpoint_group = self._get_checkpoint_process_group()
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
        if getattr(self, "student_ema", None) is not None:
            state["student_ema"] = self.student_ema.shadow
        dcp.save(state, checkpoint_id=dist_state_path, process_group=checkpoint_group)
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
                process_group=checkpoint_group,
            )
            logger.info(
                "[checkpoint][save][role] role=fake_real path={} status=restorable",
                role_path,
            )
