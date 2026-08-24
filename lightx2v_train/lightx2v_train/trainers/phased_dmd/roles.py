import copy

from loguru import logger

from lightx2v_train.model_capabilities import (
    DistributionMatchingCapability,
    ParallelCapability,
    TrainableModelCapability,
)
from lightx2v_train.model_zoo import build_loaded_model
from lightx2v_train.tricks import IdaModelPair

from ..dmd.roles import (
    DmdRoleRegistry,
    RoleSpec,
)


class PhasedRoleRegistry(DmdRoleRegistry):
    """Extend the DMD role view with Low and Low-High resources."""

    role_specs = DmdRoleRegistry.role_specs + (
        RoleSpec(
            name="student_2",
            model_attribute="student_2_model",
            optimizer_attribute="student_2_optimizer",
            scheduler_attribute="student_2_lr_scheduler",
            train_type_attribute="student_2_train_type",
            weight_directory_full="student_2_model",
            weight_directory_lora="student_2_lora",
        ),
        RoleSpec(
            name="fake_2",
            model_attribute="fake_2_model",
            optimizer_attribute="fake_2_optimizer",
            scheduler_attribute="fake_2_lr_scheduler",
            train_type_attribute="fake_2_train_type",
            weight_directory_full="fake_2_model",
            weight_directory_lora="fake_2_lora",
        ),
        RoleSpec(
            name="fake_low_high",
            model_attribute="fake_low_high_model",
            optimizer_attribute="fake_low_high_optimizer",
            scheduler_attribute="fake_low_high_lr_scheduler",
            train_type_attribute="fake_low_high_train_type",
            weight_directory_full="fake_low_high_model",
            weight_directory_lora="fake_low_high_lora",
            optional=True,
        ),
        RoleSpec(
            name="fake_real_high",
            model_attribute="fake_real_high_model",
            optimizer_attribute="fake_real_high_optimizer",
            scheduler_attribute="fake_real_high_lr_scheduler",
            train_type_attribute="fake_real_high_train_type",
            weight_directory_full="fake_real_high_model",
            weight_directory_lora="fake_real_high_lora",
            optional=True,
        ),
        RoleSpec(
            name="fake_real_low",
            model_attribute="fake_real_low_model",
            optimizer_attribute="fake_real_low_optimizer",
            scheduler_attribute="fake_real_low_lr_scheduler",
            train_type_attribute="fake_real_low_train_type",
            weight_directory_full="fake_real_low_model",
            weight_directory_lora="fake_real_low_lora",
            optional=True,
        ),
    )

    def __setattr__(self, name, value):
        if name in {"owner", "_specs"} or "owner" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        setattr(self.owner, name, value)

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def setup_trainable_model(self, model, role):
        if role == "student_2":
            train_type = self.student_2_train_type
            lora_config = self.student_2_lora_config
        elif role == "fake_2":
            train_type = self.fake_2_train_type
            lora_config = self.fake_2_lora_config
        elif role == "fake_low_high":
            train_type = self.fake_low_high_train_type
            lora_config = self.fake_low_high_lora_config
        elif role == "fake_real_high":
            train_type = self.fake_real_high_train_type
            lora_config = self.fake_real_high_lora_config
        elif role == "fake_real_low":
            train_type = self.fake_real_low_train_type
            lora_config = self.fake_real_low_lora_config
        else:
            raise ValueError(f"Unsupported phased model role: {role}")
        model.ensure_capabilities().require(TrainableModelCapability).configure(train_type, lora_config or {})

    def restore_trainable_model(self, model, role):
        train_type = self.runtime(role).train_type
        model.ensure_capabilities().require(TrainableModelCapability).restore(train_type)

    def role_model_config(self, role):
        excluded_roles = {
            "student",
            "fake",
            "teacher",
            "student_2",
            "fake_2",
            "fake_low_high",
            "fake_real",
            "fake_real_high",
            "fake_real_low",
            "teacher_2",
        }
        base_model_config = {key: copy.deepcopy(value) for key, value in self.model_config.items() if key not in excluded_roles}
        override = self.model_config.get(role, {})
        if not isinstance(override, dict):
            raise ValueError(f"model.{role} must be a mapping.")
        override = copy.deepcopy(override)
        if role == "teacher_2":
            override.pop("share_with_teacher", None)
        role_config = copy.deepcopy(self.config)
        role_config["model"] = base_model_config
        role_config["model"].update(override)
        return role_config

    def ida_model_pairs(self):
        return {
            "high": IdaModelPair(
                student=self.student.denoiser(),
                fake=self.fake.denoiser(),
            ),
            "low": IdaModelPair(
                student=self.student_2.denoiser(),
                fake=self.fake_2.denoiser(),
            ),
        }

    def active_student_state(self, region):
        if region == "high":
            return (
                self.optimizer,
                self.lr_scheduler,
                self.trainable_params,
                self._set_student_gradient_sync,
            )
        return (
            self.student_2_optimizer,
            self.student_2_lr_scheduler,
            self.student_2_trainable_params,
            self._set_student_2_gradient_sync,
        )

    def active_fake_state(self, region):
        if region == "high":
            return (
                self.fake_optimizer,
                self.fake_lr_scheduler,
                self.fake_trainable_params,
                self._set_fake_gradient_sync,
            )
        return (
            self.fake_2_optimizer,
            self.fake_2_lr_scheduler,
            self.fake_2_trainable_params,
            self._set_fake_2_gradient_sync,
        )

    def active_fake_real_state(self, region):
        role = "fake_real_high" if region == "high" else "fake_real_low"
        runtime = self.runtime(role)
        params = getattr(
            self.owner,
            f"{role}_trainable_params",
        )
        return (
            runtime.optimizer,
            runtime.scheduler,
            params,
            lambda enabled: self.set_role_gradient_sync(
                role,
                enabled,
            ),
        )

    def set_role_gradient_sync(self, role, enabled):
        runtime = self.runtime(role)
        runtime.model.capabilities.require(ParallelCapability).set_gradient_sync(enabled)

    def trainable_names(self):
        names = ["student", "fake", "student_2", "fake_2"]
        if getattr(self.owner, "fake_low_high_model", None) is not None:
            names.append("fake_low_high")
        if getattr(self.owner, "fake_real_high_model", None) is not None:
            names.append("fake_real_high")
        if getattr(self.owner, "fake_real_low_model", None) is not None:
            names.append("fake_real_low")
        return tuple(names)

    def trainable_models(self):
        return tuple((name, self.runtime(name).model) for name in self.trainable_names())

    def trainable_states(self):
        return tuple(
            (
                name,
                self.runtime(name).model,
                self.runtime(name).optimizer,
            )
            for name in self.trainable_names()
        )

    def student_role_for_region(self, region):
        if region == "high":
            return "student"
        if region == "low":
            return "student_2"
        raise ValueError(f"Unknown phased region: {region}")

    def fake_role_for_region(self, region):
        if region == "high":
            return "fake"
        if region == "low":
            return "fake_2"
        raise ValueError(f"Unknown phased region: {region}")

    def weight_directory_name(self, name):
        if name == "student":
            return None
        return super().weight_directory_name(name)

    def setup_resources(self, resume_ckpt_path=None, base_setup=None):
        student_checkpoint_path = self.student_checkpoint_path
        if resume_ckpt_path is not None:
            self.student_checkpoint_path = None
        try:
            base_setup(resume_ckpt_path=None)
        finally:
            self.student_checkpoint_path = student_checkpoint_path

        student_2_model_config = self._role_model_config("student_2")
        self.student_2_model = build_loaded_model(
            student_2_model_config,
            load_transformer=True,
            load_vae=False,
            load_condition_encoder=False,
        )
        self.student_2_model.reuse_frozen_components_from(self.model)
        self.student_2 = self.student_2_model.capabilities.require(DistributionMatchingCapability)
        self._setup_trainable_model(
            self.student_2_model,
            role="student_2",
        )
        self.student_2_model.capabilities.require(ParallelCapability).apply(self.config)
        if self.gradient_checkpointing:
            self.student_2_model.capabilities.require(TrainableModelCapability).enable_gradient_checkpointing()

        fake_2_model_config = self._role_model_config("fake_2")
        self.fake_2_model = build_loaded_model(
            fake_2_model_config,
            load_transformer=True,
            load_vae=False,
            load_condition_encoder=False,
        )
        self.fake_2_model.reuse_frozen_components_from(self.model)
        self.fake_2 = self.fake_2_model.capabilities.require(DistributionMatchingCapability)
        self._setup_trainable_model(self.fake_2_model, role="fake_2")
        self.fake_2_model.capabilities.require(ParallelCapability).apply(self.config)
        if self.gradient_checkpointing:
            self.fake_2_model.capabilities.require(TrainableModelCapability).enable_gradient_checkpointing()

        self.fake_low_high_model = None
        fake_low_high_model_config = None
        fake_low_high_model_path = None
        if self.enable_fake_low_high:
            fake_low_high_model_config = self._role_model_config("teacher")
            self.fake_low_high_model = build_loaded_model(
                fake_low_high_model_config,
                load_transformer=True,
                load_vae=False,
                load_condition_encoder=False,
            )
            self.fake_low_high_model.reuse_frozen_components_from(self.model)
            fake_low_high_model_path = fake_low_high_model_config["model"]["pretrained_model_name_or_path"]
            self.fake_low_high = self.fake_low_high_model.capabilities.require(DistributionMatchingCapability)
            self._setup_trainable_model(
                self.fake_low_high_model,
                role="fake_low_high",
            )
            self.fake_low_high_model.capabilities.require(ParallelCapability).apply(self.config)
            if self.gradient_checkpointing:
                self.fake_low_high_model.capabilities.require(TrainableModelCapability).enable_gradient_checkpointing()

        self.fake_real_high_model = None
        self.fake_real_low_model = None
        for region, role, source_role in (
            ("high", "fake_real_high", "fake"),
            ("low", "fake_real_low", "fake_2"),
        ):
            if not self.real_data_fake_trick.enabled_for(region):
                continue
            role_config = self._role_model_config(source_role)
            model = build_loaded_model(
                role_config,
                load_transformer=True,
                load_vae=False,
                load_condition_encoder=False,
            )
            model.reuse_frozen_components_from(self.model)
            self._setup_trainable_model(model, role=role)
            model.capabilities.require(ParallelCapability).apply(self.config)
            if self.gradient_checkpointing:
                model.capabilities.require(TrainableModelCapability).enable_gradient_checkpointing()
            setattr(self, f"{role}_model", model)
            capability = model.capabilities.require(DistributionMatchingCapability)
            setattr(self, role, capability)
            logger.info(
                "[train] phased_dmd independent {} path={} train_type={}",
                role,
                role_config["model"]["pretrained_model_name_or_path"],
                getattr(self, f"{role}_train_type"),
            )

        teacher_2_override = self.model_config.get("teacher_2", {})
        if not isinstance(teacher_2_override, dict):
            raise ValueError("model.teacher_2 must be a mapping.")
        share_teacher_2 = bool(teacher_2_override.get("share_with_teacher", False))
        if share_teacher_2:
            teacher_model_config = self._role_model_config("teacher")["model"]
            teacher_2_model_config = self._role_model_config("teacher_2")["model"]
            if teacher_model_config != teacher_2_model_config:
                raise ValueError("model.teacher_2 can share with teacher only when their model configurations match.")
            self.teacher_2_model = self.teacher_model
            self.teacher_2 = self.teacher
            logger.info("[train] phased_dmd teacher_2 shares teacher weights")
        else:
            teacher_2_model_config = self._role_model_config("teacher_2")
            self.teacher_2_model = build_loaded_model(
                teacher_2_model_config,
                load_transformer=True,
                load_vae=False,
                load_condition_encoder=False,
            )
            self.teacher_2_model.reuse_frozen_components_from(self.model)
            self.teacher_2 = self.teacher_2_model.capabilities.require(DistributionMatchingCapability)
            self.teacher_2.denoiser().requires_grad_(False)
            self.teacher_2.set_training(False)
            self.teacher_2_model.capabilities.require(ParallelCapability).apply(self.config)
            self.teacher_2.set_training(False)

        self.student_2_trainable_params = list(self.student_2_model.capabilities.require(TrainableModelCapability).parameters())
        self.fake_2_trainable_params = list(self.fake_2_model.capabilities.require(TrainableModelCapability).parameters())
        self.fake_low_high_trainable_params = list(self.fake_low_high_model.capabilities.require(TrainableModelCapability).parameters()) if self.fake_low_high_model is not None else []
        self.fake_real_high_trainable_params = list(self.fake_real_high_model.capabilities.require(TrainableModelCapability).parameters()) if self.fake_real_high_model is not None else []
        self.fake_real_low_trainable_params = list(self.fake_real_low_model.capabilities.require(TrainableModelCapability).parameters()) if self.fake_real_low_model is not None else []
        self.student_2_optimizer = self._build_optimizer(
            self.student_2_trainable_params,
            self.student_2_optimizer_config,
        )
        self.fake_2_optimizer = self._build_optimizer(
            self.fake_2_trainable_params,
            self.fake_2_optimizer_config,
        )
        self.fake_low_high_optimizer = (
            self._build_optimizer(
                self.fake_low_high_trainable_params,
                self.fake_low_high_optimizer_config,
            )
            if self.fake_low_high_model is not None
            else None
        )
        self.fake_real_high_optimizer = (
            self._build_optimizer(
                self.fake_real_high_trainable_params,
                self.fake_real_high_optimizer_config,
            )
            if self.fake_real_high_model is not None
            else None
        )
        self.fake_real_low_optimizer = (
            self._build_optimizer(
                self.fake_real_low_trainable_params,
                self.fake_real_low_optimizer_config,
            )
            if self.fake_real_low_model is not None
            else None
        )

        high_steps, low_steps = self._region_training_steps()
        self.lr_scheduler = self._build_lr_scheduler(
            self.optimizer,
            num_training_steps=max(1, high_steps),
        )
        self.fake_lr_scheduler = self._build_lr_scheduler(
            self.fake_optimizer,
            num_warmup_steps=0,
            num_training_steps=max(1, high_steps * self.fake_update_ratio),
        )
        self.student_2_lr_scheduler = self._build_lr_scheduler(
            self.student_2_optimizer,
            num_training_steps=max(1, low_steps),
        )
        self.fake_2_lr_scheduler = self._build_lr_scheduler(
            self.fake_2_optimizer,
            num_warmup_steps=0,
            num_training_steps=max(1, low_steps * self.fake_update_ratio),
        )
        self.fake_low_high_lr_scheduler = (
            self._build_lr_scheduler(
                self.fake_low_high_optimizer,
                num_warmup_steps=0,
                num_training_steps=max(
                    1,
                    self.max_train_iters * self.fake_update_ratio,
                ),
            )
            if self.fake_low_high_optimizer is not None
            else None
        )
        self.fake_real_high_lr_scheduler = (
            self._build_lr_scheduler(
                self.fake_real_high_optimizer,
                num_warmup_steps=0,
                num_training_steps=max(
                    1,
                    high_steps * self.fake_update_ratio,
                ),
            )
            if self.fake_real_high_optimizer is not None
            else None
        )
        self.fake_real_low_lr_scheduler = (
            self._build_lr_scheduler(
                self.fake_real_low_optimizer,
                num_warmup_steps=0,
                num_training_steps=max(
                    1,
                    low_steps * self.fake_update_ratio,
                ),
            )
            if self.fake_real_low_optimizer is not None
            else None
        )
        if self.infer_every_iters:
            if not hasattr(self.inferencer, "set_low_model"):
                raise RuntimeError("phased_dmd inference requires an inferencer that supports set_low_model().")
            self.inferencer.set_low_model(self.student_2_model)

        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)

        self._setup_ida_trick()

        logger.info(
            "[train] phased_dmd High Student={} Low Student={}",
            self.model_config["pretrained_model_name_or_path"],
            student_2_model_config["model"]["pretrained_model_name_or_path"],
        )
        logger.info(
            "[train] phased_dmd High Fake={} Low Fake={} Low-High Fake={} enabled={}",
            self.model_config.get("fake", {}).get(
                "pretrained_model_name_or_path",
                self.model_config["pretrained_model_name_or_path"],
            ),
            fake_2_model_config["model"]["pretrained_model_name_or_path"],
            fake_low_high_model_path,
            self.enable_fake_low_high,
        )
        logger.info(
            "[train] phased_dmd schedule={} boundary={} updates=1G+{}F fake_real_high={} fake_real_low={}",
            list(self.denoising_step_list),
            self.match_timestep,
            self.fake_update_ratio,
            self.real_data_fake_trick.enabled_for("high"),
            self.real_data_fake_trick.enabled_for("low"),
        )
