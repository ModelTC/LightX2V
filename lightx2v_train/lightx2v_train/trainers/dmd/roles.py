from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    Optional,
)


@dataclass(frozen=True)
class RoleSpec:
    """Describe stable checkpoint and training metadata for one model role."""

    name: str
    model_attribute: str
    optimizer_attribute: Optional[str] = None
    scheduler_attribute: Optional[str] = None
    train_type_attribute: Optional[str] = None
    weight_directory_full: Optional[str] = None
    weight_directory_lora: Optional[str] = None
    optional: bool = False


@dataclass
class RoleRuntime:
    """Expose live resources owned by a trainer for a declared role."""

    spec: RoleSpec
    model: object
    optimizer: object = None
    scheduler: object = None
    train_type: Optional[str] = None


class DmdRoleRegistry:
    """Provide a common view over trainer-owned DMD role resources."""

    role_specs = (
        RoleSpec(
            name="student",
            model_attribute="model",
            optimizer_attribute="optimizer",
            scheduler_attribute="lr_scheduler",
            train_type_attribute="student_train_type",
        ),
        RoleSpec(
            name="fake",
            model_attribute="fake_model",
            optimizer_attribute="fake_optimizer",
            scheduler_attribute="fake_lr_scheduler",
            train_type_attribute="fake_train_type",
            weight_directory_full="fake_model",
            weight_directory_lora="fake_lora",
        ),
        RoleSpec(
            name="fake_real",
            model_attribute="fake_real_model",
            optimizer_attribute="fake_real_optimizer",
            scheduler_attribute="fake_real_lr_scheduler",
            train_type_attribute="fake_real_train_type",
            weight_directory_full="fake_real_model",
            weight_directory_lora="fake_real_lora",
            optional=True,
        ),
    )

    def __init__(self, owner):
        self.owner = owner
        self._specs = {spec.name: spec for spec in self.role_specs}

    def names(self) -> Iterable[str]:
        return self._specs.keys()

    def spec(self, name: str) -> RoleSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ValueError(f"Unknown DMD model role: {name}") from exc

    def runtime(self, name: str) -> RoleRuntime:
        spec = self.spec(name)
        model = getattr(self.owner, spec.model_attribute, None)
        if model is None and not spec.optional:
            raise RuntimeError(f"DMD role {name!r} has not been initialized.")
        return RoleRuntime(
            spec=spec,
            model=model,
            optimizer=self._optional_attribute(spec.optimizer_attribute),
            scheduler=self._optional_attribute(spec.scheduler_attribute),
            train_type=self._optional_attribute(spec.train_type_attribute),
        )

    def runtimes(self) -> Dict[str, RoleRuntime]:
        return {name: self.runtime(name) for name in self.names()}

    def weight_directory_name(self, name: str) -> Optional[str]:
        runtime = self.runtime(name)
        if runtime.train_type == "lora":
            return runtime.spec.weight_directory_lora
        return runtime.spec.weight_directory_full

    def _optional_attribute(self, name: Optional[str]):
        if name is None:
            return None
        return getattr(self.owner, name, None)
