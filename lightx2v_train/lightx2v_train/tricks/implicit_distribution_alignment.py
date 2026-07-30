from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .base import TrainerTrick


@dataclass(frozen=True)
class IdaConfig:
    enabled: bool = False
    decay: float = 0.97

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "IdaConfig":
        if not isinstance(config, Mapping):
            raise ValueError("training.dmd.ida must be a mapping.")
        return cls(
            enabled=bool(config.get("enabled", False)),
            decay=float(config.get("decay", 0.97)),
        )


@dataclass(frozen=True)
class IdaModelPair:
    student: torch.nn.Module
    fake: torch.nn.Module


@dataclass(frozen=True)
class IdaSetupContext:
    model_pairs: Mapping[str, IdaModelPair]


@dataclass(frozen=True)
class IdaStepContext:
    role: str


class ImplicitDistributionAlignmentTrick(
    TrainerTrick[
        IdaSetupContext,
        None,
        None,
        IdaStepContext,
    ]
):
    """Align fake parameters to the latest student with SenseFlow IDA."""

    name = "implicit_distribution_alignment"

    def __init__(self, config: IdaConfig):
        super().__init__(enabled=config.enabled)
        self.config = config
        self._parameter_pairs: dict[
            str,
            tuple[
                tuple[
                    str,
                    torch.nn.Parameter,
                    torch.nn.Parameter,
                ],
                ...,
            ],
        ] = {}
        self.validate()

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
    ) -> "ImplicitDistributionAlignmentTrick":
        return cls(IdaConfig.from_mapping(config))

    def validate(self) -> None:
        if not 0.0 < self.config.decay <= 1.0:
            raise ValueError("training.dmd.ida.decay must satisfy 0 < decay <= 1.")

    def setup(self, context: IdaSetupContext) -> None:
        if not self.enabled:
            return
        if not context.model_pairs:
            raise ValueError("SenseFlow IDA requires at least one model pair.")
        parameter_pairs = {}
        for role, model_pair in context.model_pairs.items():
            parameter_pairs[role] = self._build_parameter_pairs(
                role,
                model_pair,
            )
        self._parameter_pairs = parameter_pairs

    def _build_parameter_pairs(
        self,
        role: str,
        model_pair: IdaModelPair,
    ) -> tuple[
        tuple[str, torch.nn.Parameter, torch.nn.Parameter],
        ...,
    ]:
        if model_pair.student is model_pair.fake:
            raise ValueError(f"SenseFlow IDA role {role!r} requires distinct models.")

        student_parameters = {name: parameter for name, parameter in model_pair.student.named_parameters()}
        fake_parameters = {name: parameter for name, parameter in model_pair.fake.named_parameters()}
        trainable_student_names = {name for name, parameter in student_parameters.items() if parameter.requires_grad}
        if not trainable_student_names:
            raise ValueError(f"SenseFlow IDA role {role!r} has no trainable student parameters.")
        if student_parameters.keys() != fake_parameters.keys():
            student_only = sorted(student_parameters.keys() - fake_parameters.keys())
            fake_only = sorted(fake_parameters.keys() - student_parameters.keys())
            raise ValueError(f"SenseFlow IDA role {role!r} parameter names do not match: student_only={student_only[:5]}, fake_only={fake_only[:5]}.")

        pairs = []
        for name, student_parameter in student_parameters.items():
            fake_parameter = fake_parameters[name]
            if student_parameter.shape != fake_parameter.shape:
                raise ValueError(f"SenseFlow IDA role {role!r} parameter {name!r} has student shape {tuple(student_parameter.shape)} and fake shape {tuple(fake_parameter.shape)}.")
            if name not in trainable_student_names:
                continue
            pairs.append(
                (
                    name,
                    student_parameter,
                    fake_parameter,
                )
            )
        return tuple(pairs)

    @torch.no_grad()
    def after_student_step(self, context: IdaStepContext) -> None:
        if not self.enabled or self.config.decay == 1.0:
            return
        parameter_pairs = self._parameter_pairs.get(context.role)
        if parameter_pairs is None:
            available = ", ".join(sorted(self._parameter_pairs))
            raise ValueError(f"Unknown SenseFlow IDA role {context.role!r}. Available roles: {available}")

        student_weight = 1.0 - self.config.decay
        for _, student_parameter, fake_parameter in parameter_pairs:
            fake_parameter.mul_(self.config.decay).add_(
                student_parameter.to(dtype=fake_parameter.dtype),
                alpha=student_weight,
            )

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "ida_enabled": self.enabled,
            "ida_decay": self.config.decay,
        }
