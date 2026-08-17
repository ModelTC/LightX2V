from __future__ import annotations

from contextlib import contextmanager

import torch
from peft import LoraConfig

from lightx2v_train.model_capabilities import AdapterBankCapability, BoundCapability


class PeftAdapterBankCapability(BoundCapability, AdapterBankCapability):
    """Strict named-adapter lifecycle for PEFT-compatible denoisers."""

    def __init__(self, model) -> None:
        super().__init__(model)
        self._active_adapter: str | None = None
        self._known_adapters: set[str] = set()
        self._parameter_pair_cache = {}

    @staticmethod
    def _validate_name(name: str) -> str:
        name = str(name).strip()
        if not name or "." in name:
            raise ValueError(f"Adapter names must be non-empty and may not contain '.', got {name!r}.")
        return name

    @staticmethod
    def _adapter_token(adapter_name: str) -> str:
        return f".{adapter_name}."

    def _denoiser(self):
        return self.model.denoiser_module()

    def _named_adapter_parameters(self, adapter_name: str) -> dict[str, torch.nn.Parameter]:
        adapter_name = self._validate_name(adapter_name)
        token = self._adapter_token(adapter_name)
        parameters = {
            name: parameter
            for name, parameter in self._denoiser().named_parameters()
            if token in f".{name}."
        }
        if not parameters:
            raise RuntimeError(f"No parameters found for adapter {adapter_name!r}.")
        self._known_adapters.add(adapter_name)
        return parameters

    def _parameter_pairs(
        self,
        source_adapter: str,
        target_adapter: str,
    ) -> tuple[tuple[torch.nn.Parameter, torch.nn.Parameter], ...]:
        source_adapter = self._validate_name(source_adapter)
        target_adapter = self._validate_name(target_adapter)
        if source_adapter == target_adapter:
            raise ValueError("Source and target adapters must be different.")
        cache_key = (source_adapter, target_adapter)
        cached = self._parameter_pair_cache.get(cache_key)
        if cached is not None:
            return cached

        source_token = self._adapter_token(source_adapter)
        target_token = self._adapter_token(target_adapter)
        all_parameters = dict(self._denoiser().named_parameters())
        source_parameters = self._named_adapter_parameters(source_adapter)
        target_parameters = self._named_adapter_parameters(target_adapter)

        pairs = []
        mapped_target_names = set()
        for source_name, source_parameter in source_parameters.items():
            padded_name = f".{source_name}."
            if padded_name.count(source_token) != 1:
                raise RuntimeError(f"Adapter token is ambiguous in parameter name {source_name!r}.")
            target_name = padded_name.replace(source_token, target_token, 1)[1:-1]
            target_parameter = all_parameters.get(target_name)
            if target_parameter is None:
                raise RuntimeError(f"Adapter {target_adapter!r} is missing parameter {target_name!r}.")
            if source_parameter.shape != target_parameter.shape:
                raise RuntimeError(
                    f"Adapter parameter shape mismatch: {source_name}={tuple(source_parameter.shape)}, "
                    f"{target_name}={tuple(target_parameter.shape)}."
                )
            mapped_target_names.add(target_name)
            pairs.append((source_parameter, target_parameter))

        unexpected_targets = set(target_parameters) - mapped_target_names
        if unexpected_targets:
            raise RuntimeError(
                f"Adapter {target_adapter!r} has parameters without a source match: "
                f"{sorted(unexpected_targets)}"
            )
        result = tuple(pairs)
        self._parameter_pair_cache[cache_key] = result
        return result

    def configure_pair(
        self,
        rank: int,
        alpha: int,
        target_modules,
        student_adapter: str,
        teacher_adapter: str,
        initialize_teacher: bool,
    ) -> None:
        student_adapter = self._validate_name(student_adapter)
        teacher_adapter = self._validate_name(teacher_adapter)
        if student_adapter == teacher_adapter:
            raise ValueError("DOPSD student and teacher adapter names must be different.")
        if int(rank) <= 0 or int(alpha) <= 0:
            raise ValueError(f"LoRA rank and alpha must be positive, got rank={rank}, alpha={alpha}.")

        adapter_config = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        denoiser = self._denoiser()
        denoiser.requires_grad_(False)
        denoiser.add_adapter(adapter_config, adapter_name=student_adapter)
        denoiser.add_adapter(adapter_config, adapter_name=teacher_adapter)
        self._known_adapters.update((student_adapter, teacher_adapter))
        self._set_active(student_adapter)
        if initialize_teacher:
            self.copy(student_adapter, teacher_adapter)
        self.set_trainable(student_adapter)
        # Parallel wrappers may replace parameter objects after configuration.
        self._parameter_pair_cache.clear()

    def parameters(self, adapter_name: str):
        self._parameter_pair_cache.clear()
        return tuple(self._named_adapter_parameters(adapter_name).values())

    def _set_active(self, adapter_name: str) -> None:
        adapter_name = self._validate_name(adapter_name)
        if adapter_name not in self._known_adapters:
            self._named_adapter_parameters(adapter_name)
        self._denoiser().set_adapter(adapter_name)
        self._active_adapter = adapter_name

    @contextmanager
    def activate(self, adapter_name: str, training: bool | None = None):
        denoiser = self._denoiser()
        previous_adapter = self._active_adapter
        previous_training = denoiser.training
        self._set_active(adapter_name)
        mode_changed = training is not None and training != previous_training
        if mode_changed:
            denoiser.train(training)
        try:
            yield
        finally:
            if mode_changed:
                denoiser.train(previous_training)
            if previous_adapter is not None and previous_adapter != adapter_name:
                self._set_active(previous_adapter)

    def set_trainable(self, adapter_name: str) -> None:
        trainable_parameters = self._named_adapter_parameters(adapter_name)
        denoiser = self._denoiser()
        denoiser.requires_grad_(False)
        denoiser.train()
        for parameter in trainable_parameters.values():
            parameter.requires_grad_(True)
        self._set_active(adapter_name)

    @torch.no_grad()
    def copy(self, source_adapter: str, target_adapter: str) -> None:
        for source_parameter, target_parameter in self._parameter_pairs(source_adapter, target_adapter):
            target_parameter.copy_(source_parameter)

    @torch.no_grad()
    def ema_update(self, source_adapter: str, target_adapter: str, decay: float) -> None:
        decay = float(decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")
        for source_parameter, target_parameter in self._parameter_pairs(source_adapter, target_adapter):
            target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)

    def load(self, path, adapter_name: str, weights_subdir: str | None = None) -> None:
        self.model.load_lora_weights_for_resume(
            path,
            adapter_name=self._validate_name(adapter_name),
            weights_subdir=weights_subdir,
        )

    def save(self, path, adapter_name: str, weights_subdir: str | None = None) -> None:
        self.model.save_lora_weights(
            path,
            adapter_name=self._validate_name(adapter_name),
            weights_subdir=weights_subdir,
        )
