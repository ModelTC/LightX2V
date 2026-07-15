from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from loguru import logger

try:
    from flashinfer.autotuner import AutoTuner as FlashInferAutoTuner
    from flashinfer.autotuner import autotune as flashinfer_autotune
except Exception:
    FlashInferAutoTuner = None
    flashinfer_autotune = None


FlashInferAutotuneMode = Literal["off", "tune", "load"]
AutotuneFactory = Callable[..., AbstractContextManager]
TunerProvider = Callable[[], Any]
StatusDeviceResolver = Callable[[], torch.device]


@dataclass(frozen=True)
class FlashInferAutotuneOptions:
    """Normalized, side-effect-free FlashInfer autotune configuration."""

    mode: FlashInferAutotuneMode
    cache_path: Path | None = None
    tuning_buckets: tuple[int, ...] | None = None
    round_up: bool | None = None
    tune_max_num_tokens: int = 8192

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def tune_mode(self) -> bool:
        return self.mode == "tune"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> FlashInferAutotuneOptions:
        mode = cls.resolve_mode(config)
        if mode == "off":
            return cls(mode=mode)
        return cls(
            mode=mode,
            cache_path=cls._resolve_cache_path(config.get("flashinfer_autotune_cache")),
            tuning_buckets=cls._parse_tuning_buckets(config.get("flashinfer_tuning_buckets")),
            round_up=cls._parse_optional_bool(config.get("flashinfer_autotune_round_up")),
            tune_max_num_tokens=int(config.get("flashinfer_tune_max_num_tokens", 8192)),
        )

    @classmethod
    def resolve_mode(cls, config: Mapping[str, Any]) -> FlashInferAutotuneMode:
        mode = config.get("flashinfer_autotune_mode")
        if mode is None and "flashinfer_autotune" in config:
            mode = "tune" if cls._parse_bool(config.get("flashinfer_autotune")) else "off"
        if mode is None:
            return "off"
        if isinstance(mode, bool):
            return "tune" if mode else "off"

        normalized = str(mode).strip().lower()
        bool_aliases = {
            "1": "tune",
            "true": "tune",
            "yes": "tune",
            "y": "tune",
            "on": "tune",
            "0": "off",
            "false": "off",
            "no": "off",
            "n": "off",
        }
        normalized = bool_aliases.get(normalized, normalized)
        if normalized not in ("off", "tune", "load"):
            raise ValueError("flashinfer_autotune_mode must be one of: off, tune, load.")
        return normalized

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("1", "true", "yes", "y", "on"):
                return True
            if normalized in ("0", "false", "no", "n", "off"):
                return False
        return bool(value)

    @classmethod
    def _parse_optional_bool(cls, value: Any) -> bool | None:
        return None if value is None else cls._parse_bool(value)

    @staticmethod
    def _resolve_cache_path(value: Any) -> Path | None:
        if value in (None, ""):
            return None
        return Path(value).expanduser()

    @staticmethod
    def _parse_tuning_buckets(value: Any) -> tuple[int, ...] | None:
        if value in (None, ""):
            return None
        if isinstance(value, int):
            return (value,)
        if isinstance(value, (list, tuple)):
            buckets = tuple(int(item) for item in value if str(item).strip())
        else:
            buckets = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
        if not buckets:
            raise ValueError("flashinfer_tuning_buckets must contain at least one integer when provided.")
        return tuple(sorted(set(buckets)))


@dataclass(frozen=True)
class DistributedAutotuneContext:
    """Distributed resources selected by the runner's parallel topology."""

    coordination_required: bool
    process_group: dist.ProcessGroup | None = None
    status_device_resolver: StatusDeviceResolver | None = None
    is_cache_writer: bool = False


class FlashInferAutotuneController:
    """Own the local or distributed FlashInfer autotune lifecycle."""

    def __init__(
        self,
        options: FlashInferAutotuneOptions,
        distributed_context: DistributedAutotuneContext,
        moe_impl: str,
        *,
        autotune_factory: AutotuneFactory | None = None,
        tuner_provider: TunerProvider | None = None,
    ):
        self.options = options
        self.distributed_context = distributed_context
        self.moe_impl = moe_impl
        self._autotune_factory = autotune_factory
        self._tuner_provider = tuner_provider

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        distributed_context: DistributedAutotuneContext,
        *,
        autotune_factory: AutotuneFactory | None = None,
        tuner_provider: TunerProvider | None = None,
    ) -> FlashInferAutotuneController:
        """Build a controller without validating inactive autotune settings."""
        moe_impl = config.get("moe_impl", "eager")
        mode = FlashInferAutotuneOptions.resolve_mode(config)
        options = FlashInferAutotuneOptions(mode=mode)
        if mode != "off" and moe_impl == "flashinfer":
            options = FlashInferAutotuneOptions.from_config(config)
        return cls(
            options=options,
            distributed_context=distributed_context,
            moe_impl=moe_impl,
            autotune_factory=autotune_factory,
            tuner_provider=tuner_provider,
        )

    def context(self) -> AbstractContextManager:
        if not self.options.enabled:
            return nullcontext()
        if self.moe_impl != "flashinfer":
            logger.warning("flashinfer_autotune_mode is set but moe_impl is not 'flashinfer'; autotune is disabled for this run.")
            return nullcontext()

        autotune_factory = self._resolve_autotune_factory()
        self._prepare_cache_path()
        logger.info(
            "HunyuanImage3 FlashInfer autotune enabled: "
            f"mode={self.options.mode}, cache={self.options.cache_path}, "
            f"buckets={self.options.tuning_buckets}, round_up={self.options.round_up}, "
            f"tune_max_num_tokens={self.options.tune_max_num_tokens}"
        )

        if self.distributed_context.coordination_required:
            return self._distributed_autotune_context(autotune_factory)
        return self._local_autotune_context(autotune_factory)

    def _resolve_autotune_factory(self) -> AutotuneFactory:
        factory = self._autotune_factory or flashinfer_autotune
        if factory is None:
            raise ImportError("flashinfer_autotune_mode requires flashinfer.autotuner.autotune.")
        return factory

    def _resolve_tuner(self) -> Any:
        if self._tuner_provider is not None:
            return self._tuner_provider()
        if FlashInferAutoTuner is None:
            raise ImportError("Distributed FlashInfer autotune requires flashinfer.autotuner.AutoTuner.")
        return FlashInferAutoTuner.get()

    def _prepare_cache_path(self) -> None:
        cache_path = self.options.cache_path
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.options.mode == "load" and cache_path is None:
            raise ValueError("flashinfer_autotune_mode='load' requires flashinfer_autotune_cache.")

    def _local_autotune_context(self, autotune_factory: AutotuneFactory) -> AbstractContextManager:
        cache_path = self.options.cache_path
        if self.options.mode == "load" and (cache_path is None or not cache_path.is_file()):
            raise FileNotFoundError(f"FlashInfer autotune cache does not exist for load mode: {cache_path}")
        return autotune_factory(
            tune_mode=self.options.tune_mode,
            cache=None if cache_path is None else str(cache_path),
            tuning_buckets=self.options.tuning_buckets,
            round_up=self.options.round_up,
        )

    @contextmanager
    def _distributed_autotune_context(self, autotune_factory: AutotuneFactory):
        tuner = self._resolve_tuner()
        runtime = self.distributed_context
        group = runtime.process_group
        if group is None:
            raise RuntimeError("Distributed FlashInfer autotune requires an initialized CFG or sequence parallel group.")

        if dist.get_backend(group) == "nccl":
            if runtime.status_device_resolver is None:
                raise RuntimeError("Distributed FlashInfer autotune requires an NCCL status device resolver.")
            status_device = torch.device(runtime.status_device_resolver())
            torch.cuda.set_device(status_device)
        else:
            status_device = torch.device("cpu")

        cache_path = None if self.options.cache_path is None else str(self.options.cache_path)
        load_error = self._load_distributed_cache(tuner, cache_path)
        load_succeeded = torch.tensor([0 if load_error is not None else 1], device=status_device, dtype=torch.int32)
        dist.all_reduce(load_succeeded, op=dist.ReduceOp.MIN, group=group)
        if not bool(load_succeeded.item()):
            raise RuntimeError(f"Distributed FlashInfer autotune cache preflight failed for {cache_path}.") from load_error

        with autotune_factory(
            tune_mode=self.options.tune_mode,
            cache=None,
            tuning_buckets=self.options.tuning_buckets,
            round_up=self.options.round_up,
        ):
            yield

        save_error = self._save_distributed_cache(tuner, cache_path)
        save_succeeded = torch.tensor([0 if save_error is not None else 1], device=status_device, dtype=torch.int32)
        source_rank = dist.get_global_rank(group, 0)
        dist.broadcast(save_succeeded, src=source_rank, group=group)
        if not bool(save_succeeded.item()):
            raise RuntimeError(f"Distributed FlashInfer autotune cache could not be saved to {cache_path}.") from save_error

    def _load_distributed_cache(self, tuner: Any, cache_path: str | None) -> Exception | None:
        if cache_path is not None:
            if Path(cache_path).is_file():
                try:
                    if not tuner.load_configs(cache_path):
                        return RuntimeError("FlashInfer autotune cache metadata does not match this runtime; choose a new cache path or remove the incompatible cache.")
                except Exception as error:
                    return error
            elif self.options.mode == "load":
                return FileNotFoundError(f"FlashInfer autotune cache does not exist for load mode: {cache_path}")
        elif self.options.mode == "load":
            return ValueError("FlashInfer load mode requires a cache path.")
        return None

    def _save_distributed_cache(self, tuner: Any, cache_path: str | None) -> Exception | None:
        if not self.options.tune_mode or cache_path is None or not self.distributed_context.is_cache_writer:
            return None
        try:
            tuner.save_configs(cache_path)
        except Exception as error:
            logger.exception(f"Failed to persist distributed FlashInfer autotune cache: {cache_path}")
            return error
        return None
