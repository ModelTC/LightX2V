import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

FI_FORCE_RETUNE_ENV = "LIGHTX2V_FI_FORCE_RETUNE"

try:
    from flashinfer.autotuner import autotune as flashinfer_autotune
except ImportError:
    flashinfer_autotune = None


def fi_force_retune(env_name: str = FI_FORCE_RETUNE_ENV) -> bool:
    return os.environ.get(env_name, "0").strip().lower() in ("1", "true", "yes", "on")


def fi_sm_arch() -> int:
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def fi_autotune_cache_path(cache_namespace: str, model_sig: str) -> str:
    root = Path.home() / ".cache" / "lightx2v" / "autotune" / cache_namespace
    return str(root / model_sig / f"sm{fi_sm_arch()}.json")


def _resolve_tune_mode(cache_path: str, *, tune_mode: bool | None) -> bool:
    if tune_mode is True:
        return True
    if tune_mode is False:
        return False
    return not os.path.isfile(os.path.expanduser(cache_path))


def _tune_mode_label(tune_mode: bool | None, effective: bool) -> str:
    if tune_mode is None:
        return f"auto->{effective}"
    return str(effective)


@dataclass
class FlashInferAutotune:
    """Generic FlashInfer autotune session (cache + tune_mode dispatch)."""

    enabled: bool = False
    cache_path: Optional[str] = None
    force_retune_env: str = FI_FORCE_RETUNE_ENV
    log_prefix: str = "Flashinfer autotune"

    def cache_rebuild_needed(self) -> bool:
        if not self.enabled or not self.cache_path:
            return False
        if fi_force_retune(self.force_retune_env):
            return True
        return not os.path.isfile(os.path.expanduser(self.cache_path))

    @contextmanager
    def session(self, *, tune_mode: bool | None = None):
        """FlashInfer autotune session.

        ``tune_mode``:
            None: cache hit → cache-only; cache miss → lazy online rebuild.
            True: profile uncovered shapes (offline tune / one-shot rebuild step).
            False: cache-only even when cache is missing (benchmark fallback path).
        """
        if not self.enabled or not self.cache_path or flashinfer_autotune is None:
            yield
            return

        cache_path = os.path.expanduser(self.cache_path)
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        effective_tune_mode = _resolve_tune_mode(cache_path, tune_mode=tune_mode)
        mode_label = _tune_mode_label(tune_mode, effective_tune_mode)

        if fi_force_retune(self.force_retune_env) and effective_tune_mode and os.path.isfile(cache_path):
            os.remove(cache_path)
            logger.info(f"Removed {self.log_prefix} cache ({self.force_retune_env}=1): {cache_path}; will profile once in this session, then cache-only for later steps/runs")

        if os.path.isfile(cache_path):
            logger.info(f"{self.log_prefix}: loading cache from {cache_path} (tune_mode={mode_label})")
        elif effective_tune_mode:
            logger.info(f"{self.log_prefix}: cache not found at {cache_path}, lazy-rebuilding online (tune_mode={mode_label}); first inference after cache loss will be slower")
        else:
            logger.warning(f"{self.log_prefix}: cache not found at {cache_path} and tune_mode=False; will use fallback tactics until cache is built.")

        with flashinfer_autotune(effective_tune_mode, cache=cache_path):
            yield
