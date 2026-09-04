"""Strict SafeTensors loader for the converted pi0.5-LIBERO checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors.torch import load_model

from ..config import Pi0Config

LOGGER = logging.getLogger(__name__)


def _validate_transformers_runtime() -> None:
    """Fail early unless the official patched Transformers runtime is active."""
    import transformers

    if transformers.__version__ != "4.53.2":
        raise RuntimeError(
            "OpenPI requires its private patched transformers==4.53.2 runtime; "
            f"the current process imported transformers=={transformers.__version__}. "
            "Launch with scripts/openpi/run_libero_*.sh or prepend "
            "OPENPI_TRANSFORMERS_RUNTIME_PATH to PYTHONPATH."
        )
    try:
        from transformers.models.siglip import check
    except ImportError as exc:
        raise RuntimeError("OpenPI Transformers patches are missing (siglip/check.py not found)") from exc
    if not check.check_whether_transformers_replace_is_installed_correctly():
        raise RuntimeError("OpenPI Transformers 4.53.2 is present but the official replacement patches are missing")


def load_pi05_libero_weights(
    weight_path: str | Path,
    config: Pi0Config,
    device: torch.device | str,
):
    """Build the exact official parameter tree and load it with strict key checks."""
    _validate_transformers_runtime()
    config.validate_pi05_libero()
    weight_path = Path(weight_path).expanduser().resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(f"Converted OpenPI SafeTensors file not found: {weight_path}")

    # Import only after selecting OpenPI's private Transformers runtime.
    from ..pi0 import PI0Pytorch

    model = PI0Pytorch(config)
    load_model(model, weight_path, strict=True, device="cpu")

    # Match upstream's mixed BF16/FP32 inference policy.
    model.paligemma_with_expert.to_bfloat16_for_selected_params(config.dtype)
    model.to(torch.device(device))
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Loaded pi05_libero PyTorch weights strictly: %.3fB parameters", parameter_count / 1e9)
    return model
