"""Strict SafeTensors loader for the converted pi0.5-LIBERO checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_model

from ..config import Pi0Config

LOGGER = logging.getLogger(__name__)


def _validate_checkpoint_precision(weight_path: Path, config: Pi0Config) -> None:
    if not config.require_fp32_checkpoint:
        return

    lower_precision = []
    with safe_open(weight_path, framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            dtype = checkpoint.get_slice(name).get_dtype()
            if dtype.startswith(("F", "BF")) and dtype != "F32":
                lower_precision.append((name, dtype))
                if len(lower_precision) == 8:
                    break
    if lower_precision:
        preview = ", ".join(f"{name}={dtype}" for name, dtype in lower_precision)
        raise RuntimeError(f"OpenPI fine-tuning requires an FP32 source checkpoint; found {preview}")


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
    _validate_checkpoint_precision(weight_path, config)

    # Import only after selecting OpenPI's private Transformers runtime.
    from ..pi0 import PI0Pytorch

    model = PI0Pytorch(config)
    load_model(model, weight_path, strict=True, device="cpu")

    model.to(torch.device(device))
    if config.resolved_parameter_dtype == "float32":
        model.assert_fp32_parameters()
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Loaded pi05_libero PyTorch weights strictly: %.3fB parameters", parameter_count / 1e9)
    return model
