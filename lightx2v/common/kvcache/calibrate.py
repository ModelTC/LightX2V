"""
KV-cache quantisation calibration.

Step 1 — Calibration run
~~~~~~~~~~~~~~~~~~~~~~~~
Use a config with ``"calibrate": true`` and ``self_attn_1_type`` set to
the **non-quant** attention (e.g. ``"sage_attn2"``).  This creates a
``CalibRollingKVCachePool`` that stores bf16 KV normally while
collecting K-mean and V per-channel abs-max.

Config example (calibration)::

    {
        "self_attn_1_type": "sage_attn2",
        "ar_config": {
            ...
            "sage_quant_kv": {
                "calibrate": true,
                "smooth_k": true
            }
        }
    }

After inference, call ``save_calibration`` to export the stats::

    from lightx2v.common.kvcache.calibrate import save_calibration
    runner.run_main()
    save_calibration(runner.model.kv_cache_manager, "calib_kv.pt")

Step 2 — Quantised inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Switch to the quant attention and provide the calibration file::

    {
        "self_attn_1_type": "sage_attn2_kvquant",
        "ar_config": {
            ...
            "sage_quant_kv": {
                "smooth_k": true,
                "calib_path": "calib_kv.pt"
            }
        }
    }
"""

from __future__ import annotations

import torch
from loguru import logger

from .quant import CalibRollingKVCachePool


def save_calibration(
    kv_cache_manager,
    output_path: str,
) -> dict[str, torch.Tensor]:
    """Export and save KV cache calibration from a completed run.

    Parameters
    ----------
    kv_cache_manager : KVCacheManager
        The manager whose ``self_attn_kv_cache`` is a
        ``CalibRollingKVCachePool`` that has been used for at least one
        full inference pass.
    output_path : str
        File path to save the calibration dict (``torch.save`` format).

    Returns
    -------
    dict with keys ``'km'`` and ``'v_scale'``.
    """
    pool = kv_cache_manager.self_attn_kv_cache
    if not isinstance(pool, CalibRollingKVCachePool):
        raise TypeError(f"Expected CalibRollingKVCachePool, got {type(pool).__name__}. Make sure the config has sage_quant_kv.calibrate=true and self_attn_1_type is NOT sage_attn2_kvquant.")

    calib = pool.export_calibration()
    torch.save(calib, output_path)
    logger.info(
        "KV calibration saved to {} — km {}, v_scale {}",
        output_path,
        list(calib["km"].shape),
        list(calib["v_scale"].shape),
    )
    return calib
