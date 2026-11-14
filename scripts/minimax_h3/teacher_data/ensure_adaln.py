#!/usr/bin/env python3
"""Validate the exact merged Ref2AV AdaLN cache, building it only if necessary."""

import argparse
import sys
import time
from pathlib import Path

from common import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "tools/cache_minimax_h3_adaln"))
    from builder import build_persistent_adaln_cache

    from lightx2v.models.networks.minimax_h3.adaln_cache import _validate_cache, resolve_adaln_cache
    from lightx2v.utils.set_config import build_startup_config

    config = build_startup_config({"model_cls": "minimax_h3", "model_variant": "ref2av", "model_path": args.model_path, "config_json": args.config_json})
    transformer = Path(config["dit_original_ckpt"]).resolve()
    if transformer != (Path(args.model_path) / "transformer_ref").resolve():
        raise ValueError(f"AdaLN must use the merged transformer_ref partition, got {transformer}")
    path, spec = resolve_adaln_cache(config)
    if not _validate_cache(path, spec):
        if path.exists():
            path.rename(path.with_name(path.name + f".invalid.{time.time_ns()}"))
        build_persistent_adaln_cache(config)
    if not _validate_cache(path, spec):
        raise RuntimeError(f"AdaLN validation failed: {path}")
    atomic_json(args.receipt, {"cache_path": str(path), "transformer_path": str(transformer), "spec": spec})
    print(f"Validated merged Ref2AV AdaLN cache: {path}", flush=True)


if __name__ == "__main__":
    main()
