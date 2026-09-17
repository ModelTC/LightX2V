#!/usr/bin/env python3
"""Convert an OpenPI pi0.5 Orbax checkpoint with the upstream converter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
OPENPI_DATA_ROOT = Path(os.environ.get("OPENPI_DATA_ROOT", str(WORKSPACE_ROOT / "openpi_data"))).expanduser().resolve()
DEFAULT_OPENPI_ROOT = WORKSPACE_ROOT / "openpi"
DEFAULT_SOURCE = OPENPI_DATA_ROOT / "openpi-assets/checkpoints/pi05_libero"
DEFAULT_TOKENIZER = OPENPI_DATA_ROOT / "big_vision/paligemma_tokenizer.model"
DEFAULT_TRANSFORMERS_OVERLAY = OPENPI_DATA_ROOT / "python_deps/openpi_official_pytorch_runtime"

EXPECTED_TENSORS = 812
DTYPE_BY_PRECISION = {"float32": "F32", "bfloat16": "BF16"}


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _default_output(source: Path, precision: str) -> Path:
    suffix = "_pytorch_fp32" if precision == "float32" else "_pytorch"
    return source.with_name(f"{source.name}{suffix}")


def _conversion_environment(args: argparse.Namespace) -> dict[str, str]:
    python_paths = []
    if args.transformers_runtime.is_dir():
        python_paths.append(str(args.transformers_runtime))
    python_paths.append(str(args.openpi_root / "src"))
    if current := os.environ.get("PYTHONPATH"):
        python_paths.append(current)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "USE_FLAX": "0",
        }
    )
    return environment


def _check_inputs(args: argparse.Namespace) -> None:
    required = (
        args.openpi_root / "examples/convert_jax_model_to_pytorch.py",
        args.source / "params/_METADATA",
        args.source / "assets/physical-intelligence/libero/norm_stats.json",
        args.tokenizer,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("missing conversion input:\n- " + "\n- ".join(missing))
    if "pi05" not in str(args.source).lower():
        raise RuntimeError("the upstream converter selects pi0.5 layers from a source path containing 'pi05'")
    if args.output == args.source or args.output.is_relative_to(args.source) or args.source.is_relative_to(args.output):
        raise RuntimeError(f"source and output must not overlap: {args.source}, {args.output}")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        if not args.dry_run:
            raise RuntimeError(f"output must not exist or must be an empty directory: {args.output}")
        print(f"note: output already exists and must be changed before conversion: {args.output}")

    probe = "import transformers; from transformers.models.siglip import check; assert transformers.__version__ == '4.53.2'; assert check.check_whether_transformers_replace_is_installed_correctly()"
    try:
        subprocess.run([sys.executable, "-c", probe], env=_conversion_environment(args), check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("the conversion Python cannot load OpenPI's patched transformers==4.53.2; run scripts/openpi/1_setup_pytorch_runtime.sh first") from error


def _run_converter(args: argparse.Namespace, output: Path) -> None:
    command = [
        sys.executable,
        str(args.openpi_root / "examples/convert_jax_model_to_pytorch.py"),
        "--checkpoint-dir",
        str(args.source),
        "--config-name",
        args.config_name,
        "--output-path",
        str(output),
        "--precision",
        args.precision,
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=args.openpi_root, env=_conversion_environment(args), check=True)


def _copy_assets(args: argparse.Namespace, output: Path) -> None:
    shutil.copytree(args.source / "assets", output / "assets", dirs_exist_ok=True)
    tokenizer = output / "assets/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.tokenizer, tokenizer)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_output(args: argparse.Namespace, output: Path) -> dict[str, object]:
    required = (
        output / "model.safetensors",
        output / "config.json",
        output / "assets/paligemma_tokenizer.model",
        output / "assets/physical-intelligence/libero/norm_stats.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("converted checkpoint is incomplete:\n- " + "\n- ".join(missing))

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if config.get("precision") != args.precision:
        raise RuntimeError(f"config precision is {config.get('precision')!r}, expected {args.precision!r}")

    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        dtypes = Counter(checkpoint.get_slice(key).get_dtype() for key in keys)
    expected_dtypes = Counter({DTYPE_BY_PRECISION[args.precision]: EXPECTED_TENSORS})
    if len(keys) != EXPECTED_TENSORS or dtypes != expected_dtypes:
        raise RuntimeError(f"expected {EXPECTED_TENSORS} {args.precision} tensors, got {dict(dtypes)}")

    rows = [f"{_sha256(path)}  {path.relative_to(output).as_posix()}" for path in required]
    (output / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"output": str(args.output), "precision": args.precision, "tensors": len(keys), "dtypes": dict(dtypes)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", default=os.environ.get("OPENPI_JAX_CHECKPOINT", str(DEFAULT_SOURCE)))
    parser.add_argument("--output", default=os.environ.get("OPENPI_PYTORCH_CHECKPOINT"))
    parser.add_argument(
        "--precision",
        choices=tuple(DTYPE_BY_PRECISION),
        default=os.environ.get("OPENPI_CONVERT_PRECISION", "float32"),
    )
    parser.add_argument("--config-name", default=os.environ.get("OPENPI_CONFIG_NAME", "pi05_libero"))
    parser.add_argument("--openpi-root", default=os.environ.get("OPENPI_PATH", str(DEFAULT_OPENPI_ROOT)))
    parser.add_argument("--tokenizer", default=os.environ.get("OPENPI_TOKENIZER_PATH", str(DEFAULT_TOKENIZER)))
    parser.add_argument(
        "--transformers-runtime",
        default=os.environ.get("OPENPI_TRANSFORMERS_RUNTIME_PATH", str(DEFAULT_TRANSFORMERS_OVERLAY)),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.source = _path(args.source)
    args.output = _path(args.output) if args.output else _default_output(args.source, args.precision)
    args.openpi_root = _path(args.openpi_root)
    args.tokenizer = _path(args.tokenizer)
    args.transformers_runtime = _path(args.transformers_runtime)
    _check_inputs(args)

    plan = {
        "source": str(args.source),
        "output": str(args.output),
        "precision": args.precision,
        "converter": str(args.openpi_root / "examples/convert_jax_model_to_pytorch.py"),
        "python": sys.executable,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("Dry run complete; no checkpoint was written.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.staging.", dir=args.output.parent))
    try:
        _run_converter(args, stage)
        _copy_assets(args, stage)
        report = _validate_output(args, stage)
        if args.output.exists():
            args.output.rmdir()
        stage.rename(args.output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))
    print(f"OpenPI PyTorch checkpoint is ready: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
