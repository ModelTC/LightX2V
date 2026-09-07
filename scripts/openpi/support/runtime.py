#!/usr/bin/env python3
"""Prepare the patched Transformers runtime and validate OpenPI/LIBERO.

Transformers remains isolated because OpenPI carries task-specific replacement
files. MuJoCo is installed into and imported directly from the selected base
Python environment.
"""

from __future__ import annotations

import argparse
import filecmp
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
OPENPI_DATA_ROOT = Path(os.environ.get("OPENPI_DATA_ROOT", str(WORKSPACE_ROOT / "openpi_data"))).expanduser().resolve()
DEFAULT_MODEL = OPENPI_DATA_ROOT / "openpi-assets/checkpoints/pi05_libero_pytorch_fp32"
DEFAULT_TRANSFORMERS_OVERLAY = OPENPI_DATA_ROOT / "python_deps/openpi_official_pytorch_runtime"
DEFAULT_LIBERO_ROOT = WORKSPACE_ROOT / "openpi/third_party/libero"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/openpi/pi05_libero.json"
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "configs/openpi/pi05_libero_eval.json"

TRANSFORMERS_EXPECTED = {
    "transformers": ("transformers", "4.53.2"),
    "huggingface-hub": ("huggingface_hub", "0.32.3"),
    "tokenizers": ("tokenizers", "0.21.1"),
}
TRANSFORMERS_PACKAGES = tuple(f"{distribution}=={version}" for distribution, (_module, version) in TRANSFORMERS_EXPECTED.items())

# The official LIBERO client uses this MuJoCo version. All other simulator and
# model dependencies remain untouched in the base Python environment.
BASE_MUJOCO_VERSION = "3.2.3"
BASE_MUJOCO_PACKAGE = f"mujoco=={BASE_MUJOCO_VERSION}"

PATCH_FILES = (
    "models/gemma/configuration_gemma.py",
    "models/gemma/modeling_gemma.py",
    "models/paligemma/modeling_paligemma.py",
    "models/siglip/check.py",
    "models/siglip/modeling_siglip.py",
)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _assert_safe_target(target: Path) -> None:
    target = _resolved(target)
    protected = {
        Path.home().resolve(),
        PROJECT_ROOT.resolve(),
        WORKSPACE_ROOT.resolve(),
        OPENPI_DATA_ROOT.resolve(),
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path(tempfile.gettempdir()).resolve(),
    }
    if target == Path(target.anchor) or target.parent == Path(target.anchor):
        raise RuntimeError(f"refusing broad runtime target: {target}")
    if target.is_relative_to(PROJECT_ROOT.resolve()):
        raise RuntimeError(f"runtime target must stay outside the source checkout: {target}")
    for path in protected:
        if target == path or path.is_relative_to(target):
            raise RuntimeError(f"runtime target is a protected directory or its parent: {target}")


def _patch_root() -> Path:
    return PROJECT_ROOT / "lightx2v/models/networks/openpi/transformers_replace"


def _validate_patch_sources() -> None:
    missing = [str(_patch_root() / relative) for relative in PATCH_FILES if not (_patch_root() / relative).is_file()]
    if missing:
        raise RuntimeError("missing OpenPI Transformers replacement files:\n- " + "\n- ".join(missing))


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _pip_install(target: Path, requirements: tuple[str, ...], dry_run: bool) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-deps",
        "--no-compile",
        "--target",
        str(target),
        *requirements,
    ]
    if dry_run:
        print("+", " ".join(command))
        return
    _run(command)


def _copy_transformers_patches(target: Path) -> None:
    for relative in PATCH_FILES:
        source = _patch_root() / relative
        destination = target / "transformers" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _check_patch_overlay(target: Path) -> None:
    _validate_patch_sources()
    mismatches = []
    for relative in PATCH_FILES:
        source = _patch_root() / relative
        installed = target / "transformers" / relative
        if not installed.is_file() or not filecmp.cmp(source, installed, shallow=False):
            mismatches.append(str(installed))
    if mismatches:
        raise RuntimeError("OpenPI Transformers replacement mismatch:\n- " + "\n- ".join(mismatches))
    print("Transformers replacement set: exact OpenPI vendored copy")


PROBE = r"""
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected = json.loads(sys.argv[2])
result = {}
for distribution, values in expected.items():
    module_name, wanted = values
    actual = importlib.metadata.version(distribution)
    module = importlib.import_module(module_name)
    origin = Path(module.__file__).resolve()
    if actual != wanted:
        raise RuntimeError(f"{distribution}: expected {wanted}, got {actual}")
    if not origin.is_relative_to(root):
        raise RuntimeError(f"{module_name} imported outside overlay: {origin}")
    result[distribution] = {"version": actual, "origin": str(origin)}
print(json.dumps(result, sort_keys=True))
"""


def _probe_overlay(target: Path, expected: dict[str, tuple[str, str]]) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(target),
            "PYTHONDONTWRITEBYTECODE": "1",
            "MUJOCO_GL": "disable",
            "USE_FLAX": "0",
        }
    )
    subprocess.run(
        [sys.executable, "-c", PROBE, str(target), json.dumps(expected)],
        check=True,
        env=env,
    )


def _prepare_transformers(target: Path, dry_run: bool) -> None:
    target = _resolved(target)
    _assert_safe_target(target)
    _validate_patch_sources()
    if target.exists():
        try:
            _probe_overlay(target, TRANSFORMERS_EXPECTED)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"refusing to modify an invalid Transformers overlay: {target}") from exc
        try:
            _check_patch_overlay(target)
        except RuntimeError:
            pass
        else:
            print(f"transformers overlay already valid: {target}")
            return
        print(f"repair Transformers replacements: {target}")
        if dry_run:
            for relative in PATCH_FILES:
                print(f"copy {_patch_root() / relative} -> {target / 'transformers' / relative}")
            return
        _copy_transformers_patches(target)
        _check_patch_overlay(target)
        _probe_overlay(target, TRANSFORMERS_EXPECTED)
        print(f"transformers overlay ready: {target}")
        return

    print(f"prepare transformers overlay: {target}")
    if dry_run:
        _pip_install(target, TRANSFORMERS_PACKAGES, dry_run=True)
        for relative in PATCH_FILES:
            print(f"copy {_patch_root() / relative} -> {target / 'transformers' / relative}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent))
    try:
        _pip_install(stage, TRANSFORMERS_PACKAGES, dry_run=False)
        _copy_transformers_patches(stage)
        _check_patch_overlay(stage)
        _probe_overlay(stage, TRANSFORMERS_EXPECTED)
        if target.exists():
            raise RuntimeError(f"runtime target appeared while preparing it: {target}")
        stage.replace(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"transformers overlay ready: {target}")


def _base_site_packages() -> Path:
    return _resolved(sysconfig.get_paths()["purelib"])


def _base_mujoco_ready() -> bool:
    try:
        if importlib.metadata.version("mujoco") != BASE_MUJOCO_VERSION:
            return False
        spec = importlib.util.find_spec("mujoco")
        if spec is None or spec.origin is None:
            return False
        return _resolved(spec.origin).is_relative_to(_base_site_packages())
    except (ImportError, importlib.metadata.PackageNotFoundError, ValueError):
        return False


def _prepare_base_mujoco(dry_run: bool) -> None:
    if _base_mujoco_ready():
        print(f"base MuJoCo already valid: {BASE_MUJOCO_VERSION} ({_base_site_packages() / 'mujoco'})")
        return
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-deps",
        "--upgrade",
        BASE_MUJOCO_PACKAGE,
    ]
    if dry_run:
        print("+", " ".join(command))
        return
    _run(command)
    if not _base_mujoco_ready():
        raise RuntimeError(f"MuJoCo {BASE_MUJOCO_VERSION} was not installed in the base site-packages")


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return value


def _check_static_inputs(args: argparse.Namespace) -> None:
    model = _resolved(args.model_path)
    required_model_files = (
        model / "model.safetensors",
        model / "config.json",
        model / "assets/paligemma_tokenizer.model",
        model / "assets/physical-intelligence/libero/norm_stats.json",
    )
    missing = [str(path) for path in required_model_files if not path.is_file()]
    if missing:
        raise RuntimeError("missing checkpoint artifacts:\n- " + "\n- ".join(missing))
    checkpoint_config = _load_json(model / "config.json", "checkpoint config")
    precision = checkpoint_config.get("precision")
    expected_dtype = {"float32": "F32", "bfloat16": "BF16"}.get(precision)
    if expected_dtype is None:
        raise RuntimeError(f"checkpoint precision must be float32 or bfloat16, got {precision!r}")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("base environment is missing safetensors") from exc
    tensor_dtypes: dict[str, str] = {}
    with safe_open(model / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            dtype = checkpoint.get_slice(name).get_dtype()
            tensor_dtypes[dtype] = tensor_dtypes.get(dtype, 0) + 1
    expected_tensors = {expected_dtype: 812}
    if tensor_dtypes != expected_tensors:
        raise RuntimeError(f"expected 812 {precision} checkpoint tensors, got {tensor_dtypes}")
    print(f"checkpoint tensor manifest: 812/812 {expected_dtype}")

    _load_json(_resolved(args.model_config), "model config")
    _load_json(_resolved(args.eval_config), "evaluation config")

    libero_root = _resolved(args.libero_root)
    required_libero = (
        libero_root / "libero/libero/bddl_files",
        libero_root / "libero/libero/init_files",
        libero_root / "libero/libero/assets",
    )
    missing = [str(path) for path in required_libero if not path.is_dir()]
    if missing:
        raise RuntimeError("incomplete official LIBERO checkout:\n- " + "\n- ".join(missing))


COMBINED_PROBE = r"""
import importlib.metadata
import json
import sys
import sysconfig
from pathlib import Path

transformers_root = Path(sys.argv[1]).resolve()
libero_root = Path(sys.argv[2]).resolve()
base_site = Path(sysconfig.get_paths()["purelib"]).resolve()

import numpy
import torch
import transformers
import mujoco
import robosuite

transformers_origin = Path(transformers.__file__).resolve()
if not transformers_origin.is_relative_to(transformers_root):
    raise RuntimeError(f"transformers imported outside overlay: {transformers_origin}")
mujoco_origin = Path(mujoco.__file__).resolve()
if importlib.metadata.version("mujoco") != "3.2.3" or mujoco.__version__ != "3.2.3":
    raise RuntimeError(
        f"base MuJoCo must be 3.2.3, got distribution={importlib.metadata.version('mujoco')} module={mujoco.__version__}"
    )
if not mujoco_origin.is_relative_to(base_site):
    raise RuntimeError(f"mujoco must be imported from base site-packages {base_site}, got {mujoco_origin}")
torch_origin = Path(torch.__file__).resolve()
if torch_origin.is_relative_to(transformers_root):
    raise RuntimeError(f"torch was shadowed by an overlay: {torch_origin}")
if importlib.metadata.version("robosuite") != "1.4.1":
    raise RuntimeError(f"validated base robosuite must remain 1.4.1, got {importlib.metadata.version('robosuite')}")
for module in (numpy, robosuite):
    origin = Path(module.__file__).resolve()
    if origin.is_relative_to(transformers_root):
        raise RuntimeError(f"base package {module.__name__} was shadowed by an overlay: {origin}")

sys.path.insert(0, str(libero_root))
import libero
import libero.libero as libero_package
from libero.libero import benchmark

namespace_paths = [str(Path(path).resolve()) for path in libero.__path__]
for module in (libero_package, benchmark):
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(libero_root):
        raise RuntimeError(f"{module.__name__} imported outside the official LIBERO root: {origin}")
payload = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_origin": str(torch_origin),
    "cuda_available": torch.cuda.is_available(),
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "mujoco": mujoco.__version__,
    "mujoco_origin": str(mujoco_origin),
    "base_site_packages": str(base_site),
    "pillow": importlib.metadata.version("Pillow"),
    "pyopengl": importlib.metadata.version("PyOpenGL"),
    "glfw": importlib.metadata.version("glfw"),
    "robosuite": importlib.metadata.version("robosuite"),
    "libero_namespace": namespace_paths,
    "libero_suites": sorted(benchmark.get_benchmark_dict()),
}
print(json.dumps(payload, sort_keys=True))
if sys.argv[3] == "1" and not payload["cuda_available"]:
    raise RuntimeError("CUDA is not available to the base interpreter")
"""


def _check_runtime(args: argparse.Namespace) -> None:
    expected_python = _resolved(args.expected_python)
    if _resolved(sys.executable) != expected_python:
        raise RuntimeError(f"runtime check must use {expected_python}, got {_resolved(sys.executable)}")

    transformers_runtime = _resolved(args.transformers_runtime)
    _probe_overlay(transformers_runtime, TRANSFORMERS_EXPECTED)
    _check_patch_overlay(transformers_runtime)
    _check_static_inputs(args)

    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="lightx2v-openpi-runtime-check-") as cache_dir:
        libero_config_dir = Path(cache_dir) / "libero_config"
        libero_config_dir.mkdir()
        benchmark_root = _resolved(args.libero_root) / "libero/libero"
        (libero_config_dir / "config.yaml").write_text(
            "\n".join(
                (
                    f"benchmark_root: {benchmark_root}",
                    f"bddl_files: {benchmark_root / 'bddl_files'}",
                    f"init_states: {benchmark_root / 'init_files'}",
                    f"datasets: {_resolved(args.libero_root) / 'libero/datasets'}",
                    f"assets: {benchmark_root / 'assets'}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        env.update(
            {
                "PYTHONPATH": os.pathsep.join((str(transformers_runtime), str(PROJECT_ROOT))),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "MUJOCO_GL": "disable",
                "LIBERO_CONFIG_PATH": str(libero_config_dir),
                "NUMBA_CACHE_DIR": cache_dir,
                "USE_FLAX": "0",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        command = [
            sys.executable,
            "-c",
            COMBINED_PROBE,
            str(transformers_runtime),
            str(_resolved(args.libero_root)),
            "0" if args.no_cuda else "1",
        ]
        completed = subprocess.run(command, env=env, text=True, capture_output=True)
        if completed.returncode != 0:
            if completed.stdout:
                print(completed.stdout, end="", file=sys.stderr)
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            raise RuntimeError(f"combined OpenPI runtime probe failed with exit code {completed.returncode}")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    print(completed.stdout.strip())
    print("OpenPI runtime check: OK")


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--transformers-runtime",
        default=os.environ.get("OPENPI_TRANSFORMERS_RUNTIME_PATH", str(DEFAULT_TRANSFORMERS_OVERLAY)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare patched Transformers and base MuJoCo runtimes")
    _add_paths(prepare)
    prepare.add_argument("--component", choices=("all", "transformers", "mujoco"), default="all")
    prepare.add_argument("--dry-run", action="store_true")

    check = subparsers.add_parser("check", help="validate runtime, checkpoint, configs, and official LIBERO paths")
    _add_paths(check)
    check.add_argument("--expected-python", default=os.environ.get("OPENPI_PYTHON", sys.executable))
    check.add_argument("--model-path", default=os.environ.get("OPENPI_MODEL_PATH", str(DEFAULT_MODEL)))
    check.add_argument("--model-config", default=os.environ.get("OPENPI_CONFIG", str(DEFAULT_MODEL_CONFIG)))
    check.add_argument("--eval-config", default=os.environ.get("OPENPI_EVAL_CONFIG", str(DEFAULT_EVAL_CONFIG)))
    check.add_argument("--libero-root", default=os.environ.get("OPENPI_LIBERO_ROOT", str(DEFAULT_LIBERO_ROOT)))
    check.add_argument("--no-cuda", action="store_true", help="allow validation on a host without a visible CUDA device")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            if args.component in {"all", "transformers"}:
                _prepare_transformers(_resolved(args.transformers_runtime), args.dry_run)
            if args.component in {"all", "mujoco"}:
                _prepare_base_mujoco(args.dry_run)
        else:
            _check_runtime(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
