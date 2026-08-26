#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../.." && pwd)}"
workspace_root="$(dirname -- "${lightx2v_path}")"
openpi_path="${OPENPI_PATH:-${workspace_root}/openpi}"
openpi_data_root="${OPENPI_DATA_ROOT:-${workspace_root}/openpi_data}"
runtime_path="${OPENPI_TRANSFORMERS_RUNTIME_PATH:-${openpi_data_root}/python_deps/openpi_pytorch_runtime}"
openpi_venv_python="${OPENPI_VENV_PYTHON:-${openpi_path}/.venv/bin/python}"
patch_source="${lightx2v_path}/lightx2v/models/networks/openpi/transformers_replace"
runtime_marker=".lightx2v_openpi_runtime"

transformers_version="4.53.2"
huggingface_hub_version="0.32.3"
tokenizers_version="0.21.1"

python_request="${OPENPI_RUNTIME_PYTHON:-python}"

if ! python_bin="$(command -v -- "${python_request}")" || [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found: ${python_request}" >&2
    exit 1
fi

runtime_path="$(
    "${python_bin}" - "${runtime_path}" "${lightx2v_path}" "${openpi_path}" "${openpi_data_root}" <<'PY'
import sys
import tempfile
from pathlib import Path

target = Path(sys.argv[1]).expanduser().resolve()
protected = {
    "home directory": Path.home().resolve(),
    "LightX2V project": Path(sys.argv[2]).expanduser().resolve(),
    "OpenPI project": Path(sys.argv[3]).expanduser().resolve(),
    "OpenPI data root": Path(sys.argv[4]).expanduser().resolve(),
    "OpenPI Python dependency root": (Path(sys.argv[4]).expanduser().resolve() / "python_deps"),
    "active Python environment": Path(sys.prefix).resolve(),
    "Python base environment": Path(sys.base_prefix).resolve(),
    "system temporary directory": Path(tempfile.gettempdir()).resolve(),
}

if target == Path(target.anchor) or target.parent == Path(target.anchor):
    raise SystemExit(f"Refusing to use a broad runtime target: {target}")
for label, path in protected.items():
    if target == path or path.is_relative_to(target):
        raise SystemExit(f"Refusing to use {target}: it is {label} or one of its parents")

print(target)
PY
)"

patch_files=(
    "models/gemma/configuration_gemma.py"
    "models/gemma/modeling_gemma.py"
    "models/paligemma/modeling_paligemma.py"
    "models/siglip/check.py"
    "models/siglip/modeling_siglip.py"
)

for relative_path in "${patch_files[@]}"; do
    if [[ ! -f "${patch_source}/${relative_path}" ]]; then
        echo "Required OpenPI Transformers patch is missing: ${patch_source}/${relative_path}" >&2
        exit 1
    fi
done

overlay_openpi_patches() {
    local target_root="$1"
    if [[ ! -d "${target_root}/transformers" ]]; then
        echo "Transformers package is missing under ${target_root}" >&2
        return 1
    fi
    cp -a "${patch_source}/." "${target_root}/transformers/"
}

runtime_has_exact_layout() {
    local target_root="$1"
    [[ -d "${target_root}/transformers" ]] \
        && [[ -d "${target_root}/transformers-${transformers_version}.dist-info" ]] \
        && [[ -d "${target_root}/huggingface_hub" ]] \
        && [[ -d "${target_root}/huggingface_hub-${huggingface_hub_version}.dist-info" ]] \
        && [[ -d "${target_root}/tokenizers" ]] \
        && [[ -d "${target_root}/tokenizers-${tokenizers_version}.dist-info" ]]
}

guard_runtime() {
    local target_root="$1"
    PYTHONDONTWRITEBYTECODE=1 USE_FLAX=0 PYTHONPATH="${target_root}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python_bin}" - "${target_root}" "${patch_source}" <<'PY'
import filecmp
import importlib.metadata
from pathlib import Path
import sys

runtime_root = Path(sys.argv[1]).resolve()
patch_root = Path(sys.argv[2]).resolve()

expected_versions = {
    "transformers": "4.53.2",
    "huggingface-hub": "0.32.3",
    "tokenizers": "0.21.1",
}
for distribution, expected in expected_versions.items():
    actual = importlib.metadata.version(distribution)
    if actual != expected:
        raise RuntimeError(f"{distribution} version mismatch: expected {expected}, got {actual}")

import huggingface_hub
import tokenizers
import transformers

for module in (transformers, huggingface_hub, tokenizers):
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(runtime_root):
        raise RuntimeError(f"{module.__name__} was imported outside the private runtime: {module_path}")

patch_files = (
    "models/gemma/configuration_gemma.py",
    "models/gemma/modeling_gemma.py",
    "models/paligemma/modeling_paligemma.py",
    "models/siglip/check.py",
    "models/siglip/modeling_siglip.py",
)
for relative_path in patch_files:
    source = patch_root / relative_path
    installed = runtime_root / "transformers" / relative_path
    if not installed.is_file() or not filecmp.cmp(source, installed, shallow=False):
        raise RuntimeError(f"OpenPI Transformers patch mismatch: {installed}")

from transformers.models.gemma.configuration_gemma import GemmaConfig  # noqa: F401
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm  # noqa: F401
from transformers.models.paligemma import modeling_paligemma  # noqa: F401
from transformers.models.siglip import check, modeling_siglip  # noqa: F401

print(f"private transformers={transformers.__version__} ({transformers.__file__})")
print(f"private huggingface-hub={huggingface_hub.__version__} ({huggingface_hub.__file__})")
print(f"private tokenizers={tokenizers.__version__} ({tokenizers.__file__})")
print("OpenPI Transformers replacement guard: OK")
PY
}

if runtime_has_exact_layout "${runtime_path}"; then
    if [[ -f "${runtime_path}/${runtime_marker}" ]]; then
        echo "Refreshing OpenPI patches in existing private runtime: ${runtime_path}"
        overlay_openpi_patches "${runtime_path}"
        if guard_runtime "${runtime_path}"; then
            echo "OpenPI private PyTorch runtime is ready: ${runtime_path}"
            exit 0
        fi
        echo "Existing private runtime failed validation." >&2
    elif guard_runtime "${runtime_path}"; then
        # Adopt only an already-correct runtime.
        touch "${runtime_path}/${runtime_marker}"
        echo "Validated and adopted existing OpenPI private runtime: ${runtime_path}"
        exit 0
    else
        echo "Refusing to patch an unowned existing runtime: ${runtime_path}" >&2
        echo "Choose an empty OPENPI_TRANSFORMERS_RUNTIME_PATH instead." >&2
        exit 1
    fi
fi

if [[ -e "${runtime_path}" || -L "${runtime_path}" ]] && [[ ! -f "${runtime_path}/${runtime_marker}" ]]; then
    echo "Refusing to replace an existing directory not owned by this setup script: ${runtime_path}" >&2
    echo "Choose an empty OPENPI_TRANSFORMERS_RUNTIME_PATH instead." >&2
    exit 1
fi

runtime_parent="$(dirname -- "${runtime_path}")"
mkdir -p "${runtime_parent}"
stage_dir="$(mktemp -d "${runtime_parent}/.openpi_pytorch_runtime.XXXXXX")"

cleanup() {
    if [[ -n "${stage_dir:-}" && -d "${stage_dir}" ]]; then
        rm -rf -- "${stage_dir}"
    fi
}
trap cleanup EXIT

copied_from_openpi_venv=false
if [[ -x "${openpi_venv_python}" ]]; then
    source_site="$(${openpi_venv_python} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    if [[ -d "${source_site}/transformers" \
        && -d "${source_site}/transformers-${transformers_version}.dist-info" \
        && -d "${source_site}/huggingface_hub" \
        && -d "${source_site}/huggingface_hub-${huggingface_hub_version}.dist-info" \
        && -d "${source_site}/tokenizers" \
        && -d "${source_site}/tokenizers-${tokenizers_version}.dist-info" ]]; then
        echo "Building private runtime from local OpenPI environment: ${source_site}"
        cp -a \
            "${source_site}/transformers" \
            "${source_site}/transformers-${transformers_version}.dist-info" \
            "${source_site}/huggingface_hub" \
            "${source_site}/huggingface_hub-${huggingface_hub_version}.dist-info" \
            "${source_site}/tokenizers" \
            "${source_site}/tokenizers-${tokenizers_version}.dist-info" \
            "${stage_dir}/"
        copied_from_openpi_venv=true
    fi
fi

if [[ "${copied_from_openpi_venv}" != true ]]; then
    echo "Exact local packages were not found; using pip --target fallback."
    "${python_bin}" -m pip install \
        --disable-pip-version-check \
        --no-input \
        --no-deps \
        --target "${stage_dir}" \
        "transformers==${transformers_version}" \
        "huggingface-hub==${huggingface_hub_version}" \
        "tokenizers==${tokenizers_version}"
fi

overlay_openpi_patches "${stage_dir}"
guard_runtime "${stage_dir}"
touch "${stage_dir}/${runtime_marker}"

if [[ -e "${runtime_path}" || -L "${runtime_path}" ]]; then
    if [[ ! -f "${runtime_path}/${runtime_marker}" ]]; then
        echo "Refusing to replace an existing directory not owned by this setup script: ${runtime_path}" >&2
        echo "Choose an empty OPENPI_TRANSFORMERS_RUNTIME_PATH instead." >&2
        exit 1
    fi
    backup_path="${runtime_path}.invalid.$(date -u +%Y%m%dT%H%M%SZ).$$"
    mv -- "${runtime_path}" "${backup_path}"
    echo "Previous invalid runtime was preserved at: ${backup_path}"
fi
mv -- "${stage_dir}" "${runtime_path}"
stage_dir=""

echo "OpenPI private PyTorch runtime is ready: ${runtime_path}"
echo "Use it without changing base packages:"
echo "  PYTHONPATH=${runtime_path}\${PYTHONPATH:+:\$PYTHONPATH} ${python_bin} <command>"
