#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="$(realpath -m -- "${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../.." && pwd)}")"
workspace_root="$(dirname -- "${lightx2v_path}")"
openpi_path="$(realpath -m -- "${OPENPI_PATH:-${workspace_root}/openpi}")"
openpi_data_root="$(realpath -m -- "${OPENPI_DATA_ROOT:-${workspace_root}/openpi_data}")"
source_checkpoint="$(realpath -m -- "${OPENPI_JAX_CHECKPOINT:-${openpi_data_root}/openpi-assets/checkpoints/pi05_libero}")"
output_checkpoint="$(realpath -m -- "${OPENPI_PYTORCH_CHECKPOINT:-${openpi_data_root}/openpi-assets/checkpoints/pi05_libero_pytorch}")"
tokenizer_path="$(realpath -m -- "${OPENPI_TOKENIZER_PATH:-${openpi_data_root}/big_vision/paligemma_tokenizer.model}")"
python_bin="${OPENPI_CONVERT_PYTHON:-${openpi_path}/.venv/bin/python}"
patch_dir="${openpi_path}/src/openpi/models_pytorch/transformers_replace"
checkpoint_marker=".lightx2v_openpi_checkpoint"

if ! python_bin="$(command -v -- "${python_bin}")" || [[ ! -x "${python_bin}" ]]; then
    echo "Conversion Python executable not found: ${OPENPI_CONVERT_PYTHON:-${openpi_path}/.venv/bin/python}" >&2
    exit 1
fi

"${python_bin}" - <<'PY'
import importlib.metadata
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit(
        "OPENPI_CONVERT_PYTHON must belong to an isolated virtual environment; "
        f"refusing to patch the base environment at {sys.prefix}"
    )
version = importlib.metadata.version("transformers")
if version != "4.53.2":
    raise SystemExit(f"Conversion environment requires transformers==4.53.2, got {version}")
PY

for required in \
    "${openpi_path}/examples/convert_jax_model_to_pytorch.py" \
    "${source_checkpoint}/params/_METADATA" \
    "${source_checkpoint}/assets/physical-intelligence/libero/norm_stats.json" \
    "${tokenizer_path}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required conversion input is missing: ${required}" >&2
        exit 1
    fi
done

# The upstream converter overwrites files below its target, so guard that
# target before invoking it.
output_checkpoint="$(
    "${python_bin}" - \
        "${output_checkpoint}" \
        "${source_checkpoint}" \
        "${lightx2v_path}" \
        "${openpi_path}" \
        "${openpi_data_root}" <<'PY'
import sys
import tempfile
from pathlib import Path

target = Path(sys.argv[1]).expanduser().resolve()
source = Path(sys.argv[2]).expanduser().resolve()
protected = {
    "home directory": Path.home().resolve(),
    "LightX2V project": Path(sys.argv[3]).expanduser().resolve(),
    "OpenPI project": Path(sys.argv[4]).expanduser().resolve(),
    "OpenPI data root": Path(sys.argv[5]).expanduser().resolve(),
    "conversion Python environment": Path(sys.prefix).resolve(),
    "conversion Python base environment": Path(sys.base_prefix).resolve(),
    "system temporary directory": Path(tempfile.gettempdir()).resolve(),
}

if target == Path(target.anchor) or target.parent == Path(target.anchor):
    raise SystemExit(f"Refusing to use a broad checkpoint target: {target}")
if target == source or target.is_relative_to(source) or source.is_relative_to(target):
    raise SystemExit(f"Output checkpoint must not overlap the JAX source checkpoint: {target}")
for label, path in protected.items():
    if target == path or path.is_relative_to(target):
        raise SystemExit(f"Refusing to use {target}: it is {label} or one of its parents")

print(target)
PY
)"

if [[ -e "${output_checkpoint}" || -L "${output_checkpoint}" ]]; then
    if [[ ! -d "${output_checkpoint}" ]]; then
        echo "PyTorch checkpoint target is not a directory: ${output_checkpoint}" >&2
        exit 1
    fi
    output_has_entries="$(find "${output_checkpoint}" -mindepth 1 -maxdepth 1 -print -quit)"
    if [[ -n "${output_has_entries}" && ! -f "${output_checkpoint}/${checkpoint_marker}" ]]; then
        if [[ "${OPENPI_FORCE_CONVERT:-0}" == "1" ]]; then
            echo "Refusing forced overwrite of an unowned checkpoint directory: ${output_checkpoint}" >&2
            exit 1
        fi
        if [[ ! -f "${output_checkpoint}/model.safetensors" || ! -f "${output_checkpoint}/SHA256SUMS" ]]; then
            echo "Refusing to write into a non-empty directory not owned by this converter: ${output_checkpoint}" >&2
            echo "Choose an empty OPENPI_PYTORCH_CHECKPOINT directory instead." >&2
            exit 1
        fi
        (
            cd "${output_checkpoint}"
            sha256sum -c SHA256SUMS
        )
        touch "${output_checkpoint}/${checkpoint_marker}"
        echo "Validated and adopted existing converted checkpoint: ${output_checkpoint}"
    fi
fi

transformers_dir="$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/transformers"
if [[ ! -d "${transformers_dir}" ]]; then
    echo "Transformers package is missing from the conversion environment: ${transformers_dir}" >&2
    exit 1
fi

if [[ -f "${output_checkpoint}/model.safetensors" && "${OPENPI_FORCE_CONVERT:-0}" != "1" ]]; then
    if [[ -f "${output_checkpoint}/SHA256SUMS" ]]; then
        (
            cd "${output_checkpoint}"
            sha256sum -c SHA256SUMS
        )
    else
        echo "Existing owned checkpoint has no SHA256SUMS; rebuilding it after validation." >&2
    fi
    echo "Converted checkpoint already exists: ${output_checkpoint}/model.safetensors"
    echo "Set OPENPI_FORCE_CONVERT=1 only when you intentionally want to overwrite it."
else
    # Patch only the isolated conversion environment checked above.
    cp -a "${patch_dir}/." "${transformers_dir}/"
    (
        cd "${openpi_path}"
        "${python_bin}" -u examples/convert_jax_model_to_pytorch.py \
            --checkpoint-dir "${source_checkpoint}" \
            --config-name pi05_libero \
            --output-path "${output_checkpoint}" \
            --precision bfloat16
    )
fi

# Keep runtime assets beside the converted checkpoint.
mkdir -p "${output_checkpoint}/assets"
cp -a "${source_checkpoint}/assets/." "${output_checkpoint}/assets/"
cp -a "${tokenizer_path}" "${output_checkpoint}/assets/paligemma_tokenizer.model"

"${python_bin}" - "${output_checkpoint}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

from safetensors import safe_open

root = Path(sys.argv[1])
required = (
    root / "model.safetensors",
    root / "config.json",
    root / "assets/physical-intelligence/libero/norm_stats.json",
    root / "assets/paligemma_tokenizer.model",
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Converted checkpoint is incomplete: {missing}")
with safe_open(root / "model.safetensors", framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    manifest_rows = []
    parameter_count = 0
    dtypes = set()
    for key in keys:
        tensor_slice = handle.get_slice(key)
        shape = tuple(tensor_slice.get_shape())
        dtype = tensor_slice.get_dtype()
        manifest_rows.append(f"{key}|{dtype}|{shape}")
        parameter_count += math.prod(shape)
        dtypes.add(dtype)

manifest_sha256 = hashlib.sha256(("\n".join(manifest_rows) + "\n").encode()).hexdigest()
expected_manifest_sha256 = "ee81d609ff73d395731f9f3df3b0caefcbd17f83d2ff153d26166e1bd024e20d"
if manifest_sha256 != expected_manifest_sha256:
    raise RuntimeError(
        "Converted pi05_libero key/shape/dtype manifest mismatch: "
        f"expected {expected_manifest_sha256}, got {manifest_sha256}"
    )

print(
    json.dumps(
        {
            "checkpoint": str(root),
            "tensor_keys": len(keys),
            "parameters": parameter_count,
            "tensor_manifest_sha256": manifest_sha256,
            "bytes": (root / "model.safetensors").stat().st_size,
        },
        indent=2,
    )
)
PY

(
    cd "${output_checkpoint}"
    sha256sum \
        model.safetensors \
        config.json \
        assets/paligemma_tokenizer.model \
        assets/physical-intelligence/libero/norm_stats.json \
        > SHA256SUMS
)
touch "${output_checkpoint}/${checkpoint_marker}"

echo "pi05_libero PyTorch checkpoint is ready: ${output_checkpoint}"
