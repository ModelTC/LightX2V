#!/usr/bin/env bash

set -euo pipefail

openpi_path="${OPENPI_PATH:-/data/liuhongda/openpi}"
source_checkpoint="${OPENPI_JAX_CHECKPOINT:-/data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_libero}"
output_checkpoint="${OPENPI_PYTORCH_CHECKPOINT:-/data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_libero_pytorch}"
tokenizer_path="${OPENPI_TOKENIZER_PATH:-/data/liuhongda/openpi_data/big_vision/paligemma_tokenizer.model}"
python_bin="${OPENPI_CONVERT_PYTHON:-${openpi_path}/.venv/bin/python}"
transformers_dir="${openpi_path}/.venv/lib/python3.11/site-packages/transformers"
patch_dir="${openpi_path}/src/openpi/models_pytorch/transformers_replace"

for required in \
    "${python_bin}" \
    "${openpi_path}/examples/convert_jax_model_to_pytorch.py" \
    "${source_checkpoint}/params/_METADATA" \
    "${source_checkpoint}/assets/physical-intelligence/libero/norm_stats.json" \
    "${tokenizer_path}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required conversion input is missing: ${required}" >&2
        exit 1
    fi
done

if [[ -f "${output_checkpoint}/model.safetensors" && "${OPENPI_FORCE_CONVERT:-0}" != "1" ]]; then
    if [[ -f "${output_checkpoint}/SHA256SUMS" ]]; then
        (
            cd "${output_checkpoint}"
            sha256sum -c SHA256SUMS
        )
    else
        echo "Existing checkpoint has no SHA256SUMS; validating its tensor manifest before creating one." >&2
    fi
    echo "Converted checkpoint already exists: ${output_checkpoint}/model.safetensors"
    echo "Set OPENPI_FORCE_CONVERT=1 only when you intentionally want to overwrite it."
else
    # The official PyTorch implementation needs five OpenPI replacements on
    # top of transformers==4.53.2.  This changes only OpenPI's own .venv, not
    # the user's base environment.
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

# Upstream's converter looks for checkpoint_dir.parent/assets, while the
# released local layout stores assets under checkpoint_dir/assets.
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
if len(keys) != 812 or parameter_count != 3_616_757_520 or dtypes != {"BF16"}:
    raise RuntimeError(
        "Converted pi05_libero tensor inventory mismatch: "
        f"keys={len(keys)}, parameters={parameter_count}, dtypes={sorted(dtypes)}"
    )
if manifest_sha256 != expected_manifest_sha256:
    raise RuntimeError(
        "Converted pi05_libero key/shape/dtype manifest mismatch: "
        f"expected {expected_manifest_sha256}, got {manifest_sha256}"
    )

required_prefixes = (
    "paligemma_with_expert.paligemma.",
    "paligemma_with_expert.gemma_expert.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
)
missing_prefixes = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in keys)]
if missing_prefixes:
    raise RuntimeError(f"SafeTensors is missing OpenPI parameter groups: {missing_prefixes}")
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

echo "pi05_libero PyTorch checkpoint is ready: ${output_checkpoint}"
