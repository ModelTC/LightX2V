#!/bin/bash
set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${PYTHON_BIN:-python}
repo_dir=${SOL_ATTN_REPO_DIR:-${lightx2v_path}/.cache/Sana-sol-engine}
sm90_ref=9bfca5c4bf35774a1d44c27b0c3c91041fb8dad0
sm120_ref=71350faed59a6865173ddfd5a7d4471da4bde328

# Check compatibility before changing the Python environment. SM120 support
# landed after the source revision validated for SM90, so each architecture has
# its own pinned revision and CUTLASS DSL version.
device_info=$("${python_bin}" - <<'PY'
import sys

import torch
from packaging.version import Version

if not torch.cuda.is_available():
    raise SystemExit("A visible CUDA GPU is required.")
arch = torch.cuda.get_device_capability()
if arch not in ((9, 0), (12, 0)):
    raise SystemExit(f"This LightX2V install helper supports SM90 and SM120; found SM{arch[0]}{arch[1]}.")
if torch.version.cuda is None or Version(torch.version.cuda) < Version("12.8"):
    raise SystemExit(f"CUDA >=12.8 is required; torch reports CUDA {torch.version.cuda}.")
if Version(torch.__version__.split("+")[0]) < Version("2.10"):
    if arch == (12, 0):
        raise SystemExit(f"SM120 Sol-Attn requires PyTorch >=2.10; found {torch.__version__}.")
    print(
        f"Warning: upstream requires PyTorch >=2.10; found {torch.__version__}. "
        "The pinned SM90 revision was smoke-tested with PyTorch 2.8 + CUDA 12.8.",
        file=sys.stderr,
    )
print(f"{arch[0]}.{arch[1]}|{torch.version.cuda}|{torch.__version__}|{torch.cuda.get_device_name()}")
PY
)
IFS='|' read -r gpu_arch torch_cuda torch_version gpu_name <<<"${device_info}"

case "${gpu_arch}" in
    9.0)
        sol_attn_ref=${SOL_ATTN_REF:-${sm90_ref}}
        cutlass_dsl_version=${SOL_ATTN_CUTLASS_DSL_VERSION:-4.5.3}
        ;;
    12.0)
        sol_attn_ref=${SOL_ATTN_REF:-${sm120_ref}}
        # vLLM 0.23 pins 4.5.2, and the SM120 kernel works with that version.
        cutlass_dsl_version=${SOL_ATTN_CUTLASS_DSL_VERSION:-4.5.2}
        ;;
esac

cuda_major=${torch_cuda%%.*}
case "${cuda_major}" in
    12|13) cutlass_extra=cu${cuda_major} ;;
    *)
        echo "Unsupported CUDA major version: ${torch_cuda}" >&2
        exit 2
        ;;
esac

echo "Detected ${gpu_name} (SM${gpu_arch/./}), torch=${torch_version}, CUDA=${torch_cuda}"
"${python_bin}" -m pip install "nvidia-cutlass-dsl[${cutlass_extra}]==${cutlass_dsl_version}"

mkdir -p "$(dirname "${repo_dir}")"
if [[ -d "${repo_dir}/.git" ]]; then
    if [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no)" ]]; then
        echo "Refusing to update dirty Sol-Attn checkout: ${repo_dir}" >&2
        exit 3
    fi
    git -C "${repo_dir}" fetch origin sol-engine
elif [[ -e "${repo_dir}" ]]; then
    echo "SOL_ATTN_REPO_DIR exists but is not a Git checkout: ${repo_dir}" >&2
    exit 3
else
    git clone --branch sol-engine --single-branch https://github.com/NVlabs/Sana.git "${repo_dir}"
fi
git -C "${repo_dir}" checkout "${sol_attn_ref}"
"${python_bin}" -m pip install -e "${repo_dir}/techniques/sparse_backends"

"${python_bin}" - <<'PY'
from sol_attn import get_sol_attn_backend, sol_attn

print("Sol-Attn import OK:", sol_attn)
print("Selected backend:", get_sol_attn_backend("cuda"))
PY
