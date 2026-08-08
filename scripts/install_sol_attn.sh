#!/bin/bash
set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${PYTHON_BIN:-python}
repo_dir=${SOL_ATTN_REPO_DIR:-${lightx2v_path}/.cache/Sana-sol-engine}
sol_attn_ref=${SOL_ATTN_REF:-9bfca5c4bf35774a1d44c27b0c3c91041fb8dad0}

# This source revision has been validated on H200 with CUTLASS DSL 4.5.3.
# Newer 4.7.x releases changed CuTe APIs used by the released SM90 kernel.
"${python_bin}" -m pip install "nvidia-cutlass-dsl==4.5.3" "cuda-python>=13.0,<13.2"

"${python_bin}" - <<'PY'
import torch
from packaging.version import Version

if not torch.cuda.is_available():
    raise SystemExit("A visible CUDA GPU is required.")
arch = torch.cuda.get_device_capability()
if arch != (9, 0):
    raise SystemExit(f"This LightX2V install helper currently targets H200/H100 SM90; found SM{arch[0]}{arch[1]}.")
if torch.version.cuda is None or Version(torch.version.cuda) < Version("12.8"):
    raise SystemExit(f"CUDA >=12.8 is required; torch reports CUDA {torch.version.cuda}.")
if Version(torch.__version__.split("+")[0]) < Version("2.10"):
    print(
        f"Warning: the upstream README requires PyTorch >=2.10; found {torch.__version__}. "
        "The pinned SM90 source revision was smoke-tested with PyTorch 2.8 + CUDA 12.8."
    )
print(f"Detected {torch.cuda.get_device_name()} (SM{arch[0]}{arch[1]}), torch={torch.__version__}, CUDA={torch.version.cuda}")
PY

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
from sol_attn import sol_attn

print("Sol-Attn import OK:", sol_attn)
PY
