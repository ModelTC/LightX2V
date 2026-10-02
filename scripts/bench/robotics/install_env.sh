#!/usr/bin/env bash
set -euo pipefail

benchmark="${1:-}"
case "$benchmark" in
    libero|robotwin) ;;
    *) echo "Usage: bash scripts/bench/robotics/install_env.sh {libero|robotwin}"; exit 1 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."
env_dir="$PWD/.venvs/$benchmark"
simulator="$PWD/lightx2v_ros/src/simulator/simulator"

# Keep the base interpreter on the same filesystem as the environments.
export UV_PYTHON_INSTALL_DIR="$PWD/.venvs/python"
uv venv --python 3.10 --managed-python --allow-existing "$env_dir"
export PATH="$env_dir/bin:$PATH"
# egl-probe's build invokes cmake before the simulator packages are installed.
uv pip install --python "$env_dir/bin/python" 'cmake<4' ninja
uv pip install --python "$env_dir/bin/python" --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
uv pip install --python "$env_dir/bin/python" -r "$script_dir/requirements_$benchmark.txt" -e .

if [[ "$benchmark" == libero ]]; then
    git submodule update --init "$simulator/libero_node/LIBERO" "$simulator/libero_node/LIBERO-plus"
else
    robotwin="$simulator/robotwin_node/RoboTwin"
    git submodule update --init "$robotwin"
    if [[ ! -d "$robotwin/envs/curobo" ]]; then
        git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$robotwin/envs/curobo"
    fi
    MAX_JOBS="${MAX_JOBS:-8}" uv pip install --python "$env_dir/bin/python" \
        --no-build-isolation --no-deps -e "$robotwin/envs/curobo"

    # Compatibility fixes from the pinned RoboTwin installation instructions.
    site_dir=$("$env_dir/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
    sed -i 's/open(urdf_file, "r")/open(urdf_file, "r", encoding="utf-8")/; s/open(srdf_file, "r")/open(srdf_file, "r", encoding="utf-8")/' "$site_dir/sapien/wrapper/urdf_loader.py"
    sed -i 's/if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:/if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:/' "$site_dir/mplib/planner.py"
fi

echo "Activate with: source $env_dir/bin/activate"
