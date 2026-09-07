#!/usr/bin/env bash
set -eo pipefail

variant=${1:?Usage: bash eval_robotwin_distilled_1step.sh dmd|consistency [Hydra overrides...]}
shift
case "$variant" in
  dmd) name=robotwin_dmd_v2_step30000 ;;
  consistency) name=robotwin_consistency_ema_step30000 ;;
  *) printf 'Unknown variant: %s\n' "$variant" >&2; exit 2 ;;
esac

fastwam_root=${FASTWAM_ROOT:-/mnt/afs_1/lvchengtao/code/wam/MeanFlowWAM}
export_root=${EXPORT_ROOT:-/mnt/miaohua/charles/models/fastwam_release/distilled}
source /mnt/miaohua/charles/envs/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
set -u
python="$CONDA_PREFIX/bin/python"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/miaohua/charles/models/fastWAM-compat
export DIFFSYNTH_SKIP_DOWNLOAD=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUROBO_TORCH_COMPILE_DISABLE=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}

# RoboTwin's activation hook configures its bundled NVIDIA libraries.
unset ROBOTWIN_FORCE_RASTER LIBGL_ALWAYS_SOFTWARE MESA_LOADER_DRIVER_OVERRIDE
: "${ROBOTWIN_NVIDIA_GL_ROOT:?RoboTwin activation must configure ROBOTWIN_NVIDIA_GL_ROOT}"
export VK_ICD_FILENAMES="$ROBOTWIN_NVIDIA_GL_ROOT/nvidia_icd_abs.json"
export LD_PRELOAD="$ROBOTWIN_NVIDIA_GL_ROOT/libGL.so.1.7.0"

"$python" - <<'PY'
from importlib.metadata import version
import pkg_resources

actual = version("sapien")
if actual != "3.0.0b1":
    raise SystemExit(
        f"FastWAM RoboTwin evaluation requires sapien==3.0.0b1, found {actual}. "
        "See https://github.com/yuantianyuan01/FastWAM/issues/23"
    )
PY
cd "$fastwam_root"
export PYTHONPATH="$fastwam_root/src:$fastwam_root${PYTHONPATH:+:$PYTHONPATH}"
out=${OUT:-$fastwam_root/evaluate_results/robotwin/${name}/1step_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$out"

nvidia-smi -L
vulkaninfo --summary | grep -E 'GPU[0-9]+|deviceName|driverName|driverInfo'

"$python" -u experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_distilled_1step \
  ckpt="$export_root/$name.pt" \
  EVALUATION.dataset_stats_path=/mnt/afs_1/lvchengtao/code/wam/MeanFlowWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.eval_num_episodes=100 \
  EVALUATION.num_inference_steps=1 \
  EVALUATION.sigma_shift=5.0 \
  EVALUATION.instruction_type=unseen \
  EVALUATION.skip_get_obs_within_replan=true \
  EVALUATION.timing_enabled=true \
  model.redirect_common_files=false \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2 \
  EVALUATION.output_dir="$out" \
  "$@" 2>&1 | tee "$out/manager.log"
