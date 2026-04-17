#!/bin/bash
# Run WorldMirror 3D reconstruction (Sequence-Parallel, multi-GPU via torchrun)
#
# 默认开启 Gaussian splats 插值漫游视频渲染（RENDER_VIDEO=1），
# 与 run_worldmirror_recon.sh 行为一致；渲染只在 rank 0 执行。
set -e

export PYTHONPATH="${PYTHONPATH}:/workspace/LightX2V"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

MODEL_PATH=${MODEL_PATH:-/data/nvme1/models/HY-World-2.0}
CONFIG_JSON=${CONFIG_JSON:-/workspace/LightX2V/configs/worldmirror/worldmirror_recon.json}
INPUT_PATH=${INPUT_PATH:-/workspace/HY-World-2.0/examples/worldrecon/realistic/Workspace}
SAVE_RESULT_PATH=${SAVE_RESULT_PATH:-/workspace/LightX2V/inference_output}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-12355}

# 未指定 CUDA_VISIBLE_DEVICES 时，自动挑 NPROC_PER_NODE 张最空闲的卡，
# 避免与其他进程共卡时出现 CUDA context 冲突 / segfault。
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | sort -t, -k2 -n -r | head -n"$NPROC_PER_NODE" \
            | awk -F, '{print $1}' | tr -d ' ' | paste -sd, -)
        export CUDA_VISIBLE_DEVICES
        echo "[run_worldmirror_recon_sp] auto-picked CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
fi

# 渲染视频开关（默认开启；设为 RENDER_VIDEO=0 可关闭）
# RENDER_INTERP_PER_PAIR: 每对相机之间插值的帧数（默认 15）
# RENDER_DEPTH=1: 额外渲染一段 depth 视频
RENDER_VIDEO=${RENDER_VIDEO:-1}
RENDER_INTERP_PER_PAIR=${RENDER_INTERP_PER_PAIR:-15}
RENDER_DEPTH=${RENDER_DEPTH:-0}

RENDER_ARGS=()
if [ "$RENDER_VIDEO" = "1" ]; then
    RENDER_ARGS+=(--save_rendered --render_interp_per_pair "$RENDER_INTERP_PER_PAIR")
    if [ "$RENDER_DEPTH" = "1" ]; then
        RENDER_ARGS+=(--render_depth)
    fi
fi

mkdir -p "$SAVE_RESULT_PATH"

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT -m lightx2v.infer \
    --model_cls worldmirror \
    --task recon \
    --model_path "$MODEL_PATH" \
    --config_json "$CONFIG_JSON" \
    --input_path "$INPUT_PATH" \
    --save_result_path "$SAVE_RESULT_PATH" \
    "${RENDER_ARGS[@]}"

CASE_DIR="$SAVE_RESULT_PATH/$(basename "$INPUT_PATH")"
LATEST_RUN_DIR=$(ls -1dt "$CASE_DIR"/*/ 2>/dev/null | head -n1)
LATEST_RUN_DIR=${LATEST_RUN_DIR%/}

echo "Output saved under: ${LATEST_RUN_DIR:-$CASE_DIR}"
if [ "$RENDER_VIDEO" = "1" ] && [ -n "$LATEST_RUN_DIR" ]; then
    RENDERED_MP4="$LATEST_RUN_DIR/rendered/rendered_rgb.mp4"
    if [ -f "$RENDERED_MP4" ]; then
        echo "Rendered video: $RENDERED_MP4"
    fi
fi
