#!/bin/bash
set -e

export PYTHONPATH="${PYTHONPATH}:/workspace/LightX2V"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

MODEL_PATH=${MODEL_PATH:-/data/nvme1/models/HY-World-2.0}
CONFIG_JSON=${CONFIG_JSON:-/workspace/LightX2V/configs/worldmirror/worldmirror_recon_fp8.json}
INPUT_PATH=${INPUT_PATH:-/workspace/HY-World-2.0/examples/worldrecon/realistic/Workspace}
SAVE_RESULT_PATH=${SAVE_RESULT_PATH:-/workspace/LightX2V/inference_output}

mkdir -p "$SAVE_RESULT_PATH"

python -m lightx2v.infer \
    --model_cls worldmirror \
    --task recon \
    --model_path "$MODEL_PATH" \
    --config_json "$CONFIG_JSON" \
    --input_path "$INPUT_PATH" \
    --save_result_path "$SAVE_RESULT_PATH"

echo "Output saved under: $SAVE_RESULT_PATH/$(basename "$INPUT_PATH")/"
