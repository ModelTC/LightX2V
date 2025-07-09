#!/bin/bash

# Set paths
lightx2v_path=
model_path=

# Check parameters
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using default value: ${cuda_devices}"
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
fi

if [ -z "${lightx2v_path}" ]; then
    echo "Error: lightx2v_path is not set. Please set this variable first."
    exit 1
fi

if [ -z "${model_path}" ]; then
    echo "Error: model_path is not set. Please set this variable first."
    exit 1
fi

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export ENABLE_PROFILING_DEBUG=true
export ENABLE_GRAPH_MODE=false
export DTYPE=BF16

echo "=========================================="
echo "Starting distributed inference API server"
echo "Model path: $model_path"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "API port: 8000"
echo "=========================================="

# Start API server with distributed inference service
python -m lightx2v.api_server \
--model_cls wan2.1 \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan/wan_i2v_dist.json \
--port 8000 \
--start_inference \
--nproc_per_node 1

echo "Service stopped"
