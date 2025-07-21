#!/bin/bash

# Default values
lightx2v_path=
model_path=

# check section
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0,1,2,3,4,5,6,7
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using default value: ${cuda_devices}, change at shell script or set env variable."
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
fi

if [ -z "${num_gpus}" ]; then
    num_gpus=8
fi

# Check required parameters
if [ -z "$lightx2v_path" ]; then
    echo "Error: lightx2v_path not set"
    exit 1
fi

if [ -z "$model_path" ]; then
    echo "Error: model_path not set"
    exit 1
fi

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export ENABLE_PROFILING_DEBUG=true
export ENABLE_GRAPH_MODE=false
export DTYPE=BF16

# Start multiple servers
python -m lightx2v.api_multi_servers \
    --num_gpus $num_gpus \
    --start_port 8000 \
    --model_cls wan2.1 \
    --task t2v \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/wan/wan_t2v.json
