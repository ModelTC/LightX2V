#!/bin/bash

if [ -z "${lightx2v_path}" ]; then
    echo "Error: lightx2v_path is not set. Please set this variable first."
    exit 1
fi

if [ -z "${model_path}" ]; then
    echo "Error: model_path is not set. Please set this variable first."
    exit 1
fi

export PYTHONPATH=${lightx2v_path}:$PYTHONPATH

# always set false to avoid some warnings
export TOKENIZERS_PARALLELISM=false

# =====================================================================================
# ⚠️  IMPORTANT CONFIGURATION PARAMETERS - READ CAREFULLY AND MODIFY WITH CAUTION ⚠️
# =====================================================================================

# Model Inference Data Type Setting (IMPORTANT!)
# Key parameter affecting model accuracy and performance
# Available options: [BF16, FP16]
# If not set, default value: BF16
export DTYPE=BF16

# Sensitive Layer Data Type Setting (IMPORTANT!)
# Used for layers requiring higher precision
# Available options: [FP32, None]
# If not set, default value: None (follows DTYPE setting)
# Note: If set to FP32, it will be slower, so we recommend set ENABLE_GRAPH_MODE to true.
export SENSITIVE_LAYER_DTYPE=FP32

# Performance Profiling Debug Mode (Debug Only)
# Enables detailed performance analysis output, such as time cost and memory usage
# Available options: [true, false]
# If not set, default value: false
# Note: This option can be set to false for production.
export ENABLE_PROFILING_DEBUG=true

# Graph Mode Optimization (Performance Enhancement)
# Enables torch.compile for graph optimization, can improve inference performance
# Available options: [true, false]
# If not set, default value: false
# Note: First run may require compilation time, subsequent runs will be faster
# Note: When you use lightx2v as a service, you can set this option to true.
export ENABLE_GRAPH_MODE=true


echo "==============================================================================="
echo "LightX2V Base Environment Variables Summary:"
echo "-------------------------------------------------------------------------------"
echo "lightx2v_path: ${lightx2v_path}"
echo "model_path: ${model_path}"
echo "-------------------------------------------------------------------------------"
echo "Model Inference Data Type: ${DTYPE}"
echo "Sensitive Layer Data Type: ${SENSITIVE_LAYER_DTYPE}"
echo "Performance Profiling Debug Mode: ${ENABLE_PROFILING_DEBUG}"
echo "Graph Mode Optimization: ${ENABLE_GRAPH_MODE}"
echo "==============================================================================="
