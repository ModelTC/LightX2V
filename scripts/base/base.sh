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
export SENSITIVE_LAYER_DTYPE=None

# Performance Profiling Debug Level (Debug Only)
# Enables detailed performance analysis output, such as time cost and memory usage
# Available options: [0, 1, 2]
# If not set, default value: 0
# Note: This option can be set to 0 for production.
export PROFILING_DEBUG_LEVEL=2


echo "==============================================================================="
echo "LightX2V Base Environment Variables Summary:"
echo "-------------------------------------------------------------------------------"
echo "lightx2v_path: ${lightx2v_path}"
echo "model_path: ${model_path}"
echo "-------------------------------------------------------------------------------"
echo "Model Inference Data Type: ${DTYPE}"
echo "Sensitive Layer Data Type: ${SENSITIVE_LAYER_DTYPE}"
echo "Performance Profiling Debug Level: ${PROFILING_DEBUG_LEVEL}"
echo "==============================================================================="
