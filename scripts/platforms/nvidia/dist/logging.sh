#!/bin/bash

LOG_DIR=${LOG_DIR:-${lightx2v_path}/logs/platforms/nvidia/dist}
mkdir -p "${LOG_DIR}"

script_name=$(basename "$0" .sh)
timestamp=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${script_name}_${timestamp}.log}

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"
