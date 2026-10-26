#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-${lightx2v_path}/configs/self_forcing/wan_t2v_sf_fp8_f16_accum_lighttae_5090.json}"
export OUTPUT_PATH="${OUTPUT_PATH:-${lightx2v_path}/save_results/output_lightx2v_wan_t2v_sf_fp8.mp4}"
exec bash "${lightx2v_path}/scripts/self_forcing/run_wan_t2v_sf.sh"
