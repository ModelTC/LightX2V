#!/usr/bin/env bash
set -Eeuo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_minimax_h3_fl2av_pdmd_official_fsdp32_32gpu_acp.sh" 8 "$@"
