#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_minimax_h3_video_pdd_fsdp32_32gpu_acp.sh" fl2av "$@"
