#!/usr/bin/env bash
set -Eeuo pipefail

H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec bash "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_turbo_fsdp32_32gpu_acp.sh" fl2av "$@"
