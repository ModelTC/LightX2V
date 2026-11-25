#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_minimax_h3_distill_32gpu.sh" fl2av "$@"
