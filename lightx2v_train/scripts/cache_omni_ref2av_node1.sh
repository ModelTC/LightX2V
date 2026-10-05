#!/usr/bin/env bash
set -Eeuo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/build_minimax_h3_omni_imageonly_cache.sh" 1 "$@"
