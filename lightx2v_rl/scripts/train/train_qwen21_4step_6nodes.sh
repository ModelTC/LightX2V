#!/usr/bin/env bash
# Compatibility entry point: four training nodes + two 9B reward nodes.
set -euo pipefail
RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$RL_SCRIPT_DIR/train_qwen21_4step.sh" "$@"
