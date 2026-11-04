#!/usr/bin/env bash
# Submit the same command on all seven nodes through the training platform.
# Ranks 0..3 train; ranks 4..6 each serve eight UnifiedReward replicas.
set -euo pipefail
RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LIGHTX2V_RL_CONFIG=qwen_image21_4step_geo_7nodes_ur_only
export LIGHTX2V_UNIFIEDREWARD_PROFILE="${LIGHTX2V_UNIFIEDREWARD_PROFILE:-lightx2v_rl/scripts/rewards/profiles/unifiedreward_edit27b_h100_concurrency2.sh}"
exec bash "$RL_SCRIPT_DIR/train_qwen21_4step.sh" "$@"
