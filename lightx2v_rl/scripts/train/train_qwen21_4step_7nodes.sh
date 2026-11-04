#!/usr/bin/env bash
# Four training nodes + three reward nodes (20 UnifiedReward 27B + 4 HPSv3).
set -euo pipefail
RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LIGHTX2V_RL_CONFIG="${LIGHTX2V_RL_CONFIG:-qwen_image21_4step_geo_7nodes_hps02}"
export LIGHTX2V_UNIFIEDREWARD_PROFILE="${LIGHTX2V_UNIFIEDREWARD_PROFILE:-lightx2v_rl/scripts/rewards/profiles/unifiedreward_edit27b_h100.sh}"
exec bash "$RL_SCRIPT_DIR/train_qwen21_4step.sh" "$@"
