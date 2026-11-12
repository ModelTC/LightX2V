#!/usr/bin/env bash
# Shared four-step launcher. Submit once per node through the training platform.
set -euo pipefail
RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RL_SCRIPT_DIR/../../.."

export LIGHTX2V_RL_CONFIG="${LIGHTX2V_RL_CONFIG:-qwen_image21_4step_geo_6nodes}"
export LIGHTX2V_QWEN_4STEP_MODEL="${LIGHTX2V_QWEN_4STEP_MODEL:-lightx2v_rl/models/Qwen-Image-2.1-Lightning-v2-base}"
RL_REWARD_PROFILE="${LIGHTX2V_UNIFIEDREWARD_PROFILE:-lightx2v_rl/scripts/rewards/profiles/unifiedreward_edit9b_h100.sh}"
if [[ ! -f "$RL_REWARD_PROFILE" ]]; then
    printf 'UnifiedReward profile unavailable: %s\n' "$RL_REWARD_PROFILE" >&2
    exit 2
fi
source "$RL_REWARD_PROFILE"
exec bash "$RL_SCRIPT_DIR/train_qwen21_platform.sh" "$@"
