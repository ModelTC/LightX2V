#!/usr/bin/env bash
# Two active requests per UR replica; validate multi-reference GPU memory.
# Pair with reward.custom_reward_function.reward_kwargs.max_concurrent_samples=2.
source "$(dirname -- "${BASH_SOURCE[0]}")/unifiedreward_edit27b_h100.sh"
export LIGHTX2V_UNIFIEDREWARD_MAX_NUM_SEQS=2
