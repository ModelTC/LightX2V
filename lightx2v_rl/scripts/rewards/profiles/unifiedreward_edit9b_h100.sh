#!/usr/bin/env bash
# Source from the repository root before launching a server or platform job.
# Set the 9B defaults, including when switching from the 27B profile.
# Apply any deliberate environment overrides after sourcing this profile.
export LIGHTX2V_UNIFIEDREWARD_MODEL=lightx2v_rl/models/rewards/UnifiedReward-Edit-qwen35-9b
export LIGHTX2V_UNIFIEDREWARD_TP=1
export LIGHTX2V_UNIFIEDREWARD_GPU_MEMORY=0.85
export LIGHTX2V_UNIFIEDREWARD_MAX_MODEL_LEN=65536
export LIGHTX2V_UNIFIEDREWARD_ENFORCE_EAGER=0
# Delegate these choices to vLLM/the model processor.
# Clear inherited 27B overrides rather than imposing its conservative limits.
unset LIGHTX2V_UNIFIEDREWARD_DTYPE
unset LIGHTX2V_UNIFIEDREWARD_MAX_NUM_SEQS
unset LIGHTX2V_UNIFIEDREWARD_MAX_BATCHED_TOKENS
unset LIGHTX2V_UNIFIEDREWARD_MM_PROCESSOR_KWARGS
# Canvas + reference + generated result; override after sourcing for more refs.
export LIGHTX2V_UNIFIEDREWARD_LIMIT_MM_PER_PROMPT='{"image":3}'
# Avoid the vLLM 0.20.2 two-image Mamba alignment stall.
export LIGHTX2V_UNIFIEDREWARD_PREFIX_CACHING=0
export LIGHTX2V_REWARD_STARTUP_TIMEOUT=1800
export LIGHTX2V_PLATFORM_STARTUP_TIMEOUT=1800
