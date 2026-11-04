#!/usr/bin/env bash
# Source from the repository root before launching a server or platform job.
# BF16 settings for one H100 80G per UnifiedReward replica.
# Apply any deliberate environment overrides after sourcing this profile.
export LIGHTX2V_UNIFIEDREWARD_MODEL=lightx2v_rl/models/rewards/UnifiedReward-Edit-qwen35-27b
export LIGHTX2V_UNIFIEDREWARD_TP=1
export LIGHTX2V_UNIFIEDREWARD_DTYPE=bfloat16
export LIGHTX2V_UNIFIEDREWARD_GPU_MEMORY=0.90
export LIGHTX2V_UNIFIEDREWARD_MAX_MODEL_LEN=32768
export LIGHTX2V_UNIFIEDREWARD_MAX_NUM_SEQS=1
export LIGHTX2V_UNIFIEDREWARD_MAX_BATCHED_TOKENS=4096
export LIGHTX2V_UNIFIEDREWARD_ENFORCE_EAGER=1
# vLLM 0.20.2 can livelock on two-image requests when encoder-cache caps
# interact with Mamba's block-aligned prefix caching. Disabling APC also
# selects mamba_cache_mode=none; keep the conservative prefill budget.
export LIGHTX2V_UNIFIEDREWARD_PREFIX_CACHING=0
# Canvas + reference + generated result; override after sourcing for more refs.
export LIGHTX2V_UNIFIEDREWARD_LIMIT_MM_PER_PROMPT='{"image":3,"video":0}'
# These are pixel AREAS: preserve 2048x2048 generated images, bound large sources.
# Both size keys are required by the Transformers 5 Qwen image processor.
export LIGHTX2V_UNIFIEDREWARD_MM_PROCESSOR_KWARGS='{"size":{"shortest_edge":65536,"longest_edge":4194304}}'
# Larger replicas may take longer to load from shared storage and compile.
export LIGHTX2V_REWARD_STARTUP_TIMEOUT=3600
export LIGHTX2V_PLATFORM_STARTUP_TIMEOUT=3600
