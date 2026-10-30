#!/usr/bin/env bash
set -Eeuo pipefail

# Historical projection-only PDMD entry point; retired before any model or GPU work.
printf '%s\n' \
    'This projection-only PDMD launcher has been retired.' \
    'Use lightx2v_train/scripts/run_minimax_h3_ref2av_omni_imageonly_pdmd8_official_fsdp32_32gpu_acp.sh for the official recipe (32 GPUs).' >&2
exit 2
