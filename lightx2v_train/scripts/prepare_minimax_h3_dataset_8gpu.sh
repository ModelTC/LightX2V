#!/usr/bin/env bash
# The default dataset pipeline now reuses videos and only encodes missing latents.
set -Eeuo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/prepare_minimax_h3_dataset_vae_8gpu.sh" "$@"
