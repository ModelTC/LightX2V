#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# -lt 1 || ! "$1" =~ ^[01]$ ]]; then
    echo "Usage: bash $0 NODE_INDEX_0_OR_1 [launch.py options]" >&2
    exit 2
fi
node_index="$1"
shift
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${H3_PYTHON:-python}" -u "$script_dir/launch.py" node --node-index "$node_index" "$@"
