#!/bin/bash
set -e

lightx2v_path=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

export PYTHONPATH="${lightx2v_path}/lightx2v_kernel/python:${lightx2v_path}:${PYTHONPATH}"

python "${lightx2v_path}/lightx2v_kernel/test/kvcache/test_nvfp4_kv.py"
