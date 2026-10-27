#!/usr/bin/env bash
set -Eeuo pipefail

# Same command on four ACP workers. Keep this mixed-rank recipe explicit.
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
export H3_FL2AV_PDMD_CONFIG="$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_fl2av_3task_pdmd8_shift6_3_r128_a128_adaln16_fsdp32.yaml"
export H3_FL2AV_PDMD_OUTPUT="${H3_FL2AV_PDMD_OUTPUT:-$H3_CODE_ROOT/outputs/h3_fl2av_3task_pdmd8_shift6_3_r128a128_adaln16_fsdp32}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_fl2av_3task_pdmd8_shift6_3_r128a128_adaln16_fsdp32}"

# Inspect the configured layer selection and installed injection path on CPU.
# Full metadata/cache preflight and --dry-run handling belong to the common launcher.
"$H3_PYTHON" - "$H3_CODE_ROOT" "$H3_FL2AV_PDMD_CONFIG" <<'PY'
import ast
import sys
from pathlib import Path

import yaml

root, path = map(Path, sys.argv[1:])
config = yaml.safe_load(path.read_text())
training = config['training']
lora = training['student']['lora']
patterns = {'adaln_proj.linear': 16}
if (lora.get('rank'), lora.get('alpha')) != (128, 128):
    raise SystemExit('This launcher requires regular LoRA rank128/alpha128.')
if lora.get('rank_pattern') != patterns or lora.get('alpha_pattern') != patterns:
    raise SystemExit('This launcher requires AdaLN rank16/alpha16.')
if 'adaln_proj.linear' not in lora.get('target_modules', []):
    raise SystemExit('AdaLN must be included in target_modules.')
if training['student'].get('checkpoint_path'):
    raise SystemExit('This recipe initializes from the H3 base; Turbo is only a layer-selection reference.')
dmd = training['dmd']
if (dmd.get('official_rollout_video_shift'), dmd.get('official_rollout_audio_shift')) != (6.0, 3.0):
    raise SystemExit('This launcher requires training video/audio shifts 6/3.')
source = root / 'lightx2v_train/lightx2v_train/model_zoo/minimax_h3/minimax_h3_t2av.py'
tree = ast.parse(source.read_text())
methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'add_lora']
if not methods or not {'rank_pattern', 'alpha_pattern'} <= {arg.arg for arg in methods[0].args.kwonlyargs}:
    raise SystemExit('H3 mixed-rank LoRA code is missing; sync the code bundle as well as the YAML.')
common = root / 'lightx2v_train/lightx2v_train/model_zoo/capability_adapters/common.py'
capability = next(node for node in ast.parse(common.read_text()).body
                  if isinstance(node, ast.ClassDef) and node.name == 'CommonTrainableCapability')
forwarding = [node for node in ast.walk(capability)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == 'add_lora']
if not any(any(keyword.arg is None for keyword in call.keywords) for call in forwarding):
    raise SystemExit('LoRA capability does not forward per-layer patterns; sync the complete code bundle.')
print('LoRA: regular rank=128 alpha=128; adaln_proj.linear rank=16 alpha=16; alpha/rank=1 for both.')
print('Student initializes from transformer/; Turbo weights are not loaded. Use a new output directory.')
PY

exec bash "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_fl2av_pdmd_official_fsdp32_32gpu_acp.sh" 8 "$@"
