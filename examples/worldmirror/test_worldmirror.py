"""Test HY-WorldMirror-2.0 reconstruction through the LightX2V runner.

Equivalent to the HY-World-2.0 CLI:
    python -m hyworld2.worldrecon.pipeline \
        --input_path /path/to/HY-World-2.0/examples/worldrecon/realistic/Workspace \
        --pretrained_model_name_or_path /path/to/HY-World-2.0 \
        --no_interactive

Run:
    python examples/worldmirror/test_worldmirror.py
"""

import argparse
import json
import os
import sys

# Make ``lightx2v`` importable when this script is executed directly from
# any directory (e.g. ``python examples/worldmirror/test_worldmirror.py``).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs/worldmirror/worldmirror_recon.json")
DEFAULT_MODEL_PATH = "/path/to/HY-World-2.0"
DEFAULT_INPUT_PATH = "/path/to/HY-World-2.0/examples/worldrecon/realistic/Workspace"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--save_result_path", default=None)
    parser.add_argument("--strict_output_path", default=None, help="If set, write outputs directly here (no subdir/timestamp)")
    parser.add_argument("--enable_bf16", action="store_true")
    args = parser.parse_args()

    from lightx2v.models.runners.runner_factory import build_runner
    from lightx2v.utils.set_config import build_startup_config

    with open(args.config_path, "r") as f:
        config_dict = json.load(f)

    config_dict["model_path"] = args.model_path
    if args.enable_bf16:
        config_dict["enable_bf16"] = True

    runner = build_runner(build_startup_config(config_dict))

    input_info = runner.prepare_request(
        {
            "input_path": args.input_path,
            "save_result_path": args.save_result_path,
            "strict_output_path": args.strict_output_path,
            "return_result_tensor": True,
        }
    )
    result = runner.run_request(input_info)
    print(f"[test_worldmirror] output: {result}")
    return result


if __name__ == "__main__":
    main()
