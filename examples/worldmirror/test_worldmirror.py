"""Test HY-WorldMirror-2.0 reconstruction through the LightX2V runner.

Equivalent to the HY-World-2.0 CLI:
    python -m hyworld2.worldrecon.pipeline \
        --input_path examples/worldrecon/realistic/Workspace \
        --pretrained_model_name_or_path /data/nvme1/models/HY-World-2.0 \
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

DEFAULT_CONFIG_PATH = "/workspace/LightX2V/configs/worldmirror/worldmirror_recon.json"
DEFAULT_MODEL_PATH = "/data/nvme1/models/HY-World-2.0"
DEFAULT_INPUT_PATH = "/workspace/HY-World-2.0/examples/worldrecon/realistic/Workspace"
DEFAULT_OUTPUT_PATH = "/workspace/LightX2V/save_results/HY-WorldMirror/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--strict_output_path", default=None, help="If set, write outputs directly here (no subdir/timestamp)")
    parser.add_argument("--enable_bf16", action="store_true")
    args = parser.parse_args()

    # Importing lightx2v no longer eagerly pulls in every runner — see
    # lightx2v/__init__.py's PEP 562 __getattr__ — so we can just import
    # the runner we need directly.
    from lightx2v.models.runners.worldmirror.worldmirror_runner import WorldMirrorRunner  # noqa: F401
    from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
    from lightx2v.utils.lockable_dict import LockableDict
    from lightx2v.utils.registry_factory import RUNNER_REGISTER

    with open(args.config_path, "r") as f:
        config_dict = json.load(f)

    config_dict["model_path"] = args.model_path
    if args.enable_bf16:
        config_dict["enable_bf16"] = True

    os.makedirs(args.output_path, exist_ok=True)

    config = LockableDict(config_dict)

    runner = RUNNER_REGISTER[config["model_cls"]](config)
    runner.init_modules()

    input_info = init_empty_input_info(config["task"])
    update_input_info_from_dict(
        input_info,
        {
            "input_path": args.input_path,
            "save_result_path": args.output_path,
            "strict_output_path": args.strict_output_path,
            "return_result_tensor": True,
        },
    )

    result = runner.run_pipeline(input_info)
    print(f"[test_worldmirror] output: {result}")
    return result


if __name__ == "__main__":
    main()
