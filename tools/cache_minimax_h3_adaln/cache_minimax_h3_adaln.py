import argparse

from builder import build_persistent_adaln_cache
from loguru import logger

from lightx2v.utils.set_config import build_startup_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the on-disk AdaLN and final-norm cache required by MiniMax-H3 inference.",
    )
    parser.add_argument("--model_path", required=True, help="MiniMax-H3 model root")
    parser.add_argument("--config_json", required=True, help="Inference JSON config")
    parser.add_argument("--model_cls", choices=("minimax_h3", "minimax_h3_causal", "minimax_h3_world"), default="minimax_h3")
    parser.add_argument(
        "--model-variant",
        choices=("fl2av", "ref2av"),
        help="Select the base (fl2av) or reference (ref2av) transformer cache profiles for minimax_h3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_startup_config(
        {
            "model_cls": args.model_cls,
            "task": "refa2v" if args.model_cls == "minimax_h3_causal" else None,
            "model_variant": args.model_variant,
            "model_path": args.model_path,
            "config_json": args.config_json,
        }
    )
    cache_path = build_persistent_adaln_cache(config)
    logger.info("MiniMax-H3 AdaLN cache saved to {}", cache_path)


if __name__ == "__main__":
    main()
