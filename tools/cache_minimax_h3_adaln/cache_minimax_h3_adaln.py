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
    parser.add_argument("--model_cls", choices=("minimax_h3", "minimax_h3_image", "minimax_h3_causal", "taomate_h3", "streamrender_h3"), default="minimax_h3")
    parser.add_argument("--task", choices=("refa2v", "refa2av"), help="Required for minimax_h3_causal")
    parser.add_argument(
        "--model-variant",
        choices=("fl2av", "ref2av"),
        help="Select the base (fl2av) or reference (ref2av) transformer cache profiles for minimax_h3",
    )
    args = parser.parse_args()
    if args.model_cls == "minimax_h3_causal" and args.task is None:
        parser.error("minimax_h3_causal requires --task refa2v or --task refa2av")
    if args.model_cls != "minimax_h3_causal" and args.task is not None:
        parser.error("--task is only supported for minimax_h3_causal")
    return args


def main() -> None:
    args = parse_args()
    task = args.task if args.model_cls == "minimax_h3_causal" else {"minimax_h3_image": "t2i", "taomate_h3": "t2av", "streamrender_h3": "refv2v"}.get(args.model_cls)
    config = build_startup_config(
        {
            "model_cls": args.model_cls,
            "task": task,
            "model_variant": args.model_variant,
            "model_path": args.model_path,
            "config_json": args.config_json,
        }
    )
    cache_path = build_persistent_adaln_cache(config)
    logger.info("MiniMax-H3 AdaLN cache saved to {}", cache_path)


if __name__ == "__main__":
    main()
