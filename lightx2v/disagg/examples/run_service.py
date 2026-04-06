import argparse
import json
import logging

from loguru import logger

from lightx2v.disagg.services.controller import ControllerService
from lightx2v.disagg.services.decoder import DecoderService
from lightx2v.disagg.services.encoder import EncoderService
from lightx2v.disagg.services.transformer import TransformerService
from lightx2v.disagg.utils import set_config
from lightx2v.utils.utils import seed_all

logging.basicConfig(level=logging.INFO)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a disaggregated LightX2V service process")
    parser.add_argument("--model_cls", type=str, default="wan2.1")
    parser.add_argument("--task", type=str, default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
            "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
            "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        ),
    )
    parser.add_argument(
        "--save_result_path",
        type=str,
        default="/root/zht/LightX2V/save_results/test_disagg.mp4",
    )

    parser.add_argument(
        "--service",
        type=str,
        choices=["encoder", "transformer", "decoder", "controller", "auto"],
        default="auto",
        help="Service role. auto = infer from config_json.disagg_mode",
    )
    parser.add_argument(
        "--engine_rank",
        type=int,
        default=None,
        help="Override engine rank for encoder/transformer/decoder service.",
    )
    return parser


def _normalize_disagg_config(config: dict) -> dict:
    disagg_cfg = config.get("disagg_config")
    if isinstance(disagg_cfg, dict):
        mapping = {
            "bootstrap_addr": "data_bootstrap_addr",
            "bootstrap_room": "data_bootstrap_room",
            "encoder_engine_rank": "encoder_engine_rank",
            "transformer_engine_rank": "transformer_engine_rank",
            "decoder_engine_rank": "decoder_engine_rank",
            "protocol": "protocol",
            "local_hostname": "local_hostname",
            "metadata_server": "metadata_server",
        }
        for src_key, dst_key in mapping.items():
            if src_key in disagg_cfg:
                config[dst_key] = disagg_cfg[src_key]
    return config


def _load_raw_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_service_mode(args: argparse.Namespace, raw_cfg: dict) -> str:
    if args.service != "auto":
        return args.service
    mode = raw_cfg.get("disagg_mode")
    if mode in {"encoder", "transformer", "decoder", "controller"}:
        return mode
    raise ValueError("Cannot resolve service mode: use --service or set disagg_mode in config_json")


def _build_runtime_config(args: argparse.Namespace) -> tuple[dict, dict]:
    raw_cfg = _load_raw_json(args.config_json)

    config = set_config(
        model_path=args.model_path,
        task=args.task,
        model_cls=args.model_cls,
        config_path=args.config_json,
    )

    config = _normalize_disagg_config(config)
    raw_cfg = _normalize_disagg_config(raw_cfg)

    config["seed"] = args.seed
    config["prompt"] = args.prompt
    config["negative_prompt"] = args.negative_prompt
    config["save_path"] = args.save_result_path
    return config, raw_cfg


def main():
    args = _build_parser().parse_args()
    config, raw_cfg = _build_runtime_config(args)
    service_mode = _resolve_service_mode(args, raw_cfg)

    if args.engine_rank is not None and service_mode in {"encoder", "transformer", "decoder"}:
        rank_key = f"{service_mode}_engine_rank"
        config[rank_key] = int(args.engine_rank)

    seed_all(args.seed)
    logger.info("Starting disagg service mode={}", service_mode)

    if service_mode == "encoder":
        EncoderService(config).run()
    elif service_mode == "transformer":
        TransformerService(config).run()
    elif service_mode == "decoder":
        DecoderService(config).run()
    elif service_mode == "controller":
        ControllerService().run(config)
    else:
        raise ValueError(f"Unsupported service mode: {service_mode}")


if __name__ == "__main__":
    main()
