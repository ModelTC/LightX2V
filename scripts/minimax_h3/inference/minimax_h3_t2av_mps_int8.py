"""Resident four-step H3 T2AV with a merged, standalone Apple INT8 checkpoint."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Original H3 directory containing text encoder, tokenizer and VAEs")
    parser.add_argument("--checkpoint", required=True, help="Converted merged INT8 transformer directory, scope=dit_proj")
    parser.add_argument("--metallib-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="save_results/minimax_h3_t2av_int8.mp4")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[3] / "configs/platforms/mps/minimax_h3_t2av_4step_int8.json"))
    args = parser.parse_args()
    os.environ.update(
        PLATFORM="mps",
        DTYPE="BF16",
        SENSITIVE_LAYER_DTYPE="FP32",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_ENABLE_MPS_FALLBACK="1",
        LIGHTX2V_METALLIB_DIR=str(Path(args.metallib_dir).expanduser().resolve()),
    )
    from lightx2v import LightX2VPipeline

    pipe = LightX2VPipeline(model_path=args.model_path, model_cls="minimax_h3", model_variant="fl2av")
    pipe.enable_quantize(dit_quantized=True, quant_scheme="int8-metal", dit_quantized_ckpt=args.checkpoint)
    pipe.create_generator(config_json=args.config)
    pipe.generate(task="t2av", seed=args.seed, prompt=args.prompt, save_result_path=args.output)


if __name__ == "__main__":
    main()
