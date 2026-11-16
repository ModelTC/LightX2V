import argparse
import json
from pathlib import Path

from lightx2v import LightX2VPipeline

parser = argparse.ArgumentParser(description="Replay NeoPP KV captured from LightLLM.")
parser.add_argument("--model_path", required=True)
parser.add_argument("--config_json", required=True)
parser.add_argument("--task", choices=["t2i", "i2i"], default="t2i")
parser.add_argument("--cond_kv", required=True)
parser.add_argument("--uncond_kv")
parser.add_argument("--index_offset_cond", type=int, required=True)
parser.add_argument("--index_offset_uncond", type=int)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--size", type=int, nargs=2, required=True, metavar=("HEIGHT", "WIDTH"))
parser.add_argument("--save_result_path", required=True)
args = parser.parse_args()

config = json.loads(Path(args.config_json).read_text())
config.update(
    {
        "model_path": args.model_path,
        "model_cls": "neopp",
        "task": args.task,
    }
)

with LightX2VPipeline(config) as pipe:
    pipe.runner.load_kvcache(args.cond_kv, args.uncond_kv)

    inference_params = {name: pipe.runner.config[name] for name in ("cfg_interval", "cfg_scale", "cfg_norm", "timestep_shift") if name in pipe.runner.config}
    pipe.runner.set_inference_params(
        index_offset_cond=args.index_offset_cond,
        index_offset_uncond=args.index_offset_uncond,
        **inference_params,
    )
    pipe.generate(
        seed=args.seed,
        size=args.size,
        save_result_path=args.save_result_path,
    )
