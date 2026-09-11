import argparse

import torch.distributed as dist

from lightx2v import LightX2VPipeline

parser = argparse.ArgumentParser(description="Replay NeoPP KV captured from LightLLM.")
parser.add_argument("--model_path", required=True)
parser.add_argument("--config_json", required=True)
parser.add_argument("--task", choices=["t2i", "i2i"], default="t2i")
parser.add_argument("--cond_kv", required=True)
parser.add_argument("--uncond_kv")
parser.add_argument("--index_offset_cond", type=int, required=True)
parser.add_argument("--index_offset_uncond", type=int)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--size", type=int, nargs=2, required=True, metavar=("HEIGHT", "WIDTH"))
parser.add_argument("--save_result_path", required=True)
args = parser.parse_args()

pipe = LightX2VPipeline(
    model_path=args.model_path,
    model_cls="neopp",
    task=args.task,
    support_tasks=["t2i", "i2i"],
)
pipe.create_generator(config_json=args.config_json)
pipe.modify_config({"load_kv_cache_in_pipeline_for_debug": False, "save_result_for_debug": True})
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

if dist.is_initialized():
    dist.destroy_process_group()
