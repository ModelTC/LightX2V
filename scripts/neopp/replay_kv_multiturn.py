"""Replay three captured requests through one NeoPP pipeline.

Set model_path, kv_dir and output_dir to local paths before running.
See README.md for FP8, parallel and 2K replay settings.
"""

from pathlib import Path

import torch.distributed as dist

from lightx2v import LightX2VPipeline

repo_root = Path(__file__).resolve().parents[2]
model_path = "/path/to/neopp_dense"
config_path = repo_root / "configs/neopp/neopp_dense.json"
kv_dir = Path("/path/to/neopp_dense_kv_1k")
output_dir = Path("/path/to/save_results")
output_prefix = "output_lightx2v_neopp_dense_1k"
size = [1024, 1024]

# Filenames and position offsets belong to the same captured request.
# Use seed=None to continue the pipeline's RNG state on a later turn.
turns = [
    ("to_x2v_cond_kv_0_298.pt", "to_x2v_uncond_kv_0_9.pt", 298, 9, 200),
    ("to_x2v_cond_kv_1_366.pt", "to_x2v_uncond_kv_1_12.pt", 366, 12, 201),
    ("to_x2v_cond_kv_2_441.pt", "to_x2v_uncond_kv_2_15.pt", 441, 15, 202),
]

pipe = LightX2VPipeline(model_path=model_path, model_cls="neopp", support_tasks=["t2i", "i2i"])
pipe.create_generator(config_json=str(config_path))
pipe.modify_config({"load_kv_cache_in_pipeline_for_debug": False, "save_result_for_debug": True})

for turn, (cond, uncond, offset_cond, offset_uncond, seed) in enumerate(turns):
    pipe.runner.load_kvcache(str(kv_dir / cond), str(kv_dir / uncond) if uncond is not None else None)
    pipe.runner.set_inference_params(
        index_offset_cond=offset_cond,
        index_offset_uncond=offset_uncond,
        cfg_interval=(-1, 2),
        cfg_scale=4.0,
        cfg_norm="none",
        timestep_shift=3.0,
    )
    pipe.generate(
        task="t2i",
        seed=seed,
        size=size,
        save_result_path=str(output_dir / f"{output_prefix}_{turn}.png"),
    )

if dist.is_initialized():
    dist.destroy_process_group()
