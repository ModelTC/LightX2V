"""Resident, same-input DiT latency; no text encoding, VAE or checkpoint I/O."""

import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch

from lightx2v_train.model_zoo.minimax_h3.condition_encoder import MiniMaxH3ConditionEncoder

from .data import PromptDataset
from .runtime import AVSample, load_model, shift_sigma


@torch.no_grad()
def benchmark(config):
    output = Path(config["output_dir"])
    encoder = MiniMaxH3ConditionEncoder(config["condition_model_path"], device="cuda", dtype=torch.bfloat16, cpu_offload=True)
    record = PromptDataset(config["prompts"], config["seed"]).records[config["benchmark"]["prompt_index"]]
    geometry = {key: record[key] for key in ("height", "width", "num_frames")}
    geometry.update({key: config["benchmark"][key] for key in geometry if key in config["benchmark"]})
    condition = encoder.encode(record["prompt"])
    sample = AVSample(condition, geometry, torch.device("cuda"))
    torch.save({k: v.cpu() for k, v in condition.items()}, output / "benchmark_condition.pt")
    del encoder, condition
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(config["seed"])
    noisy = {name: torch.randn(shape, device="cuda") for name, shape in sample.shapes.items()}
    sigmas = {name: shift_sigma(0.5, shift) for name, shift in config["flow_shift"].items()}
    permutations = torch.load(output / "partition.pt", weights_only=True)
    results = {
        "config": config,
        "geometry": geometry,
        "sigmas": sigmas,
        "scope": "resident BF16 DiT, one sigma, batch=1, single GPU",
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "video_tokens": sample.shapes["video"][1],
        "audio_tokens": sample.shapes["audio"][1],
        "text_tokens": sample.condition.shape[1],
        "variants": {},
    }
    predictions = {}
    for variant in ("dense", "moe"):
        torch.manual_seed(config["seed"])
        model = load_model(config, variant, permutations, sharded=False).eval()
        if variant == "moe":
            weights = torch.load(output / "moe.pt", map_location="cpu", weights_only=True)
            for index, state in weights.items():
                model.transformer_blocks[int(index)].ff.load_state_dict(state)
            del weights
        model.requires_grad_(False)
        for _ in range(config["benchmark"]["warmup"]):
            sample.forward(model, noisy, sigmas)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(config["benchmark"]["repeats"]):
            started = time.perf_counter()
            prediction = sample.forward(model, noisy, sigmas)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
        predictions[variant] = {name: value.float().cpu() for name, value in prediction.items()}
        results["variants"][variant] = {
            "median_seconds": statistics.median(timings),
            "seconds": timings,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "rope_buffer_dtypes": sorted({str(b.dtype) for b in model.rope.buffers()}),
        }
        print(json.dumps({variant: results["variants"][variant]}), flush=True)
        del model, prediction
        gc.collect()
        torch.cuda.empty_cache()
    dense, moe = predictions["dense"], predictions["moe"]
    results["speedup"] = results["variants"]["dense"]["median_seconds"] / results["variants"]["moe"]["median_seconds"]
    results["prediction_relative_rmse"] = {name: ((moe[name] - value).square().mean() / value.square().mean()).sqrt().item() for name, value in dense.items()}
    (output / "benchmark.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"speedup": results["speedup"], "prediction_relative_rmse": results["prediction_relative_rmse"]}))
