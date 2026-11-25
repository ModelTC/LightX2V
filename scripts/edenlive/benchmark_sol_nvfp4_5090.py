"""Measure DiT model calls on their CUDA stream and export the same preset."""
import functools
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch

from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.runner_factory import build_runner
from lightx2v.utils.set_config import build_startup_config

from nvfp4_dynamic import install

repo = Path(__file__).resolve().parents[2]
model, data, out = (Path(arg).resolve() for arg in sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
variant = "wan22_5b_ar_sol_nvfp4_lightvae_5090"
template = repo / "configs/edenlive/realtime" / f"{variant}.json"
options = json.loads(template.read_text())
for key, value in options.items():
    if isinstance(value, str):
        options[key] = value.replace("@MODEL@", str(model)).replace("@DATA@", str(data))
        if value.startswith(("@MODEL@", "@DATA@")) and not Path(options[key]).exists():
            raise FileNotFoundError(options[key])
resolved = out / "effective_config.json"
resolved.write_text(json.dumps(options, indent=2) + "\n")
sample = data / "input"
config = build_startup_config(dict(
    model_cls="wan22_edenlive_ar", task="v2v", model_path=str(model),
    config_json=str(resolved),
))
install(share_projection=config["nvfp4_share_projection"])
config["warmup"] = False
runner = build_runner(config)
request = dict(
    task="v2v", prompt=(sample / "prompt.txt").read_text(), seed=2026092901,
    image_path=str(sample / "cover.png"), video_path=str(sample / "recording.mp4"),
)

def callback_only(frames, callback=lambda *_: None):
    runner.set_chunk_callback(callback)
    runner._stream_output_file_only = True
    try:
        BaseRunner.run_request(runner, runner.prepare_request(dict(request, num_frames=frames)))
    finally:
        runner._stream_output_file_only = False
    torch.cuda.synchronize()

try:
    with config.temporarily_unlocked(recursive=False):
        config["async_vae_decode"] = False
    callback_only(81)
    with config.temporarily_unlocked(recursive=False):
        config["async_vae_decode"] = True
    callback_only(201)
    dit_events, vae_events, chunks = [], [], []
    original_infer = runner.model.infer
    original_decode = runner.decode_segment_latents

    def time_cuda(original, events):
        @functools.wraps(original)
        def measured(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            done = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            done.record()
            events.append((start, done))
            return result
        return measured

    def on_chunk(pixels, index, is_last):
        chunks.append((perf_counter(), pixels.shape[2]))

    runner.model.infer = time_cuda(original_infer, dit_events)
    runner.decode_segment_latents = time_cuda(original_decode, vae_events)
    request_start = perf_counter()
    try:
        callback_only(717, on_chunk)
    finally:
        runner.model.infer = original_infer
        runner.decode_segment_latents = original_decode
    assert len(dit_events) == 300 and len(vae_events) == len(chunks) == 60
    assert sum(frames for _, frames in chunks) == 717
    rows = [[a.elapsed_time(b) for a, b in dit_events[i:i+5]] for i in range(0, 300, 5)]
    vae_ms = [a.elapsed_time(b) for a, b in vae_events]
    gaps = [(chunks[i][0] - chunks[i-1][0]) * 1000 for i in range(6, len(chunks))]
    steady = rows[6:]
    report = {
        "scope": "Warm 717 frames, no MP4 encoding in timing; CUDA events around model.infer and decode_segment_latents",
        "step_means_ms": [statistics.mean(column) for column in zip(*steady)],
        "denoise_mean_ms": statistics.mean(sum(row[:4]) for row in steady),
        "commit_mean_ms": statistics.mean(row[4] for row in steady),
        "dit_total_mean_ms": statistics.mean(map(sum, steady)),
        "vae_mean_ms": statistics.mean(vae_ms[6:]),
        "chunk_mean_ms": statistics.mean(gaps),
        "chunk_p95_ms": statistics.quantiles(gaps, n=100, method="inclusive")[94],
        "steady_fps": 12000 / statistics.mean(gaps),
        "rgb_ready_s": chunks[-1][0] - request_start,
        "chunk_events_ms": rows,
        "vae_chunk_events_ms": vae_ms,
        "stable_chunk_gaps_ms": gaps,
    }
    (out / f"{variant}_profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PROFILE", {key: value for key, value in report.items() if not key.endswith("_ms") or isinstance(value, float)}, flush=True)
    runner.set_chunk_callback(None)
    video = out / f"{variant}.mp4"
    runner.run_request(runner.prepare_request(dict(request, num_frames=717, save_result_path=str(video))))
    print("VIDEO", video, flush=True)
finally:
    runner.close()
