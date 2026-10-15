"""Measure decoded video parity and write a sparse-frame comparison sheet."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image


def read_video(path):
    with av.open(path) as container:
        fps = float(container.streams.video[0].average_rate)
        frames = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    return frames, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference, fps = read_video(args.reference)
    candidate, candidate_fps = read_video(args.candidate)
    assert reference.shape == candidate.shape, (reference.shape, candidate.shape)
    assert fps == candidate_fps, (fps, candidate_fps)
    difference = reference.astype(np.float32) - candidate.astype(np.float32)
    mse = float(np.mean(difference**2))
    metrics = dict(
        frames=len(reference),
        height=reference.shape[1],
        width=reference.shape[2],
        fps=fps,
        psnr_db=float(10 * np.log10(255**2 / mse)) if mse else None,
        mean_abs=float(np.mean(abs(difference))),
        max_abs=float(np.max(abs(difference))),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    indices = np.linspace(0, len(reference) - 1, 4, dtype=int)
    sheet = np.concatenate([np.concatenate((candidate[index], reference[index]), axis=1) for index in indices], axis=0)
    Image.fromarray(sheet).resize((1280, round(sheet.shape[0] * 1280 / sheet.shape[1]))).save(output.with_suffix(".jpg"))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
