#!/usr/bin/env python3
"""Run one frozen GenEval prompt through the pinned SDXL inference PR."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if not args.paths.is_absolute():
        args.paths = args.paths.resolve() if args.paths.exists() else (HERE / args.paths).resolve()
    paths = read_json(args.paths)
    records = [json.loads(line) for line in (HERE / "configs/common/geneval.jsonl").read_text().splitlines()]
    item = records[args.index]
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / f"{item['id']}.png"
    model_config = HERE / "configs/mlu/model.json"
    source = read_json(HERE / "configs/mlu/source-lock.json")["lightx2v"]
    actual = subprocess.check_output(["git", "-C", paths["lightx2v"], "rev-parse", "HEAD"], text=True).strip()
    if actual != source["revision"]:
        raise ValueError(f"Pinned inference source mismatch: {actual}")
    subprocess.run(["git", "-C", paths["lightx2v"], "diff", "--exit-code", "HEAD", "--", "*.py", "*.sh", "*.json"], check=True, stdout=subprocess.DEVNULL)

    env = dict(os.environ, PLATFORM="cambricon_mlu", MLU_VISIBLE_DEVICES="0", DTYPE="BF16")
    code = r'''
import json
import torch_mlu
import torch
from pathlib import Path
import sys
root, config_path, model_path, prompt, output_path = sys.argv[1:]
sys.path.insert(0, root)
from lightx2v.models.runners.runner_factory import build_runner
from lightx2v.utils.set_config import build_startup_config
config = json.loads(Path(config_path).read_text())
config["model_path"] = model_path
runner = build_runner(build_startup_config(config))
try:
    request = runner.prepare_request({"prompt":prompt, "seed":42, "size":[1024,1024], "save_result_path":output_path})
    runner.run_request(request)
finally:
    runner.close()
from PIL import Image
import numpy as np
with Image.open(output_path) as image:
    pixels = np.asarray(image.convert("RGB"))
if image.size != (1024,1024) or not np.isfinite(pixels).all():
    raise RuntimeError("SDXL output is malformed or non-finite")
print(json.dumps({"status":"PASS", "device":"mlu:0", "size":[1024,1024], "dtype":str(pixels.dtype)}))
'''
    try:
        result = subprocess.run(
            [paths["python"], "-c", code, paths["lightx2v"], str(model_config), paths["weights"], item["prompt"], str(output)],
            env=env,
            check=True,
            text=True,
            capture_output=True,
            timeout=3600,
        )
    except subprocess.CalledProcessError as error:
        print(error.stdout or "", file=sys.stderr, end="")
        print(error.stderr or "", file=sys.stderr, end="")
        raise
    (args.output / "smoke.json").write_text(json.dumps({"status": "PASS", "fixture_id": item["id"], "inference_pr": source["pull_request"], "result": result.stdout.strip()}, indent=2) + "\n")
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
