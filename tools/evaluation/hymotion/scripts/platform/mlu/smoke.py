#!/usr/bin/env python3
"""Run one frozen HumanML3D request through the pinned HY-Motion inference PR."""

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
    records = [json.loads(line) for line in (HERE / "configs/common/humanml3d.jsonl").read_text().splitlines()]
    item = records[args.index]
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / f"{item['id']}.npz"
    model_config = HERE / "configs/mlu/model.json"
    source = read_json(HERE / "configs/mlu/source-lock.json")["lightx2v"]
    actual = subprocess.check_output(["git", "-C", paths["lightx2v"], "rev-parse", "HEAD"], text=True).strip()
    if actual != source["revision"]:
        raise ValueError(f"Pinned inference source mismatch: {actual}")
    subprocess.run(["git", "-C", paths["lightx2v"], "diff", "--exit-code", "HEAD", "--", "*.py", "*.sh", "*.json"], check=True, stdout=subprocess.DEVNULL)

    env = dict(
        os.environ,
        PLATFORM="cambricon_mlu",
        MLU_VISIBLE_DEVICES="0",
    DTYPE="BF16",
        QWEN_MODEL_PATH=paths["qwen"],
        CLIP_MODEL_PATH=paths["clip"],
    )
    code = r'''
import json
import torch_mlu
import torch
from pathlib import Path
import sys
root, config_path, model_path, qwen_path, clip_path, prompt, frame_count, output_path = sys.argv[1:]
sys.path.insert(0, root)
from lightx2v.models.runners.runner_factory import build_runner
from lightx2v.utils.set_config import build_startup_config
config = json.loads(Path(config_path).read_text())
config.update(model_path=model_path, qwen_model_path=qwen_path, clip_model_path=clip_path)
runner = build_runner(build_startup_config(config))
try:
    request = runner.prepare_request({"prompt":prompt, "num_frames":int(frame_count), "seed":42, "save_result_path":output_path})
    runner.run_request(request)
finally:
    runner.close()
import numpy as np
with np.load(output_path, allow_pickle=False) as result:
    arrays = {key: result[key] for key in result.files}
if not arrays or any(not np.isfinite(value).all() for value in arrays.values() if np.issubdtype(value.dtype, np.number)):
    raise RuntimeError("HY-Motion output is empty or non-finite")
print(json.dumps({"status":"PASS", "device":"mlu:0", "arrays":{key:list(value.shape) for key,value in arrays.items()}}))
'''
    try:
        result = subprocess.run(
            [paths["python"], "-c", code, paths["lightx2v"], str(model_config), paths["weights"], paths["qwen"], paths["clip"], item["prompt"], str(item["expected_output_frames_30fps"]), str(output)],
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
