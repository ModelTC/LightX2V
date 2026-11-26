"""Four independent GPU workers cache original H3 teacher AV and prompt pairs."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests
import torch
from omegaconf import OmegaConf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lightx2v_train.model_zoo.minimax_h3.condition_encoder import MiniMaxH3ConditionEncoder

from .data import PromptDataset, load_media
from .encoding import OnlineAVEncoder, StageTimer
from .runtime import AVSample


def atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def cache_spec(config):
    components = {}
    for root, names in ((config["model_path"], ("vae", "audio_vae")), (config["condition_model_path"], ("text_encoder", "tokenizer", "processor"))):
        for name in names:
            directory = Path(root, name).resolve()
            components[str(directory)] = {path.name: [path.stat().st_size, path.stat().st_mtime_ns] for path in sorted(directory.iterdir()) if path.is_file() or path.is_symlink()}
    return {
        "version": 1,
        "metadata_sha256": hashlib.sha256(Path(config["prompts"]).read_bytes()).hexdigest(),
        "components": components,
        "posterior": "mode",
        "latent_dtype": "float32",
        "prompt_dtype": "bfloat16",
        "video_tile": [256, 256, 64, 64],
        "audio_sample_rate": 32000,
    }


def prepare(config):
    directory = Path(config["cache_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    spec = cache_spec(config)
    path = directory / "spec.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Cache recipe/model/metadata changed; use a new cache directory")
    atomic_json(path, spec)
    (directory / "samples").mkdir(exist_ok=True)
    (directory / "logs").mkdir(exist_ok=True)
    print(json.dumps({"event": "cache_prepared", "directory": str(directory)}), flush=True)


def download(session, url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    with session.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    temporary.replace(path)


def encode(config):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    device = torch.device("cuda", torch.cuda.current_device())
    directory = Path(config["cache_dir"])
    metadata = Path(config["prompts"])
    records = PromptDataset(metadata, config["seed"]).records
    source = json.loads((metadata.parent / "source.json").read_text())
    base_url = source["url"].rsplit("/", 1)[0]
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))))
    indices = list(range(rank, len(records), world))
    pending = [index for index in indices if not (directory / "samples" / f"{index:06d}.pt").exists()]
    print(json.dumps({"rank": rank, "event": "load_encoders", "pending": len(pending)}), flush=True)
    if pending:
        # Cache workers own an entire GPU: retain the text encoder rather than
        # repeatedly transferring it from CPU, as the training-time encoder did.
        av_encoder = OnlineAVEncoder(config["model_path"], device)
        text_encoder = MiniMaxH3ConditionEncoder(config["condition_model_path"], device=device, dtype=torch.bfloat16, cpu_offload=False)
    with (directory / "logs" / f"rank_{rank}.jsonl").open("a") as log:
        for ordinal, index in enumerate(pending):
            record = records[index]
            started = time.perf_counter()
            torch.manual_seed(config["seed"] + index)
            media = metadata.parent / record["video_path"]
            download(session, f"{base_url}/{record['video_path']}", media)
            timer = StageTimer(device)
            with timer.measure("media_decode"):
                video, audio = load_media(media, record)
            latents = av_encoder.encode(video, audio, timer)
            del video, audio
            with timer.measure("text_encode"):
                condition = text_encoder.encode(record["prompt"])
            expected = AVSample(condition, record, device).shapes
            for name, value in latents.items():
                if value.shape != expected[name] or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid {name} latent at record {index}")
            if not torch.isfinite(condition["prompt_embeds"]).all():
                raise ValueError(f"Invalid prompt encoding at record {index}")
            payload = {
                "record_index": index,
                "condition": {name: value.detach().cpu().contiguous() for name, value in condition.items()},
                "latents": {name: value.detach().cpu().contiguous() for name, value in latents.items()},
            }
            path = directory / "samples" / f"{index:06d}.pt"
            temporary = path.with_suffix(".partial")
            torch.save(payload, temporary)
            temporary.replace(path)
            progress = {
                "rank": rank,
                "record_index": index,
                "completed": len(indices) - len(pending) + ordinal + 1,
                "assigned": len(indices),
                "seconds": time.perf_counter() - started,
                **timer.seconds,
            }
            message = json.dumps(progress)
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()
            del condition, latents, payload
    session.close()
    print(json.dumps({"rank": rank, "event": "cache_rank_complete", "records": len(indices)}), flush=True)


def finalize(config):
    directory = Path(config["cache_dir"])
    records = PromptDataset(config["prompts"], config["seed"]).records
    missing = [index for index in range(len(records)) if not (directory / "samples" / f"{index:06d}.pt").is_file()]
    if missing:
        raise RuntimeError(f"Cache incomplete: {len(missing)} missing records; first={missing[:10]}")
    spec = json.loads((directory / "spec.json").read_text())
    if spec != cache_spec(config):
        raise ValueError("Cache source or model changed during encoding")
    atomic_json(directory / "complete.json", {"metadata_sha256": spec["metadata_sha256"], "records": len(records)})
    print(json.dumps({"event": "cache_complete", "records": len(records), "directory": str(directory)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("prepare", "encode", "finalize"), required=True)
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    {"prepare": prepare, "encode": encode, "finalize": finalize}[args.mode](config)


if __name__ == "__main__":
    main()
