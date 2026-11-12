"""Stream unique Fine-T2I captions without storing the image dataset."""

import argparse
import json
import random
from pathlib import Path

from common import atomic_json, atomic_jsonl, fingerprint, read_prompts

REPOSITORY = "ma-xu/fine-t2i"
SUBSET = "synthetic_enhanced_prompt_random_resolution"


def select_records(samples, count):
    seen, rows = set(), []
    for sample in samples:
        prompt = sample["txt"]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        prompt = prompt.strip()
        identity = " ".join(prompt.split()).casefold()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        metadata = sample.get("json", {})
        if isinstance(metadata, (str, bytes)):
            metadata = json.loads(metadata)
        rows.append(
            {
                "id": len(rows),
                "prompt": prompt,
                "source_key": str(sample.get("__key__", "")),
                "source_shard": str(sample.get("__url__", "")),
                "source_metadata": metadata,
            }
        )
        if len(rows) % 500 == 0:
            print(f"[select] unique={len(rows)}/{count} source={rows[-1]['source_shard']}", flush=True)
        if len(rows) == count:
            return rows
    raise ValueError(f"Stream exhausted after {len(rows)} unique prompts; requested {count}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--source-shards", type=int, default=32)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    args = parser.parse_args(argv)
    recipe = {"repository": REPOSITORY, "subset": SUBSET, "count": args.count, "seed": args.seed, "source_shards": args.source_shards, "shuffle_buffer": args.shuffle_buffer, "streaming": True}
    receipt_path = args.output.with_suffix(".selection.json")
    if args.output.exists():
        receipt = json.loads(receipt_path.read_text())
        rows = read_prompts(args.output, args.count)
        if receipt["recipe"] != recipe or receipt["prompts_sha256"] != fingerprint(rows):
            raise ValueError("Existing prompt selection has a different recipe/content")
        print(f"Reusing {len(rows)} prompts: {args.output}", flush=True)
        return

    from datasets import Image, load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    revision = api.dataset_info(REPOSITORY).sha
    files = sorted(name for name in api.list_repo_files(REPOSITORY, repo_type="dataset", revision=revision) if name.startswith(SUBSET + "/train-") and name.endswith(".tar"))
    random.Random(args.seed).shuffle(files)
    files = files[: args.source_shards]
    if not files:
        raise ValueError("No Fine-T2I WebDataset shards found")
    urls = [f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{name}" for name in files]
    dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
    # Keep JPEGs encoded while the tar reader advances; only text and JSON are retained.
    dataset = dataset.cast_column("jpg", Image(decode=False))
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    print(f"[select] streaming {len(files)} shards; target={args.count} unique prompts", flush=True)
    rows = select_records(dataset, args.count)
    atomic_jsonl(args.output, rows)
    atomic_json(receipt_path, {"recipe": recipe, "revision": revision, "source_files": files, "prompts_sha256": fingerprint(rows)})
    print(f"Selected {len(rows)} unique prompts: {args.output}", flush=True)


if __name__ == "__main__":
    main()
