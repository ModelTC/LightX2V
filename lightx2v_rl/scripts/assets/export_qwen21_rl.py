#!/usr/bin/env python3
"""Export a trusted Qwen-Image-2.1 FSDP2 checkpoint as one full DiT safetensors file.

Run with the RL training Python. No GPU, Ray cluster or torchrun is required.
Specify the actor checkpoint, base pipeline and output directory explicitly.
--dry-run checks paths/configs without loading weights.
The VAE and text encoder remain in the base pipeline and are not exported here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
MODEL_CLASS = "QwenImage21Transformer2DModel"
PIPELINE_CLASS = "QwenImage21Pipeline"


def repository_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY / path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def shared_filesystem_publication(target: Path):
    """Publish verified files without requiring renameat2(RENAME_NOREPLACE).

    Exclusive mkdir reserves the destination without replacing another export.
    Publication is per-file, not directory-atomic: consumers must wait for this
    script to finish. The incomplete marker survives any publication failure.
    Retain staging on every failure so a large completed export is not deleted.
    """
    lock = target.parent / f".{target.name}.merge.lock"
    with lock.open("x"):
        staging = None
        validated = False
        try:
            if os.path.lexists(target):
                raise FileExistsError(target)
            staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.merge-", dir=target.parent))
            print(f"Export staging: {staging}", flush=True)
            yield staging
            validated = True
            # The merger has already verified all tensors and checksums here.
            # mkdir(exist_ok=False) also protects against a target appearing
            # while reconstruction was running, including an empty directory.
            target.mkdir()
            marker = target / ".export-incomplete"
            with marker.open("x") as handle:
                handle.write(f"Export publication in progress. Staging: {staging.name}\n")
            print("Weights verified; publishing to the reserved output directory...", flush=True)
            for path in sorted(staging.iterdir()):
                os.rename(path, target / path.name)
            staging.rmdir()
            marker.unlink()
        except BaseException:
            if staging is not None and staging.exists():
                status = "verified" if validated else "possibly incomplete/unverified"
                print(f"Export failed; retaining {status} staging: {staging}", file=sys.stderr, flush=True)
            if (target / ".export-incomplete").exists():
                print(f"Publication incomplete; do not load {target}", file=sys.stderr, flush=True)
            raise
        finally:
            lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Actor checkpoint directory; relative paths are relative to the repository",
    )
    parser.add_argument("--base-model", required=True, help="Complete base Diffusers pipeline used for training")
    parser.add_argument("--output", required=True, help="New output directory; existing exports are never overwritten")
    parser.add_argument("--dtype", choices=["preserve", "float32", "bfloat16"], default="preserve")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = repository_path(args.checkpoint).resolve(strict=True)
    base = repository_path(args.base_model).resolve(strict=True)
    output = repository_path(args.output)
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output = output.resolve()
    for existing in (source, base):
        if existing == output or existing in output.parents or output in existing.parents:
            raise ValueError("Checkpoint, base and output directories must not overlap")

    fsdp = read_json(source / "fsdp_config.json")
    world_size = fsdp.get("world_size")
    if fsdp.get("FSDP_version") != 2 or type(world_size) is not int or world_size < 1:
        raise ValueError("Expected an FSDP2 checkpoint with a positive world_size")
    ranks = [source / f"model_world_size_{world_size}_rank_{i}.pt" for i in range(world_size)]
    if set(source.glob("model_world_size_*_rank_*.pt")) != set(ranks):
        raise ValueError("Missing or unexpected model rank files")
    if any(not path.is_file() or path.stat().st_size == 0 for path in ranks):
        raise ValueError("A model rank file is missing or empty")

    source_config = read_json(source / "huggingface/config.json")
    if source_config.get("_class_name") not in (MODEL_CLASS, "FSDP" + MODEL_CLASS):
        raise ValueError(f"Unexpected actor class: {source_config.get('_class_name')}")
    base_config = read_json(base / "transformer/config.json")
    if base_config.get("_class_name") != MODEL_CLASS:
        raise ValueError("The base must contain a Qwen-Image-2.1 transformer")
    if read_json(base / "model_index.json").get("_class_name") != PIPELINE_CLASS:
        raise ValueError("The base must be a Qwen-Image-2.1 pipeline")
    # FSDP changes the saved class name, not the architecture. Keep every actual
    # architecture setting from the actor and require an exact base match.
    source_arch = {key: value for key, value in source_config.items() if not key.startswith("_")}
    base_arch = {key: value for key, value in base_config.items() if not key.startswith("_")}
    if source_arch != base_arch:
        raise ValueError("Actor/base architecture configurations differ")
    normalized_config = {**source_config, "_class_name": MODEL_CLASS}
    normalized_config.pop("_name_or_path", None)

    print(f"Checkpoint: {source}", flush=True)
    print(f"Base:       {base}", flush=True)
    print(f"Output:     {output / 'diffusion_pytorch_model.safetensors'}", flush=True)
    rank_gib = sum(path.stat().st_size for path in ranks) / 1024**3
    print(f"Model ranks: {world_size}; total input size: {rank_gib:.2f} GiB; dtype: {args.dtype}", flush=True)
    if args.dry_run:
        print("Paths and architecture configs passed. No weights loaded or output created.")
        return

    # Use the pinned submodule's checked FSDP reconstruction,
    # finite-value checks and source/output round-trip validation.
    sys.path.insert(0, str(REPOSITORY / "third_party/verl-omni"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from verl_omni.model_merger import ModelMergerConfig, fsdp_model_merger, merge_model
    from verl_omni.model_merger.architectures import _TRANSFORMERS

    # This revision of the generic merger predates Qwen Image 2.1. Registration
    # is process-local; neither the submodule nor the source checkpoint is edited.
    _TRANSFORMERS[PIPELINE_CLASS] = MODEL_CLASS
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        # Only override publication for this export; retain all upstream weight
        # reconstruction/validation and leave the submodule's files untouched.
        original_publication = fsdp_model_merger.publication_directory
        stack.callback(setattr, fsdp_model_merger, "publication_directory", original_publication)
        fsdp_model_merger.publication_directory = shared_filesystem_publication
        temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix="qwen21-actor-config-"))
        config_dir = Path(temporary)
        (config_dir / "config.json").write_text(json.dumps(normalized_config, indent=2) + "\n", encoding="utf-8")
        result = merge_model(
            ModelMergerConfig(
                operation="merge",
                backend="fsdp",
                local_dir=str(source),
                base_model=str(base),
                hf_model_config_path=str(config_dir),
                target_dir=str(output),
                output_format="transformer",
                dtype=args.dtype,
                max_shard_size=1024**4,
                trust_checkpoint=True,
            )
        )
    weights = sorted(result.output_dir.rglob("*.safetensors"))
    expected = result.output_dir / "diffusion_pytorch_model.safetensors"
    if weights != [expected]:
        raise RuntimeError(f"Expected exactly one full-model safetensors file, got: {weights}")
    print(f"Export complete: {expected} ({expected.stat().st_size / 1024**3:.2f} GiB)", flush=True)
    print(f"Verification manifest: {result.manifest_path}", flush=True)


if __name__ == "__main__":
    main()
