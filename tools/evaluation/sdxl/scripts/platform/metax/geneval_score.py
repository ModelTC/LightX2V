"""Score with pinned GenEval and MMCV's released PyTorch attention implementation."""

import functools
import hashlib
import json
import runpy
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_geneval(expected, generated, paths, output):
    import mmcv
    import mmcv.ops.multi_scale_deform_attn as deform_attention
    import open_clip

    if mmcv.__version__ != "1.7.2":
        raise ValueError("The MetaX GenEval runtime requires MMCV 1.7.2")
    here = Path(__file__).resolve().parents[3]
    lock = json.loads((here / "configs/metax/source-lock.json").read_text())["geneval"]
    repository = Path(paths["geneval"])
    for entry in lock["files"]:
        if sha256(repository / entry["path"]) != entry["sha256"]:
            raise ValueError("GenEval source differs from its pinned original")
    directory = output / "geneval"
    directory.mkdir(exist_ok=False)
    for item in expected:
        sample = directory / item["id"] / "samples"
        sample.mkdir(parents=True)
        (sample.parent / "metadata.jsonl").write_text(json.dumps(item["metadata"]) + "\n")
        (sample / "0.png").symlink_to(Path(generated[item["id"]]["path"]))
    report = output / "geneval-scores.jsonl"
    create_model = open_clip.create_model_and_transforms
    cuda_available = deform_attention.IS_CUDA_AVAILABLE
    mlu_available = deform_attention.IS_MLU_AVAILABLE
    original_argv = sys.argv
    try:
        # MMCV's CPU extension has no MACA deformable-attention kernel. Its
        # released grid_sample implementation accepts MetaX CUDA tensors too.
        deform_attention.IS_CUDA_AVAILABLE = False
        deform_attention.IS_MLU_AVAILABLE = False
        open_clip.create_model_and_transforms = functools.partial(create_model, cache_dir=paths["geneval_clip"])
        script = repository / "evaluation/evaluate_images.py"
        sys.argv = [str(script), str(directory.resolve()), "--outfile", str(report.resolve()), "--model-path", paths["geneval_weights"], "--model-config", paths["geneval_config"]]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv
        open_clip.create_model_and_transforms = create_model
        deform_attention.IS_CUDA_AVAILABLE = cuda_available
        deform_attention.IS_MLU_AVAILABLE = mlu_available
    entries = [json.loads(line) for line in report.read_text().splitlines() if line.strip()]
    if len(entries) != len(expected) or len({entry["filename"] for entry in entries}) != len(expected):
        raise ValueError("Incomplete or repeated GenEval scores")
    groups = {}
    for entry in entries:
        groups.setdefault(entry["tag"], []).append(bool(entry["correct"]))
    if set(groups) != {"single_object", "two_object", "counting", "colors", "position", "color_attr"}:
        raise ValueError("Missing GenEval dimensions")
    dimensions = {key: sum(values) / len(values) for key, values in sorted(groups.items())}
    provenance = {
        "backend": "MMCV multi_scale_deformable_attn_pytorch on MetaX",
        "gen_eval_revision": lock["revision"],
        "mmcv_version": mmcv.__version__,
        "mmcv_attention_source_sha256": sha256(deform_attention.__file__),
        "evaluator_sha256": sha256(repository / "evaluation/evaluate_images.py"),
    }
    (output / "metax-geneval-runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {"count": len(expected), "geneval_by_dimension": dimensions, "geneval_overall": sum(dimensions.values()) / 6}
