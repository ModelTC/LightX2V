"""Run the pinned FP32 GenEval evaluator on Ascend without changing its scores."""

import ast
import functools
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def device_program(source, filename):
    """Adapt only the upstream CUDA selection and its availability assertion."""
    program = ast.parse(source, filename=filename)
    assignments = [node for node in program.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "DEVICE"]
    assertions = [node for node in program.body if isinstance(node, ast.Assert)
                  and ast.unparse(node.test) == "DEVICE == 'cuda'"]
    if len(assignments) != 1 or len(assertions) != 1:
        raise ValueError("Unexpected pinned GenEval device selection")
    assignments[0].value = ast.Constant("npu")
    assertions[0].test = ast.parse("torch.npu.is_available()", mode="eval").body
    return compile(ast.fix_missing_locations(program), filename, "exec")


def score_geneval(expected, generated, paths, output):
    import mmcv
    import open_clip
    import torch
    import torch_npu  # noqa: F401
    import mmcv.ops.multi_scale_deform_attn as attention

    if mmcv.__version__ != "1.7.2" or not torch.npu.is_available():
        raise ValueError("Ascend GenEval requires MMCV 1.7.2 and an available NPU")
    here = Path(__file__).resolve().parents[3]
    lock = json.loads((here / "configs/ascend_npu/source-lock.json").read_text())["geneval"]
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
    script = repository / "evaluation/evaluate_images.py"
    create_model = open_clip.create_model_and_transforms
    original_argv = sys.argv
    try:
        # open_clip's original default precision is FP32. Preserve it and route
        # its existing cache parameter to the verified local CLIP checkpoint.
        open_clip.create_model_and_transforms = functools.partial(create_model, cache_dir=paths["geneval_clip"])
        sys.argv = [str(script), str(directory.resolve()), "--outfile", str(report.resolve()),
                    "--model-path", paths["geneval_weights"], "--model-config", paths["geneval_config"]]
        # MMCV 1.7.2 dispatches NPU tensors to its released, equivalent PyTorch
        # grid_sample implementation. Neither CUDA nor MLU dispatch is patched.
        namespace = dict(__name__="__main__", __file__=str(script), __package__=None)
        exec(device_program(script.read_text(), str(script)), namespace)
    finally:
        sys.argv = original_argv
        open_clip.create_model_and_transforms = create_model
    entries = [json.loads(line) for line in report.read_text().splitlines() if line.strip()]
    if len(entries) != len(expected) or len({entry["filename"] for entry in entries}) != len(expected):
        raise ValueError("Incomplete or repeated GenEval scores")
    groups = {}
    for entry in entries:
        groups.setdefault(entry["tag"], []).append(bool(entry["correct"]))
    if set(groups) != {"single_object", "two_object", "counting", "colors", "position", "color_attr"}:
        raise ValueError("Missing GenEval dimensions")
    dimensions = {key: sum(values) / len(values) for key, values in sorted(groups.items())}
    provenance = dict(backend="MMCV multi_scale_deformable_attn_pytorch on Ascend", precision="float32",
                      geneval_revision=lock["revision"], mmcv_version=mmcv.__version__,
                      mmcv_attention_source_sha256=sha256(attention.__file__), evaluator_sha256=sha256(script),
                      adapter_sha256=sha256(__file__))
    (output / "ascend-geneval-runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return dict(count=len(expected), geneval_by_dimension=dimensions, geneval_overall=sum(dimensions.values()) / 6)
