"""Read-only CPU checks for new-then-old H3 DMD/PDMD launch recipes."""

import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def manifest_paths(value):
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return [path]
    require(path.is_dir(), f"Missing cache source: {path}")
    direct = path / "metadata.jsonl"
    if direct.is_file():
        return [direct]
    paths = sorted({item.resolve() for item in path.rglob("metadata.jsonl") if item.is_file()})
    require(paths, f"No metadata.jsonl manifests in {path}")
    return paths


def scan_source(source, *, task, phase):
    paths = manifest_paths(source["data_path"])
    counts, total = Counter(), 0
    for path in paths:
        first = True
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_task = str(row.get("task", "")).lower()
                allowed = {"t2av"} if task == "fl2av" and phase == "new" else {"t2av", "i2av", "fl2av"} if task == "fl2av" else {"ref2av", "ref2va"}
                require(row_task in allowed, f"Unexpected task={row_task!r} in {phase} source {path}")
                height, width = int(row["target_height"]), int(row["target_width"])
                frames = int(row.get("target_num_frames", row.get("num_frames", 0)))
                require(frames == 124 and height > 0 and width > 0 and height % 32 == 0 and width % 32 == 0, f"Expected 124 frames and positive 32-aligned native geometry: {path}")
                if task == "ref2av" and phase == "old":
                    image_count = int(row.get("reference_image_count", row.get("ref_image_count", 0)))
                    require(image_count > 0 and row.get("reference_video_count", 0) == 0 and row.get("reference_audio_count", 0) == 0, f"Expected image-only Ref cache row: {path}")
                elif task == "ref2av":
                    visual_count = int(row.get("reference_image_count", row.get("ref_image_count", 0))) + int(row.get("reference_video_count", 0))
                    require(visual_count > 0, f"Expected an image/video reference in new Ref data: {path}")
                condition = Path(row.get("condition_path", ""))
                if not condition.is_absolute():
                    condition = path.parent / condition
                if first:
                    require(condition.is_file(), f"Missing cached condition: {condition}")
                    first = False
                counts[row_task] += 1
                total += 1
        receipt = path.with_suffix(".complete.json")
        if phase == "new":
            require(receipt.is_file(), f"New data is not fully merged: missing {receipt}; finish download/encode first")
        if receipt.is_file():
            record = json.loads(receipt.read_text())
            require(record.get("manifest_sha256") == hashlib.sha256(path.read_bytes()).hexdigest(), f"Manifest differs from its completion receipt: {path}")
    require(total > 0, f"Empty {phase} cache source")
    if task == "fl2av" and phase == "old":
        require(all(counts[name] for name in ("t2av", "i2av", "fl2av")), "Old FL2V data must contain T2AV/I2AV/FL2AV")
    return {"rows": total, "tasks": dict(counts), "manifests": [str(path) for path in paths]}


def main(task, method, config_path):
    source = Path(__file__).resolve().parents[1] / "lightx2v_train/runtime/config.py"
    spec = importlib.util.spec_from_file_location("_h3_curriculum_config", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config(config_path)
    model, training, data, distributed = config["model"], config["training"], config["data"]["train"], config["distributed"]
    expected_model = "minimax_h3_t2av" if task == "fl2av" else "minimax_h3_ref2av"
    require(model["name"] == expected_model and data["kind"] == task, "Model/curriculum task differs from the selected launcher task")
    require(data["name"] == "minimax_h3_curriculum_cache_dataset", "Select the new-then-old curriculum cache dataset")
    require(data.get("drop_last", False) is False and data["batch_size"] == 1, "Curriculum requires batch_size=1 and drop_last=false")
    require(set(data["sources"]) == {"new", "old"}, "Curriculum requires exactly new and old sources")
    require(distributed["fsdp2"]["enabled"] and distributed["fsdp2"]["size"] == 32 and not distributed["sequence_parallel"]["enabled"], "This launcher requires FSDP32 and no sequence parallel")
    component = "transformer" if task == "fl2av" else "transformer_ref"
    require((Path(model["pretrained_model_name_or_path"]) / component / "config.json").is_file(), f"H3 model root must contain {component}/config.json")
    dmd = training["dmd"]
    matching = model["capabilities"]["distribution_matching"]
    require(
        training["method"] == "dmd" and dmd["num_inference_steps"] == 8 and dmd["fake_update_ratio"] == 5 and dmd["update_order"] == "fake_first",
        "Recipe requires eight steps and five fake updates before each student",
    )
    require(bool(dmd.get("official_pdmd")) == (method == "pdmd") and bool(matching.get("projected_dmd")) == (method == "pdmd"), "DMD/official PDMD flags differ from the selected method")
    shifts = (6.0, 3.0) if task == "fl2av" else (12.0, 3.0)
    require((matching["video_flow_shift"], matching["audio_flow_shift"]) == shifts, "Unexpected video/audio flow shifts")
    lora = training["student"]["lora"]
    require(
        (lora["rank"], lora["alpha"]) == (128, 128) and lora["rank_pattern"] == {"adaln_proj.linear": 16, "norm_out.linear": 16} and lora["alpha_pattern"] == lora["rank_pattern"],
        "Student must match regular128/128 and block/output AdaLN16/16",
    )
    student_init = training["student"].get("checkpoint_path")
    if student_init:
        require(Path(student_init).exists(), f"Missing student-only warm start: {student_init}")
    adapter = model["training_adapter"]
    require(type(adapter["enabled"]) is bool and math.isfinite(float(adapter["scale"])), "TrainingAdapter enabled must be boolean and scale finite")
    require(adapter["roles"] == ["fake", "teacher"], "The corrected DeCFG recipe applies the preset to fake/teacher scores only; student and inference must exclude it")
    if adapter["enabled"]:
        require(Path(adapter["path"]).is_file(), f"Missing TrainingAdapter: {adapter['path']}; run download_minimax_h3_training_adapter.sh")
    require(Path(data["sources"]["new"]["data_path"]).resolve() != Path(data["sources"]["old"]["data_path"]).resolve(), "New and old data paths must differ")
    summary = {phase: scan_source(data["sources"][phase], task=task, phase=phase) for phase in ("new", "old")}
    iterations = int(training["max_train_iters"])
    fake_ratio = int(dmd["fake_update_ratio"])
    student_updates = iterations // (fake_ratio + 1) if method == "pdmd" else iterations
    fake_updates = iterations - student_updates if method == "pdmd" else iterations * fake_ratio
    print(f"Verified {task.upper()} {method.upper()}8: new_rows={summary['new']['rows']}, old_rows={summary['old']['rows']}; every cycle NEW -> OLD -> repeat; full coverage with boundary padding.")
    print(f"New tasks={summary['new']['tasks']}; old tasks={summary['old']['tasks']}; shifts={shifts[0]}/{shifts[1]}; 4 x 8 GPUs.")
    print(f"Budget: student_updates={student_updates}, fake_updates={fake_updates}, max_train_iters={iterations}, save_every_iters={training['save_every_iters']}.")
    print(f"TrainingAdapter enabled={adapter['enabled']} path={adapter['path']} scale={adapter['scale']} roles={adapter['roles']}; frozen preset, learned LoRA separate.")
    if adapter["enabled"]:
        print("Experimental DMD/PDMD score preset: fake/teacher only; student rollout and inference exclude DeCFG.")
    print(f"student_init={student_init or 'H3 base (fresh)'}\nconfig={config_path}\noutput={training['output_dir']}")


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"H3 curriculum preflight failed: {error}") from error
