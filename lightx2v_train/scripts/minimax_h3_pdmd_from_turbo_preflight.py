"""CPU-only header/config checks before H3 Turbo-EMA -> PDMD initialization."""

import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

PATTERNS = {"adaln_proj.linear": 16, "norm_out.linear": 16}
TARGETS = {"to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", *PATTERNS}


def adapter_header(path):
    """Read only safetensors metadata, without torch, a model, or GPU access."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= min(size - 8, 100_000_000):
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(handle.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors tensor table: {path}")
    tensors = {}
    ranges = []
    byte_sizes = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8}
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        normalized = key.removeprefix("transformer.")
        normalized = normalized.replace(".lora.down.weight", ".lora_A.weight").replace(".lora.up.weight", ".lora_B.weight")
        if not normalized.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"Student initialization must contain LoRA-only tensors; unexpected key: {key}")
        shape, offsets, dtype = entry.get("shape"), entry.get("data_offsets"), entry.get("dtype")
        if not isinstance(shape, list) or len(shape) != 2 or any(type(dimension) is not int or dimension <= 0 for dimension in shape):
            raise ValueError(f"Invalid LoRA matrix shape for {key}: {shape}")
        if dtype not in byte_sizes or not isinstance(offsets, list) or len(offsets) != 2 or any(type(offset) is not int for offset in offsets):
            raise ValueError(f"Invalid LoRA tensor metadata for {key}")
        start, end = offsets
        if start < 0 or end - start != math.prod(shape) * byte_sizes[dtype] or end > size - 8 - header_size:
            raise ValueError(f"Truncated or invalid LoRA tensor data for {key}")
        if normalized in tensors:
            raise ValueError(f"Duplicate normalized LoRA key: {key}")
        tensors[normalized] = shape
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError(f"Noncontiguous or overlapping LoRA tensor data: {path}")
        cursor = end
    if not tensors or cursor != size - 8 - header_size:
        raise ValueError(f"Empty or truncated safetensors export: {path}")
    return tensors


def validate_source(path, lora):
    weights = path / "pytorch_lora_weights.safetensors" if path.is_dir() else path
    if not weights.is_file() or weights.suffix != ".safetensors":
        raise ValueError(f"Missing student initialization LoRA safetensors: {weights}")
    tensors = adapter_header(weights)
    modules = {key.rsplit(".lora_", 1)[0] for key in tensors}
    categories = set()
    for module in modules:
        a, b = tensors.get(module + ".lora_A.weight"), tensors.get(module + ".lora_B.weight")
        if a is None or b is None:
            raise ValueError(f"Missing LoRA A/B pair: {module}")
        matched = [target for target in TARGETS if module == target or module.endswith("." + target)]
        if len(matched) != 1:
            raise ValueError(f"Unexpected LoRA target module: {module}")
        target = matched[0]
        rank = PATTERNS.get(target, lora["rank"])
        if a[0] != rank or b[1] != rank:
            raise ValueError(f"LoRA rank mismatch for {module}: expected {rank}, A={a}, B={b}")
        categories.add(target if target in PATTERNS else "regular")
    if categories != {"regular", *PATTERNS}:
        raise ValueError("Student initialization must include regular, block AdaLN, and norm_out.linear LoRA tensors.")
    sidecar = weights.parent / "adapter_config.json"
    if sidecar.is_file():
        adapter = json.loads(sidecar.read_text())
        expected = {"r": lora["rank"], "lora_alpha": lora["alpha"], "rank_pattern": PATTERNS, "alpha_pattern": PATTERNS}
        for key, value in expected.items():
            if adapter.get(key) != value:
                raise ValueError(f"Source adapter_config.json {key}={adapter.get(key)!r}, expected {value!r}")
        if set(adapter.get("target_modules", [])) != TARGETS:
            raise ValueError("Source adapter_config.json target_modules differ from the PDMD adapter targets.")
        if adapter.get("use_rslora", False) or adapter.get("use_dora", False) or adapter.get("bias", "none") != "none" or adapter.get("modules_to_save"):
            raise ValueError("Source adapter uses unsupported scaling or additional trainable parameters.")
    return weights, len(tensors)


def main(config_path):
    source = Path(__file__).resolve().parents[1] / "lightx2v_train/runtime/config.py"
    spec = importlib.util.spec_from_file_location("_pdmd_from_turbo_config", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config(config_path)
    training = config["training"]
    student = training["student"]
    lora = student["lora"]
    if student.get("train_type") != "lora" or (lora.get("rank"), lora.get("alpha")) != (128, 128):
        raise ValueError("This launcher requires student LoRA rank128/alpha128.")
    if lora.get("rank_pattern") != PATTERNS or lora.get("alpha_pattern") != PATTERNS or set(lora.get("target_modules", [])) != TARGETS:
        raise ValueError("This launcher requires block AdaLN AND norm_out.linear rank16/alpha16 with the Turbo target set.")
    if not student.get("checkpoint_path") or student.get("checkpoint_strict", True) is not True:
        raise ValueError("Set training.student.checkpoint_path and checkpoint_strict=true for exact adapter initialization.")
    if student.get("ema", {}).get("enabled", False):
        raise ValueError("PDMD uses live student weights; source EMA is loaded once as initialization.")
    if config["distributed"]["fsdp2"].get("size") != 32 or config["distributed"]["sequence_parallel"].get("enabled", False):
        raise ValueError("This launcher requires FSDP32 with sequence parallelism disabled.")
    dmd = training["dmd"]
    if training.get("method") != "dmd" or not dmd.get("official_pdmd") or dmd.get("num_inference_steps") != 8:
        raise ValueError("This launcher requires official PDMD with eight inference steps.")
    if (dmd.get("official_rollout_video_shift"), dmd.get("official_rollout_audio_shift")) != (6.0, 3.0):
        raise ValueError("This launcher requires video/audio rollout shifts 6/3.")
    weights, count = validate_source(Path(student["checkpoint_path"]), lora)
    updates = int(training["max_train_iters"])
    critic_steps = int(dmd["fake_update_ratio"])
    student_updates = updates // (critic_steps + 1)
    print(f"Verified student EMA initialization header: {weights}; LoRA tensors={count}; exact full-model key/shape validation occurs during setup.")
    print("LoRA: regular rank=128 alpha=128; block AdaLN and norm_out.linear rank=16 alpha=16.")
    print("Initialization only: student=source EMA; frozen teacher=H3 base; fake=H3 base; optimizers start fresh. Future PDMD resumes restore its own state.")
    print(f"PDMD budget: optimizer_updates={updates}, student_updates={student_updates}, critic_updates={updates - student_updates}; save_every_iters={training['save_every_iters']}.")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"H3 PDMD from Turbo preflight failed: {error}") from error
