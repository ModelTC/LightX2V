"""Fine-T2I captions -> H3-Image PE -> cached conditions -> image distillation."""

import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

DIFFUSERS_REVISION = "875373ac7f5856071e345a3e394c74efccc6b9e6"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
AFS_ROOT = Path("/mnt/lm_data_afs/gushiqiao")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("pdmd", "pdmd_gan", "pdd"), default=os.environ.get("H3_IMAGE_TRAINING_METHOD", "pdmd"))
    parser.add_argument("--code-root", type=Path, default=os.environ.get("H3_CODE_ROOT", DEFAULT_ROOT))
    parser.add_argument("--python", default=os.environ.get("H3_PYTHON", sys.executable))
    parser.add_argument("--model-path", type=Path, default=os.environ.get("H3_IMAGE_MODEL_PATH", AFS_ROOT / "models/MiniMax-H3-Image"))
    parser.add_argument("--text-encoder-path", type=Path, default=os.environ.get("H3_IMAGE_TEXT_ENCODER_PATH"))
    parser.add_argument("--pe-model-path", type=Path, default=os.environ.get("H3_IMAGE_PE_MODEL_PATH", AFS_ROOT / "models/MiniMax-H3-Image-PE"))
    parser.add_argument("--dataset-dir", type=Path, default=os.environ.get("H3_IMAGE_DATASET_DIR", AFS_ROOT / "datasets/minimax_h3_image_finet2i_10k_2048"))
    parser.add_argument("--prompts", type=Path, default=os.environ.get("H3_IMAGE_PROMPTS"))
    parser.add_argument("--pe-cache", type=Path, default=os.environ.get("H3_IMAGE_PE_CACHE"))
    parser.add_argument("--cache-dir", type=Path, default=os.environ.get("H3_IMAGE_CACHE_DIR"))
    parser.add_argument("--gan-real-dir", type=Path, default=os.environ.get("H3_IMAGE_GAN_REAL_DIR"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=os.environ.get("H3_IMAGE_CONFIG"))
    parser.add_argument("--count", type=int, default=int(os.environ.get("H3_IMAGE_PROMPT_COUNT", "10000")))
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--pe-backend", choices=("transformers", "http"), default=os.environ.get("H3_IMAGE_PE_BACKEND", "transformers"))
    parser.add_argument("--pe-endpoint", default=os.environ.get("H3_IMAGE_PE_ENDPOINT", "http://127.0.0.1:30000"))
    parser.add_argument("--pe-served-model", default=os.environ.get("H3_IMAGE_PE_SERVED_MODEL", "MiniMax-H3-Image-PE"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.text_encoder_path = args.text_encoder_path or args.model_path / "text_encoder"
    args.prompts = args.prompts or args.dataset_dir / "prompts_raw.jsonl"
    args.pe_cache = args.pe_cache or args.dataset_dir / "pe"
    args.cache_dir = args.cache_dir or args.dataset_dir / "condition_cache"
    args.gan_real_dir = args.gan_real_dir or args.dataset_dir / "gan_real_cache"
    if args.method == "pdd":
        args.config = args.config or args.code_root / "lightx2v_train/configs/train/pdd/minimax_h3_image_pdd2_2048_fsdp8.yaml"
        args.output_dir = args.output_dir or Path(os.environ.get("H3_PDD_OUTPUT", "/mnt/devsft_afs_2/gushiqiao/outputs/h3_image_pdd2_10k_2048"))
        args.max_iters = args.max_iters if args.max_iters is not None else int(os.environ.get("H3_PDD_MAX_ITERS", "30000"))
    elif args.method == "pdmd_gan":
        args.config = args.config or args.code_root / "lightx2v_train/configs/train/dmd/minimax_h3_image_pdmd2_gan_2048_r128_a128_fsdp8.yaml"
        args.output_dir = args.output_dir or Path(os.environ.get("H3_IMAGE_PDMD_OUTPUT", "/mnt/devsft_afs_2/gushiqiao/outputs/h3_image_pdmd2_gan_10k_2048"))
        args.max_iters = args.max_iters if args.max_iters is not None else int(os.environ.get("H3_IMAGE_MAX_ITERS", "15000"))
    else:
        args.config = args.config or args.code_root / "lightx2v_train/configs/train/dmd/minimax_h3_image_pdmd2_2048_r128_a128_fsdp8.yaml"
        args.output_dir = args.output_dir or Path(os.environ.get("H3_IMAGE_PDMD_OUTPUT", "/mnt/devsft_afs_2/gushiqiao/outputs/h3_image_pdmd2_10k_2048"))
        args.max_iters = args.max_iters if args.max_iters is not None else int(os.environ.get("H3_IMAGE_MAX_ITERS", "15000"))
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if args.count < 1 or args.max_iters < 1:
        parser.error("--count and --max-iters must be positive")
    return args


def commands(args):
    data = args.code_root / "lightx2v_train/data_process/minimax_h3_image"
    distributed = [args.python, "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=8", "--max_restarts=0"]
    expand = distributed + [str(data / "expand_prompts.py"), "--prompts", str(args.prompts),
                            "--pe-model-path", str(args.pe_model_path), "--output-dir", str(args.pe_cache),
                            "--expected-samples", str(args.count), "--backend", args.pe_backend]
    if args.pe_backend == "http":
        expand += ["--endpoint", args.pe_endpoint, "--served-model", args.pe_served_model]
    plan = {
        "select": [args.python, "-u", str(data / "select_prompts.py"), "--output", str(args.prompts), "--count", str(args.count)],
        "pe": expand,
        "cache": distributed + [str(data / "prepare_cache.py"), "--prompts", str(args.pe_cache / "prompts_expanded.jsonl"),
                                 "--model-path", str(args.model_path), "--text-encoder-path", str(args.text_encoder_path),
                                 "--output-dir", str(args.cache_dir), "--height", "2048", "--width", "2048",
                                 "--expected-samples", str(args.count), "--expected-world-size", "8"],
    }
    if args.method == "pdmd_gan":
        plan["gan_real"] = distributed + [str(data / "prepare_gan_real_cache.py"), "--condition-cache", str(args.cache_dir / "metadata.jsonl"),
                                         "--model-path", str(args.model_path), "--output-dir", str(args.gan_real_dir),
                                         "--expected-samples", str(args.count), "--expected-world-size", "8"]
    plan["train"] = distributed + [str(args.code_root / "lightx2v_train/train.py"), "--config", str(args.config)]
    return plan


def require_files(root, files, suggestion):
    missing = [str(root / filename) for filename in files if not (root / filename).is_file() or (root / filename).stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing model files:\n  " + "\n  ".join(missing[:12]) + "\nDownload command:\n" + suggestion)


def check_weight_index(root, name, suggestion, *, require_shards=True):
    require_files(root, [name], suggestion)
    index = json.loads((root / name).read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards:
        raise ValueError(f"Empty weight index: {root / name}")
    if require_shards:
        require_files(root, shards, suggestion)


def validate_models(args):
    image_download = shlex.join(["hf", "download", "MiniMaxAI/MiniMax-H3-Image", "--include", "transformer_diffusers/*", "pdd_config.json", "tokenizer/*", "processor/*", "--local-dir", str(args.model_path)])
    require_files(args.model_path, ["pdd_config.json", "transformer_diffusers/config.json", "tokenizer/tokenizer.json",
                                    "tokenizer/tokenizer_config.json", "processor/preprocessor_config.json"], image_download)
    check_weight_index(args.model_path / "transformer_diffusers", "diffusion_pytorch_model.safetensors.index.json", image_download)
    if args.method == "pdd":
        native = args.model_path / "transformer"
        native_download = shlex.join(["hf", "download", "MiniMaxAI/MiniMax-H3-Image", "--include", "transformer/*", "--local-dir", str(args.model_path)])
        require_files(native, ["model.safetensors.index.json"], native_download)
        mapping = json.loads((native / "model.safetensors.index.json").read_text())["weight_map"]
        head_shards = {mapping[key] for key in ("final_layer.video_out.weight", "final_layer.video_out.bias")}
        require_files(native, head_shards, native_download)
    te_download = shlex.join(["hf", "download", "MiniMaxAI/MiniMax-H3-Image", "--include", "text_encoder/*", "--local-dir", str(args.text_encoder_path.parent)])
    require_files(args.text_encoder_path, ["config.json"], te_download)
    check_weight_index(args.text_encoder_path, "model.safetensors.index.json", te_download)
    pe_download = shlex.join(["hf", "download", "MiniMaxAI/MiniMax-H3-Image-PE", "--local-dir", str(args.pe_model_path)])
    require_files(args.pe_model_path, ["config.json", "tokenizer.json", "tokenizer_config.json", "prompt/t2i/system_prompt.txt",
                                       "prompt/t2i/task_prompt.txt", "prompt/t2i/request_params.json"], pe_download)
    check_weight_index(args.pe_model_path, "model.safetensors.index.json", pe_download, require_shards=args.pe_backend == "transformers")


def validate_environment(args):
    import torch
    import yaml

    try:
        from diffusers import MiniMaxH3ImageTransformer2DModel  # noqa: F401
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor  # noqa: F401
        if args.pe_backend == "transformers":
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            CONFIG_MAPPING["qwen3_5"]
            importlib.import_module("xgrammar")
    except (ImportError, KeyError) as error:
        raise RuntimeError(
            "H3-Image model support is missing from this Python environment. "
            f"Use diffusers from git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_REVISION}, "
            "Transformers with Qwen3-VL/Qwen3.5 support, and xgrammar for local PE."
        ) from error
    if torch.cuda.device_count() != 8:
        raise ValueError(f"This launcher needs exactly 8 visible GPUs; found {torch.cuda.device_count()}. Set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7.")
    config = yaml.safe_load(args.config.read_text())
    fsdp = config["distributed"]["fsdp2"]
    if config["model"]["name"] != "minimax_h3_image" or not fsdp["enabled"] or fsdp["size"] != 8 or not fsdp.get("stream_load_pretrained"):
        raise ValueError("H3-Image requires its image model and eight-way FSDP2 with stream_load_pretrained=true.")
    if args.method == "pdd":
        if config["training"]["method"] != "pdd" or config["model"].get("image_head_mode") != "base":
            raise ValueError("Image PDD requires training.method=pdd and model.image_head_mode=base.")
        if config["training"]["pdd"]["num_inference_steps"] != 2:
            raise ValueError("This image PDD launcher requires two inference steps.")
    else:
        dmd = config["training"]["dmd"]
        if dmd["num_inference_steps"] != 2 or dmd["official_rollout_sigmas"] != [1.0, 0.75, 0.0]:
            raise ValueError("This two-step launcher requires physical rollout sigmas [1.0, 0.75, 0.0].")
        gan_enabled = dmd.get("gan", {}).get("enabled", False)
        if bool(gan_enabled) != (args.method == "pdmd_gan"):
            raise ValueError("The launcher method and training.dmd.gan.enabled disagree; select pdmd_gan for GAN training.")


def read_rows(path):
    text = path.read_text()
    return json.loads(text) if path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]


def validate_prompts(path, count, *, unique=False):
    rows = read_rows(path)
    if len(rows) != count:
        raise ValueError(f"Expected {count} prompts, found {len(rows)}: {path}")
    ids, texts = set(), set()
    for row in rows:
        identity = row.get("id")
        prompt = row.get("prompt")
        if not isinstance(identity, int) or identity in ids or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Prompt rows require unique integer id and nonempty prompt: {path}")
        ids.add(identity)
        normalized = " ".join(prompt.split()).casefold()
        if unique and normalized in texts:
            raise ValueError(f"Duplicate prompt at id={identity}: {path}")
        texts.add(normalized)


def validate_cache(path, count):
    rows = read_rows(path)
    if len(rows) != count:
        raise ValueError(f"Expected {count} cached conditions, found {len(rows)}: {path}")
    for row in rows:
        condition = Path(row["condition_path"])
        if not condition.is_absolute():
            condition = path.parent / condition
        if not condition.is_file():
            raise FileNotFoundError(f"Cached condition missing: {condition}")


def validate_gan_cache(path, count):
    from lightx2v_train.trainers.dmd.dmd2_gan_data import load_image_gan_real_records

    _, rows, _ = load_image_gan_real_records(path, geometry=(2048, 2048))
    if len(rows) != count:
        raise ValueError(f"Expected {count} GAN positive images, found {len(rows)}: {path}")


def run(args):
    plan = commands(args)
    print(f"H3-Image: {args.count} Fine-T2I prompts -> PE -> 2048x2048 conditions -> two-step {args.method.upper()} on 8 GPUs.", flush=True)
    if args.method == "pdd":
        print("Teacher uses the preserved original single head; its continuous-score use remains experimental. PDD has no fake model.", flush=True)
    else:
        print("Teacher is the released distilled six-head model; continuous-sigma head reuse is experimental.", flush=True)
    if args.method == "pdmd_gan":
        print("GAN positives: released six-step H3-Image teacher latents at 2048x2048, paired with the same PE text conditions.", flush=True)
    for stage, command in plan.items():
        print(f"[h3-image][{stage}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        print("Dry run: no downloads, model loads, or files written.", flush=True)
        return

    validate_models(args)
    validate_environment(args)
    env = {**os.environ, "H3_IMAGE_MODEL_PATH": str(args.model_path),
           "H3_IMAGE_CACHE": str(args.cache_dir / "metadata.jsonl"),
           "H3_IMAGE_PDMD_OUTPUT": str(args.output_dir), "H3_IMAGE_MAX_ITERS": str(args.max_iters)}
    if args.method == "pdd":
        env.update(H3_PDD_OUTPUT=str(args.output_dir), H3_PDD_MAX_ITERS=str(args.max_iters))
    if args.method == "pdmd_gan":
        env["H3_IMAGE_GAN_REAL_CACHE"] = str(args.gan_real_dir / "metadata.jsonl")
    package = str(args.code_root / "lightx2v_train")
    env["PYTHONPATH"] = package + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {"method": args.method, "commands": plan, "prompt_count": args.count,
               "diffusers_revision": DIFFUSERS_REVISION, "cache": env["H3_IMAGE_CACHE"]}
    if args.method == "pdd":
        receipt.update(max_student_updates=args.max_iters)
    else:
        receipt.update(max_role_updates=args.max_iters, student_sigmas=[1.0, 0.75, 0.0])
    if args.method == "pdmd_gan":
        receipt.update(gan_real_cache=env["H3_IMAGE_GAN_REAL_CACHE"], gan_real_source="h3-image-six-step-teacher", gan_real_condition="pe-caption")
    (args.output_dir / "stages.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def execute(stage):
        print(f"[h3-image] starting {stage}", flush=True)
        subprocess.run(plan[stage], cwd=package, env=env, check=True)

    if args.prompts.exists():
        validate_prompts(args.prompts, args.count, unique=True)
        print(f"[h3-image] reusing supplied prompts: {args.prompts}", flush=True)
    else:
        execute("select")
        validate_prompts(args.prompts, args.count, unique=True)
    execute("pe")
    validate_prompts(args.pe_cache / "prompts_expanded.jsonl", args.count)
    execute("cache")
    validate_cache(args.cache_dir / "metadata.jsonl", args.count)
    if args.method == "pdmd_gan":
        execute("gan_real")
        validate_gan_cache(args.gan_real_dir / "metadata.jsonl", args.count)
    execute("train")


if __name__ == "__main__":
    run(arguments())
