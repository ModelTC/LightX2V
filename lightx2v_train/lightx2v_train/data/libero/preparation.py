import gc
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import torch
from loguru import logger

from lightx2v_train.runtime.distributed import barrier, is_main_process

from .dataset import DEFAULT_DATASET_DIRS, _default_shape_meta
from .lerobot_dataset import LiberoLeRobotDataset
from .robot_video_dataset import PROMPT_TEMPLATE

TEXT_ENCODER_ID = "wan22ti2v5b"
TEXT_EMBEDDING_DIM = 4096


def _resolve_path(value):
    return Path(value).expanduser().resolve()


def _dataset_dirs(split_config):
    values = split_config.get("dataset_dirs") or DEFAULT_DATASET_DIRS
    if isinstance(values, (str, Path)):
        values = [values]
    return [_resolve_path(value) for value in values]


def _atomic_json_dump(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, ensure_ascii=True, indent=2)
    os.replace(temporary_path, path)


def _atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _to_serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _sliding_window_with_replication(value, window_size):
    offsets = torch.arange(window_size, dtype=torch.long)
    indices = torch.arange(value.shape[0], dtype=torch.long)[:, None] + offsets[None, :]
    indices.clamp_(max=value.shape[0] - 1)
    return value[indices]


def _episode_field_stats(value):
    return {
        "min": value.amin(0),
        "max": value.amax(0),
        "q01": torch.quantile(value, 0.01, dim=0, keepdim=False),
        "q99": torch.quantile(value, 0.99, dim=0, keepdim=False),
        "mean": value.mean(0),
        "var": value.var(0),
    }


def _aggregate_field_stats(episode_stats):
    values = {name: torch.stack([stats[name] for stats in episode_stats]) for name in ("min", "max", "q01", "q99", "mean", "var")}
    stepwise_mean = values["mean"].mean(0)
    stepwise_std = (values["var"] + (values["mean"] - stepwise_mean) ** 2).mean(0).sqrt()
    global_mean = values["mean"].mean((0, 1))
    global_std = (values["var"] + (values["mean"] - global_mean) ** 2).mean((0, 1)).sqrt()
    stepwise_min = values["min"].amin(0)
    stepwise_max = values["max"].amax(0)
    stepwise_q01 = values["q01"].amin(0)
    stepwise_q99 = values["q99"].amax(0)
    return {
        "stepwise_min": stepwise_min,
        "stepwise_max": stepwise_max,
        "global_min": stepwise_min.amin(0),
        "global_max": stepwise_max.amax(0),
        "stepwise_q01": stepwise_q01,
        "stepwise_q99": stepwise_q99,
        "global_q01": stepwise_q01.amin(0),
        "global_q99": stepwise_q99.amax(0),
        "stepwise_mean": stepwise_mean,
        "stepwise_std": stepwise_std,
        "global_mean": global_mean,
        "global_std": global_std,
    }


def calculate_dataset_stats(split_config):
    shape_meta = deepcopy(split_config.get("shape_meta") or _default_shape_meta())
    if len(shape_meta["action"]) != 1 or len(shape_meta["state"]) != 1:
        raise ValueError("Automatic LIBERO stats currently require one action field and one state field.")

    image_keys = [f"observation.images.{item['key']}" for item in shape_meta["images"]]
    dataset = LiberoLeRobotDataset(
        dataset_dirs=[str(path) for path in _dataset_dirs(split_config)],
        image_keys=image_keys,
        state_key="observation.state",
        action_key="action",
        num_frames=int(split_config.get("num_frames", 33)),
        global_sample_stride=int(split_config.get("global_sample_stride", 1)),
        val_set_proportion=float(split_config.get("val_set_proportion", 0.0)),
        is_training_set=True,
        video_backend=split_config.get("video_backend"),
    )
    action_horizon = int(split_config.get("num_frames", 33)) - 1

    def process_episode(episode_position):
        raw = dataset._load_episode(episode_position)
        state = raw[dataset.state_key].float().unsqueeze(1)
        action = _sliding_window_with_replication(raw[dataset.action_key].float(), action_horizon)
        return _episode_field_stats(state), _episode_field_stats(action)

    worker_count = min(16, len(dataset.episodes), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        episode_results = list(executor.map(process_episode, range(len(dataset.episodes))))

    state_key = shape_meta["state"][0]["key"]
    action_key = shape_meta["action"][0]["key"]
    return {
        "state": {state_key: _aggregate_field_stats([result[0] for result in episode_results])},
        "action": {action_key: _aggregate_field_stats([result[1] for result in episode_results])},
        "num_episodes": len(dataset.episodes),
        "num_transition": len(dataset),
    }


def _collect_prompts(split_configs):
    prompts = []
    seen = set()
    for split_config in split_configs:
        for dataset_dir in _dataset_dirs(split_config):
            tasks_path = dataset_dir / "meta" / "tasks.jsonl"
            if not tasks_path.is_file():
                raise FileNotFoundError(f"LIBERO tasks file does not exist: {tasks_path}")
            with tasks_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    task = str(json.loads(line)["task"])
                    prompt = PROMPT_TEMPLATE.format(task=task)
                    if prompt not in seen:
                        seen.add(prompt)
                        prompts.append(prompt)
    return prompts


def _text_cache_path(cache_dir, prompt, context_len):
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.t5_len{context_len}.{TEXT_ENCODER_ID}.pt"


def _valid_text_cache(path, context_len):
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return tuple(payload["context"].shape) == (context_len, TEXT_EMBEDDING_DIM) and tuple(payload["mask"].shape) == (context_len,)
    except Exception:
        return False


def _validate_text_caches(split_configs):
    for split_config in split_configs:
        cache_dir = _resolve_path(split_config["text_embedding_cache_dir"])
        context_len = int(split_config.get("context_len", 128))
        prompts = _collect_prompts([split_config])
        missing = [_text_cache_path(cache_dir, prompt, context_len) for prompt in prompts if not _valid_text_cache(_text_cache_path(cache_dir, prompt, context_len), context_len)]
        if missing:
            raise FileNotFoundError(f"LIBERO text embedding cache is incomplete: missing={len(missing)}/{len(prompts)} first={missing[0]}")


def precompute_text_embeddings(model_path, cache_dir, context_len, prompts):
    missing_prompts = [prompt for prompt in prompts if not _valid_text_cache(_text_cache_path(cache_dir, prompt, context_len), context_len)]
    if not missing_prompts:
        logger.info("[data-preflight] text embeddings already cached: prompts={} path={}", len(prompts), cache_dir)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Automatic FastWAM text embedding generation requires CUDA.")

    from lightx2v_train.model_zoo.native.wan.modules.t5 import T5EncoderModel

    model_root = _resolve_path(model_path)
    checkpoint_path = model_root / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer_path = model_root / "google" / "umt5-xxl"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Wan text encoder checkpoint does not exist: {checkpoint_path}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Wan tokenizer directory does not exist: {tokenizer_path}")

    device = torch.device("cuda", torch.cuda.current_device())
    logger.info(
        "[data-preflight] generating text embeddings: missing={}/{} context_len={} path={}",
        len(missing_prompts),
        len(prompts),
        context_len,
        cache_dir,
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        encoder = T5EncoderModel(
            text_len=context_len,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=str(checkpoint_path),
            tokenizer_path=str(tokenizer_path),
        )
        encoder.model.eval()
        batch_size = 16
        with torch.no_grad():
            for start in range(0, len(missing_prompts), batch_size):
                batch_prompts = missing_prompts[start : start + batch_size]
                input_ids, mask = encoder.tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                input_ids = input_ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                context = encoder.model(input_ids, mask)
                for index, prompt in enumerate(batch_prompts):
                    _atomic_torch_save(
                        {
                            "context": context[index].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                            "mask": mask[index].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                        },
                        _text_cache_path(cache_dir, prompt, context_len),
                    )

    del context, input_ids, mask, encoder
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("[data-preflight] generated {} text embeddings in {}", len(missing_prompts), cache_dir)


def _is_missing(value):
    return value is None or not str(value).strip()


def prepare_libero_fastwam_assets(config):
    data_config = config["data"]
    train_config = data_config["train"]
    val_config = data_config.get("val")
    output_dir = _resolve_path(config["training"]["output_dir"])

    configured_stats = train_config.get("pretrained_norm_stats")
    if _is_missing(configured_stats):
        stats_path = output_dir / "dataset_stats.json"
        if is_main_process() and not stats_path.is_file():
            logger.info("[data-preflight] calculating LIBERO normalization statistics")
            stats = calculate_dataset_stats(train_config)
            _atomic_json_dump(stats, stats_path)
            logger.info(
                "[data-preflight] saved normalization statistics: episodes={} transitions={} path={}",
                stats["num_episodes"],
                stats["num_transition"],
                stats_path,
            )
        barrier()
    else:
        stats_path = _resolve_path(configured_stats)
        if not stats_path.is_file():
            raise FileNotFoundError(f"Configured LIBERO normalization stats do not exist: {stats_path}")
        eval_stats_path = output_dir / "dataset_stats.json"
        if is_main_process() and stats_path != eval_stats_path:
            eval_stats_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stats_path, eval_stats_path)
        barrier()

    train_config["pretrained_norm_stats"] = str(stats_path)
    if val_config is not None and _is_missing(val_config.get("pretrained_norm_stats")):
        val_config["pretrained_norm_stats"] = str(stats_path)

    configured_cache = train_config.get("text_embedding_cache_dir")
    auto_text_cache = _is_missing(configured_cache)
    cache_dir = output_dir / "text_embeds_cache" if auto_text_cache else _resolve_path(configured_cache)
    train_config["text_embedding_cache_dir"] = str(cache_dir)
    if val_config is not None and _is_missing(val_config.get("text_embedding_cache_dir")):
        val_config["text_embedding_cache_dir"] = str(cache_dir)

    if auto_text_cache:
        generated_splits = [train_config]
        if val_config is not None and _resolve_path(val_config["text_embedding_cache_dir"]) == cache_dir:
            generated_splits.append(val_config)
        context_lengths = {int(split.get("context_len", 128)) for split in generated_splits}
        if len(context_lengths) != 1:
            raise ValueError(f"Automatic text cache generation requires one context_len, got {sorted(context_lengths)}")
        context_len = context_lengths.pop()
        prompts = _collect_prompts(generated_splits)
        if is_main_process():
            precompute_text_embeddings(
                model_path=config["model"]["model_path"],
                cache_dir=cache_dir,
                context_len=context_len,
                prompts=prompts,
            )
        barrier()

    _validate_text_caches([split for split in (train_config, val_config) if split is not None])
