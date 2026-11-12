"""Manual two-GPU PDD trainer save/resume regression with a tiny H3-Image.

Run with ``torchrun --standalone --nproc_per_node=2 -m
lightx2v_train.model_zoo.tests.verify_pdd_training --output-dir /tmp/h3-pdd-test``.
The output directory must be new. This is not collected by CPU test discovery.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml
from lightx2v_train.data import build_data
from lightx2v_train.model_zoo import build_model
from lightx2v_train.model_zoo.minimax_h3_image.tests.test_model import SIGMAS
from lightx2v_train.model_zoo.native.minimax_h3_image import transformer_class
from lightx2v_train.runtime.distributed import barrier, cleanup_distributed, get_rank, init_distributed
from lightx2v_train.trainers.pdd.trainer import PDDTrainer
from safetensors.torch import load_file, save_file


def prepare_fixture(root):
    root.mkdir(parents=True, exist_ok=False)
    model_dir = root / "model"
    native = transformer_class()(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=24,
        patch_size=(1, 2, 2),
        text_dim=5120,
        freq_dim=4,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=1,
        num_pdd_heads=6,
    )
    native.save_pretrained(model_dir / "transformer_diffusers", safe_serialization=True, max_shard_size="30KB")
    (model_dir / "pdd_config.json").write_text(json.dumps({"num_inference_steps": 6, "sigmas": SIGMAS}))
    original = model_dir / "transformer"
    original.mkdir()
    weights = {f"final_layer.video_out.{name}": value for name, value in native.pdd_proj_out[0].state_dict().items()}
    save_file(weights, str(original / "model.safetensors"))
    (original / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "model.safetensors" for name in weights}}))
    rows = []
    for index in range(8):
        filename = root / f"condition-{index}.pt"
        prompt = f"Tiny PDD regression {index}"
        torch.save(
            {
                "cache_schema_version": 1,
                "cache_fingerprint": "tiny-pdd-v1",
                "conditioning": {"prompt": prompt, "positive": {"prompt_embeds": torch.randn(1, 3, 5120, dtype=torch.bfloat16), "text_token_tags": torch.ones(3, dtype=torch.long)}},
                "inputs": {},
                "meta": {},
            },
            filename,
        )
        rows.append(
            {
                "id": index,
                "task": "t2i",
                "prompt": prompt,
                "num_frames": 1,
                "target_height": 64,
                "target_width": 64,
                "latent_shape": [24, 1, 4, 4],
                "condition_path": str(filename),
                "cache_fingerprint": "tiny-pdd-v1",
            }
        )
    (root / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def configuration(root):
    template = Path(__file__).resolve().parents[3] / "configs/train/pdd/minimax_h3_image_pdd2_2048_fsdp8.yaml"
    config = yaml.safe_load(template.read_text())
    config["model"]["pretrained_model_name_or_path"] = str(root / "model")
    config["distributed"]["fsdp2"]["size"] = 2
    config["data"]["train"].update(data_path=str(root / "metadata.jsonl"), num_workers=0)
    training = config["training"]
    training.update(output_dir=str(root / "train"), max_train_iters=3, save_every_iters=1, save_total_limit=3)
    training["lora"].update(rank=4, alpha=4, rank_pattern={"adaln_proj.linear": 2, "norm_out.linear": 2}, alpha_pattern={"adaln_proj.linear": 2, "norm_out.linear": 2})
    training["pdd"]["generation_shapes"] = [{"value": [64, 64]}]
    return config


def run_training(config):
    model = build_model(config)
    model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
    trainer = PDDTrainer(config)
    trainer.set_model(model)
    trainer.set_data(build_data(config, train_or_val="train"))
    trainer.train()
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    barrier()


def verify_checkpoint(directory, iteration):
    marker = json.loads((directory / "pdd.complete.json").read_text())
    assert marker["iteration"] == iteration
    for rank in range(2):
        state = torch.load(directory / f"pdd.rank{rank:05d}.pt", weights_only=False)
        assert state["iteration"] == iteration
    heads = load_file(directory / "pdd_heads.safetensors")
    assert heads["proj_out.weight"].shape == (32 * 96, 16)
    assert heads["proj_out.bias"].shape == (32 * 96,)
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in heads.values())
    metadata = json.loads((directory / "h3_image_sampling.json").read_text())
    assert metadata["method"] == "pdd" and metadata["image_head_mode"] == "base"
    assert metadata["requires_pdd_heads"] and "student_sigmas" not in metadata
    adapters = list(directory.glob("*.safetensors"))
    assert len(adapters) == 2, adapters
    lora = load_file(next(path for path in adapters if path.name != "pdd_heads.safetensors"))
    assert lora and all("lora_" in name or ".lora." in name for name in lora)
    return heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    config = configuration(root)
    init_distributed(config)
    try:
        torch.manual_seed(42)
        if get_rank() == 0:
            prepare_fixture(root)
        barrier()
        run_training(config)
        before = verify_checkpoint(root / "train/checkpoint-000000003", 3)
        barrier()
        config["training"]["max_train_iters"] = 5
        run_training(config)
        if get_rank() == 0:
            after = verify_checkpoint(root / "train/checkpoint-000000005", 5)
            assert not torch.equal(before["proj_out.weight"], after["proj_out.weight"])
            print("PASS: real PDDTrainer FSDP2, 3 updates + resume to 5, LoRA/full-head persistence", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
