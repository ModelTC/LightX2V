"""Manual two-GPU Image PDMD+GAN training/checkpoint/resume regression.

Run ``torchrun --standalone --nproc_per_node=2 -m
lightx2v_train.model_zoo.minimax_h3_image.tests.verify_pdmd_gan
--output-dir /tmp/h3-image-gan-test`` with a new output directory.
Uses a tiny real H3 transformer and small synthetic latent targets, not 33B.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml
from lightx2v_train.data import build_data
from lightx2v_train.model_zoo import build_model
from lightx2v_train.model_zoo.tests.verify_pdd_training import prepare_fixture
from lightx2v_train.runtime.distributed import barrier, cleanup_distributed, get_rank, init_distributed
from safetensors.torch import load_file

from lightx2v_train.trainers import build_trainer


def prepare_gan_fixture(root):
    prepare_fixture(root)
    rows = []
    for line in (root / "metadata.jsonl").read_text().splitlines():
        row = json.loads(line)
        identity = {"real_sample_key": f"tiny-real-{row['id']}", "real_cache_fingerprint": "tiny-image-gan-v1"}
        path = root / f"real-{row['id']}.pt"
        torch.save({"latents": torch.randn(24, 1, 4, 4), "vae_normalized": True, **identity}, path)
        rows.append({**row, **identity, "image_latent_path": str(path), "vae_normalized": True})
    (root / "real_metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def configuration(root):
    template = Path(__file__).resolve().parents[4] / "configs/train/dmd/minimax_h3_image_pdmd2_gan_2048_r128_a128_fsdp8.yaml"
    config = yaml.safe_load(template.read_text())
    config["model"]["pretrained_model_name_or_path"] = str(root / "model")
    config["distributed"]["fsdp2"]["size"] = 2
    config["data"]["train"].update(data_path=str(root / "metadata.jsonl"), num_workers=0)
    training = config["training"]
    training.update(output_dir=str(root / "train"), max_train_iters=5, save_every_iters=3, save_total_limit=3)
    training["student"]["lora"].update(rank=4, alpha=4, rank_pattern={"adaln_proj.linear": 2, "norm_out.linear": 2}, alpha_pattern={"adaln_proj.linear": 2, "norm_out.linear": 2})
    training["dmd"]["generation_shapes"] = [{"value": [64, 64]}]
    training["dmd"]["gan"].update(real_cache=str(root / "real_metadata.jsonl"), feature_block=0, head_hidden_dim=8, head_token_bins=2)
    return config


def run_training(config):
    model = build_model(config)
    model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
    trainer = build_trainer(config)
    trainer.set_model(model)
    trainer.set_data(build_data(config, train_or_val="train"))
    trainer.train()
    assert trainer.dmd2_gan_head is not None
    assert all(parameter.requires_grad for parameter in trainer.fake_trainable_params)
    assert all(parameter.dtype == torch.float32 for parameter in trainer.fake_trainable_params)
    assert all(parameter.grad is None for parameter in trainer.teacher_model.transformer.parameters())
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    barrier()


def verify_checkpoint(directory, iteration):
    assert (directory / "official_pdmd.complete").is_file()
    state = torch.load(directory / "trainer_state.pt", map_location="cpu", weights_only=False)
    assert state["iteration"] == iteration and state["dmd2_gan_enabled"]
    assert state["dmd2_gan_config"]["feature_dim"] == 16
    assert state["dmd2_gan_config"]["head"] == "fp32_target_image_ordered_pool_mlp"
    for rank in range(2):
        cursor = torch.load(directory / f"official_pdmd.rank{rank:05d}.pt", map_location="cpu", weights_only=False)
        assert cursor["iteration"] == iteration
        assert cursor["consumed_microbatches"] == iteration - iteration // 6
        assert len(cursor["cached"]) == (1 if iteration % 6 == 5 else 0)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    root = parser.parse_args().output_dir.resolve()
    config = configuration(root)
    init_distributed(config)
    try:
        torch.manual_seed(42)
        if get_rank() == 0:
            prepare_gan_fixture(root)
        barrier()
        run_training(config)
        # Resume exactly between the fifth fake update and the student update:
        # the preceding fake's cached trajectory must survive the checkpoint.
        before = verify_checkpoint(root / "train/checkpoint-000000005", 5)
        student_before = load_file(root / "train/checkpoint-000000005/pytorch_lora_weights.safetensors")
        barrier()
        config["training"]["max_train_iters"] = 8
        run_training(config)
        after = verify_checkpoint(root / "train/checkpoint-000000008", 8)
        if get_rank() == 0:
            assert {float(value["step"]) for value in before["dmd2_gan_optimizer"]["state"].values()} == {5.0}
            assert {float(value["step"]) for value in after["dmd2_gan_optimizer"]["state"].values()} == {7.0}
            student_after = load_file(root / "train/checkpoint-000000008/pytorch_lora_weights.safetensors")
            assert any(not torch.equal(value, student_after[name]) for name, value in student_before.items())
            assert any(not torch.equal(value, after["dmd2_gan_head"][name]) for name, value in before["dmd2_gan_head"].items())
            print("PASS: Image official PDMD+GAN FSDP2, fake5 checkpoint + cached-trajectory student resume, eight role updates total", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
