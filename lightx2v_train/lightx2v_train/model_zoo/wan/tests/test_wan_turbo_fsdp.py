"""Real two-rank FP32 Wan Turbo lifecycle and exact JVP/FSDP optimizer smoke."""

import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Sampler

from lightx2v_train.model_zoo.native.wan.modules.model import WanModel
from lightx2v_train.runtime.distributed import cleanup_distributed, init_distributed
from lightx2v_train.trainers.turbo.trainer import TurboTrainer
from lightx2v_train.utils.registry import build_model


class _ConfiguredSampler(Sampler):
    def configure_from_microbatch_offset(self, *args, **kwargs):
        pass

    def __iter__(self):
        return iter([0])

    def __len__(self):
        return 1


def _tiny_loaded_model(config, **kwargs):
    del kwargs
    transformer = WanModel(patch_size=(1, 1, 1), text_len=4, in_dim=4, dim=32, ffn_dim=64, freq_dim=8, text_dim=12, out_dim=4, num_heads=2, num_layers=1, attention_backend="sdpa").float()
    torch.nn.init.normal_(transformer.head.head.weight, std=0.05)
    model = build_model(config)
    with patch.object(model, "_load_transformer", return_value=transformer.to(model.device)):
        model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
    model.ensure_capabilities()
    return model


def _worker(rank, init_path, output_dir):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method="file://" + init_path, timeout=timedelta(seconds=120))
    config_path = Path(__file__).resolve().parents[5] / "lightx2v_train/configs/train/turbo/wan2_1_t2v_1_3b_turbo8_fp32_gpu0_smoke.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["model"].update(pretrained_model_name_or_path="/unused-wan", max_sequence_length=4, patch_size=[1, 1, 1], load_vae=False, load_text_encoder=False)
    config["distributed"]["fsdp2"] = {
        "enabled": True,
        "size": 2,
        "reshard_after_forward": {"root_reshard": False, "block_reshard": True},
        "mixed_precision": {"param_dtype": "fp32", "reduce_dtype": "fp32", "cast_forward_inputs": False},
    }
    config["training"].update(max_train_iters=3, output_dir=output_dir)
    config["training"]["turbo"].update(dcm_steps=1, scm_steps=1, joint_steps=1, num_inference_steps=2, fake_update_ratio=1, generation_shapes=[{"value": [5, 16, 16]}])
    config["training"]["student"]["lora"].update(rank=4, alpha=4)
    config["inference"] = {"method": "none"}
    config["resume"] = {"auto_resume": False}
    try:
        init_distributed(config)
        torch.manual_seed(219)
        model = _tiny_loaded_model(config)
        trainer = TurboTrainer(config)
        trainer.set_model(model)
        sample = {
            "inputs": {"latents": torch.randn(4, 2, 2, 2)},
            "conditioning": {"prompt": "tiny fp32 FSDP", "positive": {"prompt_embed": torch.randn(4, 12)}, "negative": {"prompt_embed": torch.zeros(4, 12)}},
            "meta": {},
        }
        loader = DataLoader([sample], sampler=_ConfiguredSampler(), batch_size=1)
        trainer.set_data(loader)
        with patch("lightx2v_train.trainers.dmd.runtime.build_loaded_model", side_effect=_tiny_loaded_model):
            trainer.setup()
        assert all(isinstance(role.denoiser(), FSDPModule) for role in (trainer.student, trainer.fake, trainer.teacher))
        assert all(isinstance(block, FSDPModule) for block in model.transformer.blocks)
        state = torch.randn(1, 4, 2, 2, 2, device=model.device)
        sigma = torch.tensor(0.4, device=model.device)
        primal, tangent = trainer.student.predict_velocity_jvp(
            state, sigma, {"prompt_embed": torch.randn(1, 4, 12, device=model.device)}, torch.randn_like(state), torch.tensor(0.1, device=model.device)
        )
        assert primal.dtype == tangent.dtype == torch.float32
        assert torch.isfinite(primal).all() and torch.isfinite(tangent).all() and tangent.abs().sum() > 0
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            scm = trainer._optimizer_step(iter(loader), "student", "scm", 1)
            joint = trainer._optimizer_step(iter(loader), "student", "joint", 2)
            fake = trainer._optimizer_step(iter(loader), "fake", "joint", 2)
        assert trainer.student_ema.num_updates == 2 and trainer.fake_updates == 1
        assert trainer.optimizer.state and trainer.fake_optimizer.state
        for parameter in model.transformer.parameters():
            if parameter.grad is not None:
                gradient = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
                assert gradient.dtype == torch.float32 and torch.isfinite(gradient).all()
        print(f"rank={rank} Wan FSDP2 FP32 lifecycle+JVP+sCM+joint+fake+EMA passed: losses={scm['loss']},{joint['loss']},{fake['loss']}", flush=True)
        trainer.finish_monitor()
    finally:
        cleanup_distributed()


class WanTurboFSDPTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Two CUDA GPUs required")
    def test_two_rank_fp32_production_lifecycle_exact_jvp_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(str(Path(directory) / "init"), str(Path(directory) / "output")), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
