"""CPU/Gloo smoke test of the real two-stage trainer, export and resume path."""

import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from lightx2v_train.model_zoo.native.wan.modules.vae import WanVAE_
from lightx2v_train.model_zoo.native.wan.pruned_vae import WAN_VAE_CONFIG
from lightx2v_train.runtime.ddp import unwrap_ddp_module
from lightx2v_train.utils.registry import build_model, build_trainer


TRAIN_ROOT = Path(__file__).resolve().parents[1]
TINY_CONFIG = {**WAN_VAE_CONFIG, "dim": 2}


def _config(directory, component, stage, *, grouped=False):
    keep = 3 if component == "encoder" else 5
    source = TRAIN_ROOT / f"configs/train/vae/wan21_{component}_prune_{stage}_keep{keep}_ddp.yaml"
    config = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    config["model"].update(
        pretrained_model_name_or_path=str(directory / "teacher.pth"),
        vae_architecture=TINY_CONFIG,
        running_dtype="fp32",
    )
    pruning = config["model"][f"pruned_{component}"]
    if stage == "search":
        pruning["search"].update(lora_rank=2, lora_alpha=4)
        if grouped:
            pruning["search"].pop("keep_residuals")
            pruning["search"].update(grouping="stage", keep_per_group=1)
        config["training"]["pruning"]["export_dir"] = str(directory / component / "export")
    else:
        pruning["selection_path"] = str(directory / component / "export/kept_layers.json")
    training = config["training"]
    training.update(
        output_dir=str(directory / component / stage), max_train_iters=2,
        gradient_accumulation_iters=2, lr_warmup_iters=0, save_every_iters=1,
    )
    weights = {"reconstruction": 1.0}
    if component == "encoder":
        weights["posterior"] = 1.0
    if stage == "recover":
        weights.update(feature=0.01, auxiliary=0.1)
    training["vae_distillation"]["stages"] = [{
        "name": "smoke", "start_iter": 0, "crop_num_frames": 5, "crop_size": 16, "weights": weights,
    }]
    training["vae_distillation"]["gan"] = {"enabled": False}
    config["inference"].update(
        infer_every_iters=1, output_dir=str(directory / component / stage / "previews"),
        save_comparison=False, save_png_frame_count=0,
    )
    return config


def _train(config, loader):
    model = build_model(config)
    model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
    trainer = build_trainer(config)
    trainer.set_model(model)
    trainer.set_data(loader, loader)
    # Exercise reconstruction/metric/restore logic without depending on a video codec.
    with patch("lightx2v_train.infer.vae.save_mp4"):
        trainer.train()
    if trainer.current_train_iteration != 2:
        raise AssertionError("Smoke training did not finish its two optimizer updates.")
    if any(parameter.grad is not None for teacher in (model.teacher_encoder, model.teacher_decoder)
           for parameter in teacher.parameters()):
        raise AssertionError("Frozen teacher received a parameter gradient.")
    return trainer


def _worker(rank, world_size, directory, grouped=False):
    directory = Path(directory)
    torch.set_num_threads(1)
    logger.disable("lightx2v_train")
    with patch("torch.cuda.is_available", return_value=False):
        dist.init_process_group(
            "gloo", init_method=f"file://{directory / 'rendezvous'}", rank=rank, world_size=world_size,
            timeout=timedelta(seconds=90),
        )
        try:
            torch.manual_seed(40 + rank)
            sample = {"inputs": {"video": torch.rand(3, 5, 16, 16)},
                      "meta": {"source_num_frames": 5, "video_path": "smoke.mp4"}}
            loader = DataLoader([sample, sample], batch_size=1)
            for component in ("encoder", "decoder"):
                search_config = _config(directory, component, "search", grouped=grouped)
                search = _train(search_config, loader)
                selection_path = directory / component / "export/kept_layers.json"
                if not selection_path.exists():
                    raise AssertionError("Search failed to publish an export.")
                student = unwrap_ddp_module(search.model.transformer)
                if abs(student.temperature - 0.1) > 1e-6:
                    raise AssertionError("Search temperature did not reach the underlying DDP module.")
                if grouped:
                    selection = json.loads(selection_path.read_text())
                    if selection["search_grouping"] != "stage" or selection["kept_per_group"] != [1] * 5:
                        raise AssertionError("Grouped search did not preserve its stage budgets in the export.")
                    if len(student.candidate_masks) != (32 if component == "encoder" else 162):
                        raise AssertionError("Grouped search constructed the wrong candidate space.")
                    for group in student.residual_groups:
                        torch.testing.assert_close(
                            student.candidate_masks[:, group].sum(-1), torch.ones(len(student.candidate_masks)),
                        )
                resumed = _train(search_config, loader)
                torch.testing.assert_close(resumed.gate_ema, search.gate_ema)
                recovery_config = _config(directory, component, "recover")
                recovered = _train(recovery_config, loader)
                resumed = _train(recovery_config, loader)
                for expected, actual in zip(recovered.model.transformer.parameters(), resumed.model.transformer.parameters(), strict=True):
                    torch.testing.assert_close(expected, actual)
        finally:
            dist.destroy_process_group()


class WanTrainingSmokeTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("WAN_RUN_DDP_SMOKE") == "1", "Opt-in CPU/Gloo subprocess test")
    def test_two_rank_search_recovery_preview_checkpoint_resume(self):
        self.run_two_rank_smoke(grouped=False)

    @unittest.skipUnless(os.environ.get("WAN_RUN_DDP_SMOKE") == "1", "Opt-in CPU/Gloo subprocess test")
    def test_two_rank_stage_search_recovery_preview_checkpoint_resume(self):
        self.run_two_rank_smoke(grouped=True)

    def run_two_rank_smoke(self, *, grouped):
        with tempfile.TemporaryDirectory(prefix="wan-vae-ddp-smoke-") as directory:
            torch.manual_seed(4)
            torch.save(WanVAE_(**TINY_CONFIG).state_dict(), Path(directory) / "teacher.pth")
            mp.spawn(_worker, args=(2, directory, grouped), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
