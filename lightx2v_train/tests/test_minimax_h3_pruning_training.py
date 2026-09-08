import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_vae_distillation_capability import MiniMaxH3VAEDistillationCapability
from lightx2v_train.model_zoo.minimax_h3.minimax_h3_pruned_vae import MiniMaxH3PrunedVAEModel
from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.vae.distillation import VAEDistillationTrainer
from lightx2v_train.trainers.vae.pruning import VAEPruningTrainer
from lightx2v_train.utils.registry import build_sample_processor
from tests.test_minimax_h3_pruned_vae import tiny_config, tiny_teacher


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = "lightx2v_train.model_zoo.minimax_h3.minimax_h3_pruned_vae"


class FeatureLossTest(unittest.TestCase):
    def test_mse_and_masked_mse_match_formulas_and_detach_teacher(self):
        owner = torch.nn.Module()
        owner.device = torch.device("cpu")
        student = torch.randn(2, 8, 3, 2, 2, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        for kind in ("mse", "masked_mse"):
            capability = MiniMaxH3VAEDistillationCapability(owner, {"feature_loss_type": kind})
            actual = capability._feature_loss(student, teacher)
            left, right = student, teacher.detach()
            if kind == "masked_mse":
                left_var, left_mean = torch.var_mean(left, dim=(1, 2, 3, 4), keepdim=True)
                right_var, right_mean = torch.var_mean(right, dim=(1, 2, 3, 4), keepdim=True)
                left = left * ((left - left_mean).abs() < 2 * left_var.sqrt())
                right = right * ((right - right_mean).abs() < 2 * right_var.sqrt())
            torch.testing.assert_close(actual, (left - right).square().mean())
            actual.backward()
            self.assertIsNone(teacher.grad)
            self.assertIsNotNone(student.grad)

    def test_arbitrary_ordered_anchors_but_no_duplicate_or_empty_anchors(self):
        owner = torch.nn.Module()
        for indices in ([2, 5, 8], [0]):
            capability = MiniMaxH3VAEDistillationCapability(owner, {"feature_weight": 1, "teacher_feature_indices": indices})
            self.assertEqual(capability.teacher_feature_indices, tuple(indices))
        for indices in ([], [2, 2], [-1], [5, 2]):
            with self.assertRaises(ValueError):
                MiniMaxH3VAEDistillationCapability(owner, {"feature_weight": 1, "teacher_feature_indices": indices})


class PruningTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.teacher_config = tiny_config()
        (self.root / "config.json").write_text(json.dumps(self.teacher_config))
        self.teacher = tiny_teacher(self.teacher_config)
        self.teacher.config = SimpleNamespace(**self.teacher_config)
        self.loader_patch = patch(f"{WRAPPER}.load_minimax_h3_video_vae", side_effect=lambda *args, **kwargs: copy.deepcopy(self.teacher))
        self.loader_patch.start()
        self.addCleanup(self.loader_patch.stop)

    def config(self, stage):
        config_stage = {"search": "search", "recover": "recover_aux_3k"}[stage]
        config = load_config(str(ROOT / "configs/train/vae" / f"minimax_h3_vit_prune_{config_stage}_4gpu_ddp.yaml"))
        config["model"].update(pretrained_model_name_or_path=str(self.root), running_dtype="fp32")
        decoder = config["model"]["pruned_decoder"]
        decoder.update(tile_sample_min_height=64, tile_sample_min_width=64,
                       tile_sample_min_overlap_height=16, tile_sample_min_overlap_width=16)
        config["training"].update(max_train_iters=2, gradient_accumulation_iters=2,
                                 lr_warmup_iters=0, save_every_iters=1,
                                 output_dir=str(self.root / stage))
        config["resume"]["auto_resume"] = False
        config["inference"] = {"infer_every_iters": 0}
        loss = config["training"]["vae_distillation"]
        loss["stages"] = [{"name": "test", "start_iter": 0, "routes": {"temporal_pair": 1},
                           "weights": {"reconstruction": 1, "feature": 0 if stage == "search" else 0.01}}]
        loss["gan"] = {"enabled": False}
        if stage == "search":
            decoder["search"].update(lora_rank=2, gate_scale=1.0)
            config["training"]["pruning"]["export_dir"] = str(self.root / "search/export")
        else:
            decoder["selection_path"] = str(self.root / "search/export/kept_layers.json")
            loss["teacher_feature_indices"] = [2, 5]
        return config

    def model(self, config):
        model = MiniMaxH3PrunedVAEModel(config)
        model.load_components(load_transformer=True, load_vae=True, load_condition_encoder=False)
        model.log_model_structure = lambda: None
        return model

    def data(self):
        sample = {"inputs": {"video": torch.rand(3, 124, 64, 64), "latents": torch.randn(24, 37, 4, 4)},
                  "meta": {"source_num_frames": 124}}
        return DataLoader([sample], batch_size=1)

    def test_search_export_recovery_and_resume_end_to_end(self):
        config = self.config("search")
        model = self.model(config)
        self.assertIsNone(model.teacher_vae)
        trainer = VAEPruningTrainer(config)
        trainer.set_model(model)
        trainer.set_data(self.data())
        trainer.train()
        self.assertEqual(trainer.current_train_iteration, 2)
        self.assertEqual(trainer.gate_ema_iteration, 2)
        self.assertTrue(torch.isfinite(trainer.gate_ema).all())
        self.assertEqual(len(trainer.optimizer.param_groups), 2)
        rates = [group["initial_lr"] for group in trainer.optimizer.param_groups]
        self.assertAlmostEqual(rates[1] / rates[0], 10)
        selection = json.loads((self.root / "search/export/kept_layers.json").read_text())
        self.assertEqual(selection["teacher_feature_indices"], [2, 5])
        self.assertEqual(len(selection["kept_layers"]), 2)

        saved_ema = trainer.gate_ema.clone()
        config["resume"]["auto_resume"] = True
        resumed = VAEPruningTrainer(config)
        resumed.set_model(self.model(config))
        resumed.set_data(self.data())
        resumed.train()
        torch.testing.assert_close(resumed.gate_ema, saved_ema)
        self.assertEqual(resumed.current_train_iteration, 2)

        recovery_config = self.config("recover")
        recovery_model = self.model(recovery_config)
        self.assertTrue(all(not parameter.requires_grad for parameter in recovery_model.teacher_vae.parameters()))
        self.assertFalse(any("lora" in name or "gate_logits" in name for name in recovery_model.transformer.state_dict()))
        recovery = VAEDistillationTrainer(recovery_config)
        recovery.set_model(recovery_model)
        recovery.set_data(self.data())
        original = recovery_model.transformer.decoder.proj_out.weight.detach().clone()
        recovery.train()
        self.assertFalse(torch.equal(original, recovery_model.transformer.decoder.proj_out.weight))
        self.assertTrue(all(parameter.grad is None for parameter in recovery_model.teacher_vae.parameters()))
        saved = self.root / "recover/checkpoint-000000002/minimax_h3_pruned_vae.safetensors"
        self.assertTrue(saved.is_file())
        recovery_model._load_initial_weights(saved)

    def test_preview_restore_does_not_unfreeze_search_base(self):
        model = self.model(self.config("search"))
        model.set_denoiser_eval()
        model.set_full_trainable()
        trainable = [name for name, param in model.transformer.named_parameters() if param.requires_grad]
        self.assertTrue(all("lora_" in name or "gate_logits" in name for name in trainable))
        self.assertFalse(model.transformer.post_quant_conv.weight.requires_grad)

    def test_new_model_has_a_registered_sample_processor(self):
        processor = build_sample_processor({"model": {"name": "minimax_h3_pruned_vae"}})
        self.assertIsNotNone(processor)


if __name__ == "__main__":
    unittest.main()
