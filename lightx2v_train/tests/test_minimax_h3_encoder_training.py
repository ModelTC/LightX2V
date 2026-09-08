import copy
import json
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import AutoencoderKLMiniMaxH3
from loguru import logger
from torch.utils.data import DataLoader

from lightx2v_train.model_zoo.minimax_h3.minimax_h3_pruned_encoder import MiniMaxH3PrunedEncoderModel
from lightx2v_train.model_zoo.native.minimax_h3.video_vae import imagenet_preprocess, normalize_video_latents
from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.vae.distillation import VAEDistillationTrainer
from lightx2v_train.trainers.vae.encoder_pruning import VAEEncoderPruningTrainer
from tests.test_minimax_h3_pruned_encoder import tiny_encoder_config


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = "lightx2v_train.model_zoo.minimax_h3.minimax_h3_pruned_encoder"


def _distributed_training_worker(rank, root, teacher_config, search_config, recover_config):
    torch.set_num_threads(1)
    logger.remove()
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(21)
        teacher = AutoencoderKLMiniMaxH3(**teacher_config).eval()
        with patch(f"{WRAPPER}.load_minimax_h3_video_vae", side_effect=lambda *args, **kwargs: copy.deepcopy(teacher)):
            for config, trainer_class in ((search_config, VAEEncoderPruningTrainer), (recover_config, VAEDistillationTrainer)):
                model = MiniMaxH3PrunedEncoderModel(config)
                model.load_components(load_transformer=True, load_vae=True, load_condition_encoder=False)
                model.log_model_structure = lambda: None
                trainer = trainer_class(config)
                trainer.set_model(model)
                generator = torch.Generator().manual_seed(123 + rank)
                samples = [{"inputs": {"video": torch.rand(3, 124, 80, 80, generator=generator)},
                            "meta": {"source_num_frames": 124}} for _ in range(2)]
                trainer.set_data(DataLoader(samples, batch_size=1))
                original_after_backward = trainer._after_backward

                def check_synchronized_gradient():
                    original_after_backward()
                    gradients = torch.cat([parameter.grad.flatten() for parameter in trainer.trainable_params if parameter.grad is not None])
                    assert torch.isfinite(gradients).all() and gradients.abs().sum() > 0
                    peers = [torch.empty_like(gradients) for _ in range(2)]
                    dist.all_gather(peers, gradients)
                    torch.testing.assert_close(peers[0], peers[1], rtol=1e-6, atol=1e-7)
                    assert all(parameter.grad is None for parameter in model.teacher_vae.parameters())

                trainer._after_backward = check_synchronized_gradient
                trainer.train()
                assert trainer.current_train_iteration == 1
                del trainer, model
    finally:
        dist.destroy_process_group()


class EncoderTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(21)
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.teacher_config = tiny_encoder_config()
        self.teacher_config.update(
            out_channels=3, decoder_num_layers=2, decoder_num_attention_heads=2,
            decoder_attention_head_dim=8, decoder_num_register_tokens=2, decoder_ffn_mult=2,
            decoder_rope_theta=100.0, decoder_rope_dim_ratio=0.75, decoder_norm_eps=1e-5,
        )
        (self.root / "config.json").write_text(json.dumps(self.teacher_config))
        self.teacher = AutoencoderKLMiniMaxH3(**self.teacher_config).eval()
        loader = patch(f"{WRAPPER}.load_minimax_h3_video_vae", side_effect=lambda *args, **kwargs: copy.deepcopy(self.teacher))
        loader.start()
        self.addCleanup(loader.stop)

    def config(self, stage):
        config = load_config(str(ROOT / "configs/train/vae" / f"minimax_h3_encoder_prune_{stage}_keep3_2gpu_ddp.yaml"))
        config["model"].update(pretrained_model_name_or_path=str(self.root), running_dtype="fp32")
        encoder = config["model"]["pruned_encoder"]
        encoder.update(tile_sample_min_height=64, tile_sample_min_width=64,
                       tile_sample_min_overlap_height=16, tile_sample_min_overlap_width=16)
        config["training"].update(max_train_iters=1, gradient_accumulation_iters=1, lr_warmup_iters=0,
                                 save_every_iters=1, output_dir=str(self.root / stage))
        config["resume"]["auto_resume"] = False
        config["inference"] = {"infer_every_iters": 0}
        losses = config["training"]["vae_distillation"]
        losses["perceptual_weight"] = 0
        losses["gan"] = {"enabled": False}
        losses["stages"] = [{"name": "test", "start_iter": 0, "routes": {"single": 1},
                             "weights": {"reconstruction": 1, "perceptual": 0, "feature": 0.01 if stage == "recover" else 0,
                                         "auxiliary": 0, "adversarial": 0}}]
        if stage == "search":
            encoder["search"].update(lora_rank=2, gate_scale=1)
            config["training"]["pruning"]["export_dir"] = str(self.root / "search/export")
        else:
            encoder["selection_path"] = str(self.root / "search/export/kept_layers.json")
        return config

    def model(self, config):
        model = MiniMaxH3PrunedEncoderModel(config)
        model.load_components(load_transformer=True, load_vae=True, load_condition_encoder=False)
        model.log_model_structure = lambda: None
        return model

    def data(self):
        return DataLoader([{"inputs": {"video": torch.rand(3, 124, 64, 64)},
                            "meta": {"source_num_frames": 124}}], batch_size=1)

    def test_frozen_decoder_transmits_gradient_to_student_encoder(self):
        config = self.config("search")
        model = self.model(config)
        model.set_full_trainable()
        model.transformer.prepare_search_step(1)
        moments = model.student_encode_clip(torch.randn(1, 3, 17, 32, 32), torch.float32)
        latent = model.transformer.normalize_latents(moments[:, :model.latent_channels])
        reconstruction = model.decode_student_latent_window(latent, torch.float32)
        reconstruction.square().mean().backward()
        self.assertGreater(model.transformer.encoder.gate_logits.grad.abs().sum(), 0)
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in model.teacher_vae.parameters()))
        self.assertTrue(model.teacher_vae.decoder.gradient_checkpointing)

    def test_preview_encoding_matches_original_protocol_when_no_branches_removed(self):
        config = self.config("search")
        encoder = config["model"]["pruned_encoder"]
        del encoder["search"]
        encoder["kept_residual_indices"] = list(range(12))
        model = self.model(config)
        model.set_denoiser_eval()
        model.teacher_vae.enable_tiling(64, 64, 16, 16)
        video = torch.rand(1, 3, 124, 64, 80)
        with torch.no_grad():
            moments = model.teacher_vae._encode(imagenet_preprocess(video))
            expected = normalize_video_latents(model.teacher_vae, moments[:, :model.latent_channels])
            actual = model.encode_video(video)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tuple(actual.shape), (1, 2, 37, 4, 5))

    def _check_search_export_recovery_and_resume(self, grouped=False):
        search_config = self.config("search")
        if grouped:
            search_options = search_config["model"]["pruned_encoder"]["search"]
            search_options.pop("keep_residuals")
            search_options.update(grouping="stage", keep_per_group=1)
        search_model = self.model(search_config)
        search = VAEEncoderPruningTrainer(search_config)
        search.set_model(search_model)
        search.set_data(self.data())
        search.train()
        selection = json.loads((self.root / "search/export/kept_layers.json").read_text())
        self.assertEqual(len(selection["kept_residual_indices"]), 6 if grouped else 3)
        self.assertEqual(selection["original_num_residuals"], 12)
        self.assertEqual(selection["initialization"], "original_teacher_weights_without_search_lora")
        self.assertEqual(selection["grouping"], "stage" if grouped else "global")
        if grouped:
            self.assertEqual(selection["residual_groups"], [[i, i + 1] for i in range(0, 12, 2)])
            self.assertEqual(selection["group_budgets"], [1] * 6)
            for group in selection["residual_groups"]:
                self.assertEqual(len(set(group).intersection(selection["kept_residual_indices"])), 1)
            search_config["resume"]["auto_resume"] = True
            resumed_search = VAEEncoderPruningTrainer(search_config)
            resumed_search.set_model(self.model(search_config))
            resumed_search.set_data(self.data())
            resumed_search.train()
            torch.testing.assert_close(resumed_search.gate_ema, search.gate_ema, rtol=0, atol=0)

        recover_config = self.config("recover")
        model = self.model(recover_config)
        self.assertFalse(any("lora_" in name or "gate_logits" in name for name in model.transformer.state_dict()))
        for name, value in model.transformer.state_dict().items():
            if name.startswith(("encoder.", "quant_conv.")):
                torch.testing.assert_close(value, self.teacher.state_dict()[name], rtol=0, atol=0)
        original = model.transformer.quant_conv.weight.detach().clone()
        recover = VAEDistillationTrainer(recover_config)
        recover.set_model(model)
        recover.set_data(self.data())
        recover.train()
        self.assertFalse(torch.equal(original, model.transformer.quant_conv.weight))
        self.assertTrue(all(parameter.grad is None for parameter in model.teacher_vae.parameters()))
        recover_config["resume"]["auto_resume"] = True
        resumed = VAEDistillationTrainer(recover_config)
        resumed.set_model(self.model(recover_config))
        resumed.set_data(self.data())
        resumed.train()
        self.assertEqual(resumed.current_train_iteration, 1)

    def test_search_export_original_weights_then_recovery_and_resume(self):
        self._check_search_export_recovery_and_resume()

    def test_grouped_search_export_original_weights_recovery_and_resume(self):
        self._check_search_export_recovery_and_resume(grouped=True)

    def _check_two_rank_training(self, grouped=False):
        search_config, recover_config = self.config("search"), self.config("recover")
        if grouped:
            search = search_config["model"]["pruned_encoder"]["search"]
            search.pop("keep_residuals")
            search.update(grouping="stage", keep_per_group=1)
        for config in (search_config, recover_config):
            config["training"]["gradient_accumulation_iters"] = 2
            stage = config["training"]["vae_distillation"]["stages"][0]
            stage["routes"] = {"quad": 0.5, "temporal_pair": 0.5}
        mp.spawn(_distributed_training_worker, args=(str(self.root), self.teacher_config, search_config, recover_config), nprocs=2, join=True)
        self.assertTrue((self.root / "search/export/kept_layers.json").is_file())
        self.assertTrue((self.root / "recover/checkpoint-000000001/minimax_h3_pruned_encoder.safetensors").is_file())
        if grouped:
            selection = json.loads((self.root / "search/export/kept_layers.json").read_text())
            self.assertEqual(selection["group_budgets"], [1] * 6)

    def test_two_rank_multitile_checkpointed_search_and_recovery(self):
        self._check_two_rank_training()

    def test_two_rank_grouped_multitile_checkpointed_search_and_recovery(self):
        self._check_two_rank_training(grouped=True)


if __name__ == "__main__":
    unittest.main()
