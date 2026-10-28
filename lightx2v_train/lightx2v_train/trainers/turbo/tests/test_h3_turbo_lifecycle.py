"""Real H3 Turbo constructor/binding/setup with only checkpoint I/O stubbed."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from lightx2v_train.model_zoo.native.minimax_h3.modeling import _transformer_class
from lightx2v_train.trainers.turbo.trainer import TurboTrainer
from lightx2v_train.utils.registry import build_model
from torch.utils.data import DataLoader


def tiny_transformer():
    return _transformer_class()(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=12,
        freq_dim=4,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=1,
    ).eval()


def load_tiny_model(config, **kwargs):
    del kwargs
    model = build_model(config)
    with patch("lightx2v_train.model_zoo.minimax_h3.minimax_h3_t2av.load_minimax_h3_transformer", return_value=tiny_transformer()):
        model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
    model.ensure_capabilities()
    return model


class H3TurboLifecycleTests(unittest.TestCase):
    def test_constructor_binding_data_setup_three_roles_ema_and_dcm_step(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "seed": 42,
                "model": {
                    "name": "minimax_h3_t2av",
                    "pretrained_model_name_or_path": "/unused-h3-checkpoint",
                    "running_dtype": "fp32",
                    "transformer_param_dtype": "fp32",
                    "patch_size": [1, 1, 1],
                    "video_latent_channels": 2,
                    "audio_latent_channels": 2,
                    "vae_spatial_scale_factor": 16,
                    "use_autocast": False,
                    "capabilities": {"distribution_matching": {"geometry_from_metadata": True, "allowed_resolutions": [[32, 32]], "video_flow_shift": 6.0, "audio_flow_shift": 3.0}},
                },
                "distributed": {"fsdp2": {"enabled": False, "stream_load_pretrained": False}, "sequence_parallel": {"enabled": False, "size": 1}},
                "data": {"train": {"name": "minimax_h3_cache_dataset", "data_path": "/unused-manifest"}},
                "scheduler": {"num_train_timesteps": 1000, "time_shift_settings": {"do_time_shift": False}},
                "training": {
                    "method": "turbo",
                    "max_train_iters": 1,
                    "gradient_accumulation_iters": 1,
                    "gradient_checkpointing": True,
                    "max_grad_norm": 1.0,
                    "lr_scheduler": "constant",
                    "lr_warmup_iters": 0,
                    "save_every_iters": 1,
                    "save_total_limit": 1,
                    "output_dir": str(Path(directory) / "output"),
                    "turbo": {"dcm_steps": 1, "scm_steps": 0, "joint_steps": 0, "num_inference_steps": 2, "dcm_total_steps": 4, "video_shift": 6.0, "audio_shift": 3.0},
                    "student": {
                        "train_type": "lora",
                        "ema": {"enabled": True, "decay": 0.99},
                        "lora": {
                            "rank": 4,
                            "alpha": 4,
                            "param_dtype": "fp32",
                            "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "adaln_proj.linear", "norm_out.linear"],
                            "rank_pattern": {"adaln_proj.linear": 2, "norm_out.linear": 2},
                            "alpha_pattern": {"adaln_proj.linear": 2, "norm_out.linear": 2},
                        },
                        "optimizer": {"learning_rate": 2e-5},
                    },
                    "fake": {"train_type": "full", "optimizer": {"learning_rate": 4e-6}},
                    "teacher": {"guidance_scale": 1.0},
                },
                "inference": {"method": "none"},
                "resume": {"auto_resume": False},
            }
            torch.manual_seed(42)
            sample = {
                "inputs": {"video_latents": torch.randn(2, 37, 2, 2), "audio_latents": torch.randn(2, 2, 207)},
                "conditioning": {
                    "prompt": "tiny lifecycle",
                    "positive": {"prompt_embeds": torch.randn(4, 12), "text_token_tags": torch.ones(4, dtype=torch.long), "task": "t2av", "keyframe_anchors": torch.empty(0, dtype=torch.long)},
                },
                "meta": {"target_num_frames": 124, "target_height": 32, "target_width": 32},
            }
            trainer = TurboTrainer(config)
            model = load_tiny_model(config)
            trainer.set_model(model)
            self.assertIsNone(trainer.generation_shapes)
            loader = DataLoader([sample], batch_size=1)
            trainer.set_data(loader)
            with patch("lightx2v_train.trainers.dmd.runtime.build_loaded_model", side_effect=load_tiny_model) as load_role:
                trainer.setup()
            self.assertEqual(load_role.call_count, 2)
            self.assertEqual(len({id(role.model) for role in (trainer.student, trainer.fake, trainer.teacher)}), 3)
            self.assertIs(trainer.student.model, model)
            self.assertIs(trainer.dataloader_train, loader)
            self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)
            self.assertIsInstance(trainer.fake_optimizer, torch.optim.AdamW)
            self.assertEqual(trainer.student_ema.num_updates, 0)
            self.assertTrue(trainer.student_ema.shadow)
            self.assertTrue(all(not parameter.requires_grad for parameter in trainer.teacher.denoiser().parameters()))
            metrics = trainer._optimizer_step(iter(loader), "student", "dcm", 0)
            self.assertIn("dcm_video", metrics)
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
            self.assertEqual(trainer.consumed_microbatches, 1)
            self.assertEqual(trainer.student_ema.num_updates, 1)
            self.assertTrue(all(parameter.grad is None for parameter in trainer.teacher.denoiser().parameters()))
            trainer.finish_monitor()


if __name__ == "__main__":
    unittest.main()
