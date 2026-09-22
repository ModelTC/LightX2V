"""H3 audio VAE checkpoint compatibility and CPU offload regression tests."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch.nn.utils import weight_norm as legacy_weight_norm

from lightx2v.models.audio_encoders.hf.minimax_h3 import audio_vae
from lightx2v.models.video_encoders.hf.minimax_h3 import weights


class MiniMaxH3AudioVAETest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.vae_dir = Path(self.directory.name) / "audio_vae"
        self.vae_dir.mkdir()
        self.config = {
            "encoder_dim": 4,
            "encoder_rates": [2],
            "latent_dim": 8,
            "latent_channels": 2,
            "num_attention_heads": 2,
            "decoder_dim": 8,
            "decoder_rates": [2],
            "decoder_kernel_sizes": [4],
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1]],
        }
        (self.vae_dir / "config.json").write_text(json.dumps(self.config))
        with warnings.catch_warnings(), patch.object(audio_vae, "weight_norm", legacy_weight_norm):
            warnings.simplefilter("ignore", FutureWarning)
            self.reference = audio_vae.MiniMaxH3AudioVAE(self.config, device="cpu").eval().requires_grad_(False)
        self.checkpoint = self.reference.state_dict()

    def test_legacy_shards_preserve_encode_decode_and_offload(self):
        keys = list(self.checkpoint)
        save_file({key: self.checkpoint[key] for key in keys[::2]}, self.vae_dir / "model-00001.safetensors")
        save_file({key: self.checkpoint[key] for key in keys[1::2]}, self.vae_dir / "model-00002.safetensors")
        generator = torch.Generator().manual_seed(1497)
        waveform = torch.randn((2, 1, 64), generator=generator)
        latents = torch.randn((2, 2, 32), generator=generator)
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            with self.subTest(device=device):
                self.reference.to(device)
                expected_encoded = self.reference.encode(waveform)
                expected_decoded = self.reference.decode(latents, return_cpu=True)
                model = audio_vae.MiniMaxH3AudioVAE.from_pretrained(self.vae_dir, device=device, cpu_offload=True)
                self.assertEqual(set(model.load_report.loaded_keys), set(model.state_dict()))
                self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))
                # PyTorch's standard loader independently checks our key mapping.
                reference_parameters = audio_vae.MiniMaxH3AudioVAE(self.config, device="cpu")
                reference_parameters.load_state_dict(self.checkpoint)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, reference_parameters.state_dict()[key], rtol=0, atol=0)
                for _ in range(2):
                    torch.testing.assert_close(model.encode(waveform), expected_encoded, rtol=0, atol=0)
                    torch.testing.assert_close(model.decode(latents, return_cpu=True), expected_decoded, rtol=0, atol=0)
                    self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))
                    for module in model.modules():
                        if hasattr(module, "parametrizations"):
                            self.assertNotIn("weight", vars(module))

    def test_mapped_checkpoint_validation_precedes_assignment(self):
        key = "encoder.block.0.weight_g"
        cases = (
            ("shape", {**self.checkpoint, key: torch.zeros(1)}, ValueError, "Shape mismatch"),
            ("dtype", {**self.checkpoint, key: self.checkpoint[key].half()}, TypeError, "Dtype mismatch"),
            ("missing", {name: tensor for name, tensor in self.checkpoint.items() if name != key}, RuntimeError, "missing="),
            (
                "duplicate",
                {**self.checkpoint, "encoder.block.0.parametrizations.weight.original0": self.checkpoint[key].clone()},
                RuntimeError,
                "duplicate:",
            ),
        )
        for label, tensors, error, message in cases:
            with self.subTest(case=label):
                save_file(tensors, self.vae_dir / "model.safetensors")
                with patch.object(weights, "_assign_tensor") as assign, self.assertRaisesRegex(error, message):
                    audio_vae.MiniMaxH3AudioVAE.from_pretrained(self.vae_dir, device="cpu")
                assign.assert_not_called()

    def test_loader_without_mapping_preserves_parameters_and_buffers(self):
        reference = torch.nn.BatchNorm1d(3)
        checkpoint = {**reference.state_dict(), "unused.weight": torch.ones(2)}
        save_file(checkpoint, self.vae_dir / "model.safetensors")
        with torch.device("meta"):
            model = torch.nn.BatchNorm1d(3)
        report = weights.load_safetensors_subset(model, self.vae_dir)
        self.assertEqual(report.ignored_keys, 1)
        self.assertEqual(set(report.loaded_keys), set(reference.state_dict()))
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, reference.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
