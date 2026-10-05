"""Exercise edit encoding with a frozen VAE that starts on the host.

Run with ``PYTHONPATH=lightx2v_train python -m unittest discover -s
lightx2v_train/tests -p test_edit_vae_offload.py``. Set
``EDIT_VAE_TEST_DEVICE=npu:0`` to check actual CPU/NPU transfers. CPU runs
use cpu:0/cpu placement labels to exercise the transfer lifecycle while
performing real tensor operations; they cannot reproduce a physical mismatch.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

if os.environ.get("EDIT_VAE_TEST_DEVICE", "cpu").startswith("npu"):
    import torch_npu  # noqa: F401 -- registers the NPU device with PyTorch

from lightx2v_train.model_zoo.flux2.flux2_dev_edit import Flux2DevEditModel
from lightx2v_train.model_zoo.longcat_image.longcat_image_edit import LongCatImageEditModel


class _TinyVAE(nn.Module):
    """A small real encoder with Diffusers' device and latent_dist interfaces."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.arange(12).reshape(4, 3, 1, 1) / 12)
        # Flux2 patchifies four channels into sixteen before normalizing.
        self.bn = nn.BatchNorm2d(16)
        self.bn.running_mean.copy_(torch.arange(16) / 16)
        self.bn.running_var.fill_(2)
        self.config = SimpleNamespace(shift_factor=0.25, scaling_factor=2.0, batch_norm_eps=0.25)
        self.requires_grad_(False)
        self.eval()
        self._placement = torch.device("cpu")
        self.moves = []
        self.encode_devices = []
        self.fail_on_encode = None

    @property
    def device(self):
        # Preserve the CPU index label since tensors discard it on CPU.
        return self._placement

    def to(self, device, *args, **kwargs):
        result = super().to(device, *args, **kwargs)
        self._placement = torch.device(device)
        self.moves.append(self._placement)
        return result

    def encode(self, pixels):
        weight_device = self.encoder.weight.device
        if pixels.device != weight_device:
            raise AssertionError(f"VAE input {pixels.device} does not match encoder weights {weight_device}")
        self.encode_devices.append(pixels.device)
        if len(self.encode_devices) == self.fail_on_encode:
            raise RuntimeError("injected VAE encode failure")
        latent = self.encoder(pixels)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent, sample=lambda: latent + 0.125))


class EditVAEOffloadTests(unittest.TestCase):
    model_classes = (LongCatImageEditModel, Flux2DevEditModel)

    @classmethod
    def setUpClass(cls):
        requested_device = torch.device(os.environ.get("EDIT_VAE_TEST_DEVICE", "cpu"))
        cls.model_device = torch.device("cpu:0") if requested_device.type == "cpu" else requested_device
        if cls.model_device.type == "npu":
            torch.npu.set_device(cls.model_device)
        cls.tensor_device = torch.empty(0, device=cls.model_device).device

    def setUp(self):
        self.backend = Mock()
        if self.model_device.type == "npu":
            self.backend.synchronize.side_effect = torch.npu.synchronize
            self.backend.empty_cache.side_effect = torch.npu.empty_cache
        backend_patch = patch("lightx2v_train.model_zoo.base.get_backend", return_value=self.backend)
        backend_patch.start()
        self.addCleanup(backend_patch.stop)

    def make_model(self, model_class, *, resident=False):
        # Bypass model loading, preserving the actual model encoding/packing code.
        model = model_class.__new__(model_class)
        model.device = self.model_device
        model.running_dtype = torch.float32
        model.vae = _TinyVAE()
        model.vae_config = model.vae.config
        if resident:
            model.vae.to(self.model_device)
            model.vae.moves.clear()
        return model

    @staticmethod
    def source_images():
        return [torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) / 48, torch.arange(72, dtype=torch.float32).reshape(3, 4, 6) / 72]

    def encode_source(self, model, images=None):
        images = self.source_images() if images is None else images
        if isinstance(model, LongCatImageEditModel):
            return model._encode_source_latents(images[0])
        return model._encode_reference_images(images)

    def assert_outputs_match(self, actual, expected):
        if torch.is_tensor(actual):
            self.assertEqual(actual.device, self.tensor_device)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        elif isinstance(actual, tuple):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_outputs_match(value, reference)
        else:
            self.assertEqual(actual, expected)

    def assert_host_restored(self, model):
        self.assertEqual(model.vae.device, torch.device("cpu"))
        self.assertEqual(model.vae.encoder.weight.device, torch.device("cpu"))
        self.assertEqual(model.vae.bn.running_mean.device, torch.device("cpu"))
        self.assertTrue(model.vae.encode_devices)
        self.assertTrue(all(device == self.tensor_device for device in model.vae.encode_devices))

    def test_offloaded_source_matches_resident_for_two_rounds(self):
        for model_class in self.model_classes:
            with self.subTest(model=model_class.__name__):
                model = self.make_model(model_class)
                expected = self.encode_source(self.make_model(model_class, resident=True))
                for _ in range(2):
                    actual = self.encode_source(model)
                    self.assert_outputs_match(actual, expected)
                    self.assert_host_restored(model)
                self.assertEqual(model.vae.moves, [self.model_device, torch.device("cpu")] * 2)
        self.backend.empty_cache.assert_called()

    def test_flux_multiple_source_images_use_one_transfer_round_trip(self):
        model = self.make_model(Flux2DevEditModel)
        tokens, image_ids = self.encode_source(model)
        self.assertEqual(tuple(tokens.shape), (1, 10, 16))
        self.assertEqual(tuple(image_ids.shape), (1, 10, 4))
        self.assertEqual(tokens.device, self.tensor_device)
        self.assertEqual(image_ids.device, self.tensor_device)
        self.assertEqual(len(model.vae.encode_devices), 2)
        self.assertEqual(model.vae.moves, [self.model_device, torch.device("cpu")])
        self.assert_host_restored(model)

    def test_source_encode_exception_restores_host_vae(self):
        for model_class in self.model_classes:
            with self.subTest(model=model_class.__name__):
                model = self.make_model(model_class)
                # Flux2 fails after one image was successfully encoded.
                model.vae.fail_on_encode = 1 if model_class is LongCatImageEditModel else 2
                with self.assertRaisesRegex(RuntimeError, "injected VAE encode failure"):
                    self.encode_source(model)
                self.assert_host_restored(model)
                self.assertEqual(model.vae.moves, [self.model_device, torch.device("cpu")])
        self.backend.empty_cache.assert_called()

    def test_resident_source_does_not_move_vae(self):
        for model_class in self.model_classes:
            with self.subTest(model=model_class.__name__):
                model = self.make_model(model_class, resident=True)
                output = self.encode_source(model)
                self.assertEqual(output[0].device, self.tensor_device)
                self.assertEqual(model.vae.device, self.model_device)
                self.assertEqual(model.vae.moves, [])
        self.backend.empty_cache.assert_not_called()

    def test_target_mode_and_sample_restore_host_and_preserve_values(self):
        sample = {"inputs": {"target_pixel_values": self.source_images()[0]}}
        for model_class in self.model_classes:
            for mode in ("mode", "sample"):
                with self.subTest(model=model_class.__name__, mode=mode):
                    model = self.make_model(model_class)
                    reference = self.make_model(model_class, resident=True)
                    expected = reference._encode_target_latent(sample, mode=mode)
                    actual = model._encode_target_latent(sample, mode=mode)
                    self.assert_outputs_match(actual, expected)
                    self.assert_host_restored(model)
                    self.assertEqual(model.vae.moves, [self.model_device, torch.device("cpu")])

    def test_target_encode_exception_restores_host_vae(self):
        sample = {"inputs": {"target_pixel_values": self.source_images()[0]}}
        for model_class in self.model_classes:
            for mode in ("mode", "sample"):
                with self.subTest(model=model_class.__name__, mode=mode):
                    model = self.make_model(model_class)
                    model.vae.fail_on_encode = 1
                    with self.assertRaisesRegex(RuntimeError, "injected VAE encode failure"):
                        model._encode_target_latent(sample, mode=mode)
                    self.assert_host_restored(model)
                    self.assertEqual(model.vae.moves, [self.model_device, torch.device("cpu")])

    def test_resident_target_encodes_without_device_transfers(self):
        sample = {"inputs": {"target_pixel_values": self.source_images()[0]}}
        for model_class in self.model_classes:
            with self.subTest(model=model_class.__name__):
                model = self.make_model(model_class, resident=True)
                for mode in ("mode", "sample"):
                    latent = model._encode_target_latent(sample, mode=mode)
                    self.assertEqual(latent.device, self.tensor_device)
                self.assertEqual(model.vae.moves, [])
        self.backend.empty_cache.assert_not_called()

    def test_source_then_target_can_repeat_after_host_restore(self):
        sample = {"inputs": {"target_pixel_values": self.source_images()[0]}}
        for model_class in self.model_classes:
            with self.subTest(model=model_class.__name__):
                model = self.make_model(model_class)
                reference = self.make_model(model_class, resident=True)
                expected_source = self.encode_source(reference)
                expected_target = reference.encode_to_latent(sample)
                for _ in range(2):
                    self.assert_outputs_match(self.encode_source(model), expected_source)
                    self.assert_host_restored(model)
                    self.assert_outputs_match(model.encode_to_latent(sample), expected_target)
                    self.assert_host_restored(model)
                self.assertEqual(model.vae.moves, [self.model_device, torch.device("cpu")] * 4)


if __name__ == "__main__":
    unittest.main()
