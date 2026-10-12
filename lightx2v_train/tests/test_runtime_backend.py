import importlib
import os
import unittest
from unittest.mock import Mock, patch

import torch

from lightx2v_train.runtime.backend import factory
from lightx2v_train.runtime.backend.musa import MusaBackend
from lightx2v_train.runtime.backend.nvidia import NvidiaBackend


class MusaBackendTests(unittest.TestCase):
    def setUp(self):
        self.device_module = Mock()
        self.device_module.is_available.return_value = True
        self.device_module.current_device.return_value = 2
        self.musa_patch = patch.object(torch, "musa", self.device_module, create=True)
        self.musa_patch.start()
        self.addCleanup(self.musa_patch.stop)

    def test_missing_extension_reports_dependency_and_is_unavailable(self):
        with patch("lightx2v_train.runtime.backend.musa.importlib.import_module", side_effect=ModuleNotFoundError("torch_musa")):
            self.assertFalse(MusaBackend.is_available())
            with self.assertRaisesRegex(RuntimeError, "requires torch_musa"):
                MusaBackend().initialize()

    def test_availability_requires_an_available_device_module(self):
        with patch.object(MusaBackend, "_load_extension"):
            self.assertTrue(MusaBackend.is_available())
            self.device_module.is_available.return_value = False
            self.assertFalse(MusaBackend.is_available())
            with self.assertRaisesRegex(RuntimeError, r"torch\.musa is not available"):
                MusaBackend().initialize()
            with patch.object(torch, "musa", None):
                self.assertFalse(MusaBackend.is_available())

    def test_initialization_loads_extension_before_reading_device_module(self):
        def register_device_module(name):
            self.assertEqual(name, "torch_musa")
            torch.musa = self.device_module

        with (
            patch.object(torch, "musa", None),
            patch("lightx2v_train.runtime.backend.musa.importlib.import_module", side_effect=register_device_module) as extension_import,
        ):
            MusaBackend().initialize(local_rank=2)
        extension_import.assert_called_once_with("torch_musa")
        self.device_module.set_device.assert_called_once_with(2)
        self.device_module.is_available.assert_called_once_with()

    def test_musa_selection_local_rank_and_initialization_cache(self):
        # A stock CPU/CUDA PyTorch build cannot construct a MUSA device until
        # the extension registers it; mock only that constructor for this test.
        device = torch.device("cuda", 2)
        with (
            patch.dict(os.environ, {"PLATFORM": "musa", "LOCAL_RANK": "2"}, clear=True),
            patch.object(factory, "_BACKEND", None),
            patch.object(MusaBackend, "_load_extension") as load_extension,
            patch("lightx2v_train.runtime.backend.base.torch.device", return_value=device) as make_device,
        ):
            backend = factory.init_backend()
            self.assertIsInstance(backend, MusaBackend)
            self.assertIs(factory.init_backend(), backend)
            self.assertIs(factory.get_backend(), backend)
            self.assertEqual(factory.get_device(), device)
            make_device.assert_called_with("musa", 2)
            load_extension.assert_called_once_with()
            self.device_module.set_device.assert_called_once_with(2)

    def test_device_kwargs_use_current_device_for_ddp_and_mccl_barriers(self):
        device = torch.device("cuda", 2)
        with (
            patch.object(MusaBackend, "_load_extension"),
            patch("lightx2v_train.runtime.backend.base.torch.device", return_value=device),
        ):
            backend = MusaBackend().initialize()
            self.assertEqual(backend.ddp_kwargs(), {"device_ids": [2], "output_device": 2})
            for group_backend in ("mccl", "MCCL", "musa:mccl"):
                with self.subTest(group_backend=group_backend):
                    self.assertEqual(backend.barrier_kwargs(group_backend), {"device_ids": [2]})
            for group_backend in ("gloo", "nccl"):
                with self.subTest(group_backend=group_backend):
                    self.assertEqual(backend.barrier_kwargs(group_backend), {})

    def test_attention_defaults_and_sdpa_aliases_choose_native(self):
        backend = MusaBackend()
        for requested in (None, "auto", "AUTO", "native", "sdpa", "torch_sdpa"):
            with self.subTest(requested=requested):
                self.assertEqual(backend.resolve_attention_backend(requested), "native")
        self.assertEqual(backend.resolve_attention_backend("_native_flash"), "_native_flash")

    def test_other_platforms_initialize_without_vendor_extensions(self):
        import_module = importlib.import_module

        def without_vendor_extension(name, *args, **kwargs):
            if name in {"torch_musa", "torch_npu", "torchada"}:
                self.fail(f"Initializing another platform imported {name}")
            return import_module(name, *args, **kwargs)

        for platform, cuda_available, expected in (
            ("cpu", True, "cpu"),
            ("nvidia", True, "nvidia"),
            ("auto", True, "nvidia"),
            ("auto", False, "cpu"),
            (None, False, "cpu"),
        ):
            environment = {} if platform is None else {"PLATFORM": platform}
            with (
                self.subTest(platform=platform, cuda_available=cuda_available),
                patch.dict(os.environ, environment, clear=True),
                patch.object(factory, "_BACKEND", None),
                patch.object(torch.cuda, "is_available", return_value=cuda_available),
                patch.object(torch.cuda, "current_device", return_value=0),
                patch.object(NvidiaBackend, "configure_math"),
                patch("importlib.import_module", side_effect=without_vendor_extension),
            ):
                self.assertEqual(factory.init_backend().name, expected)


if __name__ == "__main__":
    unittest.main()
