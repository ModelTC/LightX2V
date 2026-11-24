"""CPU acquisition tests; the original gpu_watch module owns resource policy."""

import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[1] / "nvml_capture.py"
SPEC = importlib.util.spec_from_file_location("nvml_capture_tests", FILE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
UUID = "GPU-00000000-0000-0000-0000-000000000001"


class FakeNVML:
    nvmlMemory_v2 = 0x02000028

    def __init__(self):
        self.mig = [0, 0]
        self.mps = []
        self.compute = [SimpleNamespace(pid=123)]
        self.graphics = [SimpleNamespace(pid=123), SimpleNamespace(pid=456)]
        self.names = {123: "python", 456: "renderer"}
        self.shutdown = False
        self.memory_used = 10 * 1024 * 1024

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        self.shutdown = True

    def nvmlDeviceGetHandleByUUID(self, value):
        return value

    def nvmlDeviceGetUUID(self, handle):
        return handle.encode()

    def nvmlDeviceGetMigMode(self, handle):
        return self.mig

    def nvmlDeviceGetMPSComputeRunningProcesses(self, handle):
        return self.mps

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return self.compute

    def nvmlDeviceGetGraphicsRunningProcesses(self, handle):
        return self.graphics

    def nvmlSystemGetProcessName(self, pid):
        return self.names[pid]

    def nvmlDeviceGetMemoryInfo(self, handle, version=None):
        if version != self.nvmlMemory_v2:
            raise AssertionError("Use v2 to keep reserved memory separate")
        return SimpleNamespace(used=self.memory_used, reserved=633 * 1024 * 1024)

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=25)


class NVMLAcquisitionTests(unittest.TestCase):
    def test_compute_graphics_union_and_required_values(self):
        nvml = FakeNVML()
        result = M.sample_nvml([UUID], nvml)
        self.assertEqual(result, [{"uuid": UUID, "memory_mib": 10, "utilization_pct": 25, "processes": [{"pid": 123, "type": "C+G"}, {"pid": 456, "type": "G"}]}])
        self.assertTrue(nvml.shutdown)

    def test_used_memory_excludes_reserved_and_rounds_up_like_cli(self):
        nvml = FakeNVML()
        nvml.memory_used = 256 * 1024 * 1024 + 1
        result = M.sample_nvml([UUID], nvml)
        self.assertEqual(result[0]["memory_mib"], 257)

    def test_mig_and_unknown_inventory_fail_closed(self):
        for modes in ([1, 0], [0, 1], []):
            with self.subTest(modes=modes):
                nvml = FakeNVML()
                nvml.mig = modes
                with self.assertRaises(ValueError):
                    M.sample_nvml([UUID], nvml)
                self.assertTrue(nvml.shutdown)

    def test_mps_clients_and_unknown_inventory_fail_closed(self):
        for clients in ([SimpleNamespace(pid=123)], None):
            with self.subTest(clients=clients):
                nvml = FakeNVML()
                nvml.mps = clients
                with self.assertRaises(ValueError):
                    M.sample_nvml([UUID], nvml)
                self.assertTrue(nvml.shutdown)

    def test_mps_server_without_clients_is_rejected(self):
        nvml = FakeNVML()
        nvml.names[123] = "/usr/bin/nvidia-cuda-mps-server"
        with self.assertRaises(ValueError):
            M.sample_nvml([UUID], nvml)
        self.assertTrue(nvml.shutdown)

    def test_missing_graphics_inventory_is_not_treated_as_empty(self):
        nvml = FakeNVML()
        with patch.object(nvml, "nvmlDeviceGetGraphicsRunningProcesses", side_effect=RuntimeError("unavailable")), self.assertRaises(RuntimeError):
            M.sample_nvml([UUID], nvml)
        self.assertTrue(nvml.shutdown)

    def test_invalid_pid_and_unknown_name_are_not_filtered_out(self):
        for pid in (0, 999):
            with self.subTest(pid=pid):
                nvml = FakeNVML()
                nvml.compute = [SimpleNamespace(pid=pid)]
                with self.assertRaises((ValueError, KeyError)):
                    M.sample_nvml([UUID], nvml)

    def test_bounded_query_reuses_policy_ancestry_even_if_unavailable(self):
        raw = [{"uuid": UUID, "memory_mib": 1, "utilization_pct": 0, "processes": [{"pid": 123, "type": "C"}]}]
        policy = SimpleNamespace(lineage=lambda pid: [])
        with patch.object(M.subprocess, "check_output", return_value=json.dumps(raw).encode()) as query:
            result = M.bounded_sample([UUID], 2.5, policy)
        self.assertEqual(query.call_args.kwargs["timeout"], 2.5)
        self.assertEqual(result[0]["processes"][0]["ancestors"], [])

    def test_query_deadline_failure_is_not_swallowed(self):
        with patch.object(M.subprocess, "check_output", side_effect=subprocess.TimeoutExpired("probe", 2.5)), self.assertRaises(subprocess.TimeoutExpired):
            M.bounded_sample([UUID], 2.5, SimpleNamespace())

    def test_incomplete_device_inventory_is_rejected(self):
        with patch.object(M.subprocess, "check_output", return_value=b"[]"), self.assertRaises(ValueError):
            M.bounded_sample([UUID], 2.5, SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
