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
        self.init_calls = 0
        self.memory_used = 10 * 1024 * 1024

    def nvmlInit(self):
        self.init_calls += 1

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

    def test_initialized_inventory_does_not_reinitialize_or_shutdown(self):
        nvml = FakeNVML()
        M.sample_nvml([UUID], nvml, initialized=True)
        self.assertEqual(nvml.init_calls, 0)
        self.assertFalse(nvml.shutdown)

    def test_server_initializes_once_for_multiple_complete_queries(self):
        import io

        nvml = FakeNVML()
        output = io.StringIO()
        with patch.object(M, "load_binding", return_value=nvml), patch.object(M.sys, "stdin", io.StringIO("sample\nsample\n")), patch.object(M.sys, "stdout", output):
            M.serve([UUID])
        self.assertEqual(nvml.init_calls, 1)
        self.assertTrue(nvml.shutdown)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["gpus"][0]["uuid"] == UUID for record in records))

    def test_reuses_policy_ancestry_even_if_unavailable(self):
        raw = [{"uuid": UUID, "processes": [{"pid": 123, "type": "C"}]}]
        client = SimpleNamespace(query=lambda devices, timeout: raw)
        result = M.sample_with_policy(client, [UUID], 2.5, SimpleNamespace(lineage=lambda pid: []))
        self.assertEqual(result[0]["processes"][0]["ancestors"], [])


SERVER = r"""import json,os,sys,time
from pathlib import Path
stage=Path(os.environ['LIGHTX2V_NVML_QUERY_STAGE'])
Path(__file__).with_suffix('.starts').open('a').write(str(os.getpid())+'\n')
mode=MODE
if mode=='startup_hang':
 stage.write_text('nvmlInit');time.sleep(60)
stage.write_text('ready')
sequence=0
for request in sys.stdin:
 sequence+=1
 if mode=='query_hang':
  stage.write_text('nvmlDeviceGetMemoryInfo_v2');time.sleep(60)
 if mode=='error':result={'error':'unknown process inventory'}
 elif mode=='incomplete':result={'gpus':[]}
 elif mode=='eof':raise SystemExit(0)
 else:result={'gpus':[{'uuid':sys.argv[2],'sequence':sequence,'processes':[]}]}
 print(json.dumps(result),flush=True)
"""


class PersistentQueryTests(unittest.TestCase):
    def fixture(self, directory, mode):
        script = Path(directory) / "query.py"
        script.write_text(SERVER.replace("MODE", repr(mode)))
        return script, M.NVMLQueryProcess(script)

    def wait_stage(self, client, expected):
        import time

        client._start([UUID])
        deadline = time.monotonic() + 5
        while client.stage.read_text() != expected:
            if time.monotonic() >= deadline:
                client.close()
                self.fail("Fake query process did not reach its controlled wait point")
            time.sleep(0.01)

    def test_multiple_queries_use_one_cpu_process(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            script, client = self.fixture(directory, "ok")
            try:
                first = client.query([UUID], 2.5)
                pid = client.process.pid
                second = client.query([UUID], 2.5)
                self.assertEqual(client.process.pid, pid)
                self.assertEqual([first[0]["sequence"], second[0]["sequence"]], [1, 2])
                self.assertEqual(len(script.with_suffix(".starts").read_text().splitlines()), 1)
            finally:
                client.close()
            self.assertIsNone(client.process)

    def test_initialization_deadline_is_policy_compatible_and_reaps_child(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            script, client = self.fixture(directory, "startup_hang")
            self.wait_stage(client, "nvmlInit")
            with self.assertRaisesRegex(subprocess.SubprocessError, "last stage: nvmlInit") as caught:
                client.query([UUID], 0.5)
            self.assertIsInstance(caught.exception, subprocess.TimeoutExpired)
            self.assertIsNone(client.process)
            self.assertIsNone(client.directory)

    def test_each_later_query_still_has_a_hard_deadline(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            script, client = self.fixture(directory, "query_hang")
            self.wait_stage(client, "ready")
            with self.assertRaisesRegex(subprocess.TimeoutExpired, "nvmlDeviceGetMemoryInfo_v2"):
                client.query([UUID], 0.5)
            self.assertIsNone(client.process)

    def test_unknown_inventory_and_eof_fail_closed(self):
        import tempfile

        for mode in ("error", "incomplete", "eof"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir="/tmp") as directory:
                script, client = self.fixture(directory, mode)
                with self.assertRaises(ValueError):
                    client.query([UUID], 2.5)
                self.assertIsNone(client.process)

    def test_cannot_silently_switch_capture_devices(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            script, client = self.fixture(directory, "ok")
            client.query([UUID], 2.5)
            with self.assertRaises(ValueError):
                client.query([UUID + "other"], 2.5)
            self.assertIsNone(client.process)


if __name__ == "__main__":
    unittest.main()
