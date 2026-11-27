"""Fail-closed NPU telemetry and immutable source contracts (no accelerator needed)."""
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

RUNTIME = Path(__file__).resolve().parents[3] / 'runtime'


def load(name):
    spec = importlib.util.spec_from_file_location(name, RUNTIME / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.lineage = Mock(side_effect=lambda pid: [[pid, 77]])
        with patch.dict(sys.modules, gpu_watch=types.SimpleNamespace(check=Mock(), lineage=self.lineage)):
            self.capture = load('ascend_capture')

    def inventory(self, process='| No running processes found in NPU 2 |'):
        return '\n'.join(['| 2 910B2C | OK | 91 41 0 / 0 |',
                          '| 0 | 0000:52:00.0 | 0 0 / 0 3412 / 65536 |',
                          '| NPU Chip | Process id | Process name | Process memory(MB) |', process])

    def test_physical_identity_and_firmware_reservation(self):
        result = self.capture.parse_sample(self.inventory(), 2)
        self.assertEqual(result['uuid'], 'Ascend-0000:52:00.0')
        self.assertEqual(result['memory_mib'], 3412)
        self.assertEqual(result['processes'], [])

    def test_host_pid_ancestry_is_retained(self):
        result = self.capture.parse_sample(self.inventory('| 2 0 | 1234 | python | 1024 |'), 2)
        self.assertEqual(result['processes'][0]['ancestors'], [[1234, 77]])

    def test_missing_ancestry_unknown_inventory_and_wrong_chip_fail(self):
        self.lineage.return_value = []
        self.lineage.side_effect = None
        for process in ('| 2 0 | 1234 | python | 1024 |', '| truncated |',
                        '| 2 1 | 1234 | python | 1024 |',
                        '| No running processes found in NPU 1 |'):
            with self.subTest(process=process), self.assertRaises(ValueError):
                self.capture.parse_sample(self.inventory(process), 2)

    def test_duplicate_or_missing_device_inventory_fails(self):
        for inventory in (self.inventory().replace('OK', 'ERROR'), self.inventory() + '\n' + self.inventory(),
                          self.inventory().replace('Process id', 'unavailable')):
            with self.assertRaises(ValueError):
                self.capture.parse_sample(inventory, 2)

    def test_virtual_partition_and_device_sharing_fail(self):
        for virtual, shared in ((1, 'False'), (0, 'True')):
            answers = [types.SimpleNamespace(stdout=f'Total number of vnpu: {virtual}'),
                       types.SimpleNamespace(stdout=f'Device-share Status : {shared}'),
                       types.SimpleNamespace(stdout='VNPU mode: docker')]
            with patch.object(self.capture.subprocess, 'run', side_effect=answers), self.assertRaises(ValueError):
                self.capture.deployment(2)


class SourceTests(unittest.TestCase):
    def test_pinned_checkout_rejects_modified_and_extra_source(self):
        module = load('ascend_sources')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '--quiet', directory], check=True)
            source = root / 'model.py'
            source.write_text('value = 1\n')
            subprocess.run(['git', '-C', directory, 'add', 'model.py'], check=True)
            subprocess.run(['git', '-C', directory, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                            'commit', '--quiet', '-m', 'fixture'], check=True)
            revision = module.git(root, 'rev-parse', 'HEAD').decode().strip()
            lock = dict(base_revision=revision, runtime_files=[dict(path='model.py', sha256=hashlib.sha256(source.read_bytes()).hexdigest())])
            module.verify(root, lock)
            source.write_text('value = 2\n')
            with self.assertRaises(ValueError):
                module.verify(root, lock)
            source.write_text('value = 1\n')
            (root / 'injected.py').write_text('value = 3\n')
            with self.assertRaises(ValueError):
                module.verify(root, lock)


if __name__ == '__main__':
    unittest.main()
