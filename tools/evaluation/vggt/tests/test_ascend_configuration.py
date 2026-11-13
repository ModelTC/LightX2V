"""Exercise Ascend configuration without requiring an NPU or running a model."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluate as evaluation


SNAPSHOT = '''
| npu-smi 26.0.rc1                         Version: 26.0.rc1 |
| 3     910B2C              | OK            | 92.4                 45                      0 / 0 |
| 0                         | 0000:3F:00.0  | 0                    0 / 0                3414 / 65536 |
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
| No running processes found in NPU 3 |
'''


class AscendTests(unittest.TestCase):
    def setUp(self):
        self.previous = evaluation.ACTIVE_PLATFORM
        evaluation.select_platform('ascend_npu')
        script = evaluation.ROOT / 'scripts/platform/ascend_npu/monitor.py'
        spec = importlib.util.spec_from_file_location('ascend_monitor', script)
        self.monitor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.monitor)

    def tearDown(self):
        evaluation.select_platform(self.previous)

    def test_npu_mapping_is_isolated_and_cpu_worker_has_no_devices(self):
        args = SimpleNamespace(card=3, image='test', reuse_runs=None, repeats=2, limit=1, data_root=Path('/input'))
        _, command = evaluation.container_command(args, Path('/output'), 'lightx2v')
        _, cpu = evaluation.container_command(args, Path('/output'), 'postprocess')
        for value in ['/dev/davinci3', '/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc', 'ASCEND_RT_VISIBLE_DEVICES=3', '/usr/local/bin/eval-python']:
            self.assertIn(value, command)
        self.assertTrue(any('src=/usr/local/Ascend/driver' in value for value in command))
        self.assertNotIn('--device', cpu)
        self.assertNotIn('ASCEND_RT_VISIBLE_DEVICES=3', cpu)
        self.assertFalse(any('src=/usr/local/Ascend/driver' in value for value in cpu))
        self.assertIn('npu', evaluation.device_name())
        self.assertEqual(evaluation.load_platform_config()['registration_modules'], ['torch_npu'])

    def test_platform_dependency_file_is_in_the_configuration_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            evaluation.copy_active_configs(destination)
            expected = evaluation.ROOT / evaluation.load_platform_config()['requirements_file']
            self.assertEqual((destination / 'ascend_npu/runtime-requirements.txt').read_bytes(), expected.read_bytes())

    def test_monitor_keeps_driver_reserved_memory_and_processes(self):
        row = self.monitor.parse_snapshot(SNAPSHOT, 3)
        self.assertEqual(row['memory_mib'], 3414)
        self.assertEqual(row['capacity_mib'], 65536)
        self.assertEqual(row['pids'], [])
        busy = SNAPSHOT + '| 3       0 | 12345 | python | 2048 | 42 |\n'
        self.assertEqual(self.monitor.parse_snapshot(busy, 3)['pids'], [12345])
        with self.assertRaises(ValueError):
            self.monitor.parse_snapshot(SNAPSHOT.split('| NPU     Chip')[0], 3)
        with self.assertRaises(ValueError):
            self.monitor.parse_snapshot(SNAPSHOT, 0)

    def test_idle_memory_allowance_does_not_allow_active_processes(self):
        args = SimpleNamespace(card=3)
        for row in [dict(memory_mib=3414, util_percent=1, pids=[]), dict(memory_mib=3414, util_percent=0, pids=[42]), dict(memory_mib=5000, util_percent=0, pids=[])]:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as temporary:
                with patch.object(evaluation, 'monitor_snapshot', return_value=('snapshot', {3: row})):
                    with self.assertRaisesRegex(RuntimeError, 'not idle'):
                        evaluation.execute_stage(args, Path(temporary), 'lightx2v')



if __name__ == '__main__':
    unittest.main()
