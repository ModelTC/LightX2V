"""MLU telemetry must preserve device and process attribution, rejecting unknowns."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlu_capture

DEVICE = dict(uuid='MLU-71031045-2400-2180-0000-000000000000', pci_bus_id='0000:4c:00.0', physical_card=0)
EMPTY = '''| 0     /   MLU590-M9D   v1.0.3 |         0000:4C:00.0 | 0%                  0 |
| 1         44C     86 W/ 470 W |     0 MiB/ 81920 MiB | FULL          Default |
| Processes:
| Card MI PID Command Line MLU Memory Usage |
| No running processes found |
'''


class MLUCaptureTests(unittest.TestCase):
    def test_physical_identity_and_idle_inventory(self):
        result = mlu_capture.parse_sample(EMPTY, 0, DEVICE)
        self.assertEqual(result['uuid'], DEVICE['uuid'])
        self.assertEqual(result['memory_mib'], 0)
        self.assertEqual(result['processes'], [])

    def test_process_ancestry_and_unknown_pid(self):
        occupied = EMPTY.replace('| No running processes found |', '| 0 / 123 /usr/bin/python 100 MiB |')
        with patch.object(mlu_capture, 'lineage', return_value=[[123, 1000], [100, 900]]):
            result = mlu_capture.parse_sample(occupied, 0, DEVICE)
        self.assertEqual(result['processes'][0]['ancestors'], [[123, 1000], [100, 900]])
        with patch.object(mlu_capture, 'lineage', return_value=[]), self.assertRaises(ValueError):
            mlu_capture.parse_sample(occupied, 0, DEVICE)

    def test_identity_partitions_and_incomplete_inventory(self):
        for bad in (EMPTY.replace('4C:00.0', '51:00.0'), EMPTY.replace('FULL', 'MIM'),
                    EMPTY.replace('0     /', '0     1'), EMPTY.replace('| Processes:', ''),
                    EMPTY.replace('| No running processes found |', ''),
                    EMPTY.replace('| No running processes found |', '| 0 / 123 unavailable |')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                mlu_capture.parse_sample(bad, 0, DEVICE)

    def test_foreign_pid_missing_sample_and_reused_pid_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory)/'samples.jsonl'; task = Path(directory)/'task.json'
            task.write_text(json.dumps(dict(pid=123,start_ticks=1000,start=2,end=3,boot_id='boot',devices=[DEVICE['uuid']])))
            def row(start, end, ancestors=None):
                device = dict(DEVICE, memory_mib=0,utilization_pct=0,processes=[])
                if ancestors is not None: device['processes']=[dict(pid=123,type='C',ancestors=ancestors)]
                return dict(start=start,end=end,boot_id='boot',devices=[DEVICE['uuid']],gpus=[device])
            good=[row(1,1.1),row(2.1,2.2,[[123,1000]]),row(3.1,3.2)]
            def check(rows):
                sample.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                return mlu_capture.verify(sample,task,3,256)
            self.assertTrue(check(good)['resources_clear'])
            self.assertFalse(check([good[0],row(2.1,2.2,[[123,999]]),good[2]])['resources_clear'])
            self.assertFalse(check([good[0],row(2.1,2.2,[[456,1000]]),good[2]])['resources_clear'])
            with self.assertRaises(ValueError): check(good[:-1])
            bad=[*good];bad[1]=dict(bad[1],error='query timed out')
            with self.assertRaises(ValueError): check(bad)


if __name__=='__main__': unittest.main()
