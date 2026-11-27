"""Validate MetaX telemetry parsing and reuse the established resource policy."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import metax_campaign
import metax_capture

EMPTY = """GPU#1 MXC500 0000:09:00.0
    vram used : 879032 KB
    GPU : 0 %
| Process:
| no process found |
"""


class MetaXCaptureTests(unittest.TestCase):
    def test_idle_card_identity_and_memory(self):
        value = metax_capture.parse_sample(EMPTY, 1)
        self.assertEqual(value["uuid"], "MetaX-0000:09:00.0")
        self.assertEqual(value["processes"], [])
        self.assertGreater(value["memory_mib"], 858)
        self.assertLess(value["memory_mib"], 900)

    def test_process_ancestry_is_required(self):
        occupied = EMPTY.replace("| no process found |", "| 1 123 python 2048 |")
        with patch.object(metax_capture, "lineage", return_value=[[123, 456], [100, 400]]):
            value = metax_capture.parse_sample(occupied, 1)
        self.assertEqual(value["processes"][0]["ancestors"], [[123, 456], [100, 400]])
        with patch.object(metax_capture, "lineage", return_value=[]), self.assertRaisesRegex(ValueError, "PID"):
            metax_capture.parse_sample(occupied, 1)

    def test_wrong_card_or_missing_inventory_is_rejected(self):
        with self.assertRaises(ValueError):
            metax_capture.parse_sample(EMPTY, 0)
        with self.assertRaises(ValueError):
            metax_capture.parse_sample(EMPTY.replace("| Process:", ""), 1)
        with self.assertRaises(ValueError):
            metax_capture.parse_sample(EMPTY.replace("| no process found |", ""), 1)

    def test_chunk_plan_preserves_every_index(self):
        indices = list(range(1023))
        planned = list(metax_campaign.chunks(indices, 64))
        self.assertEqual([index for start, count in planned for index in range(start, start + count)], indices)
        self.assertTrue(all(1 <= count <= 64 for start, count in planned))
        sparse = [0, 100, 101, 400, 500]
        self.assertEqual([index for start, count in metax_campaign.chunks(sparse, 2) for index in range(start, start + count)], sparse)

    def test_only_dedicated_python_worker_is_selected(self):
        listing = "PID COMMAND\n100 timeout 3300 python /task/worker.py\n123 python /task/worker.py --completion /out/done\n"
        with patch.object(metax_campaign.subprocess, "check_output", return_value=listing):
            self.assertEqual(metax_campaign.worker_identity("task", "/task/worker.py"), 123)


if __name__ == "__main__":
    unittest.main()
