"""RelayVSR output geometry, tested without importing GPU-initializing modules."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).resolve().parents[1] / "lightx2v/models/networks/relayvsr/geometry.py"
spec = importlib.util.spec_from_file_location("relayvsr_geometry", path)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
RelayVSRGeometry = geometry.RelayVSRGeometry


class RelayVSRGeometryTests(unittest.TestCase):
    def test_sr_ratio_scales_both_axes(self):
        self.assertEqual(RelayVSRGeometry(4).output_size(SimpleNamespace(sr_ratio=1.875, size=[]), 768, 1344), (1440, 2520))
        self.assertEqual(RelayVSRGeometry(2).output_size(SimpleNamespace(sr_ratio=2, size=[]), 768, 1024), (1536, 2048))

    def test_size_scales_each_axis(self):
        # 1344x768 to the 16:9 2K canvas: horizontal 1.905x, vertical 1.875x.
        request = SimpleNamespace(sr_ratio=2.0, size=[1440, 2560])
        self.assertEqual(RelayVSRGeometry(4).output_size(request, 768, 1344), (1440, 2560))

    def test_size_ignores_sr_ratio(self):
        request = SimpleNamespace(sr_ratio=100.0, size=[1376, 2656])
        self.assertEqual(RelayVSRGeometry(4).output_size(request, 736, 1408), (1376, 2656))

    def test_missing_size_attribute_uses_sr_ratio(self):
        self.assertEqual(RelayVSRGeometry(4).output_size(SimpleNamespace(sr_ratio=2.0), 768, 1024), (1536, 2048))

    def test_size_rejects_bad_shapes(self):
        for size, message in (([1440], "height, width"), ([1440, 2560, 3], "height, width"), ([0, 2560], "positive")):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, message):
                RelayVSRGeometry(4).output_size(SimpleNamespace(sr_ratio=2.0, size=size), 768, 1344)

    def test_x2_rejects_size(self):
        with self.assertRaisesRegex(ValueError, "exact 2x"):
            RelayVSRGeometry(2).output_size(SimpleNamespace(sr_ratio=2, size=[1536, 2688]), 768, 1344)

    def test_sr_ratio_still_validated_without_size(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 4"):
            RelayVSRGeometry(4).output_size(SimpleNamespace(sr_ratio=5, size=[]), 768, 1024)


if __name__ == "__main__":
    unittest.main()
