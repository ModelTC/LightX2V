"""Check metric semantics using known geometry; no model or device required."""

import tempfile
import unittest
from pathlib import Path

import evaluate as evaluation
import numpy as np
from evaluate import compare, difference, surface_sample


def setUpModule():
    evaluation.load_numeric_dependencies()


class MetricTests(unittest.TestCase):
    def test_difference_and_shape(self):
        self.assertEqual(difference(np.zeros(5), np.ones(5))["mae"], 1)
        self.assertEqual(difference(np.zeros(5), np.ones(5))["rmse"], 1)
        self.assertTrue(difference(np.arange(5), np.arange(5))["exact"])
        self.assertFalse(difference(np.zeros(4), np.zeros(5))["same_shape"])

    def test_texture_coordinate_schema_and_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "a.npz"
            b = Path(directory) / "b.npz"
            coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
            attrs = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
            np.savez(a, texture_coords=coords, texture_attrs=attrs)
            np.savez(b, texture_coords=np.c_[np.zeros(2, dtype=np.int64), coords[::-1]], texture_attrs=attrs[::-1])
            with np.load(a) as x, np.load(b) as y:
                report = compare(x, y)
            self.assertEqual(report["texture_alignment"]["coordinate_iou"], 1)
            self.assertTrue(report["texture_alignment"]["attrs_common_coordinates"]["exact"])

    def test_mesh_rejects_negative_indices(self):
        vertices = np.eye(3)
        with self.assertRaises(ValueError):
            surface_sample(vertices, np.array([[0, 1, -1]]), 5, 42)
        with self.assertRaises(ValueError):
            surface_sample(vertices, np.array([[0, 1, 3]]), 5, 42)

    def test_identical_output_distance_is_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.npz"
            np.savez(path, texture_coords=np.array([[0, 1, 2, 3]], dtype=np.int32), texture_attrs=np.array([[0.1, 0.2]], dtype=np.float32), vertices=np.eye(3, dtype=np.float32))
            with np.load(path) as x, np.load(path) as y:
                result = compare(x, y)
            self.assertEqual(result["texture_alignment"]["coordinate_iou"], 1)
            self.assertEqual(result["raw_vertex_set_distance"]["max_sampled_nearest_distance"], 0)


if __name__ == "__main__":
    unittest.main()
