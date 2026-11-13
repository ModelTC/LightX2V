"""Check metric semantics using known geometry; no model or device required."""

import tempfile
import unittest
from pathlib import Path

import evaluate as evaluation
import numpy as np
from PIL import Image
from evaluate import auc, difference, vggt_metrics


def setUpModule():
    evaluation.load_numeric_dependencies()


class MetricTests(unittest.TestCase):
    def test_difference_and_shape(self):
        self.assertEqual(difference(np.zeros(5), np.ones(5))["mae"], 1)
        self.assertEqual(difference(np.zeros(5), np.ones(5))["rmse"], 1)
        self.assertTrue(difference(np.arange(5), np.arange(5))["exact"])
        self.assertFalse(difference(np.zeros(4), np.zeros(5))["same_shape"])

    def test_auc_endpoints(self):
        self.assertEqual(auc([0, 0, 0], 5), 1)
        self.assertEqual(auc([6, 8], 5), 0)

    def test_depth_scale_and_camera_gauge_invariance(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            extrinsics = []
            world_points = []
            yy, xx = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
            camera_points = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
            for i in range(5):
                angle = i * 0.3
                rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
                pose = np.eye(4)
                pose[:3, :3] = rotation
                pose[:3, 3] = [i * 0.2, i * 0.1, i * 0.05]
                np.savetxt(directory / f"pose_{i:02d}.txt", pose)
                np.savetxt(directory / f"intrinsics_{i:02d}.txt", np.eye(4))
                world_points.append(camera_points @ rotation.T + pose[:3, 3])
                Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(directory / f"depth_{i:02d}.png")
                pred_center = pose[:3, 3] * 3 + [2, 1, -4]
                extrinsics.append(np.c_[rotation.T, -rotation.T @ pred_center])
            output = {"depth": np.full((1, 5, 4, 4, 1), 2.0), "extrinsics": np.asarray(extrinsics)[None], "world_points": np.asarray(world_points)[None]}
            metrics = vggt_metrics(output, {"gt_directory": "."}, directory)
            self.assertEqual(metrics["depth_macro"]["abs_rel"], 0)
            self.assertEqual(metrics["depth_macro"]["delta1"], 1)
            for pair in metrics["camera_pairs"]:
                self.assertLess(pair["rotation_deg"], 1e-05)
                self.assertLess(pair["translation_direction_deg"], 1e-05)


if __name__ == "__main__":
    unittest.main()
