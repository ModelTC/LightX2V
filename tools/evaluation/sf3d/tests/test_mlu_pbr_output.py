"""Reject geometry-only MLU outputs without changing other platform validation."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

spec = importlib.util.spec_from_file_location("sf3d_eval", Path(__file__).resolve().parents[1] / "evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class MLUOutputContract(unittest.TestCase):
    def test_pbr_required_only_for_mlu(self):
        mesh = trimesh.creation.icosphere(subdivisions=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.glb"
            mesh.export(path)
            with patch.dict(os.environ, PLATFORM="cambricon_mlu"):
                with self.assertRaisesRegex(ValueError, "UVs"):
                    evaluation.validate_output(path, {})
            with patch.dict(os.environ, PLATFORM="cuda"):
                self.assertTrue(evaluation.validate_output(path, {})["finite"])

    def test_full_pbr_roundtrip_and_missing_normal(self):
        mesh = trimesh.creation.icosphere(subdivisions=0)
        texture = Image.new("RGB", (1024, 1024), (128, 128, 255))
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=texture, normalTexture=texture, roughnessFactor=.5, metallicFactor=.1)
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), material=material)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, PLATFORM="cambricon_mlu"):
            path = Path(directory) / "mesh.glb"
            mesh.export(path)
            self.assertTrue(evaluation.validate_output(path, {})["finite"])
            material.normalTexture = None
            mesh.export(path)
            with self.assertRaisesRegex(ValueError, "normal textures"):
                evaluation.validate_output(path, {})


if __name__ == "__main__":
    unittest.main()
