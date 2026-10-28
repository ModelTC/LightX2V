"""Exercise the production Turbo constructor and H3 capability binding."""

import unittest

import torch
from lightx2v_train.model_capabilities import DistributionMatchingCapability
from lightx2v_train.trainers.turbo.tests import test_h3_turbo_launch as recipes
from lightx2v_train.trainers.turbo.trainer import TurboTrainer
from lightx2v_train.utils.registry import build_model


class H3TurboStartupTests(unittest.TestCase):
    def bind_trainer(self, task):
        config = recipes.H3TurboLaunchTests.config(task)
        trainer = TurboTrainer(config)
        self.addCleanup(trainer.finish_monitor)
        model = build_model(config)
        # Initialize the production geometry/precision metadata without loading
        # the large pretrained components; binding itself needs no weights.
        model.load_components(load_transformer=False, load_vae=False, load_condition_encoder=False)
        trainer.set_model(model)
        return trainer

    def test_production_recipes_bind_without_dmd_generation_shapes(self):
        for task in recipes.CONFIGS:
            with self.subTest(task=task):
                trainer = self.bind_trainer(task)
                self.assertIsNone(trainer.generation_shapes)
                self.assertEqual(trainer.latent_dtype, torch.float32)
                self.assertEqual(trainer.num_inference_steps, 8)
                self.assertEqual(trainer.guidance_scale, 1.0)
                self.assertEqual(trainer.student.options.geometry_from_metadata, True)
                self.assertEqual(trainer.separate_cm_data, task == "fl2av")

    def test_bound_trainer_keeps_each_samples_landscape_or_portrait_geometry(self):
        for task in recipes.CONFIGS:
            trainer = self.bind_trainer(task)
            for height, width in ((768, 1344), (1344, 768)):
                with self.subTest(task=task, height=height, width=width):
                    sample = {"meta": {"target_num_frames": 124, "target_height": height, "target_width": width}}
                    shape = trainer._latent_shape(sample)
                    self.assertEqual(shape.num_frames, 124)
                    self.assertEqual(shape.latent_height, height // 16)
                    self.assertEqual(shape.latent_width, width // 16)
                    self.assertEqual(shape.latent_frames, 37)
                    self.assertEqual(shape.audio_latents, 207)

    def test_metadata_geometry_remains_strict_per_sample(self):
        trainer = self.bind_trainer("fl2av")
        with self.assertRaisesRegex(KeyError, "target_width"):
            trainer._latent_shape({"meta": {"target_num_frames": 124, "target_height": 768}})
        with self.assertRaisesRegex(ValueError, "allowed_resolutions"):
            trainer._latent_shape({"meta": {"target_num_frames": 124, "target_height": 64, "target_width": 96}})

    def test_explicit_shapes_remain_validated_in_metadata_mode(self):
        capability = self.bind_trainer("fl2av").student
        capability.validate_generation_shapes([{"value": [124, 768, 1344]}])
        for shapes in ([], [{"value": [768, 1344]}], [{"value": [124, -768, 1344]}]):
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                capability.validate_generation_shapes(shapes)

    def test_configured_geometry_still_requires_generation_shapes(self):
        config = recipes.H3TurboLaunchTests.config("fl2av")
        config["model"]["capabilities"]["distribution_matching"]["geometry_from_metadata"] = False
        model = build_model(config)
        capability = model.ensure_capabilities().require(DistributionMatchingCapability)
        with self.assertRaisesRegex(ValueError, "generation_shapes must be a non-empty list"):
            capability.validate_generation_shapes(None)
        capability.validate_generation_shapes([{"value": [124, 768, 1344]}])


if __name__ == "__main__":
    unittest.main()
