import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

from lightx2v.utils import transformer_profile as profile_module
from lightx2v.utils.transformer_profile import (
    PROFILE_LAYER_ENV,
    PROFILE_MODE_ENV,
    PROFILE_STEP_ENV,
    TransformerProfile,
    suspend_transformer_profile,
)


class TransformerProfileTest(unittest.TestCase):
    def _block_profile(self, **overrides):
        env = {
            PROFILE_MODE_ENV: "block",
            PROFILE_STEP_ENV: "2",
            PROFILE_LAYER_ENV: "3",
        }
        env.update(overrides)
        return patch.dict(os.environ, env)

    def test_block_profile_is_consumed_once(self):
        with self._block_profile():
            profile = TransformerProfile("test", infer_steps=4, num_layers=5)

        self.assertIsNone(profile.mode_for_step(1))
        mode = profile.mode_for_step(2)
        self.assertEqual(mode, "block")

        with (
            patch.object(profile_module, "_one_call_torch_profile", side_effect=lambda *_args: nullcontext()),
            patch.object(profile_module, "_latest_trace", return_value=Path("trace.json")),
            profile.record_transformer(mode),
        ):
            self.assertTrue(profile.should_record_block(3))
            with profile.record_block(3):
                pass

        self.assertIsNone(profile.mode_for_step(2))

    def test_missing_target_block_is_an_error(self):
        with self._block_profile():
            profile = TransformerProfile("test", infer_steps=4, num_layers=5)

        with self.assertRaisesRegex(RuntimeError, "was not executed"):
            with profile.record_transformer(profile.mode_for_step(2)):
                self.assertFalse(profile.should_record_block(2))

    def test_suspension_preserves_target_profile(self):
        with self._block_profile():
            profile = TransformerProfile("test", infer_steps=4, num_layers=5)

        with suspend_transformer_profile():
            self.assertIsNone(profile.mode_for_step(2))
        self.assertEqual(profile.mode_for_step(2), "block")

    def test_target_bounds_are_validated(self):
        with self._block_profile(**{PROFILE_LAYER_ENV: "5"}):
            with self.assertRaisesRegex(ValueError, "out of range"):
                TransformerProfile("test", infer_steps=4, num_layers=5)


if __name__ == "__main__":
    unittest.main()
