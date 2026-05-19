import os
import unittest
from importlib import import_module

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")


class WanSwitchStatusTest(unittest.TestCase):
    def test_first_block_switch_status_toggles_infer_conditional(self):
        transformer_infer = import_module("lightx2v.models.networks.wan.infer.feature_caching.transformer_infer")

        config = {
            "task": "i2v",
            "num_layers": 1,
            "num_heads": 1,
            "dim": 8,
            "seq_parallel": False,
            "cpu_offload": False,
            "enable_cfg": True,
            "residual_diff_threshold": 0.1,
            "downsample_factor": 2,
            "modulate_type": "torch",
            "rope_type": "torch",
        }

        infer = transformer_infer.WanTransformerInferFirstBlock(config)

        self.assertTrue(hasattr(infer, "switch_status"))
        self.assertTrue(infer.infer_conditional)

        infer.switch_status()
        self.assertFalse(infer.infer_conditional)

        infer.switch_status()
        self.assertTrue(infer.infer_conditional)


if __name__ == "__main__":
    unittest.main()
