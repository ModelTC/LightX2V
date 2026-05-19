import os
import unittest
from importlib import import_module
from types import SimpleNamespace

import torch

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")


class WanFeatureCacheCfgStateTest(unittest.TestCase):
    def test_first_block_reads_cfg_state_from_scheduler(self):
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
        scheduler = SimpleNamespace(infer_condition=True)
        infer.set_scheduler(scheduler)

        cond_residual = torch.ones(1, 4, 8)
        self.assertTrue(infer.calculate_should_calc(cond_residual))
        self.assertIsNotNone(infer.prev_first_block_residual_even)
        self.assertIsNone(infer.prev_first_block_residual_odd)

        scheduler.infer_condition = False
        uncond_residual = torch.full((1, 4, 8), 2.0)
        self.assertTrue(infer.calculate_should_calc(uncond_residual))
        self.assertIsNotNone(infer.prev_first_block_residual_odd)


if __name__ == "__main__":
    unittest.main()
