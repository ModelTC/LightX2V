"""FP32 H3-Image projections retain their compute dtype inside BF16 FSDP."""

import copy
import unittest
from unittest.mock import Mock, patch

import torch

from lightx2v_train.runtime import fsdp


class ShardPrecisionTests(unittest.TestCase):
    def apply_plan(self, plan):
        model = Mock()
        model.is_fsdp2_wrapped.return_value = False
        model.fsdp2_shard_plan.return_value = plan
        config = {"distributed": {"fsdp2": {
            "enabled": True,
            "log_memory": False,
            "mixed_precision": {
                "param_dtype": "bf16",
                "reduce_dtype": "fp32",
                "output_dtype": None,
                "cast_forward_inputs": False,
            },
        }}}
        original_config = copy.deepcopy(config)
        mesh = object()
        with (
            patch.object(fsdp, "fsdp2_enabled", return_value=True),
            patch.object(fsdp, "get_device_mesh", return_value=mesh),
            patch.object(fsdp, "_cuda_memory_gb", return_value=None),
            patch.object(fsdp, "_fully_shard_module") as shard,
            patch.object(torch.cuda, "empty_cache"),
        ):
            self.assertIs(fsdp.apply_fsdp2(model, config), model)
        self.assertEqual(config, original_config)
        model.after_fsdp2_shard.assert_called_once_with(config)
        return shard.call_args_list, mesh

    def test_old_module_and_modules_entries_share_global_policy(self):
        first, second, root = object(), object(), object()
        calls, mesh = self.apply_plan([
            {"modules": [first, second], "reshard_after_forward": True},
            {"module": root, "reshard_after_forward": False},
        ])
        self.assertEqual([call.args[0] for call in calls], [first, second, root])
        self.assertEqual([call.args[3] for call in calls], [True, True, False])
        self.assertTrue(all(call.args[1] is mesh for call in calls))
        self.assertTrue(all(call.args[2] is calls[0].args[2] for call in calls))
        self.assertEqual(calls[0].args[2].param_dtype, torch.bfloat16)
        self.assertEqual(calls[0].args[2].reduce_dtype, torch.float32)

    def test_fp32_projection_override_does_not_change_transformer_policy(self):
        projection, head, block = object(), object(), object()
        calls, _ = self.apply_plan([
            {"modules": [projection, head], "reshard_after_forward": True,
             "mixed_precision": {"param_dtype": "fp32"}},
            {"module": block, "reshard_after_forward": False},
        ])
        for call in calls[:2]:
            policy = call.args[2]
            self.assertEqual(policy.param_dtype, torch.float32)
            self.assertEqual(policy.reduce_dtype, torch.float32)
            self.assertIsNone(policy.output_dtype)
            self.assertFalse(policy.cast_forward_inputs)
        self.assertEqual(calls[2].args[2].param_dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
