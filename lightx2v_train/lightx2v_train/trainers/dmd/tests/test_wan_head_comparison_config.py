import os
import unittest
from pathlib import Path
from unittest.mock import patch

from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.dmd.residual_head import ResidualHeadConfig

CONFIG = Path(__file__).resolve().parents[4] / "configs/train/dmd/wan2_1_t2v_1_3b_head_comparison_fsdp2.yaml"


class WanHeadComparisonConfigTest(unittest.TestCase):
    def config(self, projected="false", head="false"):
        with patch.dict(
            os.environ,
            {
                "WAN_DMD_MODEL": "/path/to/model",
                "WAN_DMD_PROMPTS": "/path/to/prompts.txt",
                "WAN_DMD_OUTPUT": "/path/to/output",
                "WAN_DMD_PROJECTED": projected,
                "WAN_DMD_RESIDUAL_HEAD": head,
            },
        ):
            return load_config(str(CONFIG))

    def test_only_loss_strategy_differs_across_three_experiments(self):
        variants = [self.config(), self.config("true"), self.config(head="true")]
        for config, projected, head in zip(variants, (False, True, False), (False, False, True)):
            options = config["model"]["capabilities"]["distribution_matching"]
            self.assertIs(options.pop("projected_dmd"), projected)
            head_config = ResidualHeadConfig.from_mapping(config["training"]["dmd"]["residual_head"])
            self.assertIs(head_config.enabled, head)
            self.assertIs(config["training"]["dmd"]["residual_head"].pop("enabled"), head)
        self.assertEqual(variants[0], variants[1])
        self.assertEqual(variants[0], variants[2])

    def test_four_steps_lora_full_fake_and_shared_previews(self):
        config = self.config(head="true")
        training, inference = config["training"], config["inference"]
        self.assertEqual(training["dmd"]["num_inference_steps"], 4)
        self.assertEqual(training["dmd"]["update_order"], "fake_first")
        self.assertEqual(inference["num_inference_steps"], 4)
        self.assertEqual(training["dmd"]["generation_shapes"], [{"value": [81, 480, 832]}])
        self.assertEqual(training["student"]["train_type"], "lora")
        self.assertEqual(training["student"]["lora"]["rank"], 128)
        self.assertEqual(training["student"]["lora"]["alpha"], 8)
        self.assertEqual(training["fake"]["train_type"], "full")
        self.assertEqual(training["student"]["optimizer"]["learning_rate"], 5e-5)
        self.assertEqual(training["fake"]["optimizer"]["learning_rate"], 4e-7)
        self.assertEqual(training["dmd"]["fake_update_ratio"], 5)
        self.assertEqual(config["data"]["val"]["max_samples"], 8)
        self.assertEqual(inference["infer_every_iters"], 100)
        self.assertEqual(config["distributed"]["fsdp2"]["size"], 2)

    def test_teacher_cfg_shift_and_student_weight_ema(self):
        config = self.config()
        training = config["training"]
        self.assertEqual(training["teacher"]["guidance_scale"], 4.0)
        self.assertEqual(config["scheduler"]["time_shift_settings"]["time_shift_mu"], 6.0)
        self.assertEqual(
            training["dmd"]["score_sampling"],
            {
                "type": "continuous_uniform",
                "discrete_samples": 1000,
            },
        )
        self.assertEqual(
            training["student"]["ema"],
            {
                "enabled": True,
                "decay": 0.99,
                "use_for_inference": True,
            },
        )
        self.assertIs(config["inference"]["enable_cfg"], False)

    def test_all_transformer_roles_keep_fp32_master_parameters(self):
        model = self.config()["model"]
        self.assertEqual(model["transformer_param_dtype"], "fp32")
        self.assertEqual(model["fake"]["transformer_param_dtype"], "fp32")
        self.assertEqual(model["teacher"]["transformer_param_dtype"], "fp32")
        self.assertEqual(model["running_dtype"], "bf16")
        self.assertEqual(self.config()["distributed"]["fsdp2"]["mixed_precision"]["param_dtype"], "bf16")


if __name__ == "__main__":
    unittest.main()
