import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class SeedVRInferenceOptimizationsTest(unittest.TestCase):
    def test_main_norms_use_native_ops_with_rms_fallback(self):
        source = (REPO_ROOT / "lightx2v/models/networks/seedvr/infer/utils.py").read_text()
        tree = ast.parse(source)

        self.assertIn("F.rms_norm(x, (x.shape[-1],), weight=None, eps=eps)", source)
        self.assertIn("F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=eps)", source)
        self.assertIn('if hasattr(F, "rms_norm"):', source)
        self.assertIsInstance(tree, ast.Module)

    def test_transformer_reuses_branch_outputs_inplace(self):
        source = (REPO_ROOT / "lightx2v/models/networks/seedvr/infer/transformer_infer.py").read_text()

        for operation in [
            "vid_attn.add_(vid)",
            "txt_attn.add_(txt)",
            "vid_mlp.add_(vid_attn)",
            "txt_mlp.add_(txt_attn)",
            "F.silu(gate, inplace=True)",
            "gate.mul_(up)",
        ]:
            with self.subTest(operation=operation):
                self.assertIn(operation, source)

        self.assertNotIn("vid.add_(vid_attn)", source)
        self.assertNotIn("txt.add_(txt_attn)", source)

    def test_qk_norm_is_fused_but_can_fall_back_for_precision(self):
        source = (REPO_ROOT / "lightx2v/models/networks/seedvr/infer/transformer_infer.py").read_text()

        self.assertIn('config.get("fused_qk_rms_norm", True)', source)
        self.assertIn("norm_q.sensitive_layer_dtype == norm_q.infer_dtype", source)
        self.assertIn("apply_qk_rms_norm(query, key, norm_q, norm_k, use_triton=use_triton)", source)
        self.assertEqual(source.count("self._apply_qk_norm(block_weight"), 4)

    def test_vae_inplace_path_is_inference_only(self):
        source = (REPO_ROOT / "lightx2v/models/video_encoders/hf/seedvr/attn_video_vae.py").read_text()

        self.assertIn("if not torch.is_grad_enabled() and isinstance(activation, nn.SiLU):", source)
        self.assertIn("hidden_states.add_(temb)", source)
        self.assertIn("hidden_states.mul_(1 + scale).add_(shift)", source)
        self.assertIn("hidden_states.add_(input_tensor)", source)
        self.assertIn("hidden_states.div_(self.output_scale_factor)", source)
        self.assertIn("if torch.is_grad_enabled():", source)


if __name__ == "__main__":
    unittest.main()
