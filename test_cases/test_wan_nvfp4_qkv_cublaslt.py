import os
import sys
import types
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")


def ensure_lightx2v_pipeline_stub():
    if "lightx2v.pipeline" not in sys.modules:
        pipeline_stub = types.ModuleType("lightx2v.pipeline")
        pipeline_stub.LightX2VPipeline = object
        sys.modules["lightx2v.pipeline"] = pipeline_stub


def ensure_local_lightx2v_kernel():
    kernel_python_root = Path(__file__).resolve().parents[1] / "lightx2v_kernel" / "python"
    kernel_python_root_str = str(kernel_python_root)
    if kernel_python_root_str in sys.path:
        sys.path.remove(kernel_python_root_str)
    sys.path.insert(0, kernel_python_root_str)


class WanNvfp4QkvCublasltTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_lightx2v_pipeline_stub()
        ensure_local_lightx2v_kernel()

    def test_quantized_projection_forwards_to_cublaslt(self):
        mm_weight = import_module("lightx2v.common.ops.mm.mm_weight")
        operator = mm_weight.MMWeightWnvfp4Anvfp4dynamic(
            "blocks.0.self_attn.q.weight",
            "blocks.0.self_attn.q.bias",
        )
        input_quant = object()
        input_scale = object()
        weight = object()
        weight_scale = object()
        alpha = object()
        bias = object()
        output = object()
        operator.weight = weight
        operator.weight_scale = weight_scale
        operator.alpha = alpha
        operator.bias = bias

        with patch.object(mm_weight, "cublaslt_scaled_nvfp4_mm_bias", return_value=output) as kernel:
            actual = operator.apply_quantized_cublaslt(input_quant, input_scale)

        self.assertIs(actual, output)
        kernel.assert_called_once_with(
            input_quant,
            weight,
            input_scale,
            weight_scale,
            alpha=alpha,
            bias=bias,
            algorithm_index=-1,
        )

    def test_projection_quantizes_then_forwards_to_cublaslt(self):
        mm_weight = import_module("lightx2v.common.ops.mm.mm_weight")
        operator = mm_weight.MMWeightWnvfp4Anvfp4dynamic(
            "blocks.0.self_attn.o.weight",
            "blocks.0.self_attn.o.bias",
        )
        input_tensor = object()
        input_quant = object()
        input_scale = object()
        output = object()
        operator.act_quant_func = unittest.mock.Mock(return_value=(input_quant, input_scale))
        operator.apply_quantized_cublaslt = unittest.mock.Mock(return_value=output)

        actual = operator.apply_cublaslt(input_tensor, algorithm_index=2)

        self.assertIs(actual, output)
        operator.act_quant_func.assert_called_once_with(input_tensor)
        operator.apply_quantized_cublaslt.assert_called_once_with(
            input_quant,
            input_scale,
            2,
        )

    def test_thor_requires_nvfp4_weights(self):
        transformer_weights = import_module("lightx2v.models.networks.wan.weights.transformer_weights")
        config = {
            "rope_type": "torch_real_rope",
            "layer_norm_type": "torch",
            "rms_norm_type": "torch",
            "tensor_parallel": False,
            "seq_parallel": False,
            "thor": True,
        }

        with self.assertRaisesRegex(ValueError, "requires dit_quant_scheme='nvfp4'"):
            transformer_weights.WanSelfAttention(
                block_index=0,
                block_prefix="blocks",
                task="i2v",
                mm_type="Default",
                config=config,
            )
