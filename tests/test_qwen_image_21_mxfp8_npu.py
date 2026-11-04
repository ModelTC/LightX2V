import sys

import pytest
import torch

from lightx2v.common.ops.mm import mm_weight
from lightx2v.common.ops.mm.mm_weight import MMWeightWmxfp8Amxfp8dynamicNpu
from lightx2v.models.networks.qwen_image_21.model import QwenImage21TransformerModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.qwen_image_21 import qwen_image_21_runner
from lightx2v.models.runners.qwen_image_21.qwen_image_21_runner import QwenImage21Runner
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class _FakeTorchNpu:
    """CPU implementation that records the torch_npu operator contracts."""

    float8_e4m3fn = torch.float8_e4m3fn
    float8_e8m0fnu = object()

    def __init__(self):
        self.quant_calls = []
        self.matmul_calls = []

    def npu_dynamic_mx_quant(self, tensor, *, dst_type):
        self.quant_calls.append({"tensor": tensor, "dst_type": dst_type})
        quantized = tensor.to(torch.float8_e4m3fn)
        scale = torch.arange(tensor.shape[0] * (tensor.shape[1] // 32), dtype=torch.uint8, device=tensor.device)
        return quantized, scale.reshape(tensor.shape[0], tensor.shape[1] // 32)

    def npu_quant_matmul(self, input_tensor_quant, weight, weight_scale, **kwargs):
        self.matmul_calls.append(
            {
                "input_tensor_quant": input_tensor_quant,
                "weight": weight,
                "weight_scale": weight_scale,
                **kwargs,
            }
        )
        return torch.zeros(
            (*input_tensor_quant.shape[:-1], weight.shape[1]),
            dtype=kwargs["output_dtype"],
            device=input_tensor_quant.device,
        )


@pytest.fixture
def fake_npu(monkeypatch):
    runtime = _FakeTorchNpu()
    monkeypatch.setattr(mm_weight, "AI_DEVICE", "npu")
    monkeypatch.setitem(sys.modules, "torch_npu", runtime)

    # The operator is mocked, so keep its tensors on CPU while preserving the
    # production code's explicit moves to AI_DEVICE="npu".
    original_to = torch.Tensor.to

    def to_with_npu_mapped_to_cpu(tensor, *args, **kwargs):
        args = tuple("cpu" if isinstance(arg, str) and arg == "npu" else arg for arg in args)
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to_with_npu_mapped_to_cpu)
    return runtime


def _linear(bias=True, **kwargs):
    return MMWeightWmxfp8Amxfp8dynamicNpu(
        "linear.weight",
        "linear.bias" if bias else None,
        **kwargs,
    )


def _load_linear(runtime, *, out_features=4, in_features=32, dtype=torch.bfloat16):
    linear = _linear()
    weight = torch.randn(out_features, in_features, dtype=dtype)
    bias = torch.randn(out_features, dtype=dtype)
    linear.load({"linear.weight": weight, "linear.bias": bias})
    return linear


def test_mxfp8_npu_is_registered():
    assert MM_WEIGHT_REGISTER["mxfp8-npu"] is MMWeightWmxfp8Amxfp8dynamicNpu


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_load_quantizes_and_transposes_weight_with_fp32_bias(fake_npu, dtype):
    linear = _linear()
    weight = torch.randn(2, 32, dtype=dtype)
    bias = torch.randn(2, dtype=dtype)

    linear.load({"linear.weight": weight, "linear.bias": bias})

    quant_call = fake_npu.quant_calls[0]
    assert quant_call["tensor"].dtype == dtype
    assert quant_call["tensor"].is_contiguous()
    assert quant_call["dst_type"] is fake_npu.float8_e4m3fn
    assert linear.weight.shape == (32, 2)
    assert linear.weight.is_contiguous()
    assert torch.equal(linear.weight, weight.to(torch.float8_e4m3fn).t())
    assert linear.bias.dtype == torch.float32
    torch.testing.assert_close(linear.bias, bias.to(torch.float32), rtol=0, atol=0)
    assert linear.pin_weight is None
    assert linear.pin_weight_scale is None
    assert linear.pin_bias is None


def test_load_pads_odd_scale_groups_and_pairs_output_channels(fake_npu):
    linear = _load_linear(fake_npu, out_features=2, in_features=32)

    # Raw scales are [[0], [1]]. Padding to two columns and grouping pairs
    # must produce [group_pair, output_channel, two_scales].
    expected_scale = torch.tensor([[[0, 0], [1, 0]]], dtype=torch.uint8)
    assert linear.weight_scale.shape == (1, 2, 2)
    assert linear.weight_scale.is_contiguous()
    assert torch.equal(linear.weight_scale, expected_scale)


def test_load_preserves_even_scale_groups_without_padding(fake_npu):
    linear = _load_linear(fake_npu, out_features=2, in_features=64)

    expected_scale = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.uint8)
    assert linear.weight_scale.shape == (1, 2, 2)
    assert torch.equal(linear.weight_scale, expected_scale)


@pytest.mark.parametrize(
    "shape,dtype,error,match",
    [
        ((2, 16, 32), torch.bfloat16, ValueError, "expects a 2D weight"),
        ((2, 32), torch.float32, TypeError, "expects BF16/FP16 weights"),
        ((2, 30), torch.bfloat16, ValueError, "divisible by 32"),
    ],
)
def test_load_rejects_invalid_weights_before_quantization(fake_npu, shape, dtype, error, match):
    with pytest.raises(error, match=match):
        _linear(bias=False).load({"linear.weight": torch.zeros(shape, dtype=dtype)})
    assert fake_npu.quant_calls == []


@pytest.mark.parametrize(
    "options",
    [
        {"create_cuda_buffer": True},
        {"create_cpu_buffer": True},
        {"lazy_load": True},
    ],
)
def test_load_rejects_offline_or_buffered_loading(options):
    with pytest.raises(ValueError, match="online loading"):
        _linear(bias=False, **options).load({})


def test_load_rejects_non_npu_device(monkeypatch):
    monkeypatch.setattr(mm_weight, "AI_DEVICE", "cpu")
    with pytest.raises(RuntimeError, match="requires an Ascend NPU"):
        _linear(bias=False).load({})


def test_apply_flattens_tokens_and_restores_original_shape(fake_npu):
    linear = _load_linear(fake_npu)
    input_tensor = torch.randn(2, 3, 32, dtype=torch.bfloat16)

    output = linear.apply(input_tensor)

    quant_call = fake_npu.quant_calls[-1]
    assert quant_call["tensor"].shape == (6, 32)
    torch.testing.assert_close(quant_call["tensor"], input_tensor.reshape(6, 32), rtol=0, atol=0)
    assert quant_call["dst_type"] is fake_npu.float8_e4m3fn

    matmul_call = fake_npu.matmul_calls[0]
    assert matmul_call["input_tensor_quant"].shape == (6, 32)
    assert matmul_call["weight"] is linear.weight
    assert matmul_call["weight_scale"] is linear.weight_scale
    assert matmul_call["scale_dtype"] is fake_npu.float8_e8m0fnu
    assert matmul_call["pertoken_scale_dtype"] is fake_npu.float8_e8m0fnu
    assert matmul_call["bias"] is linear.bias
    assert matmul_call["output_dtype"] == torch.bfloat16
    assert matmul_call["group_sizes"] == [1, 1, 32]
    assert output.shape == (2, 3, 4)
    assert output.dtype == input_tensor.dtype


def test_apply_casts_bias_to_fp32(fake_npu):
    linear = _load_linear(fake_npu)
    linear.bias = linear.bias.to(torch.bfloat16)

    linear.apply(torch.randn(3, 32, dtype=torch.bfloat16))

    assert fake_npu.matmul_calls[0]["bias"].dtype == torch.float32


def test_apply_rejects_invalid_inputs_before_quantization(fake_npu):
    linear = _load_linear(fake_npu)
    quant_call_count = len(fake_npu.quant_calls)

    with pytest.raises(TypeError, match="expects BF16/FP16 activations"):
        linear.apply(torch.randn(3, 32, dtype=torch.float32))
    with pytest.raises(ValueError, match="input K=64 does not match weight K=32"):
        linear.apply(torch.randn(3, 64, dtype=torch.bfloat16))

    assert len(fake_npu.quant_calls) == quant_call_count
    assert fake_npu.matmul_calls == []


def test_online_mxfp8_uses_original_checkpoint_loader(monkeypatch):
    model = object.__new__(QwenImage21TransformerModel)
    model.config = {
        "dit_quant_scheme": "mxfp8-npu",
        "dit_original_ckpt": "/path/to/custom-bf16",
    }
    model.use_tp = False
    expected = {"linear.weight": object()}
    calls = []

    def load_ckpt(unified_dtype, sensitive_layer):
        calls.append((unified_dtype, sensitive_layer))
        return expected

    monkeypatch.setattr(model, "_load_ckpt", load_ckpt)

    result = model._load_quant_ckpt(True, {"sensitive"})

    assert result is expected
    assert calls == [(True, {"sensitive"})]


def test_runner_accepts_online_mxfp8_without_prequantized_checkpoint(monkeypatch):
    monkeypatch.setattr(qwen_image_21_runner, "AI_DEVICE", "npu")
    monkeypatch.setattr(DefaultRunner, "__init__", lambda self, config: None)

    QwenImage21Runner(
        {
            "dit_quantized": True,
            "dit_quant_scheme": "mxfp8-npu",
            "causal_condition": True,
            "enable_cfg": False,
            "infer_steps": 1,
        }
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"cpu_offload": True}, "cpu_offload"),
        ({"tensor_parallel": True}, "tensor_parallel"),
        ({"seq_parallel": True}, "seq_parallel"),
        ({"shared_cpu_weights": True}, "shared_cpu_weights"),
        ({"dit_disk_streaming": True}, "dit_disk_streaming"),
        ({"lora_configs": [{"path": "lora.safetensors"}]}, "LoRA"),
        ({"attn_type": "npu_quant_flash_attn"}, "independently of quantized attention"),
    ],
)
def test_runner_rejects_unsupported_mxfp8_combinations(monkeypatch, overrides, match):
    monkeypatch.setattr(qwen_image_21_runner, "AI_DEVICE", "npu")

    with pytest.raises(ValueError, match=match):
        QwenImage21Runner._validate_mxfp8_npu_config(overrides)


def test_runner_rejects_mxfp8_on_non_npu_device(monkeypatch):
    monkeypatch.setattr(qwen_image_21_runner, "AI_DEVICE", "cpu")

    with pytest.raises(ValueError, match="requires an Ascend NPU"):
        QwenImage21Runner._validate_mxfp8_npu_config({})
