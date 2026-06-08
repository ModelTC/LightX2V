import os
import sys
import unittest
from importlib import import_module
from types import SimpleNamespace
import types
from pathlib import Path

import torch

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
    for module_name in list(sys.modules):
        if module_name == "lightx2v_kernel" or module_name.startswith("lightx2v_kernel."):
            del sys.modules[module_name]


def make_config(**overrides):
    config = {
        "task": "i2v",
        "num_layers": 1,
        "num_heads": 1,
        "dim": 8,
        "seq_parallel": False,
        "cpu_offload": False,
        "modulate_type": "torch",
        "rope_type": "torch",
        "dit_quant_scheme": "mxfp8",
        "mxfp8_fuse_enable": True,
        "infer_steps": 4,
        "teacache_thresh": 0.1,
        "use_ret_steps": False,
        "coefficients": ([1.0], [1.0]),
    }
    config.update(overrides)
    return config


def make_linear_phase():
    return SimpleNamespace(
        norm2=SimpleNamespace(apply=lambda x: x.clone()),
        ffn_0=SimpleNamespace(apply=lambda x: x + 1),
        ffn_2=SimpleNamespace(apply=lambda x: x + 2),
    )


class WanMxfp8FuseForwardingTest(unittest.TestCase):
    def test_base_infer_ffn_respects_mxfp8_fuse_enable(self):
        ensure_lightx2v_pipeline_stub()
        ensure_local_lightx2v_kernel()
        transformer_infer = import_module("lightx2v.models.networks.wan.infer.transformer_infer")

        phase = make_linear_phase()
        x = torch.zeros(1, 8)
        attn_out = torch.zeros(1, 8)
        c_shift = torch.zeros(1, 8)
        c_scale = torch.zeros(1, 8)

        disabled = transformer_infer.WanTransformerInfer(make_config(mxfp8_fuse_enable=False))
        disabled._mxfp8_fuse_available = True
        disabled._ensure_mxfp8_quant_ffn_ready = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not be called"))
        y = disabled.infer_ffn(phase, x.clone(), attn_out.clone(), c_shift, c_scale, c_gate_msa=None)
        self.assertIsInstance(y, torch.Tensor)

        enabled = transformer_infer.WanTransformerInfer(make_config(mxfp8_fuse_enable=True))
        enabled._mxfp8_fuse_available = True
        enabled._ensure_mxfp8_quant_ffn_ready = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fuse gate check reached"))
        with self.assertRaisesRegex(RuntimeError, "fuse gate check reached"):
            enabled.infer_ffn(phase, x.clone(), attn_out.clone(), c_shift, c_scale, c_gate_msa=None)

    def test_offload_phase_two_forwards_c_gate_msa(self):
        ensure_lightx2v_pipeline_stub()
        ensure_local_lightx2v_kernel()
        offload_infer = import_module("lightx2v.models.networks.wan.infer.offload.transformer_infer")

        infer = offload_infer.WanOffloadTransformerInfer(make_config())
        gate = torch.ones(1, 8)
        infer.phase_params = {
            "attn_out": torch.zeros(1, 8),
            "c_shift_msa": torch.zeros(1, 8),
            "c_scale_msa": torch.zeros(1, 8),
            "c_gate_msa": gate,
            "y": None,
        }
        seen = {}

        def fake_infer_ffn(phase, x, attn_out, c_shift, c_scale, c_gate=None):
            seen["gate"] = c_gate
            return torch.zeros_like(x)

        infer.infer_ffn = fake_infer_ffn
        infer.post_process = lambda x, y, c_gate, pre_infer_out=None: x

        x = infer.infer_phase(2, SimpleNamespace(), torch.zeros(1, 8), SimpleNamespace(adapter_args={"hints": []}))
        self.assertIsInstance(x, torch.Tensor)
        self.assertIs(seen["gate"], gate)

    def test_self_forcing_forwards_c_gate_msa(self):
        ensure_lightx2v_pipeline_stub()
        ensure_local_lightx2v_kernel()
        self_forcing = import_module("lightx2v.models.networks.wan.infer.self_forcing.transformer_infer")

        infer = self_forcing.WanSFTransformerInfer(make_config())
        gate = torch.ones(2, 1, 8)
        infer.pre_process = lambda modulation, embed0: (gate, gate, gate, gate, gate, gate)
        infer.infer_self_attn_with_kvcache = lambda *args, **kwargs: torch.zeros(4, 8)
        infer.infer_cross_attn_with_kvcache = lambda *args, **kwargs: (torch.zeros(4, 8), torch.zeros(4, 8))
        seen = {}

        def fake_infer_ffn(phase, x, attn_out, c_shift, c_scale, c_gate=None):
            seen["gate"] = c_gate
            return torch.zeros_like(x)

        infer.infer_ffn = fake_infer_ffn
        infer.post_process = lambda x, y, c_gate, pre_infer_out=None: x

        block = SimpleNamespace(compute_phases=[SimpleNamespace(modulation=None), SimpleNamespace(), SimpleNamespace()])
        pre_infer_out = SimpleNamespace(
            x=torch.zeros(4, 8),
            embed0=torch.zeros(2, 6, 8),
            grid_sizes=SimpleNamespace(tensor=torch.ones(1, dtype=torch.int32)),
            seq_lens=torch.ones(1, dtype=torch.int32),
            freqs=torch.zeros(1),
            context=torch.zeros(1, 8),
        )
        infer.infer_block_with_kvcache(block, torch.zeros(4, 8), pre_infer_out)
        self.assertIs(seen["gate"], gate)

    def test_lingbot_forwards_c_gate_msa(self):
        ensure_lightx2v_pipeline_stub()
        ensure_local_lightx2v_kernel()
        lingbot = import_module("lightx2v.models.networks.wan.infer.lingbot.transformer_infer")

        infer = lingbot.WanLingbotTransformerInfer(make_config())
        gate = torch.ones(1, 8)
        infer.pre_process = lambda modulation, embed0: (gate, gate, gate, gate, gate, gate)
        infer.infer_self_attn = lambda *args, **kwargs: torch.zeros(1, 8)
        infer.infer_cross_attn = lambda *args, **kwargs: (torch.zeros(1, 8), torch.zeros(1, 8))
        seen = {}

        def fake_infer_ffn(phase, x, attn_out, c_shift, c_scale, c_gate=None):
            seen["gate"] = c_gate
            return torch.zeros_like(x)

        infer.infer_ffn = fake_infer_ffn
        infer.post_process = lambda x, y, c_gate, pre_infer_out=None: x

        block = SimpleNamespace(compute_phases=[SimpleNamespace(modulation=None), SimpleNamespace(), SimpleNamespace()])
        pre_infer_out = SimpleNamespace(
            x=torch.zeros(1, 8),
            embed0=torch.zeros(1, 6, 8),
            context=torch.zeros(1, 8),
            conditional_dict={},
            adapter_args={"hints": []},
        )
        infer.infer_block(block, torch.zeros(1, 8), pre_infer_out)
        self.assertIs(seen["gate"], gate)

    def test_audio_forwards_c_gate_msa(self):
        ensure_lightx2v_pipeline_stub()
        ensure_local_lightx2v_kernel()
        audio = import_module("lightx2v.models.networks.wan.infer.audio.transformer_infer")

        infer = audio.WanAudioARTransformerInfer(make_config())
        gate = torch.ones(1, 8)
        infer.pre_process = lambda modulation, embed0: (gate, gate, gate, gate, gate, gate)
        infer.infer_self_attn_with_kvcache = lambda *args, **kwargs: torch.zeros(1, 8)
        infer.infer_cross_attn_with_kvcache = lambda *args, **kwargs: (torch.zeros(1, 8), torch.zeros(1, 8))
        seen = {}

        def fake_infer_ffn(phase, x, attn_out, c_shift, c_scale, c_gate=None):
            seen["gate"] = c_gate
            return torch.zeros_like(x)

        infer.infer_ffn = fake_infer_ffn
        infer.post_process = lambda x, y, c_gate, pre_infer_out=None: x

        block = SimpleNamespace(compute_phases=[SimpleNamespace(modulation=None), SimpleNamespace(), SimpleNamespace(), SimpleNamespace()])
        pre_infer_out = SimpleNamespace(
            x=torch.zeros(1, 8),
            embed0=torch.zeros(1, 6, 8),
            grid_sizes=SimpleNamespace(tensor=torch.ones(1, 3, dtype=torch.int32)),
            seq_lens=torch.ones(1, dtype=torch.int32),
            freqs=torch.zeros(1),
            context=torch.zeros(1, 8),
            adapter_args={"audio_encoder_output": None},
        )
        infer.infer_block_with_kvcache(block, torch.zeros(1, 8), pre_infer_out)
        self.assertIs(seen["gate"], gate)

    def test_feature_caching_variants_forward_c_gate_msa(self):
        ensure_lightx2v_pipeline_stub()
        ensure_local_lightx2v_kernel()
        feature_caching = import_module("lightx2v.models.networks.wan.infer.feature_caching.transformer_infer")

        cases = [
            (
                feature_caching.WanTransformerInferTaylorCaching,
                make_config(),
                lambda infer, weights, x, embed0: infer.infer_calculating(weights, None, None, x, embed0, None, None, None),
            ),
            (
                feature_caching.WanTransformerInferAdaCaching,
                make_config(),
                lambda infer, weights, x, embed0: infer.infer_calculating(weights, None, None, x, embed0, None, None, None),
            ),
            (
                feature_caching.WanTransformerInferCustomCaching,
                make_config(),
                lambda infer, weights, x, embed0: infer.infer_calculating(weights, None, None, x, embed0, None, None, None),
            ),
        ]

        for cls, config, runner in cases:
            with self.subTest(cls=cls.__name__):
                infer = cls(config)
                infer.scheduler = SimpleNamespace(infer_condition=True)
                infer.derivative_approximation = lambda *args, **kwargs: None
                gate = torch.ones(1, 8)
                infer.infer_modulation = lambda phase, embed0: (gate, gate, gate, gate, gate, gate)
                infer.infer_self_attn = lambda *args, **kwargs: torch.zeros(1, 8)
                infer.infer_cross_attn = lambda *args, **kwargs: (torch.zeros(1, 8), torch.zeros(1, 8))
                seen = {}

                def fake_infer_ffn(phase, x, attn_out, c_shift, c_scale, c_gate=None):
                    seen["gate"] = c_gate
                    return torch.zeros_like(x)

                infer.infer_ffn = fake_infer_ffn
                infer.post_process = lambda x, y, c_gate, pre_infer_out=None: x

                weights = SimpleNamespace(blocks=[SimpleNamespace(compute_phases=[SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), SimpleNamespace()])])
                runner(infer, weights, torch.zeros(1, 8), torch.zeros(1, 6, 8))
                self.assertIs(seen["gate"], gate)


if __name__ == "__main__":
    unittest.main()
