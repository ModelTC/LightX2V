"""Dependency-free routing tests for the Wan/Thor merge (no torch imports)."""

import ast
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
MM = ROOT / "lightx2v/common/ops/mm/mm_weight.py"
WEIGHTS = ROOT / "lightx2v/models/networks/wan/weights/transformer_weights.py"
INFER = ROOT / "lightx2v/models/networks/wan/infer/transformer_infer.py"


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    # Execute only selected definitions from local source files, without package imports.
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102
    return namespace


class Registry(dict):
    def __call__(self, name):
        def register(cls):
            self[name] = cls
            return cls

        return register


class WeightModule:
    def __init__(self, *args, **kwargs):
        pass

    def add_module(self, name, module):
        setattr(self, name, module)


class Tensor:
    shape = (4, 2, 8)

    def squeeze(self):
        return self

    def view(self, *shape):
        return self


class WanMergeTests(unittest.TestCase):
    def setUp(self):
        self.registry = Registry({name: Mock(name=name) for name in ("Default", "Calib", "nvfp4", "nvfp4-split-n-workaround", "nvfp4-split-n-stride-workaround", "TensorParallel")})
        self.namespace = load_definitions(
            WEIGHTS,
            {"_mm_weight", "WanFFN"},
            {
                "MM_WEIGHT_REGISTER": self.registry,
                "LN_WEIGHT_REGISTER": {"torch": Mock()},
                "WeightModule": WeightModule,
                "dist": SimpleNamespace(get_rank=lambda group: 1, get_world_size=lambda group: 2),
            },
        )

    def test_override_precedes_calibration_and_forwards_kwargs(self):
        mm_weight = self.namespace["_mm_weight"]
        config = {"dit_quant_scheme": "nvfp4", "do_mm_calib": True}
        mm_weight(config, "w", "b")
        self.registry["Calib"].assert_called_once()
        options = {"split_n_parts": 2}
        mm_weight(config, "w", "b", mm_type_override="nvfp4-split-n-stride-workaround", mm_kwargs=options)
        self.assertEqual(self.registry["nvfp4-split-n-stride-workaround"].call_args.kwargs["split_n_parts"], 2)
        self.assertEqual(options, {"split_n_parts": 2})

    def test_tp_helper_forwards_override_and_kwargs(self):
        group = object()
        config = {"tensor_parallel": True, "device_mesh": SimpleNamespace(get_group=lambda **kwargs: group), "do_mm_calib": True}
        options = {"split_n_parts": 2}
        self.namespace["_mm_weight"](config, "w", "b", split_dim="col", mm_type_override="nvfp4", mm_kwargs=options)
        kwargs = self.registry["TensorParallel"].call_args.kwargs
        self.assertEqual(kwargs["mm_type"], "nvfp4")
        self.assertIs(kwargs["mm_kwargs"], options)
        self.assertIs(kwargs["tp_group"], group)
        self.assertEqual((kwargs["tp_rank"], kwargs["tp_size"]), (1, 2))

    def test_ffn_registry_routing(self):
        for thor, split_n, mm_type, expected in (
            (True, False, "nvfp4", "nvfp4-split-n-stride-workaround"),
            (True, True, "nvfp4", "nvfp4-split-n-stride-workaround"),
            (False, True, "nvfp4", "nvfp4-split-n-workaround"),
            (False, False, "nvfp4", "nvfp4"),
            (False, True, "Default", "Default"),
            (False, True, "Calib", "Calib"),
        ):
            with self.subTest(thor=thor, split_n=split_n, mm_type=mm_type):
                for factory in self.registry.values():
                    factory.reset_mock()
                config = {"thor": thor, "nvfp4_ffn_split_n_workaround": split_n, "dit_quant_scheme": mm_type}
                self.namespace["WanFFN"](0, "blocks", "t2v", mm_type, config)
                factory = self.registry[expected]
                self.assertEqual(factory.call_count, 2)
                for call in factory.call_args_list:
                    self.assertEqual(call.kwargs.get("split_n_parts"), 2 if thor else None)
                self.assertEqual(sum(factory.call_count for factory in self.registry.values()), 2)

    def test_split_n_requires_boolean(self):
        with self.assertRaises(TypeError):
            self.namespace["WanFFN"](0, "blocks", "t2v", "nvfp4", {"nvfp4_ffn_split_n_workaround": 1})

    def test_tp_positional_lora_chunks_and_mm_kwargs_coexist(self):
        inner = Mock()
        registry = Registry({"nvfp4": inner})
        namespace = load_definitions(
            MM,
            {"MMWeightTP"},
            {
                "MMWeightTemplate": WeightModule,
                "MMWeight": Mock(),
                "MM_WEIGHT_REGISTER": registry,
            },
        )
        cls = namespace["MMWeightTP"]
        params = list(inspect.signature(cls).parameters)
        self.assertEqual(params[-3:], ["reduce_output", "lora_column_chunks", "mm_kwargs"])
        weight = cls("w", "b", "nvfp4", None, 0, 2, "col", False, False, False, None, False, "prefix", "lora", False, 2, mm_kwargs={"split_n_parts": 2})
        self.assertEqual(weight.lora_column_chunks, 2)
        self.assertFalse(weight.reduce_output)
        self.assertEqual(inner.call_args.kwargs["split_n_parts"], 2)
        self.assertEqual(inner.call_args.kwargs["lora_path"], "lora")
        self.assertNotIn("lora_column_chunks", inner.call_args.kwargs)

    def test_split_n_registries_and_single_fp8_registration(self):
        registry = Registry()
        load_definitions(
            MM,
            {"MMWeightWnvfp4Anvfp4dynamicSplitNWorkaround", "MMWeightWnvfp4Anvfp4dynamicSplitNStrideWorkaround"},
            {
                "MM_WEIGHT_REGISTER": registry,
                "MMWeightWnvfp4Anvfp4dynamic": WeightModule,
            },
        )
        self.assertEqual(set(registry), {"nvfp4-split-n-workaround", "nvfp4-split-n-stride-workaround"})
        self.assertEqual(registry["nvfp4-split-n-stride-workaround"]("w", "b", split_n_parts=2).split_n_parts, 2)
        tree = ast.parse(MM.read_text())
        wrappers = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_fp8_scaled_mm"]
        self.assertEqual(len(wrappers), 1)
        self.assertIn("sgl_fp8_scaled_mm_meta", ast.unparse(wrappers[0]))
        self.assertIn("return sgl_fp8_scaled_mm(", ast.unparse(wrappers[0]))

    def test_attention_kwargs_reach_all_routes_with_thor_qkv(self):
        namespace = load_definitions(
            INFER,
            {"WanTransformerInfer"},
            {
                "WanMxfp8FuseMixin": type("Mixin", (), {}),
                "BaseTransformerInfer": object,
                "torch": SimpleNamespace(no_grad=lambda: lambda method: method, equal=lambda a, b: a == b),
            },
        )
        cls = namespace["WanTransformerInfer"]
        tensor = Tensor()
        for thor in (False, True):
            for route in ("local", "new", "legacy"):
                with self.subTest(thor=thor, route=route):
                    infer = cls.__new__(cls)
                    infer.__dict__.update(
                        thor=thor,
                        cos_sin=None,
                        rope_positions=None,
                        sensitive_layer_dtype="bf16",
                        infer_dtype="bf16",
                        num_heads=2,
                        head_dim=8,
                        clean_cuda_cache=False,
                        block_idx=3,
                        scheduler=object(),
                        _sol_morton_preordered=False,
                        seq_parallel=route != "local",
                        use_new_seq_p_interface=route == "new",
                        self_attn_cu_seqlens_qkv=object(),
                        seq_p_group=None,
                        seq_p_prepost_backend="torch",
                        seq_p_a2a_backend="torch",
                        seq_p_quant_scheme=None,
                        seq_p_tensor_fusion=False,
                        seq_p_head_parallel=False,
                        seq_p_fp8_comm=False,
                        seq_p_fp4_comm=False,
                        seq_p_configured_quant_scheme=None,
                        has_post_adapter=False,
                    )
                    infer._use_mxfp8_quant_fuse = lambda: False
                    infer._can_reuse_self_attn_mxfp8_quant = lambda *args: False
                    infer.modulate_func = lambda *args, **kwargs: tensor

                    def projection():
                        return SimpleNamespace(
                            apply=Mock(return_value=tensor),
                            apply_cublaslt=Mock(return_value=tensor),
                            apply_quantized_cublaslt=Mock(return_value=tensor),
                            act_quant_func=Mock(return_value=(tensor, tensor)),
                            input_global_scale=1,
                        )

                    phase = SimpleNamespace(
                        modulation=object(),
                        norm1=projection(),
                        self_attn_q=projection(),
                        self_attn_k=projection(),
                        self_attn_v=projection(),
                        self_attn_o=projection(),
                        self_attn_norm_q=projection(),
                        self_attn_norm_k=projection(),
                        rope=SimpleNamespace(apply=Mock(return_value=(tensor, tensor))),
                        self_attn_1=projection(),
                        self_attn_1_parallel=SimpleNamespace(apply=Mock(return_value=tensor), apply_new=Mock(return_value=(tensor, None))),
                    )
                    infer.pre_process = lambda *args: (tensor,) * 6
                    infer.infer_cross_attn = lambda *args: (tensor, tensor)
                    infer.infer_ffn = lambda *args: None
                    pre = SimpleNamespace(embed0=tensor, context=tensor, grid_sizes=SimpleNamespace(tuple=(1, 2, 2)))
                    marker = object()
                    options = {"cache_state": marker, "block_idx": 99}
                    self.assertIs(infer.infer_block(SimpleNamespace(compute_phases=[phase, object(), object()]), tensor, pre, options), tensor)
                    if route == "new":
                        kwargs = phase.self_attn_1_parallel.apply_new.call_args.kwargs["attention_kwargs"]
                    else:
                        attention = phase.self_attn_1 if route == "local" else phase.self_attn_1_parallel
                        kwargs = attention.apply.call_args.kwargs
                    self.assertIs(kwargs["cache_state"], marker)
                    self.assertEqual(kwargs["block_idx"], 99)
                    self.assertEqual(kwargs["grid_sizes"], (1, 2, 2))
                    self.assertEqual(options, {"cache_state": marker, "block_idx": 99})
                    if thor:
                        phase.self_attn_q.act_quant_func.assert_called_once()
                        phase.self_attn_k.act_quant_func.assert_not_called()
                        phase.self_attn_v.act_quant_func.assert_not_called()
                        for proj in (phase.self_attn_q, phase.self_attn_k, phase.self_attn_v):
                            proj.apply_quantized_cublaslt.assert_called_once_with(tensor, tensor, -1)
                        phase.self_attn_o.apply_cublaslt.assert_called_once_with(tensor, -1)
                    else:
                        phase.self_attn_o.apply.assert_called_once_with(tensor)


if __name__ == "__main__":
    unittest.main()
