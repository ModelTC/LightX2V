"""CPU-only merge regression tests; do not import the LightX2V package."""

import ast
import builtins
import inspect
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ATTN = Path(__file__).resolve().parents[1] / "lightx2v/common/ops/attn"


def load_attention(filename, dense=None, varlen=None, metadata=None, import_error=ImportError):
    registrations = {}
    ops = {}

    def register(name):
        def decorate(cls):
            if name in registrations:
                raise AssertionError(f"Duplicate attention registration: {name}")
            registrations[name] = cls
            return cls

        return decorate

    def custom_op(name, **options):
        def decorate(fn):
            if name in ops:
                raise AssertionError(f"Duplicate custom op: {name}")
            ops[name] = fn
            fn.options = options

            def register_fake(fake):
                fn.fake = fake
                return fake

            fn.register_fake = register_fake
            return fn

        return decorate

    dependencies = {
        "lightx2v.utils.registry_factory": types.SimpleNamespace(ATTN_WEIGHT_REGISTER=register),
        "template": types.SimpleNamespace(AttnWeightTemplate=object),
        "utils.sla_util": types.SimpleNamespace(
            get_block_map=Mock(),
            get_cuda_arch=lambda _: "sm110",
            block_lut_to_ordinal_metadata=Mock(),
        ),
        "utils.sla_util_blhd": types.SimpleNamespace(get_block_lut_blhd=Mock(), get_block_map_blhd=Mock()),
        "utils.sparge_util": types.SimpleNamespace(
            block_map_incremental_lut_triton=Mock(),
            block_map_ordinal_lut_triton=Mock(),
            sage2_block_sparse_attn=Mock(),
            get_block_map_meansim=Mock(),
        ),
        "kernels.sla_kernel": types.SimpleNamespace(_attention=Mock()),
        "kernels.sla_kernel_ar": types.SimpleNamespace(_attention_ar=Mock()),
    }
    real_import = builtins.__import__

    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in dependencies:
            return dependencies[name]
        if name == "flash_attn.cute":
            value = dense if fromlist[0] == "flash_attn_func" else varlen
            if value is None:
                raise import_error(fromlist[0])
            return types.SimpleNamespace(**{fromlist[0]: value})
        if name == "flash_attn.cute.block_sparsity":
            if metadata is None:
                raise import_error("BlockSparseTensorsTorch")
            return types.SimpleNamespace(BlockSparseTensorsTorch=metadata)
        if name.startswith(("flash_attn", "sageattn3_sparse", "magi_attention")):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    namespace = {"__name__": "attention_merge_test", "__builtins__": dict(vars(builtins), __import__=isolated_import)}
    with patch.object(torch.library, "custom_op", custom_op), patch.object(torch.compiler, "disable", lambda f: f):
        exec(compile((ATTN / filename).read_text(), str(ATTN / filename), "exec"), namespace)  # noqa: S102 - trusted local source
    return types.SimpleNamespace(**namespace), registrations, ops


def load_helpers():
    tree = ast.parse((ATTN / "utils/sla_util.py").read_text())
    names = {"_get_block_lut", "block_lut_to_ordinal_metadata", "get_block_lut", "get_block_map"}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch}
    exec(compile(tree, str(ATTN / "utils/sla_util.py"), "exec"), namespace)  # noqa: S102 - trusted local helpers
    return namespace


class FlashAttentionMergeTests(unittest.TestCase):
    def setUp(self):
        self.q = torch.randn(5, 2, 4)
        self.k = torch.randn(7, 2, 4)
        self.dense = Mock(side_effect=lambda q, k, v, **kw: (q.clone(), None))
        self.varlen = Mock(side_effect=lambda q, k, v, cu, **kw: (q.clone(), None))

    def test_dense_single_sequence_without_varlen_or_metadata(self):
        for shape4 in (False, True):
            for error in (ImportError, AttributeError):
                with self.subTest(shape4=shape4, error=error):
                    mod, registered, _ = load_attention("flash_attn.py", dense=self.dense, import_error=error)
                    q = self.q.unsqueeze(0) if shape4 else self.q
                    k = self.k.unsqueeze(0) if shape4 else self.k
                    out = mod.FlashAttn4Weight().apply(q, k, k, causal=True, softmax_scale=0.3)
                    self.assertEqual(out.shape, (5, 8))
                    self.assertEqual(self.dense.call_args.args[0].shape, (1, 5, 2, 4))
                    self.assertEqual(self.dense.call_args.kwargs, {"causal": True, "softmax_scale": 0.3})
                    self.assertEqual(len(registered), 4)
                    self.assertIsNone(mod.BlockSparseTensorsTorch)

    def test_single_sequence_with_cumulative_lengths_preserves_varlen(self):
        for q_end, k_end in ((5, 7), (3, 4)):
            with self.subTest(q_end=q_end, k_end=k_end):
                mod, _, _ = load_attention("flash_attn.py", self.dense, self.varlen)
                cuq, cuk = torch.tensor([0, q_end]), torch.tensor([0, k_end])
                out = mod.FlashAttn4Weight().apply(self.q, self.k, self.k, cuq, cuk, q_end, k_end)
                self.assertEqual(out.shape, (5, 8))
                self.assertIs(self.varlen.call_args.args[3], cuq)
                self.assertIs(self.varlen.call_args.kwargs["cu_seqlens_k"], cuk)
                self.dense.assert_not_called()

    def test_single_sequence_with_lengths_requires_varlen(self):
        mod, _, _ = load_attention("flash_attn.py", self.dense)
        with self.assertRaisesRegex(RuntimeError, "varlen"):
            mod.FlashAttn4Weight().apply(self.q, self.k, self.k, torch.tensor([0, 3]), torch.tensor([0, 4]), 3, 4)
        self.dense.assert_not_called()

    def test_packed_unequal_sequences_use_varlen_and_total_tokens(self):
        mod, _, _ = load_attention("flash_attn.py", self.dense, self.varlen)
        cuq, cuk = torch.tensor([0, 2, 5]), torch.tensor([0, 3, 7])
        out = mod.FlashAttn4Weight().apply(self.q, self.k, self.k, cuq, cuk, 3, 4, causal=True, softmax_scale=0.2)
        self.assertEqual(out.shape, (5, 8))
        self.dense.assert_not_called()
        args, kw = self.varlen.call_args
        self.assertIs(args[3], cuq)
        self.assertIs(kw["cu_seqlens_k"], cuk)
        self.assertEqual((kw["max_seqlen_q"], kw["max_seqlen_k"]), (3, 4))
        self.assertEqual((kw["causal"], kw["softmax_scale"]), (True, 0.2))

    def test_packed_input_never_falls_back_to_dense(self):
        mod, _, _ = load_attention("flash_attn.py", self.dense)
        with self.assertRaisesRegex(RuntimeError, "varlen"):
            mod.FlashAttn4Weight().apply(self.q, self.k, self.k, torch.tensor([0, 2, 5]), torch.tensor([0, 3, 7]), 3, 4)
        self.dense.assert_not_called()

    def test_batched_4d_varlen_and_missing_lengths(self):
        mod, _, _ = load_attention("flash_attn.py", self.dense, self.varlen)
        q = torch.randn(2, 3, 2, 4)
        cu = torch.tensor([0, 3, 6])
        out = mod.FlashAttn4Weight().apply(q, q, q, cu, cu, 3, 3)
        self.assertEqual(out.shape, (6, 8))
        self.assertEqual(self.varlen.call_args.args[0].shape, (6, 2, 4))
        with self.assertRaises(ValueError):
            mod.FlashAttn4Weight().apply(q, q, q)
        with self.assertRaises(ValueError):
            mod.FlashAttn4Weight().apply(self.q, self.k, self.k, cu_seqlens_q=cu)

    def test_varlen_remains_available_without_dense(self):
        mod, _, _ = load_attention("flash_attn.py", varlen=self.varlen, import_error=AttributeError)
        out = mod.FlashAttn4Weight().apply(self.q, self.k, self.k, torch.tensor([0, 5]), torch.tensor([0, 7]), 5, 7)
        self.assertEqual(out.shape, (5, 8))
        self.varlen.assert_called_once()

    def test_fa3_lse_stays_on_fa3_class(self):
        mod, _, _ = load_attention("flash_attn.py")
        lse = torch.arange(10).reshape(1, 2, 5)
        fn = Mock(return_value=(self.q.unsqueeze(0), lse, None))
        mod.FlashAttn3Weight.apply_with_lse.__globals__["flash_attn_func_v3"] = fn
        out, actual_lse = mod.FlashAttn3Weight().apply_with_lse(self.q, self.q, self.q, 0.5)
        self.assertEqual(out.shape, (5, 8))
        torch.testing.assert_close(actual_lse, lse.transpose(1, 2).reshape(5, 2))
        self.assertTrue(fn.call_args.kwargs["return_attn_probs"])
        self.assertFalse(hasattr(mod.FlashAttn4Weight, "apply_with_lse"))


class DynamicSparseMergeTests(unittest.TestCase):
    def test_sparse_apis_eager_and_compile_boundary(self):
        for api in ("expanded", "block_sparse_tensors"):
            for error in (ImportError, AttributeError):
                with self.subTest(api=api, error=error):
                    calls = []

                    def expanded(q, k, v, mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx, block_size, calls=calls):
                        calls.append((full_block_cnt, full_block_idx, block_size))
                        return q.clone(), None

                    def bundled(q, k, v, block_sparse_tensors, calls=calls):
                        calls.append((block_sparse_tensors.full_block_cnt, block_sparse_tensors.full_block_idx, block_sparse_tensors.block_size))
                        return q.clone(), None

                    fn = expanded if api == "expanded" else bundled
                    metadata = None if api == "expanded" else types.SimpleNamespace
                    mod, _, ops = load_attention("dynamic_sparse_attn.py", dense=fn, metadata=metadata, import_error=error)
                    self.assertEqual(mod._FA4_SPARSE_API, api)
                    self.assertEqual(inspect.signature(mod.flash_attn_func_v4), inspect.signature(fn))
                    helpers = load_helpers()
                    ns = mod.DynamicSparseAttnWeight.apply_fa4.__globals__
                    ns["get_block_lut_blhd"] = Mock(return_value=(torch.zeros(1, 2, 1, 1, dtype=torch.long), 1, 1))
                    ns["block_lut_to_ordinal_metadata"] = helpers["block_lut_to_ordinal_metadata"]
                    weight = object.__new__(mod.DynamicSparseAttnWeight)
                    weight.topk, weight.BLKQ, weight.BLKK = 0.2, 128, 128
                    q = torch.randn(5, 2, 4)
                    for compiling in (False, True):
                        with patch.object(torch.compiler, "is_compiling", return_value=compiling):
                            out = weight.apply_fa4(q, q, q, max_seqlen_q=5)
                        torch.testing.assert_close(out, q.reshape(5, 8))
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(calls[0][2], (128, 128))
                    torch.testing.assert_close(calls[0][0], calls[1][0])
                    op = ops["lightx2v_internal::fa4_blocksparse"]
                    fake = op.fake(q, q, q, None, None, None, None, 128, 128)
                    self.assertEqual(fake.shape, q.shape)
                    self.assertEqual(fake.dtype, q.dtype)

    def test_sage2_custom_op_and_apply(self):
        mod, registered, ops = load_attention("dynamic_sparse_attn.py")
        self.assertEqual(list(registered), ["dynamic_sparse_attn"])
        op = ops["lightx2v::dynamic_sparse_sage2"]
        self.assertEqual(op.options, {"mutates_args": (), "device_types": "cuda"})
        q = torch.randn(5, 2, 4)
        ns = op.__globals__
        sparse_map, lut, counts = object(), object(), object()
        ns["get_block_map"] = Mock(return_value=(sparse_map, None, None))
        ns["block_map_incremental_lut_triton"] = Mock(return_value=(lut, counts))
        ns["sage2_block_sparse_attn"] = Mock(side_effect=lambda q, *args: q.clone())
        weight = object.__new__(mod.DynamicSparseAttnWeight)
        weight.topk, weight.BLKQ, weight.BLKK, weight.arch = 0.2, 128, 64, "sm110"
        out = weight.apply_sage2(q, q, q, max_seqlen_q=5)
        torch.testing.assert_close(out, q.reshape(5, 8))
        ns["block_map_incremental_lut_triton"].assert_called_once_with(sparse_map)
        args = ns["sage2_block_sparse_attn"].call_args.args
        self.assertEqual(args[0].shape, (1, 2, 5, 4))
        self.assertTrue(args[0].is_contiguous())
        self.assertEqual(args[3:], (lut, counts, 128, 64, "sm110"))
        fake = op.fake(q, q, q, 0.2, 128, 64, "sm110")
        self.assertEqual((fake.shape, fake.dtype, fake.device), (q.shape, q.dtype, q.device))

    def test_cutedsl_fp8_retains_blhd_selection(self):
        mod, _, _ = load_attention("dynamic_sparse_attn.py")
        weight = object.__new__(mod.DynamicSparseAttnWeight)
        weight.topk, weight.BLKQ, weight.BLKK = 0.2, 256, 128
        weight.cutedsl_sparse_fmha = Mock(side_effect=lambda q, *args, **kw: q.to(torch.bfloat16))
        ns = mod.DynamicSparseAttnWeight.apply_cutedsl_fp8.__globals__
        pool = Mock(return_value=(torch.zeros(1, 2, 1, 1, dtype=torch.long), 1, 1))
        ns["get_block_lut_blhd"] = pool
        ns["block_lut_to_ordinal_metadata"] = load_helpers()["block_lut_to_ordinal_metadata"]
        q = torch.ones(5, 2, 4, dtype=torch.bfloat16)
        out = weight.apply_cutedsl_fp8(q, q, q)
        self.assertEqual(pool.call_args.args[0].shape, (1, 5, 2, 4))
        self.assertEqual(pool.call_args.args[0].dtype, torch.bfloat16)
        args, kw = weight.cutedsl_sparse_fmha.call_args
        self.assertTrue(all(t.dtype == torch.float8_e4m3fn for t in args[:3]))
        self.assertEqual(kw["output_dtype"], torch.bfloat16)
        self.assertEqual(out.shape, (5, 8))


class SparseHelperMergeTests(unittest.TestCase):
    def test_short_sequences_keep_one_block_and_gqa(self):
        helpers = load_helpers()
        q, k = torch.ones(1, 4, 2, 4), torch.ones(1, 2, 1, 4)
        for ratio in (0.0, 0.01, 0.2, 1.0):
            lut, topk, count = helpers["_get_block_lut"](q, k, ratio)
            self.assertEqual((topk, count), (1, 1))
            self.assertEqual(lut.shape, (1, 4, 2, 1))
            indices, counts = helpers["block_lut_to_ordinal_metadata"](lut, count)
            self.assertEqual(indices.dtype, torch.int32)
            self.assertTrue(torch.all(counts == 1))
        helpers["mean_pool"] = lambda x, block: x.mean(-2, keepdim=True)
        sparse_map, _, topk = helpers["get_block_map"](q, k, 0.01)
        self.assertEqual(topk, 1)
        self.assertTrue(torch.all(sparse_map == 1))

    def test_syntax_and_unique_definitions(self):
        for path in (ATTN / "dynamic_sparse_attn.py", ATTN / "flash_attn.py", ATTN / "utils/sla_util.py"):
            tree = ast.parse(path.read_text())
            definitions = [node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
            self.assertEqual(len(definitions), len(set(definitions)), str(path))
            for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
                methods = [node.name for node in cls.body if isinstance(node, ast.FunctionDef)]
                self.assertEqual(len(methods), len(set(methods)), cls.name)


if __name__ == "__main__":
    unittest.main()
