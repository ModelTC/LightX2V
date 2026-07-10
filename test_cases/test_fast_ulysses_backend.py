import unittest
import json
import sys
import types
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeFastUlyssesGroup:
    instances = []

    def __init__(self, world_size):
        self.world_size = world_size
        self.calls = []
        self.destroyed = False
        type(self).instances.append(self)

    def all_to_all_single_4d(self, x, *, mode=0, tag="", use_tma=None):
        self.calls.append((mode, tag, tuple(x.shape), use_tma))
        b, x1, x2, d = x.shape
        if mode == 0:
            return x.reshape(b, x1 * self.world_size, x2 // self.world_size, d)
        if mode == 1:
            return x.reshape(b, x1 // self.world_size, x2 * self.world_size, d)
        raise AssertionError(f"unexpected mode {mode}")

    def destroy(self):
        self.destroyed = True


class FakeAttention:
    def __init__(self):
        self.kwargs = None

    def apply(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["q"]


class FastUlyssesBackendTest(unittest.TestCase):
    def tearDown(self):
        from lightx2v.common.ops.attn.fast_ulysses_attn import FastUlyssesAttnWeight

        FastUlyssesAttnWeight._groups.clear()
        FastUlyssesAttnWeight._single_node_cache.clear()
        FastUlyssesAttnWeight.reset_runtime_stats()
        FakeFastUlyssesGroup.instances.clear()

    def test_fast_ulysses_backend_is_registered(self):
        from lightx2v.common.ops import attn  # noqa: F401
        from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

        self.assertIn("fast_ulysses", ATTN_WEIGHT_REGISTER)

    def test_unsupported_flags_fallback_before_native_import(self):
        from lightx2v.common.ops.attn.fast_ulysses_attn import FastUlyssesAttnWeight

        q = torch.zeros(4, 4, 8)
        fallback_out = torch.empty(1)
        cases = (
            ("use_fp8_comm", {"use_fp8_comm": True}),
            ("use_fp4_comm", {"use_fp4_comm": True}),
            ("use_tensor_fusion", {"use_tensor_fusion": True}),
            ("enable_head_parallel", {"enable_head_parallel": True}),
            ("q_only_img", {"q_only_img": True}),
            ("img_first", {"img_first": False}),
        )
        for message, kwargs in cases:
            with self.subTest(message=message):
                backend = FastUlyssesAttnWeight()
                with (
                    mock.patch.object(backend, "_get_group", side_effect=AssertionError("native path must not initialize")),
                    mock.patch("lightx2v.common.ops.attn.fast_ulysses_attn.UlyssesAttnWeight.apply", return_value=fallback_out) as fallback,
                ):
                    out = backend.apply(
                        q=q,
                        k=q,
                        v=q,
                        slice_qkv_len=4,
                        cu_seqlens_qkv=[0, 4],
                        attention_module=FakeAttention(),
                        seq_p_group=object(),
                        **kwargs,
                    )
                self.assertIs(out, fallback_out)
                fallback.assert_called_once()

    def test_unsupported_shapes_fallback_before_native_import(self):
        from lightx2v.common.ops.attn.fast_ulysses_attn import FastUlyssesAttnWeight

        backend = FastUlyssesAttnWeight()
        q = torch.zeros(4, 4, 8, dtype=torch.bfloat16, device="cuda")
        fallback_out = torch.empty(1)
        with (
            mock.patch.object(backend, "_get_group", side_effect=AssertionError("native path must not initialize")),
            mock.patch("lightx2v.common.ops.attn.fast_ulysses_attn.UlyssesAttnWeight.apply", return_value=fallback_out) as fallback,
        ):
            out = backend.apply(
                q=q,
                k=torch.zeros(4, 2, 8),
                v=torch.zeros(4, 2, 8),
                slice_qkv_len=4,
                cu_seqlens_qkv=[0, 4],
                attention_module=FakeAttention(),
                seq_p_group=object(),
            )
        self.assertIs(out, fallback_out)
        fallback.assert_called_once()

        with (
            mock.patch("torch.distributed.get_world_size", return_value=1),
            mock.patch.object(backend, "_get_group", side_effect=AssertionError("native path must not initialize")),
            mock.patch("lightx2v.common.ops.attn.fast_ulysses_attn.UlyssesAttnWeight.apply", return_value=fallback_out) as fallback,
        ):
            out = backend.apply(
                q=q,
                k=q,
                v=q,
                slice_qkv_len=2,
                cu_seqlens_qkv=[0, 4],
                attention_module=FakeAttention(),
                seq_p_group=object(),
            )
        self.assertIs(out, fallback_out)
        fallback.assert_called_once()

    def test_cross_node_detection_fallback_before_native_import(self):
        from lightx2v.common.ops.attn.fast_ulysses_attn import FastUlyssesAttnWeight

        backend = FastUlyssesAttnWeight()
        q = torch.zeros(4, 4, 8)
        fallback_out = torch.empty(1)
        with (
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch.object(backend, "_is_single_node", return_value=False),
            mock.patch.object(backend, "_get_group", side_effect=AssertionError("native path must not initialize")),
            mock.patch("lightx2v.common.ops.attn.fast_ulysses_attn.UlyssesAttnWeight.apply", return_value=fallback_out) as fallback,
        ):
            out = backend.apply(
                q=q,
                k=q,
                v=q,
                slice_qkv_len=4,
                cu_seqlens_qkv=[0, 4],
                attention_module=FakeAttention(),
                seq_p_group=object(),
            )
        self.assertIs(out, fallback_out)
        fallback.assert_called_once()

    def test_native_init_error_fallback(self):
        from lightx2v.common.ops.attn.fast_ulysses_attn import FastUlyssesAttnWeight

        backend = FastUlyssesAttnWeight()
        q = torch.zeros(4, 4, 8, dtype=torch.bfloat16, device="cuda")
        fallback_out = torch.empty(1)
        with (
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch.object(backend, "_is_single_node", return_value=True),
            mock.patch.object(backend, "_get_group", side_effect=ImportError("missing native extension")),
            mock.patch("lightx2v.common.ops.attn.fast_ulysses_attn.UlyssesAttnWeight.apply", return_value=fallback_out) as fallback,
        ):
            out = backend.apply(
                q=q,
                k=q,
                v=q,
                slice_qkv_len=4,
                cu_seqlens_qkv=[0, 4],
                attention_module=FakeAttention(),
                seq_p_group=object(),
            )
        self.assertIs(out, fallback_out)
        fallback.assert_called_once()

    def test_native_precondition_failures_fallback_before_native_import(self):
        from lightx2v.common.ops.attn.fast_ulysses_attn import FastUlyssesAttnWeight

        fallback_out = torch.empty(1)
        cases = (
            (
                torch.zeros(4, 4, 8),
                "cpu tensors",
                {},
            ),
            (
                torch.zeros(4, 4, 8, device="cuda"),
                "cuda float32 tensors",
                {},
            ),
            (
                torch.zeros(4, 4, 7, dtype=torch.bfloat16, device="cuda"),
                "unaligned head dim",
                {},
            ),
            (
                torch.zeros(4, 4, 8, dtype=torch.bfloat16, device="cuda"),
                "forced TMA on sm80",
                {"LIGHTX2V_FAST_ULYSSES_USE_TMA": "1"},
            ),
        )
        for q, name, env in cases:
            with self.subTest(name=name):
                backend = FastUlyssesAttnWeight()
                patches = [
                    mock.patch("torch.distributed.get_world_size", return_value=2),
                    mock.patch.object(backend, "_is_single_node", return_value=True),
                    mock.patch.object(backend, "_get_group", side_effect=AssertionError("native path must not initialize")),
                    mock.patch("lightx2v.common.ops.attn.fast_ulysses_attn.UlyssesAttnWeight.apply", return_value=fallback_out),
                    mock.patch.dict("os.environ", env, clear=False),
                ]
                if env:
                    patches.append(mock.patch("torch.cuda.get_device_capability", return_value=(8, 0)))
                with patches[0], patches[1], patches[2], patches[3] as fallback, patches[4]:
                    if env:
                        with patches[5]:
                            out = backend.apply(
                                q=q,
                                k=q,
                                v=q,
                                slice_qkv_len=4,
                                cu_seqlens_qkv=[0, 4],
                                attention_module=FakeAttention(),
                                seq_p_group=object(),
                            )
                    else:
                        out = backend.apply(
                            q=q,
                            k=q,
                            v=q,
                            slice_qkv_len=4,
                            cu_seqlens_qkv=[0, 4],
                            attention_module=FakeAttention(),
                            seq_p_group=object(),
                        )
                self.assertIs(out, fallback_out)
                fallback.assert_called_once()

    def test_validated_pure_image_path_wraps_a2a_and_attention(self):
        from lightx2v.common.ops.attn.fast_ulysses_attn import FastUlyssesAttnWeight

        q = torch.arange(4 * 4 * 8, device="cuda").reshape(4, 4, 8).to(torch.bfloat16)
        k = q + 1
        v = q + 2
        group = FakeFastUlyssesGroup(world_size=2)
        attention = FakeAttention()
        backend = FastUlyssesAttnWeight()

        with (
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.get_rank", return_value=0),
            mock.patch.object(backend, "_is_single_node", return_value=True),
            mock.patch.object(backend, "_get_group", return_value=group),
        ):
            out = backend.apply(
                q=q,
                k=k,
                v=v,
                slice_qkv_len=4,
                cu_seqlens_qkv=[0, 4],
                attention_module=attention,
                seq_p_group=object(),
                block_idx=7,
            )

        self.assertEqual(tuple(out.shape), (4, 32))
        self.assertEqual([call[0] for call in group.calls], [0, 0, 0, 1])
        self.assertEqual([call[1] for call in group.calls], ["lightx2v_q", "lightx2v_k", "lightx2v_v", "lightx2v_out"])
        self.assertEqual(tuple(attention.kwargs["q"].shape), (8, 2, 8))
        self.assertEqual(attention.kwargs["max_seqlen_q"], 8)
        self.assertEqual(attention.kwargs["max_seqlen_kv"], 8)
        self.assertEqual(FastUlyssesAttnWeight.runtime_stats()["fast_path_calls"], 1)
        self.assertEqual(FastUlyssesAttnWeight.runtime_stats()["fallback_calls"], 0)

    def test_group_cache_is_shared_across_backend_instances(self):
        from lightx2v.common.ops.attn.fast_ulysses_attn import FastUlyssesAttnWeight

        fake_module = types.SimpleNamespace(UlyssesGroup=lambda **kwargs: FakeFastUlyssesGroup(world_size=2))
        group = object()

        with (
            mock.patch.dict(sys.modules, {"lightx2v_fast_ulysses": fake_module}),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.distributed.get_process_group_ranks", return_value=[0, 1]),
        ):
            first = FastUlyssesAttnWeight()._get_group(group)
            second = FastUlyssesAttnWeight()._get_group(group)

        self.assertIs(first, second)
        self.assertEqual(len(FakeFastUlyssesGroup.instances), 1)

        FastUlyssesAttnWeight.destroy_cached_groups()
        self.assertTrue(first.destroyed)
        self.assertEqual(FastUlyssesAttnWeight._groups, {})

    def test_group_cache_uses_explicit_tensor_device(self):
        from lightx2v.common.ops.attn.fast_ulysses_attn import FastUlyssesAttnWeight

        captured = []

        def make_group(**kwargs):
            captured.append(kwargs)
            return FakeFastUlyssesGroup(world_size=2)

        fake_module = types.SimpleNamespace(UlyssesGroup=make_group)
        group = object()
        device = torch.device("cuda", 1)

        with (
            mock.patch.dict(sys.modules, {"lightx2v_fast_ulysses": fake_module}),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.distributed.get_process_group_ranks", return_value=[0, 1]),
        ):
            first = FastUlyssesAttnWeight()._get_group(group, device=device)
            second = FastUlyssesAttnWeight()._get_group(group, device=device)

        self.assertIs(first, second)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["device"], device)
        self.assertIn(((0, 1), 1), FastUlyssesAttnWeight._groups)

    def test_native_package_surface_and_build_are_a2a_only(self):
        root = Path(__file__).resolve().parents[1] / "lightx2v_fast_ulysses"
        init_text = (root / "__init__.py").read_text()
        comm_text = (root / "comm.py").read_text()
        cmake_text = (root / "CMakeLists.txt").read_text()
        bindings_text = (root / "csrc" / "bindings.cpp").read_text()

        csrc_text = "\n".join(path.read_text() for path in (root / "csrc").glob("*"))
        for name in ("rms_norm", "norm_rope", "all_to_all_single_4d_qk", "config_key_qk", "qk_norm_rope", "all_to_all_qk"):
            self.assertNotIn(name, init_text)
            self.assertNotIn(name, comm_text)
            self.assertNotIn(name, bindings_text)
            self.assertNotIn(name, csrc_text)
        self.assertNotIn("file(GLOB", cmake_text)
        self.assertNotIn("all_to_all_qk.cu", cmake_text)
        self.assertNotIn("qk_norm_rope.cu", cmake_text)

    def test_wan_i2v_fast_config_uses_fast_ulysses(self):
        root = Path(__file__).resolve().parents[1]
        fast_config = root / "configs" / "dist_infer" / "wan_i2v_dist_cfg_fast_ulysses.json"
        cfg = json.loads(fast_config.read_text())
        self.assertEqual(cfg["parallel"]["seq_p_size"], 4)
        self.assertEqual(cfg["parallel"]["cfg_p_size"], 1)
        self.assertEqual(cfg["parallel"]["seq_p_attn_type"], "fast_ulysses")


if __name__ == "__main__":
    unittest.main()
