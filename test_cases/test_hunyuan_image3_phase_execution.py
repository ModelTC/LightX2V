import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lightx2v.models.networks.hunyuan_image3.infer.post_infer import HunyuanImage3PostInfer
from lightx2v.models.networks.hunyuan_image3.infer.transformer_infer import HunyuanImage3TransformerInfer
from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner
from lightx2v.utils.set_config import _normalize_hunyuan_image3_phase_parallel


class _ParallelContext:
    def __init__(self, phase="denoise", tp_size=2, seq_size=2):
        self.phase = phase
        self.active_tp_group = object()
        self.active_tp_rank = 0
        self.active_tp_size = tp_size
        self.logical_tp_rank = 0
        self.active_seq_group = object() if seq_size > 1 else None
        self.active_seq_size = seq_size
        self.active_seq_parallel = seq_size > 1
        self.logical_gather_order = (0, 2, 1, 3) if tp_size == 4 else tuple(range(tp_size))
        self.ar_tp_size = 4
        self.activations = []

    def activate_phase(self, phase):
        self.phase = phase
        self.activations.append(phase)


class _Linear:
    weight = None
    bias = None

    def __init__(self, fn):
        self._fn = fn

    def apply(self, value):
        return self._fn(value)


class HunyuanImage3PhaseExecutionTests(unittest.TestCase):
    def test_multi_micro_config_requires_the_production_topology(self):
        def parallel(ar_tp=4, denoise_sp=2):
            return {
                "phase_aware": True,
                "storage_tensor_p_size": 2,
                "ar": {"tensor_p_size": ar_tp, "seq_p_size": 1},
                "denoise": {"tensor_p_size": 2, "seq_p_size": denoise_sp},
            }

        config = {
            "moe_impl": "flashinfer",
            "flashinfer_multi_micro": True,
            "flashinfer_multi_micro_backend": "grouped_mm",
        }
        normalized = parallel()
        self.assertTrue(_normalize_hunyuan_image3_phase_parallel(config, normalized))
        self.assertEqual(normalized["micro_shard_count"], 2)

        invalid_cases = (
            ({**config, "moe_impl": "eager"}, parallel(), "moe_impl='flashinfer'"),
            ({**config, "flashinfer_multi_micro_backend": "auto"}, parallel(), "backend='grouped_mm'"),
            (config, parallel(ar_tp=6, denoise_sp=3), "exactly two micro shards"),
            (config, {}, "parallel.phase_aware=true"),
        )
        for invalid_config, invalid_parallel, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _normalize_hunyuan_image3_phase_parallel(invalid_config, invalid_parallel)

    def test_runner_activates_ar_and_denoise_before_stage_work(self):
        context = _ParallelContext()
        runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
        runner.config = {"parallel_context": context}
        runner._resolve_bot_task = lambda: "image"

        self.assertIsNone(runner._generate_cot_text("prompt", (1024, 1024)))
        self.assertEqual(context.activations, ["ar"])

        class _DenoiseReached(RuntimeError):
            pass

        def stop_after_activation(_prepared_inputs):
            raise _DenoiseReached

        runner._resolve_denoise_cfg_mode = stop_after_activation
        with self.assertRaises(_DenoiseReached):
            runner._denoise_latents({}, (1024, 1024))
        self.assertEqual(context.activations, ["ar", "denoise"])

    def test_sp_cfg_requires_serial_mode_without_cfg_parallel(self):
        context = _ParallelContext(phase="denoise", tp_size=2, seq_size=2)
        runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
        runner.config = {
            "parallel_context": context,
            "parallel": {"cfg_mode": "batch"},
            "cfg_parallel": False,
            "enable_cfg": True,
        }
        runner.model = SimpleNamespace()

        with self.assertRaisesRegex(ValueError, "cfg_mode='serial'"):
            runner._resolve_denoise_cfg_mode({"do_cfg": True})

        runner.config["parallel"]["cfg_mode"] = "serial"
        self.assertEqual(runner._resolve_denoise_cfg_mode({"do_cfg": True}), "serial")

    def test_flashinfer_cache_isolated_by_phase_and_legacy_is_unchanged(self):
        context = _ParallelContext(phase="ar", tp_size=4, seq_size=1)
        runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
        runner.config = {"parallel_context": context, "flashinfer_autotune_cache": "results/cache.json"}

        ar_config = runner._flashinfer_autotune_config()
        self.assertEqual(Path(ar_config["flashinfer_autotune_cache"]), Path("results/cache_ar.json"))
        context.phase = "denoise"
        denoise_config = runner._flashinfer_autotune_config()
        self.assertEqual(Path(denoise_config["flashinfer_autotune_cache"]), Path("results/cache_denoise.json"))

        runner.config["flashinfer_autotune_cache_by_phase"] = {"ar": "ar.json", "denoise": "denoise.json"}
        self.assertEqual(runner._flashinfer_autotune_config()["flashinfer_autotune_cache"], "denoise.json")

        runner.config.update(flashinfer_multi_micro=True, flashinfer_autotune_mode="auto")
        multi_micro_denoise_config = runner._flashinfer_autotune_config()
        self.assertIsNot(multi_micro_denoise_config, runner.config)
        self.assertEqual(multi_micro_denoise_config["flashinfer_autotune_mode"], "off")
        context.phase = "ar"
        multi_micro_ar_config = runner._flashinfer_autotune_config()
        self.assertEqual(multi_micro_ar_config["flashinfer_autotune_mode"], "auto")
        self.assertEqual(multi_micro_ar_config["flashinfer_autotune_cache"], "ar.json")

        legacy_config = {"flashinfer_autotune_cache": "results/cache.json"}
        runner.config = legacy_config
        self.assertIs(runner._flashinfer_autotune_config(), legacy_config)

    def test_post_infer_restores_canonical_vocab_shard_order(self):
        context = _ParallelContext(phase="ar", tp_size=4, seq_size=1)
        infer = HunyuanImage3PostInfer({"parallel_context": context})
        weights = SimpleNamespace(
            final_norm=_Linear(lambda value: value),
            lm_head=_Linear(lambda value: torch.zeros(value.shape[0], 1, dtype=value.dtype)),
        )
        pre_infer_out = SimpleNamespace(image_mask=None, timesteps=None, token_hw=None)

        def fake_all_gather(outputs, _local, group):
            self.assertIs(group, context.active_tp_group)
            for physical_rank, output in enumerate(outputs):
                output.fill_(physical_rank)

        with patch("lightx2v.models.networks.hunyuan_image3.infer.post_infer.dist.all_gather", side_effect=fake_all_gather):
            output = infer.infer(weights, torch.ones(1, 3, 1), pre_infer_out)["logits"]
        self.assertEqual(output.flatten().tolist(), [0.0, 2.0, 1.0, 3.0])

    def test_attention_head_partition_tracks_active_tp_size(self):
        context = _ParallelContext(phase="ar", tp_size=4, seq_size=1)
        infer = HunyuanImage3TransformerInfer.__new__(HunyuanImage3TransformerInfer)
        infer.parallel_context = context
        infer.tp_group = None
        infer.tp_rank = 0
        infer.tp_size = 1
        infer.global_num_heads = 32
        infer.global_num_key_value_heads = 8
        infer.num_key_value_groups = 4
        infer.head_dim = 2
        captured_heads = []

        def qkv_projection(value):
            local_kv_heads = infer.global_num_key_value_heads // context.active_tp_size
            width = local_kv_heads * (infer.num_key_value_groups + 2) * infer.head_dim
            return torch.zeros(value.shape[0], width, dtype=value.dtype)

        def registered_attention(query, _key, _value, _mask, **_kwargs):
            captured_heads.append(query.shape[1])
            return query

        infer._registered_attention = registered_attention
        phase = SimpleNamespace(
            qkv_proj=_Linear(qkv_projection),
            o_proj=_Linear(lambda value: torch.zeros(value.shape[0], 64, dtype=value.dtype)),
            query_layernorm=None,
        )
        hidden_states = torch.zeros(1, 3, 64)

        ar_output = infer.infer_attention(0, phase, hidden_states, None, None, None)
        context.phase = "denoise"
        context.active_tp_size = 2
        denoise_output = infer.infer_attention(0, phase, hidden_states, None, None, None)

        self.assertEqual(captured_heads, [8, 16])
        self.assertEqual(ar_output.shape, hidden_states.shape)
        self.assertEqual(denoise_output.shape, hidden_states.shape)

    @staticmethod
    def _configure_flashinfer_test_infer(context, *, multi_micro=False):
        infer = HunyuanImage3TransformerInfer.__new__(HunyuanImage3TransformerInfer)
        infer.config = {
            "moe_impl": "flashinfer",
            "flashinfer_multi_micro": multi_micro,
        }
        infer.parallel_context = context
        infer.tp_group = None
        infer.tp_rank = 0
        infer.tp_size = 1
        infer.hidden_act = "silu"
        infer.flashinfer_tune_max_num_tokens = 123
        infer.flashinfer_multi_micro = multi_micro
        infer.flashinfer_multi_micro_backend = "grouped_mm"
        infer._moe_easy_topk = lambda _moe, states: (
            states.reshape(-1, states.shape[-1]),
            torch.ones(states.numel() // states.shape[-1], 1),
            torch.zeros(states.numel() // states.shape[-1], 1, dtype=torch.long),
        )
        return infer

    def test_flashinfer_multi_micro_disabled_keeps_legacy_two_call_path(self):
        context = _ParallelContext(phase="denoise", tp_size=2, seq_size=2)
        infer = self._configure_flashinfer_test_infer(context, multi_micro=False)
        hidden_states = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        dummy_weight = torch.empty(1, 1, 1, 1, dtype=torch.bfloat16)
        moe = SimpleNamespace(
            moe_impl="flashinfer",
            shared_mlp=None,
            flashinfer_logical_tp_size=4,
            active_flashinfer_weight_shards=lambda _device, _dtype: (
                (0, 0, dummy_weight, dummy_weight),
                (1, 1, dummy_weight, dummy_weight),
            ),
        )
        calls = []

        def fake_flashinfer(_input, _topk_idx, _topk_weight, _w1, _w2, _dtype, *, output, tp_size, tp_rank, **kwargs):
            calls.append((tp_size, tp_rank, kwargs["tune_max_num_tokens"]))
            output.fill_(tp_rank + 1)

        with patch("lightx2v.models.networks.hunyuan_image3.infer.transformer_infer.flashinfer_cutlass_fused_moe", fake_flashinfer):
            local_output = infer._infer_mlp_flashinfer(moe, hidden_states)
        self.assertTrue(torch.equal(local_output, torch.full_like(hidden_states, 3)))
        self.assertEqual(calls, [(4, 0, 123), (4, 1, 123)])

    def test_flashinfer_multi_micro_ar_still_uses_one_official_call(self):
        context = _ParallelContext(phase="ar", tp_size=4, seq_size=1)
        infer = self._configure_flashinfer_test_infer(context, multi_micro=True)
        hidden_states = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        dummy_weight = torch.empty(1, 1, 1, 1, dtype=torch.bfloat16)
        moe = SimpleNamespace(
            moe_impl="flashinfer",
            shared_mlp=None,
            flashinfer_logical_tp_size=4,
            active_flashinfer_weight_shards=lambda _device, _dtype: ((0, 2, dummy_weight, dummy_weight),),
            active_flashinfer_multi_micro_weights=lambda _device, _dtype: (_ for _ in ()).throw(AssertionError("AR must not request the denoise multi-micro pack")),
        )
        official_calls = []

        def fake_flashinfer(_input, _topk_idx, _topk_weight, _w1, _w2, _dtype, *, output, tp_size, tp_rank, **kwargs):
            official_calls.append((tp_size, tp_rank, kwargs["tune_max_num_tokens"]))
            output.fill_(7)

        def fail_custom(*_args, **_kwargs):
            raise AssertionError("AR must continue to use the official single-micro FlashInfer path")

        with (
            patch(
                "lightx2v.models.networks.hunyuan_image3.infer.transformer_infer.flashinfer_cutlass_fused_moe",
                fake_flashinfer,
            ),
            patch(
                "lightx2v.models.networks.hunyuan_image3.infer.transformer_infer.lightx2v_multi_micro_fused_moe",
                fail_custom,
            ),
        ):
            output = infer._infer_mlp_flashinfer(moe, hidden_states)

        self.assertTrue(torch.equal(output, torch.full_like(hidden_states, 7)))
        self.assertEqual(official_calls, [(4, 2, 123)])

    def test_flashinfer_multi_micro_denoise_uses_custom_once_and_allreduces_once(self):
        context = _ParallelContext(phase="denoise", tp_size=2, seq_size=2)
        infer = self._configure_flashinfer_test_infer(context, multi_micro=True)
        hidden_states = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        w1_pack = torch.empty(2, 1, 2, 4, dtype=torch.bfloat16)
        w2_pack = torch.empty(2, 1, 4, 1, dtype=torch.bfloat16)
        accessor_calls = []

        def multi_micro_weights(device, dtype):
            accessor_calls.append((device, dtype))
            return w1_pack, w2_pack

        moe = SimpleNamespace(
            moe_impl="flashinfer",
            shared_mlp=None,
            flashinfer_logical_tp_size=4,
            active_flashinfer_multi_micro_weights=multi_micro_weights,
            active_flashinfer_weight_shards=lambda _device, _dtype: (_ for _ in ()).throw(AssertionError("fused denoise must not enter the legacy shard loop")),
        )
        custom_calls = []

        def fake_custom(input, topk_idx, topk_weight, fc1_weights, fc2_weights, *, output, backend):
            custom_calls.append((input, topk_idx, topk_weight, fc1_weights, fc2_weights, output, backend))
            output.fill_(5)
            return output

        def fail_official(*_args, **_kwargs):
            raise AssertionError("fused denoise must not call official FlashInfer directly")

        with (
            patch(
                "lightx2v.models.networks.hunyuan_image3.infer.transformer_infer.flashinfer_cutlass_fused_moe",
                fail_official,
            ),
            patch(
                "lightx2v.models.networks.hunyuan_image3.infer.transformer_infer.lightx2v_multi_micro_fused_moe",
                fake_custom,
            ),
        ):
            local_output = infer._infer_mlp_flashinfer(moe, hidden_states)

        self.assertTrue(torch.equal(local_output, torch.full_like(hidden_states, 5)))
        self.assertEqual(len(accessor_calls), 1)
        self.assertEqual(len(custom_calls), 1)
        call = custom_calls[0]
        self.assertIs(call[3], w1_pack)
        self.assertIs(call[4], w2_pack)
        self.assertEqual(call[6], "grouped_mm")

        phase = SimpleNamespace(is_moe=True, moe=moe)
        with patch.object(infer, "_infer_mlp_flashinfer", return_value=local_output), patch("lightx2v.models.networks.hunyuan_image3.infer.transformer_infer.dist.all_reduce") as all_reduce:
            infer.infer_mlp(phase, hidden_states)
        all_reduce.assert_called_once_with(local_output, op=torch.distributed.ReduceOp.SUM, group=context.active_tp_group)


if __name__ == "__main__":
    unittest.main()
