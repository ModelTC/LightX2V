import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner


class HunyuanImage3Ti2iHybridParallelTests(unittest.TestCase):
    def test_generate_ti2i_reuses_conditions_and_preserves_stage_order(self):
        runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
        runner.config = {
            "enable_cfg": True,
            "cfg_distilled": False,
            "seed": 42,
        }
        input_info = SimpleNamespace(
            prompt="edit the reference image",
            prompt_enhanced=None,
            image_path="reference.png",
            infer_align_image_size=True,
            seed=23,
        )

        events = []
        batch_cond_images = [[object()]]
        text_cond_inputs = {"cond_vae_images": object(), "cond_vit_embeds": object()}
        generation_cond_inputs = {"packed_cfg_conditions": object()}
        prepared_inputs = {"generator": object()}
        latents = object()
        decoded_images = [object()]
        final_images = [object()]

        def record(name, result):
            def invoke(*_args, **_kwargs):
                events.append(name)
                return result

            return invoke

        runner._ensure_pipeline_modules = Mock(side_effect=record("ensure_modules", None))
        runner._split_image_paths = Mock(side_effect=record("split_paths", ["reference.png"]))
        runner._build_batch_cond_images = Mock(side_effect=record("build_conditions", batch_cond_images))
        runner._resolve_image_size = Mock(side_effect=record("resolve_requested_size", "auto"))
        runner._resolve_ti2i_image_size = Mock(side_effect=record("resolve_ti2i_size", (768, 1024)))
        runner._prepare_cond_inputs = Mock(side_effect=record("encode_conditions", text_cond_inputs))
        runner._generate_cot_text = Mock(side_effect=record("ar", "generated recaption"))
        runner._repeat_cond_inputs = Mock(side_effect=record("pack_cfg_conditions", generation_cond_inputs))
        runner._prepare_text_to_image_inputs = Mock(side_effect=record("prepare_denoise", prepared_inputs))
        runner._denoise_latents = Mock(side_effect=record("denoise", latents))
        runner._is_output_rank = Mock(side_effect=record("check_output_rank", True))
        runner._decode_latents = Mock(side_effect=record("decode", decoded_images))
        runner.hunyuan_image_processor = SimpleNamespace(
            postprocess_outputs=Mock(side_effect=record("postprocess", final_images))
        )

        result = runner.generate_ti2i(input_info)

        self.assertIs(result, final_images)
        self.assertEqual(
            events,
            [
                "ensure_modules",
                "split_paths",
                "build_conditions",
                "resolve_requested_size",
                "resolve_ti2i_size",
                "encode_conditions",
                "ar",
                "pack_cfg_conditions",
                "prepare_denoise",
                "denoise",
                "check_output_rank",
                "decode",
                "postprocess",
            ],
        )

        cot_call = runner._generate_cot_text.call_args
        self.assertIs(cot_call.kwargs["batch_cond_images"], batch_cond_images)
        self.assertIs(cot_call.kwargs["cond_inputs"], text_cond_inputs)

        runner._repeat_cond_inputs.assert_called_once_with(text_cond_inputs, factor=2)
        denoise_prepare_call = runner._prepare_text_to_image_inputs.call_args
        self.assertIs(denoise_prepare_call.kwargs["batch_cond_images"], batch_cond_images)
        self.assertIs(denoise_prepare_call.kwargs["cond_inputs"], generation_cond_inputs)
        runner._denoise_latents.assert_called_once_with(prepared_inputs, (768, 1024))

    def test_cfg_condition_replication_and_serial_branch_slicing(self):
        runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
        source_vae = torch.tensor([[1.0, 2.0]])
        source_timesteps = torch.tensor([3.0])
        source_vit = [torch.tensor([[4.0, 5.0]])]
        source_conditions = {
            "cond_vae_images": source_vae,
            "cond_timesteps": source_timesteps,
            "cond_vit_embeds": source_vit,
            "optional_metadata": None,
        }

        packed_conditions = runner._repeat_cond_inputs(source_conditions, factor=2)

        self.assertEqual(packed_conditions["cond_vae_images"].tolist(), [[1.0, 2.0], [1.0, 2.0]])
        self.assertEqual(packed_conditions["cond_timesteps"].tolist(), [3.0, 3.0])
        self.assertEqual(len(packed_conditions["cond_vit_embeds"]), 2)
        self.assertIs(packed_conditions["cond_vit_embeds"][0], source_vit[0])
        self.assertIs(packed_conditions["cond_vit_embeds"][1], source_vit[0])
        self.assertIsNone(packed_conditions["optional_metadata"])
        self.assertEqual(source_vae.shape[0], 1)

        prepared_inputs = {
            "input_ids": torch.tensor([[10, 11], [20, 21]]),
            "position_ids": torch.tensor([[0, 1], [2, 3]]),
            "custom_pos_emb": object(),
            "rope_image_info": ("cond-rope", "uncond-rope"),
            "batch_size": 1,
            "do_cfg": True,
            **packed_conditions,
        }
        rebuilt_position_embeddings = [object(), object()]
        runner._build_custom_pos_emb = Mock(side_effect=rebuilt_position_embeddings)

        cond_branch = runner._prepare_cfg_parallel_branch_inputs(prepared_inputs, 0, mark_parallel_branch=False)
        uncond_branch = runner._prepare_cfg_parallel_branch_inputs(prepared_inputs, 1, mark_parallel_branch=False)

        self.assertEqual(cond_branch["input_ids"].tolist(), [[10, 11]])
        self.assertEqual(uncond_branch["input_ids"].tolist(), [[20, 21]])
        for branch in (cond_branch, uncond_branch):
            self.assertEqual(branch["batch_size"], 1)
            self.assertFalse(branch["do_cfg"])
            self.assertNotIn("_cfg_parallel_branch", branch)
            self.assertEqual(branch["cond_vae_images"].tolist(), [[1.0, 2.0]])
            self.assertEqual(branch["cond_timesteps"].tolist(), [3.0])
            self.assertEqual(len(branch["cond_vit_embeds"]), 1)
            self.assertIs(branch["cond_vit_embeds"][0], source_vit[0])

        self.assertIs(cond_branch["custom_pos_emb"], rebuilt_position_embeddings[0])
        self.assertIs(uncond_branch["custom_pos_emb"], rebuilt_position_embeddings[1])

    def test_denoise_condition_is_injected_only_on_first_cached_step(self):
        runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
        runner.config = {}
        rebuilt_position_embedding = object()
        runner._build_custom_pos_emb = Mock(return_value=rebuilt_position_embedding)

        condition_values = {
            "cond_vae_images": torch.ones(1, 2, 2, 2),
            "cond_vae_image_mask": torch.tensor([[True, False, False, False]]),
            "cond_timesteps": torch.zeros(1),
            "cond_timestep_index": torch.tensor([[0]]),
            "cond_vit_embeds": [torch.ones(1, 2, 3)],
            "cond_vit_image_mask": torch.tensor([[False, True, False, False]]),
        }
        prepared_inputs = {
            "input_ids": torch.tensor([[10, 11, 12, 13]]),
            "attention_mask": None,
            "full_attn_slices": None,
            "position_ids": torch.tensor([[0, 1, 2, 3]]),
            "custom_pos_emb": object(),
            "rope_image_info": object(),
            "image_mask": torch.tensor([[False, False, True, True]]),
            "timesteps_index": torch.tensor([[0]]),
            "guidance_index": None,
            "timesteps_r_index": None,
            **condition_values,
        }
        cache = object()
        cache_position_ids = torch.tensor([[0, 2, 3]])
        cache_local_inputs = {
            "image_mask": torch.tensor([[False, True, True]]),
            "timesteps_index": torch.tensor([[0]]),
            "guidance_index": None,
            "timesteps_r_index": None,
        }
        kv_state = {
            "kv_cache": cache,
            "cache_position_ids": cache_position_ids,
            "cache_local_inputs": cache_local_inputs,
        }
        latents = torch.zeros(1, 4, 2, 2)
        timestep = torch.tensor(1.0)

        first_step_inputs = runner._build_denoise_model_inputs(
            prepared_inputs,
            latents,
            timestep,
            step_index=0,
            use_kv_cache=True,
            kv_state=kv_state,
            guidance_scale=1.0,
        )
        cached_step_inputs = runner._build_denoise_model_inputs(
            prepared_inputs,
            latents,
            timestep,
            step_index=1,
            use_kv_cache=True,
            kv_state=kv_state,
            guidance_scale=1.0,
        )

        self.assertTrue(first_step_inputs["first_step"])
        self.assertIs(first_step_inputs["input_ids"], prepared_inputs["input_ids"])
        for key, value in condition_values.items():
            self.assertIs(first_step_inputs[key], value)

        self.assertFalse(cached_step_inputs["first_step"])
        self.assertIsNone(cached_step_inputs["input_ids"])
        for key in condition_values:
            self.assertNotIn(key, cached_step_inputs)
        self.assertIs(cached_step_inputs["position_ids"], cache_position_ids)
        self.assertIs(cached_step_inputs["custom_pos_emb"], rebuilt_position_embedding)

        for model_inputs in (first_step_inputs, cached_step_inputs):
            self.assertTrue(model_inputs["use_cache"])
            self.assertIs(model_inputs["past_key_values"], cache)


if __name__ == "__main__":
    unittest.main()
