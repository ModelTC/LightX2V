import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from lightx2v.utils.set_config import set_config


def _tiny_hunyuan_config():
    return {
        "model_cls": "hunyuan_image3",
        "task": "t2i",
        "seq_parallel": False,
        "cpu_offload": False,
        "feature_caching": "NoCaching",
        "num_layers": 2,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 12,
        "moe_intermediate_size": 12,
        "num_experts": 3,
        "moe_topk": 2,
        "moe_layer_num_skipped": 1,
        "use_mixed_mlp_moe": False,
        "hidden_act": "silu",
        "mlp_bias": False,
        "rms_norm_eps": 1e-5,
        "patch_size": 1,
        "patch_embed_hidden_dim": 8,
        "vae": {"latent_channels": 4},
        "vocab_size": 32,
        "cfg_distilled": False,
        "use_meanflow": False,
    }


def _tiny_weight_dict(config):
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = hidden // heads
    qkv_out = (heads + 2 * kv_heads) * head_dim
    inter = config["intermediate_size"]
    moe_inter = config["moe_intermediate_size"]
    patch_hidden = config["patch_embed_hidden_dim"]
    latent = config["vae"]["latent_channels"]
    vocab = config["vocab_size"]
    weights = {
        "model.wte.weight": torch.randn(vocab, hidden),
        "model.ln_f.weight": torch.randn(hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
        "timestep_emb.mlp.0.weight": torch.randn(hidden, 256),
        "timestep_emb.mlp.0.bias": torch.randn(hidden),
        "timestep_emb.mlp.2.weight": torch.randn(hidden, hidden),
        "timestep_emb.mlp.2.bias": torch.randn(hidden),
        "time_embed.mlp.0.weight": torch.randn(hidden, 256),
        "time_embed.mlp.0.bias": torch.randn(hidden),
        "time_embed.mlp.2.weight": torch.randn(hidden, hidden),
        "time_embed.mlp.2.bias": torch.randn(hidden),
        "time_embed_2.mlp.0.weight": torch.randn(hidden, 256),
        "time_embed_2.mlp.0.bias": torch.randn(hidden),
        "time_embed_2.mlp.2.weight": torch.randn(hidden, hidden),
        "time_embed_2.mlp.2.bias": torch.randn(hidden),
        "patch_embed.model.0.weight": torch.randn(patch_hidden, latent, 3, 3),
        "patch_embed.model.0.bias": torch.randn(patch_hidden),
        "patch_embed.model.1.in_layers.0.weight": torch.randn(patch_hidden),
        "patch_embed.model.1.in_layers.0.bias": torch.randn(patch_hidden),
        "patch_embed.model.1.in_layers.2.weight": torch.randn(hidden, patch_hidden, 3, 3),
        "patch_embed.model.1.in_layers.2.bias": torch.randn(hidden),
        "patch_embed.model.1.emb_layers.1.weight": torch.randn(hidden * 2, hidden),
        "patch_embed.model.1.emb_layers.1.bias": torch.randn(hidden * 2),
        "patch_embed.model.1.out_layers.0.weight": torch.randn(hidden),
        "patch_embed.model.1.out_layers.0.bias": torch.randn(hidden),
        "patch_embed.model.1.out_layers.3.weight": torch.randn(hidden, hidden, 3, 3),
        "patch_embed.model.1.out_layers.3.bias": torch.randn(hidden),
        "patch_embed.model.1.skip_connection.weight": torch.randn(hidden, patch_hidden, 1, 1),
        "patch_embed.model.1.skip_connection.bias": torch.randn(hidden),
        "final_layer.model.0.in_layers.0.weight": torch.randn(hidden),
        "final_layer.model.0.in_layers.0.bias": torch.randn(hidden),
        "final_layer.model.0.in_layers.2.weight": torch.randn(patch_hidden, hidden, 3, 3),
        "final_layer.model.0.in_layers.2.bias": torch.randn(patch_hidden),
        "final_layer.model.0.emb_layers.1.weight": torch.randn(patch_hidden * 2, hidden),
        "final_layer.model.0.emb_layers.1.bias": torch.randn(patch_hidden * 2),
        "final_layer.model.0.out_layers.0.weight": torch.randn(patch_hidden),
        "final_layer.model.0.out_layers.0.bias": torch.randn(patch_hidden),
        "final_layer.model.0.out_layers.3.weight": torch.randn(patch_hidden, patch_hidden, 3, 3),
        "final_layer.model.0.out_layers.3.bias": torch.randn(patch_hidden),
        "final_layer.model.0.skip_connection.weight": torch.randn(patch_hidden, hidden, 1, 1),
        "final_layer.model.0.skip_connection.bias": torch.randn(patch_hidden),
        "final_layer.model.1.0.weight": torch.randn(patch_hidden),
        "final_layer.model.1.0.bias": torch.randn(patch_hidden),
        "final_layer.model.1.2.weight": torch.randn(latent, patch_hidden, 3, 3),
        "final_layer.model.1.2.bias": torch.randn(latent),
    }
    for layer in range(config["num_layers"]):
        prefix = f"model.layers.{layer}"
        weights[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden)
        weights[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden)
        weights[f"{prefix}.self_attn.qkv_proj.weight"] = torch.randn(qkv_out, hidden)
        weights[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden)
        weights[f"{prefix}.self_attn.query_layernorm.weight"] = torch.randn(head_dim)
        weights[f"{prefix}.self_attn.key_layernorm.weight"] = torch.randn(head_dim)
        if layer < config["moe_layer_num_skipped"]:
            weights[f"{prefix}.mlp.gate_and_up_proj.weight"] = torch.randn(inter * 2, hidden)
            weights[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden, inter)
        else:
            weights[f"{prefix}.mlp.gate.wg.weight"] = torch.randn(config["num_experts"], hidden)
            for expert in range(config["num_experts"]):
                weights[f"{prefix}.mlp.experts.{expert}.gate_and_up_proj.weight"] = torch.randn(moe_inter * 2, hidden)
                weights[f"{prefix}.mlp.experts.{expert}.down_proj.weight"] = torch.randn(hidden, moe_inter)
    return weights


def _loaded_shape(weight):
    tensor = getattr(weight, "weight", None)
    if tensor is None:
        tensor = getattr(weight, "pin_weight")
    return tensor.shape


def test_infer_entrypoint_registers_hunyuan_image3_runner():
    infer_source = Path("lightx2v/infer.py").read_text()

    assert "HunyuanImage3Runner" in infer_source
    assert '"hunyuan_image3"' in infer_source


def test_hunyuan_image3_config_merges_model_config(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "hunyuan_image_3_moe",
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "intermediate_size": 3072,
                "diff_infer_steps": 8,
                "diff_guidance_scale": 6.0,
                "flow_shift": 7.0,
            }
        )
    )
    config_json = tmp_path / "runtime.json"
    config_json.write_text(
        json.dumps(
            {
                "infer_steps": 4,
                "sample_guide_scale": 3.5,
                "target_height": 1024,
                "target_width": 1024,
            }
        )
    )

    args = argparse.Namespace(
        seed=123,
        model_cls="hunyuan_image3",
        task="t2i",
        support_tasks=[],
        model_path=str(model_path),
        config_json=str(config_json),
    )

    config = set_config(args)

    assert config["num_layers"] == 32
    assert config["num_heads"] == 32
    assert config["num_key_value_heads"] == 8
    assert config["hidden_size"] == 4096
    assert config["infer_steps"] == 4
    assert config["sample_guide_scale"] == 3.5
    assert config["diff_infer_steps"] == 8
    assert config["diff_guidance_scale"] == 6.0
    assert config["flow_shift"] == 7.0


def test_hunyuan_image3_cache_cli_args_override_config_json(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_json = tmp_path / "runtime.json"
    config_json.write_text(
        json.dumps(
            {
                "enable_kv_cache": False,
                "enable_text_kv_cache": False,
                "use_taylor_cache": False,
                "taylor_cache_interval": 5,
                "taylor_cache_order": 2,
                "taylor_cache_enable_first_enhance": False,
                "taylor_cache_first_enhance_steps": 3,
                "taylor_cache_enable_tailing_enhance": False,
                "taylor_cache_tailing_enhance_steps": 1,
                "taylor_cache_low_freqs_order": 2,
                "taylor_cache_high_freqs_order": 2,
            }
        )
    )

    args = argparse.Namespace(
        seed=123,
        model_cls="hunyuan_image3",
        task="i2i",
        support_tasks=[],
        model_path=str(model_path),
        config_json=str(config_json),
        enable_kv_cache=True,
        enable_text_kv_cache=True,
        use_taylor_cache=True,
        taylor_cache_interval=7,
        taylor_cache_order=3,
        taylor_cache_enable_first_enhance=True,
        taylor_cache_first_enhance_steps=4,
        taylor_cache_enable_tailing_enhance=True,
        taylor_cache_tailing_enhance_steps=2,
        taylor_cache_low_freqs_order=1,
        taylor_cache_high_freqs_order=3,
    )

    config = set_config(args)

    assert config["enable_kv_cache"] is True
    assert config["enable_text_kv_cache"] is True
    assert config["use_taylor_cache"] is True
    assert config["taylor_cache_interval"] == 7
    assert config["taylor_cache_order"] == 3
    assert config["taylor_cache_enable_first_enhance"] is True
    assert config["taylor_cache_first_enhance_steps"] == 4
    assert config["taylor_cache_enable_tailing_enhance"] is True
    assert config["taylor_cache_tailing_enhance_steps"] == 2
    assert config["taylor_cache_low_freqs_order"] == 1
    assert config["taylor_cache_high_freqs_order"] == 3


def test_hunyuan_image3_moe_impl_cli_arg_overrides_config_json(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_json = tmp_path / "runtime.json"
    config_json.write_text(json.dumps({"moe_impl": "eager"}))

    args = argparse.Namespace(
        seed=123,
        model_cls="hunyuan_image3",
        task="t2i",
        support_tasks=[],
        model_path=str(model_path),
        config_json=str(config_json),
        moe_impl="flashinfer",
    )

    config = set_config(args)

    assert config["moe_impl"] == "flashinfer"


def test_hunyuan_image3_ti2i_script_passes_cache_args_in_python_entrypoint():
    script_source = Path("scripts/hunyuan_image3/run_hunyuan_image3_ti2i.sh").read_text()
    python_entry = script_source.split("python -m lightx2v.infer", maxsplit=1)[1]

    for cli_arg in [
        "--moe_impl",
        "--enable_kv_cache",
        "--enable_text_kv_cache",
        "--use_taylor_cache",
        "--taylor_cache_interval",
        "--taylor_cache_order",
        "--taylor_cache_enable_first_enhance",
        "--taylor_cache_first_enhance_steps",
        "--taylor_cache_enable_tailing_enhance",
        "--taylor_cache_tailing_enhance_steps",
        "--taylor_cache_low_freqs_order",
        "--taylor_cache_high_freqs_order",
    ]:
        assert cli_arg in python_entry


def test_hunyuan_image3_t2i_script_passes_cache_args_in_python_entrypoint():
    script_source = Path("scripts/hunyuan_image3/run_hunyuan_image3_t2i.sh").read_text()
    python_entry = script_source.split("python -m lightx2v.infer", maxsplit=1)[1]

    for cli_arg in [
        "--moe_impl",
        "--enable_kv_cache",
        "--enable_text_kv_cache",
        "--use_taylor_cache",
        "--taylor_cache_interval",
        "--taylor_cache_order",
        "--taylor_cache_enable_first_enhance",
        "--taylor_cache_first_enhance_steps",
        "--taylor_cache_enable_tailing_enhance",
        "--taylor_cache_tailing_enhance_steps",
        "--taylor_cache_low_freqs_order",
        "--taylor_cache_high_freqs_order",
    ]:
        assert cli_arg in python_entry


def test_hunyuan_image3_i2i_script_passes_moe_impl_in_python_entrypoint():
    script_source = Path("scripts/hunyuan_image3/run_hunyuan_image3_i2i.sh").read_text()
    python_entry = script_source.split("python -m lightx2v.infer", maxsplit=1)[1]

    assert "--moe_impl" in python_entry


def test_hunyuan_image3_runner_is_not_upstream_pipeline_wrapper():
    runner_source = Path("lightx2v/models/runners/hunyuan_image3/hunyuan_image3_runner.py").read_text()

    assert "HunyuanImage3ForCausalMM" not in runner_source
    assert ".generate_image(" not in runner_source
    assert "device_map" not in runner_source


def test_hunyuan_image3_weights_load_real_key_layout():
    from lightx2v.models.networks.hunyuan_image3.weights.post_weights import HunyuanImage3PostWeights
    from lightx2v.models.networks.hunyuan_image3.weights.pre_weights import HunyuanImage3PreWeights
    from lightx2v.models.networks.hunyuan_image3.weights.transformer_weights import HunyuanImage3TransformerWeights

    config = _tiny_hunyuan_config()
    weight_dict = _tiny_weight_dict(config)

    pre = HunyuanImage3PreWeights(config)
    transformer = HunyuanImage3TransformerWeights(config)
    post = HunyuanImage3PostWeights(config)

    pre.load(weight_dict)
    transformer.load(weight_dict)
    post.load(weight_dict)

    assert _loaded_shape(pre.token_embedding) == (32, 16)
    assert pre.patch_embed.input_conv.weight.shape == (8, 4, 3, 3)
    assert len(transformer.blocks) == 2
    assert _loaded_shape(transformer.blocks[0].compute_phases[1].gate_and_up_proj) == (16, 24)
    assert _loaded_shape(transformer.blocks[1].compute_phases[1].experts[2].down_proj) == (12, 16)
    assert _loaded_shape(post.final_norm) == (16,)
    assert post.final_layer.output_conv.weight.shape == (4, 8, 3, 3)


def test_hunyuan_image3_pipeline_parallel_maps_weights_across_devices():
    from lightx2v.models.networks.hunyuan_image3.model import resolve_pipeline_device_for_key

    config = _tiny_hunyuan_config()
    config["num_layers"] = 8
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    assert resolve_pipeline_device_for_key("model.wte.weight", config, devices) == "cuda:0"
    assert resolve_pipeline_device_for_key("patch_embed.model.0.weight", config, devices) == "cuda:0"
    assert resolve_pipeline_device_for_key("time_embed.mlp.0.weight", config, devices) == "cuda:0"

    assert resolve_pipeline_device_for_key("model.layers.0.self_attn.qkv_proj.weight", config, devices) == "cuda:0"
    assert resolve_pipeline_device_for_key("model.layers.1.mlp.experts.0.down_proj.weight", config, devices) == "cuda:0"
    assert resolve_pipeline_device_for_key("model.layers.2.self_attn.qkv_proj.weight", config, devices) == "cuda:1"
    assert resolve_pipeline_device_for_key("model.layers.5.self_attn.qkv_proj.weight", config, devices) == "cuda:2"
    assert resolve_pipeline_device_for_key("model.layers.7.mlp.gate.wg.weight", config, devices) == "cuda:3"

    assert resolve_pipeline_device_for_key("model.ln_f.weight", config, devices) == "cuda:3"
    assert resolve_pipeline_device_for_key("lm_head.weight", config, devices) == "cuda:3"
    assert resolve_pipeline_device_for_key("final_layer.model.0.in_layers.0.weight", config, devices) == "cuda:3"


def test_hunyuan_image3_loader_streams_core_tensors_with_pipeline_mapping(tmp_path):
    from lightx2v.models.networks.hunyuan_image3.model import HunyuanImage3Model

    ckpt = tmp_path / "tiny.safetensors"
    save_file(
        {
            "model.wte.weight": torch.randn(4, 4),
            "model.layers.1.self_attn.qkv_proj.weight": torch.randn(4, 4),
            "vae.decoder.conv_in.weight": torch.randn(4, 4, 1, 1),
        },
        ckpt,
    )
    model = HunyuanImage3Model.__new__(HunyuanImage3Model)
    model.config = {**_tiny_hunyuan_config(), "num_layers": 2}
    model.pipeline_devices = ["cpu", "cpu"]
    model.pipeline_parallel = True
    model.remove_keys = []
    model.preserved_keys = ["model.", "lm_head.", "patch_embed.", "final_layer.", "time_embed.", "time_embed_2.", "timestep_emb."]

    loaded = model._load_safetensor_to_dict(str(ckpt), unified_dtype=True, sensitive_layer={})

    assert sorted(loaded) == ["model.layers.1.self_attn.qkv_proj.weight", "model.wte.weight"]
    assert loaded["model.wte.weight"].device.type == "cpu"
    assert loaded["model.layers.1.self_attn.qkv_proj.weight"].shape == (4, 4)


def test_hunyuan_image3_runner_pipeline_saves_generated_image(tmp_path):
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    save_path = tmp_path / "out.png"
    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {"task": "t2i"}
    runner._gc_frozen = True
    runner.generate_t2i = lambda input_info: [Image.new("RGB", (2, 3), color=(10, 20, 30))]

    result = runner.run_pipeline(
        SimpleNamespace(
            prompt="hello",
            save_result_path=str(save_path),
            return_result_tensor=False,
        )
    )

    assert result == {"image": None}
    assert save_path.exists()
    assert Image.open(save_path).size == (2, 3)


def test_hunyuan_image3_runner_pipeline_saves_ti2i_image(tmp_path):
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    save_path = tmp_path / "out.png"
    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {"task": "i2i"}
    runner._gc_frozen = True
    runner.generate_i2i = lambda input_info: [Image.new("RGB", (4, 5), color=(10, 20, 30))]

    result = runner.run_pipeline(
        SimpleNamespace(
            prompt="edit",
            image_path="ref.png",
            save_result_path=str(save_path),
            return_result_tensor=False,
        )
    )

    assert result == {"image": None}
    assert save_path.exists()
    assert Image.open(save_path).size == (4, 5)


def test_hunyuan_image3_resolves_text_generation_plan_for_think_recaption():
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.hunyuan_tokenizer = SimpleNamespace(
        end_of_think_token_id=10,
        end_of_recaption_token_id=12,
        recaption_token="<recaption>",
        convert_tokens_to_ids=lambda token: {"<recaption>": 11}[token],
    )

    plan = runner._resolve_text_generation_plan("think_recaption")

    assert plan.first_bot_task == "think"
    assert plan.stage_transitions == [(10, [11])]
    assert plan.final_stop_tokens == [12]


def test_hunyuan_image3_prepare_image_inputs_passes_cot_text():
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    class FakeImageProcessor:
        vae_reso_group = SimpleNamespace(base_size=1024)

        def build_gen_image_info(self, image_size, add_guidance_token=False, add_timestep_r_token=False):
            return SimpleNamespace(image_size=image_size)

        def prepare_full_attn_slices(self, tokenizer_output, batch_idx):
            return []

    class FakeTokenizer:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, **kwargs):
            self.kwargs = kwargs
            output = SimpleNamespace(
                tokens=torch.tensor([[1, 2, 3]]),
                all_image_slices=[[]],
                gen_image_mask=torch.tensor([[False, True, True]]),
                gen_timestep_scatter_index=None,
                guidance_scatter_index=None,
                gen_timestep_r_scatter_index=None,
            )
            return {"output": output, "sections": [[{"type": "text"}]]}

    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {"max_position_embeddings": 16}
    runner.hunyuan_generation_config = SimpleNamespace(max_length=16, sequence_template="instruct", drop_think=True)
    runner.hunyuan_image_processor = FakeImageProcessor()
    runner.hunyuan_tokenizer = FakeTokenizer()
    runner.hunyuan_config = SimpleNamespace(max_position_embeddings=16, rope_type="default")
    runner.hunyuan_cached_rope = lambda *args, **kwargs: None
    runner._pipeline_latent_device = lambda: torch.device("cpu")

    runner._prepare_text_to_image_inputs("prompt", (8, 8), 123, cot_text="<think>x</think><recaption>y</recaption>")

    assert runner.hunyuan_tokenizer.kwargs["batch_cot_text"] == ["<think>x</think><recaption>y</recaption>"]
    assert runner.hunyuan_tokenizer.kwargs["bot_task"] == "image"


def test_hunyuan_image3_prepare_image_inputs_passes_cond_images():
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    class FakeImageProcessor:
        vae_reso_group = SimpleNamespace(base_size=1024)

        def build_gen_image_info(self, image_size, add_guidance_token=False, add_timestep_r_token=False):
            return SimpleNamespace(image_size=image_size)

        def prepare_full_attn_slices(self, tokenizer_output, batch_idx):
            return []

    class FakeTokenizer:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, **kwargs):
            self.kwargs = kwargs
            output = SimpleNamespace(
                tokens=torch.tensor([[1, 2, 3], [1, 2, 3]]),
                all_image_slices=[[], []],
                gen_image_mask=torch.tensor([[False, True, True], [False, True, True]]),
                gen_timestep_scatter_index=torch.tensor([[0], [0]]),
                guidance_scatter_index=None,
                gen_timestep_r_scatter_index=None,
                vae_image_mask=torch.tensor([[True, False, False], [True, False, False]]),
                vit_image_mask=torch.tensor([[False, True, False], [False, True, False]]),
                cond_timestep_scatter_index=torch.tensor([[2], [2]]),
            )
            return {"output": output, "sections": [[{"type": "cond_vae_image"}, {"type": "gen_image"}]]}

    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {"max_position_embeddings": 16, "enable_cfg": True}
    runner.hunyuan_generation_config = SimpleNamespace(max_length=16, sequence_template="instruct", drop_think=True)
    runner.hunyuan_image_processor = FakeImageProcessor()
    runner.hunyuan_tokenizer = FakeTokenizer()
    runner.hunyuan_config = SimpleNamespace(max_position_embeddings=16, rope_type="default")
    runner.hunyuan_cached_rope = lambda *args, **kwargs: None
    runner._pipeline_latent_device = lambda: torch.device("cpu")

    prepared = runner._prepare_text_to_image_inputs(
        "prompt",
        (8, 8),
        123,
        cot_text="<recaption>x</recaption>",
        batch_cond_images=[["cond"]],
        cond_inputs={
            "cond_vae_images": [torch.zeros(1, 4, 2, 2), torch.zeros(1, 4, 2, 2)],
            "cond_timesteps": [torch.zeros(1), torch.zeros(1)],
            "cond_vit_embeds": [torch.zeros(1, 1, 16), torch.zeros(1, 1, 16)],
        },
    )

    assert runner.hunyuan_tokenizer.kwargs["batch_cond_images"] == [["cond"]]
    assert prepared["cond_vae_images"][0].shape == (1, 4, 2, 2)
    assert torch.equal(prepared["cond_vae_image_mask"], torch.tensor([[True, False, False], [True, False, False]]))
    assert torch.equal(prepared["cond_timestep_index"], torch.tensor([[2], [2]]))
    assert prepared["cond_vit_embeds"][0].shape == (1, 1, 16)


def test_hunyuan_image3_generate_t2i_uses_cot_for_think_recaption():
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {"task": "t2i", "bot_task": "think_recaption"}
    runner._ensure_pipeline_modules = lambda: None
    runner._resolve_image_size = lambda input_info: (8, 8)
    runner._generate_cot_text = lambda prompt, image_size: "<think>x</think><recaption>y</recaption>"
    seen = {}

    def fake_prepare(prompt, image_size, seed, cot_text=None):
        seen["cot_text"] = cot_text
        return {"batch_size": 1, "generator": "generator"}

    runner._prepare_text_to_image_inputs = fake_prepare
    runner._denoise_latents = lambda prepared_inputs, image_size: "latents"
    runner._decode_latents = lambda latents, generator: ["image"]

    images = runner.generate_t2i(SimpleNamespace(prompt="prompt", seed=123))

    assert images == ["image"]
    assert seen["cot_text"] == "<think>x</think><recaption>y</recaption>"


def test_hunyuan_image3_text_generation_applies_think_to_recaption_transition():
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    class FakeModel:
        def __init__(self):
            self.next_tokens = [10, 12]
            self.calls = 0

        def infer(self, inputs):
            vocab_size = 16
            logits = torch.zeros(1, inputs["input_ids"].shape[1], vocab_size)
            logits[:, -1, self.next_tokens[self.calls]] = 100.0
            self.calls += 1
            return {"logits": logits}

    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {"text_do_sample": False, "max_new_tokens": 5}
    runner.hunyuan_generation_config = SimpleNamespace(max_new_tokens=5, do_sample=False)
    runner.hunyuan_tokenizer = SimpleNamespace(
        think_token="<think>",
        recaption_token="<recaption>",
        decode=lambda tokens: "".join({10: "</think>", 11: "<recaption>", 12: "</recaption>"}[int(token)] for token in tokens),
    )
    runner.model = FakeModel()
    runner._build_text_model_inputs = lambda input_ids, tokenizer_output: {"input_ids": input_ids}
    plan = SimpleNamespace(first_bot_task="think", stage_transitions=[(10, [11])], final_stop_tokens=[12])

    generated_tokens = runner._generate_text_tokens(torch.tensor([[1, 2]]), SimpleNamespace(), plan)
    cot_text = runner._decode_cot_text(generated_tokens, plan)

    assert generated_tokens == [10, 11, 12]
    assert cot_text == "<think></think><recaption></recaption>"


def test_hunyuan_image3_text_generation_moves_sampled_token_to_input_device():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return

    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    input_device = torch.device("cuda:0")
    logits_device = torch.device("cuda:1")

    class FakeModel:
        def infer(self, inputs):
            logits = torch.zeros(1, inputs["input_ids"].shape[1], 8, device=logits_device)
            logits[:, -1, 3] = 100.0
            return {"logits": logits}

    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {"text_do_sample": False, "max_new_tokens": 1}
    runner.hunyuan_generation_config = SimpleNamespace(max_new_tokens=1, do_sample=False)
    runner.model = FakeModel()
    runner._build_text_model_inputs = lambda input_ids, tokenizer_output: {"input_ids": input_ids}
    plan = SimpleNamespace(first_bot_task="recaption", stage_transitions=[], final_stop_tokens=[3])

    generated_tokens = runner._generate_text_tokens(torch.tensor([[1, 2]], device=input_device), SimpleNamespace(), plan)

    assert generated_tokens == [3]


def test_hunyuan_image3_generate_cot_text_streams_process_to_stdout(capsys):
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    class FakeModel:
        def __init__(self):
            self.next_tokens = [10, 12]
            self.calls = 0

        def infer(self, inputs):
            vocab_size = 16
            logits = torch.zeros(1, inputs["input_ids"].shape[1], vocab_size)
            logits[:, -1, self.next_tokens[self.calls]] = 100.0
            self.calls += 1
            return {"logits": logits}

    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {"bot_task": "think_recaption", "text_do_sample": False, "max_new_tokens": 5, "text_stream_output": True}
    runner.hunyuan_generation_config = SimpleNamespace(max_new_tokens=5, do_sample=False)
    runner.hunyuan_tokenizer = SimpleNamespace(
        think_token="<think>",
        recaption_token="<recaption>",
        end_of_think_token_id=10,
        end_of_recaption_token_id=12,
        convert_tokens_to_ids=lambda token: {"<recaption>": 11}[token],
        decode=lambda tokens: "".join({10: "</think>", 11: "<recaption>", 12: "</recaption>"}[int(token)] for token in tokens),
    )
    runner.model = FakeModel()
    runner._resolve_system_prompt = lambda bot_task: None
    runner._prepare_text_generation_inputs = lambda prompt, bot_task, system_prompt: {
        "input_ids": torch.tensor([[1, 2]]),
        "tokenizer_output": SimpleNamespace(),
    }
    runner._build_text_model_inputs = lambda input_ids, tokenizer_output: {"input_ids": input_ids}

    cot_text = runner._generate_cot_text("prompt", (8, 8))

    captured = capsys.readouterr()
    assert cot_text == "<think></think><recaption></recaption>"
    assert "<think></think><recaption></recaption>" in captured.out


def test_hunyuan_image3_pre_infer_instantiates_timestep_guidance_and_meanflow_tokens(monkeypatch):
    from lightx2v.models.networks.hunyuan_image3.infer import pre_infer

    def fake_timestep_embedder(weights, values):
        fill = {"timestep": 1.0, "guidance": 2.0, "timestep_r": 3.0}[weights]
        return torch.full((values.numel(), 4), fill)

    monkeypatch.setattr(pre_infer, "apply_timestep_embedder", fake_timestep_embedder)
    infer = pre_infer.HunyuanImage3PreInfer({"hidden_size": 4, "cfg_distilled": True, "use_meanflow": True})
    weights = SimpleNamespace(timestep_emb="timestep", guidance_emb="guidance", timestep_r_emb="timestep_r")

    out = infer.infer(
        weights,
        {
            "inputs_embeds": torch.zeros(1, 5, 4),
            "timesteps": torch.tensor([10.0]),
            "timesteps_index": torch.tensor([[1]]),
            "guidance": torch.tensor([5.0]),
            "guidance_index": torch.tensor([[3]]),
            "timesteps_r": torch.tensor([2.0]),
            "timesteps_r_index": torch.tensor([[4]]),
        },
    )

    assert torch.equal(out.hidden_states[0, 0], torch.zeros(4))
    assert torch.equal(out.hidden_states[0, 1], torch.ones(4))
    assert torch.equal(out.hidden_states[0, 3], torch.full((4,), 2.0))
    assert torch.equal(out.hidden_states[0, 4], torch.full((4,), 3.0))


def test_hunyuan_image3_pre_infer_instantiates_cond_vae_and_vit_tokens(monkeypatch):
    from lightx2v.models.networks.hunyuan_image3.infer import pre_infer

    def fake_timestep_embedder(weights, values):
        return torch.full((values.numel(), 4), 3.0)

    monkeypatch.setattr(pre_infer, "apply_timestep_embedder", fake_timestep_embedder)
    infer = pre_infer.HunyuanImage3PreInfer({"hidden_size": 4, "cfg_distilled": False, "use_meanflow": False})
    infer._patch_embed = lambda weights, images, timesteps: (torch.full((1, 2, 4), 5.0), 1, 2)
    weights = SimpleNamespace(timestep_emb="timestep")

    out = infer.infer(
        weights,
        {
            "inputs_embeds": torch.zeros(1, 6, 4),
            "cond_vae_images": torch.zeros(1, 4, 1, 2),
            "cond_vae_image_mask": torch.tensor([[False, True, True, False, False, False]]),
            "cond_timesteps": torch.tensor([0.0]),
            "cond_timestep_index": torch.tensor([[3]]),
            "cond_vit_embeds": torch.full((1, 2, 4), 7.0),
            "cond_vit_image_mask": torch.tensor([[False, False, False, False, True, True]]),
        },
    )

    assert torch.equal(out.hidden_states[0, 1], torch.full((4,), 5.0))
    assert torch.equal(out.hidden_states[0, 2], torch.full((4,), 5.0))
    assert torch.equal(out.hidden_states[0, 3], torch.full((4,), 3.0))
    assert torch.equal(out.hidden_states[0, 4], torch.full((4,), 7.0))
    assert torch.equal(out.hidden_states[0, 5], torch.full((4,), 7.0))


def test_hunyuan_image3_timestep_embedder_matches_weight_dtype():
    from lightx2v.models.networks.hunyuan_image3.infer.utils import apply_timestep_embedder

    class FakeLinear:
        def __init__(self, in_features, out_features, dtype):
            self.weight = torch.randn(in_features, out_features, dtype=dtype)
            self.bias = torch.randn(out_features, dtype=dtype)

        def apply(self, input_tensor):
            output = torch.empty(input_tensor.shape[0], self.weight.shape[1], dtype=input_tensor.dtype)
            return torch.addmm(self.bias, input_tensor, self.weight, out=output)

    weights = SimpleNamespace(
        linear_1=FakeLinear(256, 8, torch.bfloat16),
        linear_2=FakeLinear(8, 4, torch.bfloat16),
    )

    output = apply_timestep_embedder(weights, torch.tensor([10.0]))

    assert output.dtype == torch.bfloat16
    assert output.shape == (1, 4)


def test_hunyuan_image3_apply_linear_matches_weight_dtype():
    from lightx2v.models.networks.hunyuan_image3.infer.utils import apply_linear

    class FakeLinear:
        def __init__(self, in_features, out_features, dtype):
            self.weight = torch.randn(in_features, out_features, dtype=dtype)
            self.bias = None

        def apply(self, input_tensor):
            output = torch.empty(input_tensor.shape[0], self.weight.shape[1], dtype=input_tensor.dtype)
            return torch.mm(input_tensor, self.weight, out=output)

    linear = FakeLinear(4, 6, torch.bfloat16)

    output = apply_linear(linear, torch.randn(2, 4, dtype=torch.float32))

    assert output.dtype == torch.bfloat16
    assert output.shape == (2, 6)


def test_hunyuan_image3_apply_mlp_matches_weight_dtype():
    from lightx2v.models.networks.hunyuan_image3.infer.utils import apply_mlp

    class FakeLinear:
        def __init__(self, in_features, out_features, dtype):
            self.weight = torch.randn(in_features, out_features, dtype=dtype)
            self.bias = None

        def apply(self, input_tensor):
            output = torch.empty(input_tensor.shape[0], self.weight.shape[1], dtype=input_tensor.dtype)
            return torch.mm(input_tensor, self.weight, out=output)

    gate_and_up = FakeLinear(4, 8, torch.bfloat16)
    down = FakeLinear(4, 4, torch.bfloat16)

    output = apply_mlp(gate_and_up, down, torch.randn(2, 3, 4, dtype=torch.float32))

    assert output.dtype == torch.bfloat16
    assert output.shape == (2, 3, 4)


def test_hunyuan_image3_rotary_preserves_query_key_dtype():
    from lightx2v.models.networks.hunyuan_image3.infer.utils import apply_rotary_pos_emb

    q = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)
    cos = torch.randn(1, 3, 4, dtype=torch.float32)
    sin = torch.randn(1, 3, 4, dtype=torch.float32)

    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

    assert q_out.dtype == torch.bfloat16
    assert k_out.dtype == torch.bfloat16


def test_hunyuan_image3_conv2d_matches_weight_dtype():
    from lightx2v.models.networks.hunyuan_image3.weights.common import HunyuanImage3Conv2dWeight

    conv = HunyuanImage3Conv2dWeight("weight", "bias", padding=1)
    conv.load(
        {
            "weight": torch.randn(2, 2, 3, 3, dtype=torch.bfloat16),
            "bias": torch.randn(2, dtype=torch.bfloat16),
        }
    )

    output = conv.apply(torch.randn(1, 2, 4, 4, dtype=torch.float32))

    assert output.dtype == torch.bfloat16
    assert output.shape == (1, 2, 4, 4)


def test_hunyuan_image3_moe_scatter_matches_destination_dtype():
    from lightx2v.models.networks.hunyuan_image3.infer.transformer_infer import HunyuanImage3TransformerInfer

    class FakeLinear:
        def __init__(self, in_features, out_features, dtype):
            self.weight = torch.randn(in_features, out_features, dtype=dtype)
            self.bias = None

        def apply(self, input_tensor):
            output = torch.empty(input_tensor.shape[0], self.weight.shape[1], dtype=input_tensor.dtype)
            return torch.mm(input_tensor, self.weight, out=output)

    class FakeExpert:
        def __init__(self):
            self.gate_and_up_proj = FakeLinear(4, 8, torch.bfloat16)
            self.down_proj = FakeLinear(4, 4, torch.bfloat16)

    infer = HunyuanImage3TransformerInfer(
        {
            "num_layers": 1,
            "hidden_size": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "attention_head_dim": 4,
            "hidden_act": "silu",
        }
    )
    phase = SimpleNamespace(
        is_moe=True,
        moe=SimpleNamespace(
            gate=FakeLinear(4, 2, torch.float32),
            moe_topk=1,
            experts=[FakeExpert(), FakeExpert()],
            shared_mlp=None,
        ),
    )

    output = infer.infer_mlp(phase, torch.randn(1, 3, 4, dtype=torch.float32))

    assert output.dtype == torch.float32
    assert output.shape == (1, 3, 4)


class _FakeHunyuanLinear:
    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias
        self.has_lora_branch = False

    def _get_actual_weight(self):
        return self.weight

    def apply(self, input_tensor):
        input_tensor = input_tensor.to(device=self.weight.device, dtype=self.weight.dtype)
        output = torch.mm(input_tensor, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output


def _fake_hunyuan_moe_phase(moe_impl="eager"):
    from lightx2v.models.networks.hunyuan_image3.weights.common import HunyuanImage3MoEWeights

    torch.manual_seed(7)
    hidden = 4
    intermediate = 6
    num_experts = 3
    moe = HunyuanImage3MoEWeights.__new__(HunyuanImage3MoEWeights)
    moe.num_experts = num_experts
    moe.moe_topk = 2
    moe.moe_impl = moe_impl
    moe.shared_mlp = None
    moe.gate = _FakeHunyuanLinear(torch.randn(hidden, num_experts, dtype=torch.float32) * 0.2)
    moe.experts = []
    for _ in range(num_experts):
        moe.experts.append(
            SimpleNamespace(
                gate_and_up_proj=_FakeHunyuanLinear(torch.randn(hidden, intermediate * 2, dtype=torch.bfloat16) * 0.2),
                down_proj=_FakeHunyuanLinear(torch.randn(intermediate, hidden, dtype=torch.bfloat16) * 0.2),
            )
        )
    moe.moe_weight = None
    moe.moe_weight_2 = None
    moe._flashinfer_weights_initialized = False
    moe._flashinfer_weight_device = None
    moe._flashinfer_weight_dtype = None
    return SimpleNamespace(is_moe=True, moe=moe)


def _fake_cutlass_fused_moe(input_tensor, token_selected_experts, token_final_scales, fc1_expert_weights, fc2_expert_weights, output_dtype, quant_scales, output=None, **kwargs):
    result = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], device=input_tensor.device, dtype=output_dtype)
    for token_idx in range(input_tensor.shape[0]):
        token = input_tensor[token_idx : token_idx + 1].to(output_dtype)
        for topk_idx in range(token_selected_experts.shape[1]):
            expert_idx = int(token_selected_experts[token_idx, topk_idx])
            projected = torch.mm(token, fc1_expert_weights[expert_idx].t())
            x1, x2 = projected.chunk(2, dim=-1)
            activated = x1 * torch.nn.functional.silu(x2)
            expert_out = torch.mm(activated, fc2_expert_weights[expert_idx].t())
            result[token_idx] += expert_out.squeeze(0) * token_final_scales[token_idx, topk_idx].to(output_dtype)
    if output is not None:
        output.copy_(result)
        return output
    return result


def test_hunyuan_image3_flashinfer_moe_packs_lightx2v_weights_like_torch_linear():
    phase = _fake_hunyuan_moe_phase("flashinfer")
    moe = phase.moe
    expert_gate_up = moe.experts[0].gate_and_up_proj.weight.clone()
    expert_down = moe.experts[0].down_proj.weight.clone()

    moe_weight, moe_weight_2 = moe.ensure_flashinfer_weights(torch.device("cpu"), torch.bfloat16)

    assert moe_weight.shape == (3, 12, 4)
    assert moe_weight_2.shape == (3, 4, 6)
    torch.testing.assert_close(moe_weight[0], expert_gate_up.t().contiguous())
    torch.testing.assert_close(moe_weight_2[0], expert_down.t().contiguous())
    assert moe.experts[0].gate_and_up_proj.weight.numel() == 0
    assert moe.experts[0].down_proj.weight.numel() == 0


def test_hunyuan_image3_flashinfer_moe_matches_eager_with_mocked_kernel(monkeypatch):
    from lightx2v.models.networks.hunyuan_image3.infer import transformer_infer
    from lightx2v.models.networks.hunyuan_image3.infer.transformer_infer import HunyuanImage3TransformerInfer

    monkeypatch.setattr(transformer_infer, "flashinfer_cutlass_fused_moe", _fake_cutlass_fused_moe)
    infer = HunyuanImage3TransformerInfer(
        {
            "num_layers": 1,
            "hidden_size": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "attention_head_dim": 4,
            "hidden_act": "silu",
            "moe_impl": "flashinfer",
        }
    )
    torch.manual_seed(11)
    hidden_states = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    eager_output = infer.infer_mlp(_fake_hunyuan_moe_phase("eager"), hidden_states)
    flashinfer_output = infer.infer_mlp(_fake_hunyuan_moe_phase("flashinfer"), hidden_states)

    torch.testing.assert_close(flashinfer_output.float(), eager_output.float(), atol=2e-2, rtol=2e-2)


def test_hunyuan_image3_flashinfer_moe_requires_flashinfer(monkeypatch):
    from lightx2v.models.networks.hunyuan_image3.infer import transformer_infer
    from lightx2v.models.networks.hunyuan_image3.infer.transformer_infer import HunyuanImage3TransformerInfer

    monkeypatch.setattr(transformer_infer, "flashinfer_cutlass_fused_moe", None)
    infer = HunyuanImage3TransformerInfer(
        {
            "num_layers": 1,
            "hidden_size": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "attention_head_dim": 4,
            "hidden_act": "silu",
            "moe_impl": "flashinfer",
        }
    )

    with pytest.raises(ImportError, match="flashinfer"):
        infer.infer_mlp(_fake_hunyuan_moe_phase("flashinfer"), torch.randn(1, 2, 4))


def test_hunyuan_image3_static_kv_cache_updates_requested_positions():
    from lightx2v.models.networks.hunyuan_image3.infer.kv_cache import HunyuanImage3StaticKVCache

    cache = HunyuanImage3StaticKVCache(num_layers=1, max_cache_len=5)
    first_keys = torch.arange(20, dtype=torch.float32).reshape(1, 1, 5, 4)
    first_values = first_keys + 100

    key_cache, value_cache = cache.update(first_keys, first_values, layer_idx=0, cache_position=torch.arange(5))

    assert key_cache.shape == (1, 1, 5, 4)
    assert torch.equal(key_cache, first_keys)
    assert torch.equal(value_cache, first_values)

    second_keys = torch.full((1, 1, 2, 4), -1.0)
    second_values = torch.full((1, 1, 2, 4), -2.0)
    key_cache, value_cache = cache.update(second_keys, second_values, layer_idx=0, cache_position=torch.tensor([[1, 4]]))

    assert torch.equal(key_cache[:, :, 0], first_keys[:, :, 0])
    assert torch.equal(key_cache[:, :, 1], second_keys[:, :, 0])
    assert torch.equal(key_cache[:, :, 4], second_keys[:, :, 1])
    assert torch.equal(value_cache[:, :, 1], second_values[:, :, 0])
    assert torch.equal(value_cache[:, :, 4], second_values[:, :, 1])


def test_hunyuan_image3_dynamic_kv_cache_returns_filled_prefix():
    from lightx2v.models.networks.hunyuan_image3.infer.kv_cache import HunyuanImage3StaticKVCache

    cache = HunyuanImage3StaticKVCache(num_layers=1, max_cache_len=6, dynamic=True)
    first_keys = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    first_values = first_keys + 100

    key_cache, value_cache = cache.update(first_keys, first_values, layer_idx=0, cache_position=torch.arange(3))

    assert key_cache.shape == (1, 1, 3, 4)
    assert value_cache.shape == (1, 1, 3, 4)

    next_keys = torch.full((1, 1, 2, 4), 5.0)
    next_values = torch.full((1, 1, 2, 4), 6.0)
    key_cache, value_cache = cache.update(next_keys, next_values, layer_idx=0, cache_position=torch.tensor([[3, 4]]))

    assert key_cache.shape == (1, 1, 5, 4)
    assert value_cache.shape == (1, 1, 5, 4)
    assert torch.equal(key_cache[:, :, 3], next_keys[:, :, 0])
    assert torch.equal(key_cache[:, :, 4], next_keys[:, :, 1])


def test_hunyuan_image3_runner_builds_cache_decode_positions_and_masks():
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {"cfg_distilled": True, "use_meanflow": False}
    prepared_inputs = {
        "image_mask": torch.tensor([[False, True, False, True], [False, True, False, True]]),
        "timesteps_index": torch.tensor([[0], [0]]),
        "guidance_index": torch.tensor([[2], [2]]),
        "timesteps_r_index": None,
        "attention_mask": torch.arange(2 * 1 * 4 * 4).reshape(2, 1, 4, 4),
    }

    position_ids = runner._build_denoise_cache_position_ids(prepared_inputs)
    attention_mask = runner._slice_denoise_cache_attention_mask(prepared_inputs["attention_mask"], position_ids)
    local_inputs = runner._build_denoise_cache_local_indices(prepared_inputs, position_ids)

    assert torch.equal(position_ids, torch.tensor([[0, 2, 1, 3], [0, 2, 1, 3]]))
    assert attention_mask.shape == (2, 1, 4, 4)
    assert torch.equal(attention_mask[0], prepared_inputs["attention_mask"][0].index_select(1, position_ids[0]))
    assert torch.equal(local_inputs["image_mask"], torch.tensor([[False, False, True, True], [False, False, True, True]]))
    assert torch.equal(local_inputs["timesteps_index"], torch.tensor([[0], [0]]))
    assert torch.equal(local_inputs["guidance_index"], torch.tensor([[1], [1]]))
    assert local_inputs["timesteps_r_index"] is None


def test_hunyuan_image3_transformer_attention_uses_kv_cache():
    from lightx2v.models.networks.hunyuan_image3.infer.kv_cache import HunyuanImage3StaticKVCache
    from lightx2v.models.networks.hunyuan_image3.infer.transformer_infer import HunyuanImage3TransformerInfer

    class FakeLinear:
        def __init__(self, in_features, out_features):
            self.weight = torch.randn(in_features, out_features)
            self.bias = None

        def apply(self, input_tensor):
            return torch.mm(input_tensor, self.weight)

    infer = HunyuanImage3TransformerInfer(
        {
            "num_layers": 1,
            "hidden_size": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "attention_head_dim": 4,
            "hidden_act": "silu",
        }
    )
    phase = SimpleNamespace(qkv_proj=FakeLinear(4, 12), o_proj=FakeLinear(4, 4), query_layernorm=None, key_layernorm=None)
    cache = HunyuanImage3StaticKVCache(num_layers=1, max_cache_len=5)

    first = infer.infer_attention(
        0,
        phase,
        torch.randn(1, 5, 4),
        attention_mask=None,
        position_ids=torch.arange(5).reshape(1, 5),
        custom_pos_emb=None,
        past_key_values=cache,
    )
    second = infer.infer_attention(
        0,
        phase,
        torch.randn(1, 2, 4),
        attention_mask=None,
        position_ids=torch.tensor([[0, 3]]),
        custom_pos_emb=None,
        past_key_values=cache,
    )

    assert first.shape == (1, 5, 4)
    assert second.shape == (1, 2, 4)
    assert cache.layers[0].key.shape == (1, 1, 5, 4)


def test_hunyuan_image3_text_generation_uses_dynamic_kv_cache_for_uncached_tokens():
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    class FakeModel:
        def __init__(self):
            self.next_tokens = [10, 12]
            self.calls = []

        def infer(self, inputs):
            self.calls.append(
                {
                    "input_ids": inputs["input_ids"].clone(),
                    "position_ids": inputs["position_ids"].clone(),
                    "use_cache": inputs.get("use_cache"),
                    "past_key_values": inputs.get("past_key_values"),
                }
            )
            vocab_size = 16
            logits = torch.zeros(1, inputs["input_ids"].shape[1], vocab_size)
            logits[:, -1, self.next_tokens[len(self.calls) - 1]] = 100.0
            return {"logits": logits}

    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {"text_do_sample": False, "max_new_tokens": 5, "enable_kv_cache": True}
    runner.hunyuan_generation_config = SimpleNamespace(max_new_tokens=5, do_sample=False)
    runner.hunyuan_config = SimpleNamespace(max_position_embeddings=16)
    runner.hunyuan_cached_rope = lambda *args, **kwargs: None
    runner.hunyuan_image_processor = SimpleNamespace(prepare_full_attn_slices=lambda tokenizer_output, batch_idx: [])
    runner.model = FakeModel()
    plan = SimpleNamespace(first_bot_task="think", stage_transitions=[(10, [11])], final_stop_tokens=[12])

    generated_tokens = runner._generate_text_tokens(torch.tensor([[1, 2]]), SimpleNamespace(), plan)

    assert generated_tokens == [10, 11, 12]
    assert len(runner.model.calls) == 2
    assert torch.equal(runner.model.calls[0]["input_ids"], torch.tensor([[1, 2]]))
    assert torch.equal(runner.model.calls[0]["position_ids"], torch.tensor([[0, 1]]))
    assert torch.equal(runner.model.calls[1]["input_ids"], torch.tensor([[10, 11]]))
    assert torch.equal(runner.model.calls[1]["position_ids"], torch.tensor([[2, 3]]))
    assert runner.model.calls[0]["past_key_values"] is runner.model.calls[1]["past_key_values"]
    assert runner.model.calls[0]["use_cache"] is True


def test_hunyuan_image3_model_taylor_cache_skips_transformer_between_full_steps():
    from lightx2v.models.networks.hunyuan_image3.infer.module_io import HunyuanImage3PreInferOutput
    from lightx2v.models.networks.hunyuan_image3.model import HunyuanImage3Model

    class FakePreInfer:
        def infer(self, weights, inputs):
            value = float(inputs["cache_dic"]["current_step"])
            return HunyuanImage3PreInferOutput(hidden_states=torch.full((1, 3, 4), value))

    class FakeTransformerInfer:
        def __init__(self):
            self.calls = 0

        def infer(self, weights, pre_infer_out):
            self.calls += 1
            return pre_infer_out.hidden_states + 10.0

    class FakePostInfer:
        def infer(self, weights, hidden_states, pre_infer_out):
            return {"diffusion_prediction": hidden_states}

    model = HunyuanImage3Model.__new__(HunyuanImage3Model)
    model.config = {"seq_parallel": False}
    model.pre_weight = None
    model.transformer_weights = None
    model.post_weight = None
    model.pre_infer = FakePreInfer()
    model.transformer_infer = FakeTransformerInfer()
    model.post_infer = FakePostInfer()

    cache_dic = {
        "current_step": 0,
        "cache_interval": 2,
        "max_order": 2,
        "num_steps": 3,
        "enable_first_enhance": False,
        "first_enhance_steps": 3,
        "enable_tailing_enhance": False,
        "tailing_enhance_steps": 1,
        "low_freqs_order": 2,
        "high_freqs_order": 2,
    }

    outputs = []
    for step in range(3):
        cache_dic["current_step"] = step
        outputs.append(model.infer({"cache_dic": cache_dic})["diffusion_prediction"])

    assert model.transformer_infer.calls == 2
    assert torch.equal(outputs[0], torch.full((1, 3, 4), 10.0))
    assert torch.equal(outputs[1], outputs[0])
    assert torch.equal(outputs[2], torch.full((1, 3, 4), 12.0))


def test_hunyuan_image3_runner_builds_taylor_cache_config_from_runtime_config():
    from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner

    runner = HunyuanImage3Runner.__new__(HunyuanImage3Runner)
    runner.config = {
        "use_taylor_cache": True,
        "taylor_cache_interval": 5,
        "taylor_cache_order": 2,
        "taylor_cache_enable_first_enhance": True,
        "taylor_cache_first_enhance_steps": 4,
        "taylor_cache_enable_tailing_enhance": True,
        "taylor_cache_tailing_enhance_steps": 2,
        "taylor_cache_low_freqs_order": 1,
        "taylor_cache_high_freqs_order": 2,
    }

    cache_dic = runner._build_taylor_cache_dic(num_steps=50)

    assert cache_dic == {
        "counter": 0,
        "current_step": 0,
        "cache_interval": 5,
        "max_order": 2,
        "num_steps": 50,
        "enable_first_enhance": True,
        "first_enhance_steps": 4,
        "enable_tailing_enhance": True,
        "tailing_enhance_steps": 2,
        "low_freqs_order": 1,
        "high_freqs_order": 2,
        "enable_force_control": False,
        "force_compute": False,
    }


def test_hunyuan_image3_rms_norm_ignores_inactive_diff_tensor():
    from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightFP32

    rms = RMSWeightFP32("norm.weight")
    rms.weight = torch.ones(4)
    rms.weight_diff = torch.ones(5)
    rms.has_diff = False

    output = rms.apply(torch.ones(1, 4))

    assert output.shape == (1, 4)


def test_hunyuan_image3_rms_norm_aligns_active_diff_to_weight():
    from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightFP32

    rms = RMSWeightFP32("norm.weight")
    rms.weight = torch.ones(4, device="meta", dtype=torch.bfloat16)
    rms.weight_diff = torch.ones(4, dtype=torch.float32)
    rms.has_diff = True

    actual_weight = rms._get_actual_weight()

    assert actual_weight.device == rms.weight.device
    assert actual_weight.dtype == rms.weight.dtype


def test_hunyuan_image3_rms_norm_register_diff_marks_active():
    from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightFP32

    rms = RMSWeightFP32("norm.weight")

    rms.register_diff({rms.weight_diff_name: torch.ones(4)})

    assert rms.has_diff is True
