"""Compare native EdenLive boundaries with a local upstream checkout."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def compare(name, actual, expected):
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    difference = (actual.float() - expected.float()).abs()
    result = dict(max_abs=difference.max().item(), mean_abs=difference.mean().item(), rmse=difference.square().mean().sqrt().item())
    print(name, json.dumps(result), flush=True)
    return result


def main():
    import lightx2v.common.ops  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--converted-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-cls", choices=["wan21_edenlive", "wan21_edenlive_ar", "wan22_edenlive", "wan22_edenlive_ar"], required=True)
    parser.add_argument("--boundary", choices=["dino", "dit", "dynamic", "text"], required=True)
    parser.add_argument("--wan-path")
    parser.add_argument("--offload", choices=["none", "block", "model"], default="none")
    parser.add_argument("--kv-offload", action="store_true")
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--cfg-parallel", action="store_true")
    args = parser.parse_args()
    if args.sp or args.cfg_parallel:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    sys.path.insert(0, args.upstream)
    root = Path(args.model_path)
    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    if args.boundary == "text":
        from transformers import UMT5EncoderModel

        from lightx2v.models.runners.wan import edenlive_runner  # noqa: F401
        from lightx2v.utils.registry_factory import RUNNER_REGISTER
        from lightx2v.utils.set_config import build_startup_config

        config = build_startup_config(
            dict(
                model_cls=args.model_cls,
                task="i2v",
                model_path=args.model_path,
                config_json=args.config,
            )
        )
        config["t5_original_ckpt"] = str(Path(args.wan_path) / "models_t5_umt5-xxl-enc-bf16.pth")
        config["t5_tokenizer_path"] = str(root / "umt5/tokenizer")
        config["t5_cpu_offload"] = args.offload != "none"
        encoder = RUNNER_REGISTER[args.model_cls](config).load_text_encoder()[0]
        reference = UMT5EncoderModel.from_pretrained(str(root / "umt5/text_encoder"), torch_dtype=torch.bfloat16).cuda().eval()
        for text in ("A person walks across an icy landscape.", "blurry, low quality"):
            ids, mask = encoder.tokenizer([text], return_mask=True, add_special_tokens=True)
            expected = reference(input_ids=ids.cuda(), attention_mask=mask.cuda()).last_hidden_state[0, : mask.sum()]
            actual = encoder.infer([text])[0]
            compare("text", actual, expected)
            torch.testing.assert_close(actual, expected, atol=args.atol, rtol=0)
        return
    if args.boundary == "dino":
        from edenlive.models.proxy_dino.encoder import create_checkpoint_dinov3_encoder, encode_proxy_dino

        from lightx2v.models.input_encoders.edenlive.proxy_dino import ProxyDINO

        for resize in ["bilinear", "bicubic_center_crop", "bicubic"]:
            reference = create_checkpoint_dinov3_encoder(str(root / "vision-encoder/backbone"), str(root / "vision-encoder/projection.safetensors")).cuda().float()
            native = ProxyDINO(str(root / "vision-encoder/backbone"), str(root / "vision-encoder/projection.safetensors"), resize)
            images = torch.rand(3, 3, 128, 224, device="cuda")
            captures = []
            for block in native.blocks:
                original = block.norm1.apply

                def capture(x, original=original):
                    captures.append(x.clone())
                    return original(x)

                block.norm1.apply = capture
            projection = {}
            reference.patch_projection.register_forward_hook(lambda module, values, output: projection.update(reference_input=values[0], reference_output=output))
            original_projection = native.projection.apply

            def capture_projection(x):
                result = original_projection(x)
                projection.update(native_input=x, native_output=result)
                return result

            native.projection.apply = capture_projection
            expected = encode_proxy_dino(reference, images, image_size=(480, 832), resize_mode=resize, projection_autocast=True)
            actual = native.infer(images)
            compare(resize, actual, expected.permute(2, 1, 0, 3, 4))
            torch.testing.assert_close(actual, expected.permute(2, 1, 0, 3, 4), atol=args.atol, rtol=0)
            for boundary in ("input", "output"):
                reference_tensor = projection[f"reference_{boundary}"].flatten(0, 2).to(projection[f"native_{boundary}"].dtype)
                compare(f"{resize}-projection-{boundary}", projection[f"native_{boundary}"], reference_tensor)

            pixels = native.preprocess(images)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                states = reference.backbone(pixel_values=pixels, output_hidden_states=True).hidden_states
            for index in (0, 1, 23):
                compare(f"{resize}-block-input-{index}", captures[index], states[index])
            del reference, native
        return
    from diffusers import WanTransformer3DModel
    from edenlive.core.types import RuntimeContext
    from edenlive.models.adapters.wan22_diffusers.wan_ops import allocate_wan_kv_cache, patch_wan_transformer_forward, wan_causal_forward_with_cache

    from lightx2v.common.kvcache.manager import build_self_attn_kv_cache
    from lightx2v.models.runners.wan import edenlive_runner  # noqa: F401
    from lightx2v.models.schedulers.wan.edenlive.scheduler import EdenLiveARScheduler
    from lightx2v.utils.registry_factory import RUNNER_REGISTER
    from lightx2v.utils.set_config import build_startup_config

    config = build_startup_config(dict(model_cls=args.model_cls, task="i2v", model_path=args.model_path, config_json=args.config))
    source = root / Path(config["dit_original_ckpt"]).name
    config["dit_original_ckpt"] = args.converted_path
    if args.sp or args.cfg_parallel:
        from lightx2v.utils.set_config import init_parallel

        config["parallel"] = {"cfg_p_size" if args.cfg_parallel else "seq_p_size": dist.get_world_size()}
        if args.cfg_parallel:
            config["enable_cfg"] = True
            config["sample_guide_scale"] = 3.0
        init_parallel(config)
    config["cpu_offload"] = args.offload != "none"
    config["offload_granularity"] = args.offload if args.offload != "none" else "block"
    runner = RUNNER_REGISTER[args.model_cls](config)
    scheduler = runner.scheduler
    if isinstance(scheduler, EdenLiveARScheduler):
        config["ar_config"]["kv_offload"] = args.kv_offload
    from edenlive.models.adapters.wan22_diffusers.adapter import Wan22DiffusersAdapter
    from edenlive.models.adapters.wan22_diffusers.channel_concat import enable_wan_channel_concat

    reference = Wan22DiffusersAdapter._load_widened_transformer(WanTransformer3DModel, source, torch.bfloat16)
    enable_wan_channel_concat(reference, 1024, condition_rmsnorm=config["proxy_condition_rmsnorm"])
    reference = reference.to(device="cuda", dtype=torch.bfloat16).eval()
    patch_wan_transformer_forward(reference)
    native = runner.load_transformer()
    native.set_scheduler(scheduler)
    if args.boundary == "dynamic":
        from types import SimpleNamespace

        from edenlive.models.adapters.wan22_diffusers.wan_ops import _cross_attention

        hidden = torch.randn(18, config["dim"], device="cuda", dtype=torch.bfloat16)
        text = torch.randn(1024, config["dim"], device="cuda", dtype=torch.bfloat16)
        native.transformer_infer.current_pre = SimpleNamespace(grid_sizes=SimpleNamespace(tuple=(3, 2, 3)), adapter_args={"text_segments": [2, 1]})
        phase = native.transformer_weights.blocks[0].compute_phases[1]
        actual = native.transformer_infer.infer_cross_attn_output(phase, hidden, text)
        normalized = reference.blocks[0].norm2(hidden[None].float()).to(hidden.dtype)
        expected = torch.cat(
            (_cross_attention(reference.blocks[0].attn2, normalized[:, :12], text[None, :512]), _cross_attention(reference.blocks[0].attn2, normalized[:, 12:], text[None, 512:])), dim=1
        )[0]
        compare("dynamic-segment-attention", actual, expected)
        torch.testing.assert_close(actual, expected, atol=args.atol, rtol=0)
        return
    if not isinstance(scheduler, EdenLiveARScheduler):
        from edenlive.models.adapters.wan22_diffusers.wan_ops import wan_parallel_forward

        latent_shape = [config["out_dim"], 7, 4, 6]
        scheduler.prepare(42, latent_shape, {"clean_first_frame": torch.randn(config["out_dim"], 1, 4, 6, device="cuda", dtype=torch.bfloat16)})
        scheduler.step_pre(0)
        control = torch.randn(1, 1024, 7, 4, 6, device="cuda", dtype=torch.bfloat16)
        text = torch.randn(1, 512, 4096, device="cuda", dtype=torch.bfloat16)
        image_condition = {}
        latent_input = scheduler.latents_input[None]
        image_embeds = None
        if config["use_image_encoder"]:
            image_condition["vae_encoder_out"] = torch.randn(20, 7, 4, 6, device="cuda", dtype=torch.bfloat16)
            image_condition["clip_encoder_out"] = torch.randn(257, 1280, device="cuda", dtype=torch.bfloat16)
            image_embeds = image_condition["clip_encoder_out"][None]
            latent_input = torch.cat((latent_input, image_condition["vae_encoder_out"][None]), dim=1)
        inputs = dict(proxy_features=control, text_encoder_output=dict(context=text, context_null=text), image_encoder_output=image_condition)
        timestep = scheduler.timestep_input[None] if scheduler.timestep_input.numel() > 1 else scheduler.timestep_input
        if timestep.ndim == 2:
            tokens_per_frame = (latent_shape[2] // config["patch_size"][1]) * (latent_shape[3] // config["patch_size"][2])
            timestep = timestep.repeat_interleave(tokens_per_frame, dim=1)
        expected = wan_parallel_forward(reference, torch.cat((latent_input, control), dim=1), timestep, text, encoder_hidden_states_image=image_embeds).sample[0]
        native.infer(inputs)
        compare("bidir-dit", scheduler.noise_pred, expected)
        torch.testing.assert_close(scheduler.noise_pred, expected, atol=args.atol, rtol=0)
        return
    if args.sp:
        reference_block = {}
        reference.blocks[0].register_forward_hook(lambda module, inputs, output: reference_block.update(output=output.detach()))
        original_block = native.transformer_infer.infer_block

        def block_boundary(block, x, pre, **kwargs):
            result = original_block(block, x, pre, **kwargs)
            if native.transformer_infer.block_idx == 0:
                gathered = [torch.empty_like(result) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered, result)
                compare("SP-block-0", torch.cat(gathered), reference_block["output"][0])
            return result

        native.transformer_infer.infer_block = block_boundary
    scheduler.prepare(42, [config["out_dim"], 25, 4, 6], {"clean_first_frame": torch.randn(config["out_dim"], 1, 4, 6, device="cuda", dtype=torch.bfloat16)})
    context = RuntimeContext(device=torch.device("cuda"))
    context.cache["kv_cache"] = dict(context=dict(rope_mode="fixed_window"), chunk_size=3)
    reference_cache = allocate_wan_kv_cache(reference, batch_size=1, max_tokens=18 * 6, device=torch.device("cuda"), dtype=torch.bfloat16)
    native_cache = build_self_attn_kv_cache(config, config["ar_config"], 18 * 6, torch.bfloat16, torch.device("cuda"))
    native_cache._init_kv_buffer()
    native.transformer_infer.cache_pools = {True: native_cache}
    reference_caches = {True: reference_cache}
    if args.cfg_parallel:
        reference_caches[False] = allocate_wan_kv_cache(reference, batch_size=1, max_tokens=18 * 6, device=torch.device("cuda"), dtype=torch.bfloat16)
        pool = build_self_attn_kv_cache(config, config["ar_config"], 18 * 6, torch.bfloat16, torch.device("cuda"))
        pool._init_kv_buffer()
        native.transformer_infer.cache_pools[False] = pool

    def reference_predict(x, timestep, text, image_embeds, start, cache_update, negative):
        predictions = {}
        for branch in [True, False] if args.cfg_parallel else [True]:
            model_input = x
            if not branch and config["proxy_unconditional_control"] == "zero":
                model_input = x.clone()
                model_input[:, -1024:] = 0
            predictions[branch] = wan_causal_forward_with_cache(
                reference,
                model_input,
                timestep,
                text if branch else negative,
                kv_cache=reference_caches[branch],
                current_start=start * 6,
                max_attention_tokens=18 * 6,
                sink_attention_tokens=6 * 6,
                cache_update=cache_update,
                runtime_context=context,
                encoder_hidden_states_image=image_embeds,
            ).sample[0]
        return predictions[False] + 3 * (predictions[True] - predictions[False]) if args.cfg_parallel else predictions[True]

    for start in range(0, 25, 3):
        scheduler.step_pre(0, chunk_start=start)
        count = scheduler.latents_input.shape[1]
        control = torch.randn(1, 1024, count, 4, 6, device="cuda", dtype=torch.bfloat16)
        text = torch.randn(1, 512, 4096, device="cuda", dtype=torch.bfloat16)
        negative = torch.randn_like(text) if args.cfg_parallel else None
        image_condition = {}
        image_embeds = None
        latent_input = scheduler.latents_input[None]
        if config["use_image_encoder"]:
            image_condition["vae_encoder_out"] = torch.randn(20, 25, 4, 6, device="cuda", dtype=torch.bfloat16)
            image_condition["clip_encoder_out"] = torch.randn(257, 1280, device="cuda", dtype=torch.bfloat16)
            image_embeds = image_condition["clip_encoder_out"][None]
            latent_input = torch.cat((latent_input, image_condition["vae_encoder_out"][None, :, start : start + count]), dim=1)
        inputs = dict(proxy_features=control, text_encoder_output=dict(context=text, context_null=negative), image_encoder_output=image_condition)
        x = torch.cat((latent_input, control), dim=1)
        timestep = scheduler.timestep_input[None] if scheduler.timestep_input.numel() > 1 else scheduler.timestep_input
        expected = reference_predict(x, timestep, text, image_embeds, start, False, negative)
        native.infer(inputs)
        actual = scheduler.noise_pred
        compare(f"dit-{start}", actual, expected)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=args.atol)
        scheduler.cache_update = True
        scheduler.step_index = scheduler.infer_steps - 1
        scheduler.timestep_input = torch.zeros(1, device="cuda")
        expected = reference_predict(x, scheduler.timestep_input, text, image_embeds, start, True, negative)
        native.infer(inputs)
        actual = scheduler.noise_pred
        compare(f"commit-{start}", actual, expected)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=args.atol)
    if args.sp or args.cfg_parallel:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
