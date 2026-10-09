#!/usr/bin/env python3
"""Compare real old/new H3 forward/backward on a fixed Ref2AV condition.

Run this file with torchrun and PYTHONPATH pointing to ONE checkout. It imports
that checkout's actual packing, wrapper, predictor and losses. No training
config, checkpoint or cache is modified. This is a controlled gradient test,
not a three-role DMD training run: score predictions for the student loss are
a fixed counterfactual fixture, so initial equal fake/teacher weights cannot
make the student gradient trivially zero. Fake CPU offload is diagnostic-only.
"""

import argparse
import gc
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from lightx2v_train.model_zoo.native.minimax_h3 import (
    audio_latent_num_frames,
    init_empty_minimax_h3_transformer,
    stream_load_minimax_h3_transformer,
    video_latent_num_frames,
)
from lightx2v_train.runtime.distributed import cleanup_distributed, init_distributed
from lightx2v_train.trainers.dmd.math import dmd_loss_with_stats


def local(value):
    return value.to_local() if hasattr(value, "to_local") else value


def gradient_snapshot(model):
    names, stats, sketch, hashes = [], [], {}, {}
    for name, parameter in model.transformer.named_parameters():
        if parameter.grad is None:
            continue
        value = local(parameter.grad).detach().contiguous().cpu()
        flat = value.reshape(-1)
        names.append(name)
        finite = bool(torch.isfinite(flat).all())
        stats.append([flat.numel(), flat.float().square().sum().item(), flat.abs().max().item() if flat.numel() else 0.0, int(finite)])
        sample_count = min(1024, flat.numel())
        if sample_count <= 1:
            indices = torch.zeros(sample_count, dtype=torch.long)
        else:
            # Integer arithmetic guarantees the final index is numel - 1;
            # float32 linspace can round the endpoint up to numel for large
            # parameter shards and make this diagnostic fail after backward.
            indices = torch.arange(sample_count, dtype=torch.long) * (flat.numel() - 1) // (sample_count - 1)
        sketch[name] = flat[indices].float().clone()
        hashes[name] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    return {"names": names, "stats": torch.tensor(stats, dtype=torch.float64), "sketch": sketch, "hashes": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("old", "new"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--rows", default="0,1", help="Zero-based manifest row for each of the two ranks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("fake", "student", "both"), default="both")
    parser.add_argument("--fake-cpu-offload", action="store_true")
    parser.add_argument("--height", type=int, help="Diagnostic-only target geometry override")
    parser.add_argument("--width", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--new-numerics", choices=("legacy", "fp32"), default="legacy", help="Production H3 arithmetic to test in the new checkout")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", 0)) != 2:
        parser.error("Exactly two torchrun ranks are required")
    rows = [int(value) for value in args.rows.split(",")]
    if len(rows) != 2 or min(rows) < 0:
        parser.error("--rows must contain two nonnegative manifest indices")
    rank = int(os.environ["RANK"])
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    started = time.monotonic()

    def event(stage, **fields):
        record = {"rank": rank, "stage": stage, "elapsed_s": round(time.monotonic() - started, 2), **fields}
        if torch.cuda.is_initialized():
            record["allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
            record["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        print(json.dumps(record), flush=True)
        with (args.output / f"rank{rank}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    config = {
        "model": {"running_dtype": "bf16"},
        "distributed": {"backend": "nccl", "timeout_minutes": 20, "sequence_parallel": {"enabled": False, "size": 1}, "fsdp2": {"enabled": True, "size": 2}},
        "training": {"dmd": {"num_inference_steps": 8}},
    }
    init_distributed(config)
    mesh = init_device_mesh("cuda", (2,))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    seed = 20261009 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    old = args.implementation == "old"
    if old:
        from lightx2v_train.model_zoo.minimax_h3_ref2av import MiniMaxH3Ref2AVModel as Model
        from lightx2v_train.trainers.dmd.math import weighted_mse_pair
        from lightx2v_train.trainers.dmd.minimax_h3_trainer import MiniMaxH3Ref2AVDmdTrainer as Trainer
    else:
        from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import MiniMaxH3JointLatents, MiniMaxH3LatentShape
        from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability import MiniMaxH3DistributionMatchingCapability as Capability
        from lightx2v_train.model_zoo.minimax_h3.minimax_h3_ref2av import MiniMaxH3Ref2AVModel as Model

    records = []
    with args.metadata.open(encoding="utf-8") as handle:
        for index, line in enumerate(line for line in handle if line.strip()):
            if index in rows:
                records.append((index, json.loads(line)))
            if index >= max(rows):
                break
    record = dict(records)[rows[rank]]
    cache_path = Path(record["condition_path"])
    if not cache_path.is_absolute():
        cache_path = args.metadata.parent / cache_path
    batch = torch.load(cache_path, map_location="cpu", weights_only=True)
    raw = batch["conditioning"]["positive"]
    raw = dict(raw)
    for flag, field in ((args.height, "target_height"), (args.width, "target_width"), (args.frames, "target_num_frames")):
        if flag is not None:
            raw[field] = flag
    height, width, frames = (int(raw[name]) for name in ("target_height", "target_width", "target_num_frames"))
    shape = {
        "batch_size": 1,
        "num_frames": frames,
        "latent_frames": video_latent_num_frames(frames),
        "latent_height": height // 16,
        "latent_width": width // 16,
        "audio_latents": audio_latent_num_frames(frames),
    }
    shape["video_tokens"] = (1, shape["latent_frames"] * (height // 32) * (width // 32), 96)
    shape["audio_tokens"] = (1, shape["audio_latents"] * 2, 32)
    native_shape = shape if old else MiniMaxH3LatentShape(**{key: value for key, value in shape.items() if key != "batch_size"})
    # Physical sigmas are shared exactly; deliberately exclude scheduler RNG
    # and inverse/forward-shift roundoff from the first comparison.
    base = torch.tensor(0.3, device=device, dtype=torch.float32)
    sigmas = tuple(shift * base / (1 + (shift - 1) * base) for shift in (12.0, 3.0))
    source = tuple(torch.randn(shape[key], device=device, dtype=torch.float32) for key in ("video_tokens", "audio_tokens"))
    noises = tuple(torch.randn_like(value) for value in source)
    renoised = tuple((1 - sigma) * value + sigma * noise for value, noise, sigma in zip(source, noises, sigmas))
    target = tuple(value - noise for value, noise in zip(source, noises))
    # Actual root-FSDP BF16 projections return BF16 velocities. Preserve this
    # in the controlled scoring fixture to exercise both old scalar products.
    score_fixture = tuple(torch.randn_like(value).bfloat16() for value in source)
    score_offset = tuple((0.01 * torch.randn_like(value)).bfloat16() for value in source)
    payload = {
        "metadata": {
            "implementation": args.implementation,
            "physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "rank": rank,
            "model_path": str(args.model.resolve()),
            "cache_path": str(cache_path.resolve()),
            "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            "shape": shape,
            "seed": seed,
            "student_param_dtype": "bf16",
            "fake_param_dtype": "fp32",
            "running_dtype": "bf16",
            "fake_cpu_offload": args.fake_cpu_offload,
            "new_numerics": args.new_numerics,
            "scope": "fixed_input_real_H3_fake_MSE_and_student_DMD_surrogate_not_full_training",
        },
        "tensors": {},
        "gradients": {},
    }

    def capture(name, value):
        payload["tensors"][name] = value.detach().cpu().clone()

    for index, (src, noise, inp, sigma) in enumerate(zip(source, noises, renoised, sigmas)):
        modality = ("video", "audio")[index]
        for field, value in (("source", src), ("noise", noise), ("input", inp), ("sigma", sigma)):
            capture(f"{field}.{modality}", value)
        capture(f"score_fixture.{modality}", score_fixture[index])
        capture(f"score_offset.{modality}", score_offset[index])

    def build(role):
        model = Model(config)
        model.patch_size = (1, 2, 2)
        model.video_latent_channels, model.audio_latent_channels = 24, 32
        model.vae_spatial_scale_factor, model.text_dim = 16, 5120
        model.use_autocast = False
        model.transformer = init_empty_minimax_h3_transformer(
            args.model, component_name="transformer_ref", torch_dtype=torch.float32 if role == "fake" else torch.bfloat16, attention_backend="_flash_3_hub"
        )
        if role == "student":
            model.add_lora(128, 8, ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"])
            model.set_lora_trainable()
        else:
            model.set_full_trainable()
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
        options = {"mesh": mesh, "mp_policy": mp}
        if role == "fake" and args.fake_cpu_offload:
            options["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
        for entry in model.fsdp2_shard_plan({"reshard_after_forward": {"root_reshard": False, "block_reshard": True}}):
            modules = [entry["module"]] if "module" in entry else entry["modules"]
            for module in modules:
                fully_shard(module, reshard_after_forward=entry["reshard_after_forward"], **options)
        load_device = torch.device("cpu") if role == "fake" and args.fake_cpu_offload else device
        stream_load_minimax_h3_transformer(model.transformer, args.model / "transformer_ref", device=load_device, lora_seed=42)
        # Offloaded Parameters stay CPU, but this nonpersistent rotary buffer
        # is used directly by the CUDA forward and is not a sharded Parameter.
        model.transformer.rope.inv_freq = model.transformer.rope.inv_freq.to(device)
        model.enable_gradient_checkpointing()
        model.transformer.train()
        if old:
            predictor = Trainer.__new__(Trainer)
            predictor.model = model
            predictor._layout_cache = OrderedDict()
            predictor.layout_cache_size = 16
        else:
            predictor = Capability(model, {"video_flow_shift": 12, "audio_flow_shift": 3, "legacy_numerics": args.new_numerics == "legacy"})
            predictor._modality_sigmas = lambda sigma: sigmas
        return model, predictor

    def predict(model, predictor, condition):
        if old:
            return predictor._predict_velocity(model, renoised, sigmas, condition, shape)
        output = predictor.predict_velocity(MiniMaxH3JointLatents(*renoised, native_shape), base, condition)
        return output.video, output.audio

    def prepare(model, predictor):
        condition = model.prepare_text_condition(raw)
        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed_all(seed + 1)
        condition = predictor._prepare_rollout_condition(condition, shape) if old else predictor._prepare_condition_for_rollout(condition, lambda value: value)
        layout = predictor._layout(condition, native_shape)
        for field in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices"):
            capture("layout." + field, getattr(layout, field))
        for field in ("prompt_embeds", "condition_video_latents", "noised_condition_video_latents", "condition_audio_latents"):
            if condition.get(field) is not None:
                capture("condition." + field, condition[field])
        event("condition", tokens=layout.token_tags.numel(), refs=len(condition["references"]), height=height, width=width, frames=frames)
        return condition

    def student_loss(generated, fake_x0, teacher_x0):
        return sum(dmd_loss_with_stats(x, f, t, normalize=True, normalization_epsilon=0.0, reduction="mean")[0] for x, f, t in zip(generated, fake_x0, teacher_x0))

    try:
        event("start", metadata=payload["metadata"])
        roles = ("fake", "student") if args.role == "both" else (args.role,)
        for role in roles:
            event(role + "_load_start")
            model, predictor = build(role)
            condition = prepare(model, predictor)
            event(role + "_loaded")
            if role == "fake":
                pred = predict(model, predictor, condition)
                if old:
                    loss = weighted_mse_pair(pred, target, 1.0, 1.0)
                else:
                    loss = predictor.regression_loss(MiniMaxH3JointLatents(*pred, native_shape), MiniMaxH3JointLatents(*target, native_shape))
                for modality, prediction, expected in zip(("video", "audio"), pred, target):
                    capture("fake.pred." + modality, prediction)
                    capture("fake.target." + modality, expected)
                capture("fake.loss", loss)
                event("fake_forward", loss=loss.item(), output_dtype=str(pred[0].dtype))
                loss.backward()
                torch.cuda.synchronize()
                event("fake_backward")
                payload["gradients"]["fake"] = gradient_snapshot(model)
                event("fake_gradient_saved", params=len(payload["gradients"]["fake"]["names"]))
                del pred, loss
            else:
                # First preserve each checkout's x0 formula. Second unify ONLY
                # x0 precision, isolating network/loss/backward from that drift.
                for pass_name in ("original", "common_x0"):
                    model.transformer.zero_grad(set_to_none=True)
                    pred = predict(model, predictor, condition)
                    use_old_formula = old and pass_name == "original"
                    generated = tuple(
                        value + sigma * velocity if use_old_formula else value.float() + sigma.reshape(1, 1, 1) * velocity.float() for value, velocity, sigma in zip(renoised, pred, sigmas)
                    )
                    fake_x0 = tuple(
                        value + sigma * velocity if use_old_formula else value + sigma.reshape(1, 1, 1) * velocity.float() for value, velocity, sigma in zip(renoised, score_fixture, sigmas)
                    )
                    teacher_x0 = tuple(
                        value + sigma * (velocity + delta) if use_old_formula else value + sigma.reshape(1, 1, 1) * (velocity + delta).float()
                        for value, velocity, delta, sigma in zip(renoised, score_fixture, score_offset, sigmas)
                    )
                    if not old and pass_name == "original":
                        # Exercise the new production conversion, not a copy
                        # of its formula that could hide implementation drift.
                        inp = MiniMaxH3JointLatents(*renoised, native_shape)

                        def native_x0(values):
                            result = predictor.x0_from_velocity(inp, MiniMaxH3JointLatents(*values, native_shape), base)
                            return result.video, result.audio

                        generated = native_x0(pred)
                        fake_x0 = native_x0(score_fixture)
                        teacher_x0 = native_x0(tuple(v + d for v, d in zip(score_fixture, score_offset)))
                    loss = student_loss(generated, fake_x0, teacher_x0)
                    for modality, prediction, x0 in zip(("video", "audio"), pred, generated):
                        capture(f"student.{pass_name}.pred.{modality}", prediction)
                        capture(f"student.{pass_name}.x0.{modality}", x0)
                    for modality, fx, tx in zip(("video", "audio"), fake_x0, teacher_x0):
                        capture(f"student.{pass_name}.fake_x0.{modality}", fx)
                        capture(f"student.{pass_name}.teacher_x0.{modality}", tx)
                    capture("student." + pass_name + ".loss", loss)
                    event("student_forward", pass_name=pass_name, loss=loss.item(), output_dtype=str(pred[0].dtype))
                    loss.backward()
                    torch.cuda.synchronize()
                    event("student_backward", pass_name=pass_name)
                    payload["gradients"]["student_" + pass_name] = gradient_snapshot(model)
                    del pred, generated, fake_x0, teacher_x0, loss
            model.transformer.zero_grad(set_to_none=True)
            del predictor, condition, model
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
            torch.save(payload, args.output / f"rank{rank}.pt")
        event("complete", result=str(args.output / f"rank{rank}.pt"))
    except BaseException as error:
        event("failure", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
