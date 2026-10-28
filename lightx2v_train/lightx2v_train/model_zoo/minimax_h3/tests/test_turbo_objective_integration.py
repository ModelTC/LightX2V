"""Real Turbo objectives through H3, FP64 trig conversion and exact model JVP."""

import os
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import MiniMaxH3JointLatents
from lightx2v_train.model_zoo.minimax_h3.tests.test_turbo_jvp import tiny_setup
from lightx2v_train.trainers.dmd.student_ema import StudentWeightEMA
from lightx2v_train.trainers.turbo import core
from lightx2v_train.trainers.turbo.config import TurboConfig
from lightx2v_train.trainers.turbo.sampler import predict_trig
from lightx2v_train.trainers.turbo.trainer import TurboTrainer
from lightx2v_train.tricks import ImplicitDistributionAlignmentTrick


def setup_trainer(device="cpu", dtype=torch.float32):
    models, capabilities = [], []
    for role in ("student", "fake", "teacher"):
        model, capability, clean, _, condition = tiny_setup(device, dtype)
        model.config["training"] = {"method": "turbo"}
        model.transformer.eval()
        if role == "student":
            model.add_lora(
                4,
                4,
                [*capability.default_lora_target_modules, "adaln_proj.linear", "norm_out.linear"],
                rank_pattern={"adaln_proj.linear": 2, "norm_out.linear": 2},
                alpha_pattern={"adaln_proj.linear": 2, "norm_out.linear": 2},
            )
            model.set_lora_param_dtype("fp32")
            model.set_lora_trainable()
        elif role == "fake":
            # A full-size FP32 master critic, matching the production recipe.
            # CUDA tests shard it before any forward so compute stays BF16.
            model.transformer.float()
        else:
            model.transformer.requires_grad_(False)
        models.append(model)
        capabilities.append(capability)
    trainer = object.__new__(TurboTrainer)
    trainer.turbo = TurboConfig(dcm_steps=1, scm_steps=1, joint_steps=1, video_shift=6.0, audio_shift=3.0, tangent_warmup_steps=0)
    trainer.student, trainer.fake, trainer.teacher = capabilities
    return trainer, models, clean, condition


def local_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


def run_objectives(trainer, models, clean, condition):
    checks = (
        ("dcm", lambda: trainer._dcm_loss(clean, condition), 0),
        ("scm", lambda: trainer._scm_loss(clean, condition, 1), 0),
        ("dmd", lambda: trainer._dmd_loss(clean, condition), 0),
        ("critic", lambda: trainer._critic_loss(clean, condition), 1),
    )
    results = {}
    for name, objective, owner in checks:
        for model in models:
            model.transformer.zero_grad(set_to_none=True)
        loss, metrics = objective()
        assert loss.ndim == 0 and loss.requires_grad and bool(torch.isfinite(loss)), (name, loss)
        assert loss.dtype == torch.float64, (name, loss.dtype)
        assert all(bool(torch.isfinite(value)) for value in metrics.values())
        loss.backward()
        gradients = [local_tensor(parameter.grad) for parameter in models[owner].transformer.parameters() if parameter.grad is not None]
        assert gradients and all(bool(torch.isfinite(value).all()) for value in gradients), name
        assert any(bool(value.abs().sum() > 0) for value in gradients), name
        for index, model in enumerate(models):
            if index != owner:
                assert all(parameter.grad is None for parameter in model.transformer.parameters()), (name, "gradient leaked", index)
        results[name] = float(loss.detach())
    return results


def configure_optimizer_runtime(trainer, models, clean, condition):
    trainer.model, trainer.fake_model = models[:2]
    trainer.config = {"seed": 711}
    trainer.gradient_accumulation_iters = 1
    trainer.max_grad_norm = 1.0
    trainer.consumed_microbatches = trainer.fake_updates = 0
    trainer.route_loss_scale = lambda _: 1.0
    trainer.trainable_params = [parameter for parameter in models[0].transformer.parameters() if parameter.requires_grad]
    trainer.fake_trainable_params = [parameter for parameter in models[1].transformer.parameters() if parameter.requires_grad]
    trainer.optimizer = torch.optim.AdamW(trainer.trainable_params, lr=1e-4)
    trainer.fake_optimizer = torch.optim.AdamW(trainer.fake_trainable_params, lr=1e-4)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.fake_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.fake_optimizer, lambda _: 1.0)
    trainer.student_ema = StudentWeightEMA(models[0].transformer, decay=0.9)
    trainer.ida_trick = ImplicitDistributionAlignmentTrick.from_mapping({})
    trainer._encode_conditions = lambda sample: (condition, None)
    trainer._latent_shape = lambda sample: clean.shape
    video = clean.video.reshape(1, 2, 2, 2, 2).permute(0, 4, 1, 2, 3).contiguous()
    audio = clean.audio.reshape(1, 2, 3, 2).permute(0, 1, 3, 2).contiguous()
    return {"inputs": {"video_latents": video, "audio_latents": audio}}


def run_joint_optimizer_and_ema(trainer, models, clean, condition):
    """Exercise the real sync-false SCM / sync-true DMD optimizer loop."""
    sample = configure_optimizer_runtime(trainer, models, clean, condition)
    samples = iter([sample, sample, sample])
    first = trainer._optimizer_step(samples, "student", "joint", 2)
    assert trainer.student_ema.num_updates == 1
    fake = trainer._optimizer_step(samples, "fake", "joint", 2)
    assert trainer.fake_updates == 1
    second = trainer._optimizer_step(samples, "student", "joint", 3)
    assert trainer.student_ema.num_updates == 2
    assert trainer.consumed_microbatches == 3
    assert "scm_video" in first and "dmd_video" in first
    assert "scm_audio" in second and "dmd_audio" in second
    assert all(torch.isfinite(torch.tensor(value)) for result in (first, fake, second) for value in result.values())
    assert all(bool(torch.isfinite(local_tensor(shadow)).all()) for shadow in trainer.student_ema.shadow.values())
    assert all(parameter.grad is None for parameter in models[2].transformer.parameters())
    return {"first_joint": first["loss"], "fake": fake["loss"], "second_joint": second["loss"], "ema_updates": trainer.student_ema.num_updates}


def run_dual_joint_optimizer_and_ema(trainer, models, clean, condition):
    """CM has paired AV; online DMD has a different condition and geometry."""
    cm_sample = configure_optimizer_runtime(trainer, models, clean, condition)
    cm_sample["stream"] = "cm"
    online_sample = {"inputs": {}, "stream": "online"}
    trainer.public_config = {"data": {"cm": {"name": "minimax_h3_turbo_cm_dataset"}}}
    trainer.consumed_cm_microbatches = 0
    trainer.cm_route_loss_scale = lambda _: 1.0
    online_condition = {**condition, "prompt_embeds": condition["prompt_embeds"] + 3}
    online_shape = replace(clean.shape, num_frames=39, latent_frames=3, audio_latents=4, video_tokens=(1, 12, 2), audio_tokens=(1, 8, 2))
    trainer._encode_conditions = lambda sample: (condition if sample["stream"] == "cm" else online_condition, None)
    trainer._latent_shape = lambda sample: clean.shape if sample["stream"] == "cm" else online_shape
    original_extract = trainer.student.extract_turbo_latents
    extracted = []

    def extract(sample, shape):
        assert sample["stream"] == "cm", "Online DMD/fake must not read a clean target."
        extracted.append(shape)
        return original_extract(sample, shape)

    trainer.student.extract_turbo_latents = extract
    seen = set()

    def capture(module, args, kwargs):
        del module, args
        rows = kwargs["hidden_states"].shape[1]
        expected = condition if rows == clean.shape.video_tokens[1] else online_condition
        assert rows in {clean.shape.video_tokens[1], online_shape.video_tokens[1]}
        torch.testing.assert_close(kwargs["encoder_hidden_states"], expected["prompt_embeds"], rtol=0, atol=0)
        seen.add("cm" if rows == clean.shape.video_tokens[1] else "online")

    handles = [model.transformer.register_forward_pre_hook(capture, with_kwargs=True) for model in models]
    try:
        samples, cm_samples = iter([online_sample] * 3), iter([cm_sample] * 2)
        first = trainer._optimizer_step(samples, "student", "joint", 2, cm_samples=cm_samples)
        fake = trainer._optimizer_step(samples, "fake", "joint", 2)
        second = trainer._optimizer_step(samples, "student", "joint", 3, cm_samples=cm_samples)
    finally:
        trainer.student.extract_turbo_latents = original_extract
        for handle in handles:
            handle.remove()
    assert seen == {"cm", "online"}
    assert len(extracted) == 2
    assert trainer.consumed_cm_microbatches == 2 and trainer.consumed_microbatches == 3
    assert trainer.student_ema.num_updates == 2 and trainer.fake_updates == 1
    assert all(torch.isfinite(torch.tensor(value)) for result in (first, fake, second) for value in result.values())
    assert all(parameter.grad is None for parameter in models[2].transformer.parameters())
    return {"cm_microbatches": 2, "online_microbatches": 3, "ema_updates": 2, "conditions_and_geometry": "independent", "online_clean_reads": 0}


def fsdp_objective_worker(rank, init_file):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method="file://" + init_file, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(503)
        trainer, models, clean, condition = setup_trainer(torch.device("cuda", rank), torch.bfloat16)
        mesh = init_device_mesh("cuda", (2,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
        for model in models:
            for block in [*model.transformer.token_refiner.refiner_blocks, *model.transformer.transformer_blocks]:
                fully_shard(block, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
            fully_shard(model.transformer, mesh=mesh, reshard_after_forward=False, mp_policy=policy)
        with sdpa_kernel(SDPBackend.MATH):
            results = run_objectives(trainer, models, clean, condition)
            optimizer_results = run_joint_optimizer_and_ema(trainer, models, clean, condition)
            dual_results = run_dual_joint_optimizer_and_ema(trainer, models, clean, condition)
        print(f"FSDP rank={rank} real Turbo objective losses={results}", flush=True)
        print(f"FSDP rank={rank} joint sync/backward/optimizer/EMA={optimizer_results}", flush=True)
        print(f"FSDP rank={rank} dual-stream joint optimizer/EMA/cursors={dual_results}", flush=True)
    finally:
        dist.destroy_process_group()


class TurboObjectiveIntegrationTests(unittest.TestCase):
    def test_real_h3_all_four_turbo_objectives_and_parameter_backward(self):
        torch.manual_seed(501)
        trainer, models, clean, condition = setup_trainer()
        with sdpa_kernel(SDPBackend.MATH):
            results = run_objectives(trainer, models, clean, condition)
        self.assertEqual(set(results), {"dcm", "scm", "dmd", "critic"})

    def test_real_h3_joint_two_backwards_optimizer_ema_then_fake_then_student(self):
        torch.manual_seed(504)
        trainer, models, clean, condition = setup_trainer()
        with sdpa_kernel(SDPBackend.MATH):
            results = run_joint_optimizer_and_ema(trainer, models, clean, condition)
        self.assertEqual(results["ema_updates"], 2)

    def test_real_h3_dual_stream_joint_conditions_geometry_cursors_and_ema(self):
        torch.manual_seed(505)
        trainer, models, clean, condition = setup_trainer()
        with sdpa_kernel(SDPBackend.MATH):
            results = run_dual_joint_optimizer_and_ema(trainer, models, clean, condition)
        self.assertEqual(results["online_clean_reads"], 0)

    def test_real_h3_trig_jvp_matches_whole_pipeline_reverse_mode(self):
        torch.manual_seed(502)
        trainer, models, clean, condition = setup_trainer()
        theta = (torch.tensor(0.9, dtype=torch.float64), torch.tensor(0.5, dtype=torch.float64))
        state = trainer._corrupt(clean, theta)
        self.assertEqual(state.video.dtype, torch.float64)
        with torch.no_grad():
            teacher = predict_trig(trainer.teacher, state, theta, condition)
            actual = trainer._trig_jvp(state, theta, teacher, condition)
        self.assertTrue(all(value.dtype == torch.float64 for value in actual))
        tangent = tuple(core.scm_tangent(getattr(state, name), teacher[i].F, theta[i]) for i, name in enumerate(("video", "audio")))
        video, audio = state.video.detach().requires_grad_(), state.audio.detach().requires_grad_()
        times = torch.stack(theta).requires_grad_()
        with sdpa_kernel(SDPBackend.MATH):
            reference = predict_trig(trainer.student, MiniMaxH3JointLatents(video, audio, clean.shape), times.unbind(), condition)
            cotangents = tuple(torch.randn_like(value.F) for value in reference)
            grads = torch.autograd.grad(sum((reference[i].F * cotangents[i]).sum() for i in range(2)), (video, audio, times))
        lhs = sum((actual[i] * cotangents[i]).sum() for i in range(2))
        rhs = (grads[0] * tangent[0][0]).sum() + (grads[1] * tangent[1][0]).sum() + (grads[2] * torch.stack((tangent[0][1], tangent[1][1]))).sum()
        torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Two-GPU real Turbo FSDP2 objectives")
    def test_cuda_two_rank_fsdp_all_four_objectives_and_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(fsdp_objective_worker, args=(os.path.join(directory, "init"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
