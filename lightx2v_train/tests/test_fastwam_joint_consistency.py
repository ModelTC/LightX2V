from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from lightx2v_train.model_zoo.native.wan.fastwam.action_dit import ActionDiT
from lightx2v_train.model_zoo.native.wan.fastwam.model import FastWAM
from lightx2v_train.model_zoo.native.wan.fastwam.mot import MoT
from lightx2v_train.model_zoo.native.wan.fastwam.video_dit import FastWAMVideoDiT
from lightx2v_train.schedulers.flow_matching import WanContinuousFlowMatchScheduler
from lightx2v_train.trainers.fastwam_action_consistency.trainer import FastWAMActionConsistencyTrainer
from lightx2v_train.trainers.fastwam_joint_consistency.config import FastWAMJointConsistencyConfig
from lightx2v_train.trainers.fastwam_joint_consistency.roles import role_state_dict
from lightx2v_train.trainers.fastwam_joint_consistency.trainer import FastWAMJointConsistencyTrainer, shifted_consistency_pair
from lightx2v_train.utils.registry import build_trainer
from torch import nn


class TinyFastWAM(nn.Module):
    """Real two-layer DiTs and MoT with a small frozen observation encoder."""

    _build_mot_attention_mask = FastWAM._build_mot_attention_mask

    def __init__(self):
        super().__init__()
        common = dict(hidden_dim=8, ffn_dim=16, text_dim=8, freq_dim=8, eps=1e-6, num_heads=1, attn_head_dim=8, num_layers=2)
        self.action_expert = ActionDiT(action_dim=3, **common)
        self.video_expert = FastWAMVideoDiT(in_dim=2, out_dim=2, patch_size=(1, 1, 1), has_image_input=False, seperated_timestep=True, **common)
        self.mot = MoT({"video": self.video_expert, "action": self.action_expert}, mot_checkpoint_mixed_attn=False)
        self.vae = nn.Conv3d(2, 2, 1)
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        self.device = torch.device("cpu")
        self.input_calls = 0

    @torch.no_grad()
    def build_action_distill_inputs(self, sample):
        self.input_calls += 1
        return {
            "first_frame_latents": self.vae(sample["video"]),
            "context": sample["context"],
            "context_mask": sample["context_mask"],
            "action": sample["action"],
            "action_is_pad": sample["action_is_pad"],
        }


def _sample(seed=42):
    generator = torch.Generator().manual_seed(seed)
    return {
        "video": torch.randn(2, 2, 1, 2, 2, generator=generator),
        "context": torch.randn(2, 2, 8, generator=generator),
        "context_mask": torch.tensor([[True, True], [True, False]]),
        "action": torch.randn(2, 3, 3, generator=generator),
        "action_is_pad": torch.tensor([[False, False, True], [False, True, True]]),
    }


def _trainer(directory, *, train_video=True, train_type="lora", checkpointing=False, legacy=False, setup=True, flow_weight=0.2):
    config = {
        "training": {
            "method": "fastwam_action_consistency" if legacy else "fastwam_joint_consistency",
            "output_dir": str(directory),
            "train_video": train_video,
            "max_train_iters": 2,
            "gradient_accumulation_iters": 2,
            "gradient_checkpointing": checkpointing,
            "save_final": False,
            "student": {
                "train_type": train_type,
                "lora": {"rank": 2, "alpha": 2, "dropout": 0.0},
                "optimizer": {"learning_rate": 1e-3},
            },
            "action_consistency": {"teacher_reference_steps": 3, "ema_decay": 0.5, "flow_loss_weight": flow_weight},
        },
        "data": {"train": {"batch_size": 2}},
        "inference": {"num_samples": 2},
        "logging": {"train_log_every_iters": 1},
    }
    # Every rank starts from the same pretrained weights; adapter seeds may differ.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        module = TinyFastWAM()
    trainer = build_trainer(config)
    trainer.set_model(SimpleNamespace(unwrap_module=lambda: module, autocast_context=nullcontext))
    if setup:
        trainer.setup()
    return trainer


def _update(trainer):
    trainer.optimizer.zero_grad(set_to_none=True)
    loss, _ = trainer._loss(*trainer._prepare_batch(_sample()))
    loss.backward()
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.roles.update_target(trainer.parsed.ema_decay)
    if trainer.video_roles is not None:
        trainer.video_roles.update_target(trainer.parsed.ema_decay)


def _states(trainer):
    train_type = trainer.parsed.student.train_type
    return {
        f"{branch}/{name}": deepcopy(role_state_dict(getattr(roles, name), train_type))
        for branch, roles in (("action", trainer.roles), ("video", trainer.video_roles))
        if roles is not None
        for name in ("student", "target")
    }


def test_config_and_registration(tmp_path):
    path = Path(__file__).parents[1] / "configs/train/fastwam_action_dmd/robotwin_action_1step_consistency_joint.yaml"
    config = yaml.safe_load(path.read_text())
    parsed = FastWAMJointConsistencyConfig.from_mapping(config)
    assert parsed.train_video is True
    assert parsed.student.lora["rank"] == 128
    assert parsed.target_steps == 2
    assert parsed.consistency_loss_weight == 1.0
    assert parsed.flow_loss_weight == 0.2
    config["logging"]["wandb"]["enable"] = False
    config["training"]["output_dir"] = str(tmp_path)
    assert isinstance(build_trainer(config), FastWAMJointConsistencyTrainer)
    from lightx2v_train.trainers import FastWAMJointConsistencyTrainer as exported_trainer

    assert exported_trainer is FastWAMJointConsistencyTrainer
    del config["training"]["train_video"]
    assert FastWAMJointConsistencyConfig.from_mapping(config).train_video is False
    config["training"]["train_video"] = "false"
    with pytest.raises(TypeError, match="train_video"):
        FastWAMJointConsistencyConfig.from_mapping(config)


@pytest.mark.parametrize("train_type", ["lora", "full"])
@pytest.mark.parametrize("flow_weight", [0.0, 0.2])
def test_action_loss_updates_both_students_and_ema_only(tmp_path, train_type, flow_weight):
    trainer = _trainer(tmp_path, train_type=train_type, flow_weight=flow_weight)
    branches = (trainer.roles, trainer.video_roles)
    teacher_before = [deepcopy(roles.teacher.state_dict()) for roles in branches]
    ema_before = [[target.clone() for target, _ in roles.ema_pairs] for roles in branches]
    student_before = [[parameter.clone() for parameter in roles.trainable_parameters] for roles in branches]
    inputs, condition, mask = trainer._prepare_batch(_sample())
    assert condition is inputs
    assert not inputs["first_frame_latents"].requires_grad
    assert trainer.model.unwrap_module().input_calls == 1
    # At initialization, a zero endpoint makes student and teacher x0 identical.
    sigmas = shifted_consistency_pair(torch.tensor([0.75, 0.9]), shift=5.0, target_steps=2)
    with patch.object(trainer, "_sigma_pair", return_value=sigmas):
        loss, _ = trainer._loss(inputs, condition, mask)
    loss.backward()
    assert torch.isfinite(loss)
    for roles in branches:
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in roles.trainable_parameters)
        assert all(parameter.grad is None for role in (roles.target, roles.teacher) for parameter in role.parameters())
        if train_type == "lora":
            assert all("lora_" in name for name, parameter in roles.student.named_parameters() if parameter.requires_grad)
    assert all(parameter.grad is None for parameter in trainer.model.unwrap_module().vae.parameters())
    trainer.optimizer.step()
    for roles, teacher, old_ema, old_student in zip(branches, teacher_before, ema_before, student_before):
        roles.update_target(0.5)
        assert any(not torch.equal(old, parameter) for old, parameter in zip(old_student, roles.trainable_parameters))
        for old, (target, student) in zip(old_ema, roles.ema_pairs):
            torch.testing.assert_close(target, 0.5 * old + 0.5 * student)
        torch.testing.assert_close(roles.teacher.state_dict(), teacher)


@pytest.mark.parametrize("train_type", ["lora", "full"])
@pytest.mark.parametrize("changed_role", ["student", "target", "teacher"])
def test_each_role_uses_its_own_video_condition(tmp_path, train_type, changed_role):
    trainer = _trainer(tmp_path, train_type=train_type)
    inputs, _, _ = trainer._prepare_batch(_sample())
    denoisers = {name: getattr(trainer, f"{name}_denoiser") for name in ("student", "target", "teacher")}
    with torch.no_grad():
        before = {name: denoiser.build_condition(inputs) for name, denoiser in denoisers.items()}
        video = getattr(trainer.video_roles, changed_role)
        projection = video.blocks[0].self_attn.v
        weight = projection.lora_B["default"].weight if hasattr(projection, "lora_B") else projection.weight
        weight.add_(0.5)
        after = {name: denoiser.build_condition(inputs) for name, denoiser in denoisers.items()}
    for name in denoisers:
        old_cache, new_cache = before[name].video_kv_cache, after[name].video_kv_cache
        if name == changed_role:
            assert not torch.equal(old_cache[0]["v"], new_cache[0]["v"])
        else:
            torch.testing.assert_close(old_cache, new_cache)
        assert not new_cache[0]["v"].requires_grad
    assert trainer.student_denoiser.build_condition(inputs).video_kv_cache[0]["v"].requires_grad


@pytest.mark.parametrize("train_type", ["lora", "full"])
def test_disabled_video_matches_original_loss_and_gradients(tmp_path, train_type):
    torch.manual_seed(7)
    original = _trainer(tmp_path / "original", legacy=True, train_video=False, train_type=train_type)
    torch.manual_seed(7)
    joint = _trainer(tmp_path / "joint", train_video=False, train_type=train_type)
    assert isinstance(original, FastWAMActionConsistencyTrainer)
    assert joint.video_roles is None
    batches = [trainer._prepare_batch(_sample()) for trainer in (original, joint)]
    torch.testing.assert_close(batches[0][1].video_kv_cache, batches[1][1].video_kv_cache)
    results = []
    for trainer, batch in zip((original, joint), batches):
        torch.manual_seed(9)
        loss, metrics = trainer._loss(*batch)
        loss.backward()
        results.append((loss.detach(), metrics, [parameter.grad for parameter in trainer.student_params]))
    torch.testing.assert_close(results[0], results[1])
    assert all(not parameter.requires_grad for parameter in joint.model.unwrap_module().video_expert.parameters())


def test_evaluation_reuses_separate_frozen_caches(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.set_data([], [_sample()])
    trainer.monitor = Mock()
    with (
        patch.object(trainer.target_denoiser, "build_condition", wraps=trainer.target_denoiser.build_condition) as target_prefill,
        patch.object(trainer.teacher_denoiser, "build_condition", wraps=trainer.teacher_denoiser.build_condition) as teacher_prefill,
        patch.object(trainer.student_denoiser, "build_condition", wraps=trainer.student_denoiser.build_condition) as student_prefill,
        patch.object(trainer.teacher_denoiser, "forward", wraps=trainer.teacher_denoiser.forward) as teacher_forward,
        patch.object(trainer.target_denoiser, "forward", wraps=trainer.target_denoiser.forward) as target_forward,
    ):
        trainer.evaluate(1)
    assert target_prefill.call_count == teacher_prefill.call_count == 1
    assert student_prefill.call_count == 0
    assert teacher_forward.call_count == trainer.parsed.teacher_reference_steps
    teacher_condition = teacher_forward.call_args_list[0].args[2]
    target_condition = target_forward.call_args.args[2]
    assert teacher_condition is not target_condition
    assert all(call.args[2] is teacher_condition for call in teacher_forward.call_args_list)
    for condition in (teacher_condition, target_condition):
        assert all(not value.requires_grad for layer in condition.video_kv_cache for value in layer.values())
    assert trainer.model.unwrap_module().input_calls == 1
    trainer.monitor.log_metrics.assert_called_once()
    assert set(trainer.monitor.log_metrics.call_args.args[0]) == {"eval/ema_teacher_l1", "eval/ema_gt_l1"}


@pytest.mark.parametrize("train_type", ["lora", "full"])
def test_checkpoint_restores_both_branches_and_training_state(tmp_path, train_type):
    trainer = _trainer(tmp_path / "source", train_type=train_type)
    _update(trainer)
    expected = _states(trainer)
    optimizer_state = deepcopy(trainer.optimizer.state_dict())
    scheduler_state = deepcopy(trainer.scheduler.state_dict())
    trainer.checkpoints.save(1)
    checkpoint = tmp_path / "source/checkpoint-000000001"
    assert all((checkpoint / name).exists() for name in ("student_action.pt", "ema_action.pt", "student_video.pt", "ema_video.pt"))
    expected_random = torch.rand(4)
    _update(trainer)
    continued = _states(trainer)
    restored = _trainer(tmp_path / "restored", train_type=train_type)
    assert restored.checkpoints.load(checkpoint) == 1
    torch.testing.assert_close(_states(restored), expected)
    torch.testing.assert_close(restored.optimizer.state_dict(), optimizer_state)
    torch.testing.assert_close(restored.scheduler.state_dict(), scheduler_state)
    torch.testing.assert_close(torch.rand(4), expected_random)
    _update(restored)
    torch.testing.assert_close(_states(restored), continued)


@pytest.mark.parametrize("missing", ["student_video.pt", "ema_video.pt"])
def test_joint_checkpoint_requires_both_video_weights(tmp_path, missing):
    trainer = _trainer(tmp_path)
    trainer.checkpoints.save(1)
    checkpoint = tmp_path / "checkpoint-000000001"
    (checkpoint / missing).unlink()
    with pytest.raises(RuntimeError, match=missing):
        trainer.checkpoints.load(checkpoint)


@pytest.mark.parametrize("saved_video", [False, True])
def test_checkpoint_rejects_video_mode_mismatch(tmp_path, saved_video):
    source = _trainer(tmp_path / "source", train_video=saved_video)
    source.checkpoints.save(1)
    restored = _trainer(tmp_path / "restored", train_video=not saved_video)
    with pytest.raises(RuntimeError, match="train_video"):
        restored.checkpoints.load(tmp_path / "source/checkpoint-000000001")


def test_disabled_video_loads_legacy_action_checkpoint(tmp_path):
    original = _trainer(tmp_path / "original", legacy=True, train_video=False)
    original.checkpoints.save(1)
    restored = _trainer(tmp_path / "restored", train_video=False)
    assert restored.checkpoints.load(tmp_path / "original/checkpoint-000000001") == 1
    torch.testing.assert_close(role_state_dict(restored.roles.student, "lora"), role_state_dict(original.roles.student, "lora"))
    torch.testing.assert_close(role_state_dict(restored.roles.target, "lora"), role_state_dict(original.roles.target, "lora"))


def _ddp_worker(rank, directory, train_type, checkpointing):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        with patch("torch.cuda.is_available", return_value=False):
            torch.manual_seed(100 + rank)
            trainer = _trainer(Path(directory) / "run", train_type=train_type, checkpointing=checkpointing, setup=False)
            trainer.set_data([_sample(50 + rank), _sample(70 + rank)])
            original_setup = trainer.setup
            initial = {}

            def setup():
                iteration = original_setup()
                denoiser = trainer.student_denoiser.module
                assert denoiser.mot.training
                assert denoiser.mot.mot_checkpoint_mixed_attn == checkpointing
                assert denoiser.action_module().use_gradient_checkpointing == checkpointing
                assert denoiser.video_expert.use_gradient_checkpointing == checkpointing
                initial.update(_states(trainer))
                for roles in (trainer.roles, trainer.video_roles):
                    for target, student in roles.ema_pairs:
                        torch.testing.assert_close(target, student)
                return iteration

            trainer.setup = setup
            trainer.train()
            final = _states(trainer)
            for branch in ("action", "video"):
                assert any(not torch.equal(initial[f"{branch}/student"][key], value) for key, value in final[f"{branch}/student"].items())
            for roles in (trainer.roles, trainer.video_roles):
                for role in (roles.student, roles.target):
                    flat = torch.cat([parameter.detach().flatten() for parameter in role.parameters()])
                    gathered = [torch.empty_like(flat) for _ in range(2)]
                    dist.all_gather(gathered, flat)
                    torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
                assert all(parameter.grad is None for parameter in roles.teacher.parameters())
            # These video parameters deliberately have no path to action loss.
            video = trainer.video_roles.student
            assert video.blocks[-1].self_attn.q.weight.grad is None
            assert video.head.head.weight.grad is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("train_type", "checkpointing"), [("full", False), ("lora", True)])
def test_two_rank_training_with_accumulation_and_unused_video_parameters(tmp_path, train_type, checkpointing):
    mp.spawn(_ddp_worker, args=(str(tmp_path), train_type, checkpointing), nprocs=2, join=True)
