import copy
import sys
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from lightx2v_train.trainers.fastwam_action_tbsm.config import FastWAMActionTBSMConfig
from lightx2v_train.trainers.fastwam_action_tbsm.loss import action_scattering_loss
from torch import nn

ROOT = Path(__file__).parents[1]


def test_scattering_target_scale_gradient_and_detachment():
    x = torch.zeros(1, 1, 4, requires_grad=True)
    positive = torch.tensor([[[3.0, 0.0, 0.0, 0.0]]], requires_grad=True)
    negative = torch.tensor([[[0.0, 4.0, 0.0, 0.0]]], requires_grad=True)
    loss, raw = action_scattering_loss(x, positive, negative)
    # Radius sqrt(4)=2: attraction points +x and repulsion points -y.
    target = torch.tensor([[[6.0 / (3.0 + 1e-6), -8.0 / (4.0 + 1e-6), 0.0, 0.0]]])
    expected_raw = target.square().mean()
    torch.testing.assert_close(raw, expected_raw)
    torch.testing.assert_close(loss, torch.ones_like(loss))
    loss.backward()
    torch.testing.assert_close(x.grad, -2.0 * target / (4.0 * expected_raw))
    assert positive.grad is None and negative.grad is None and not raw.requires_grad


def test_source_exchange_reverses_gradient():
    x = torch.tensor([[[0.2, -0.1]]], requires_grad=True)
    positive, negative = torch.ones_like(x), -torch.ones_like(x)
    first = torch.autograd.grad(action_scattering_loss(x, positive, negative)[0], x)[0]
    second = torch.autograd.grad(action_scattering_loss(x, negative, positive)[0], x)[0]
    torch.testing.assert_close(first, -second)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_padding_matches_cropped_chunks_and_empty_rows(dtype):
    torch.manual_seed(7)
    x = torch.randn(3, 4, 2, dtype=dtype, requires_grad=True)
    positive, negative = torch.randn_like(x), torch.randn_like(x)
    valid = torch.tensor([[True, True, False, False], [True, False, False, False], [False] * 4])
    loss, raw = action_scattering_loss(x, positive, negative, valid)
    expected = torch.stack(
        [
            action_scattering_loss(x[0:1, :2], positive[0:1, :2], negative[0:1, :2])[1],
            action_scattering_loss(x[1:2, :1], positive[1:2, :1], negative[1:2, :1])[1],
        ]
    ).mean()
    torch.testing.assert_close(raw, expected)
    mask = valid.unsqueeze(-1).expand_as(x)
    changed = [value.detach().masked_fill(~mask, float("nan")) for value in (x, positive, negative)]
    torch.testing.assert_close(action_scattering_loss(*changed, valid)[1], raw)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[~mask]) == 0
    assert torch.count_nonzero(x.grad[mask]) > 0


@pytest.mark.parametrize("empty", [False, True])
def test_coincident_and_all_padding_are_finite_zero(empty):
    x = torch.randn(2, 3, 2, requires_grad=True)
    valid = torch.zeros(2, 3, dtype=torch.bool) if empty else None
    positive = torch.randn_like(x) if empty else x.detach().clone()
    negative = torch.randn_like(x) if empty else x.detach().clone()
    loss, raw = action_scattering_loss(x, positive, negative, valid)
    assert loss.item() == raw.item() == 0.0
    loss.backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))


def test_action_magnitudes_are_preserved():
    x = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    loss, raw = action_scattering_loss(x, 2 * x.detach(), x.detach())
    assert raw > 0  # Unit-normalizing action tokens would erase this difference.
    loss.backward()
    assert x.grad[0, 0, 0] < 0


@pytest.mark.parametrize("source", ["teacher", "data"])
def test_config_is_standalone_and_keeps_training_defaults(source):
    path = ROOT / f"configs/train/fastwam_action_tbsm/robotwin_action_1step_tbsm_{source}.yaml"
    config = yaml.safe_load(path.read_text())
    parsed = FastWAMActionTBSMConfig.from_mapping(config)
    assert parsed.positive_source == source
    assert parsed.teacher_steps == parsed.teacher_reference_steps == 20
    assert parsed.student.lora["rank"] == 128
    assert parsed.ema_decay == 0.995
    assert config["training"]["method"] == "fastwam_action_tbsm"
    assert config["data"]["train"]["batch_size"] == 16
    assert config["training"]["max_train_iters"] == 30000
    assert config["training"]["student"]["optimizer"]["learning_rate"] == 1e-4
    assert config["training"]["output_dir"].endswith(f"tbsm_{source}")
    assert config["logging"]["wandb"]["name"] == f"fastwam_robotwin_action_1step_tbsm_{source}"
    assert "action_consistency" not in config["training"]


@pytest.mark.parametrize("key,value", [("positive_source", "ema"), ("teacher_steps", 0), ("ema_decay", 1.0)])
def test_invalid_tbsm_config(key, value, tmp_path):
    config = _config(tmp_path)
    config["training"]["action_tbsm"][key] = value
    with pytest.raises(ValueError):
        FastWAMActionTBSMConfig.from_mapping(config)


@contextmanager
def _cpu_training_environment():
    # Training orchestration uses tiny CPU experts. Do not load installed CUDA
    # extensions or claim these checks exercise the real FlashAttention kernels.
    # Restore only these entries, not the whole module cache: unloading newly
    # imported torch modules can cause duplicate operator registration.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
        monkeypatch.setitem(sys.modules, "flash_attn", None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        yield


class _ActionExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(2, 2, bias=False)
        self.calls = []

    def pre_dit(self, action_tokens, timestep, context, context_mask):
        self.calls.append((torch.is_grad_enabled(), action_tokens.detach().clone(), timestep.detach().clone(), context))
        return dict(tokens=action_tokens, freqs=None, t_mod=timestep, context=context, context_mask=context_mask)

    def post_dit(self, tokens, pre):
        return tokens


class _VideoExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)

    def pre_dit(self, x, timestep, context, context_mask, **kwargs):
        return dict(tokens=self.proj(x), freqs=None, t_mod=timestep, context=context, context_mask=context_mask, meta={"tokens_per_frame": 1})

    def build_video_to_video_mask(self, seq_len, tokens_per_frame, device):
        return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)


class _MoT(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefill_count = 0
        self.caches = []

    def prefill_video_cache(self, video_tokens, **kwargs):
        self.prefill_count += 1
        return [{"k": video_tokens, "v": video_tokens}]

    def forward_action_with_video_cache(self, action_tokens, action_expert, video_kv_cache, **kwargs):
        self.caches.append(video_kv_cache)
        return action_expert.q(action_tokens) + video_kv_cache[0]["v"]


class _Model:
    def __init__(self):
        from lightx2v_train.schedulers.flow_matching import WanContinuousFlowMatchScheduler

        module = nn.Module()
        module.action_expert = _ActionExpert()
        module.video_expert = _VideoExpert()
        module.mot = _MoT()
        module.device = torch.device("cpu")
        module.train_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        module.infer_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        module.build_action_distill_inputs = lambda sample: sample
        module._build_mot_attention_mask = lambda video_seq_len, action_seq_len, **kwargs: torch.ones(
            video_seq_len + action_seq_len,
            video_seq_len + action_seq_len,
            dtype=torch.bool,
        )
        self.module = module

    def unwrap_module(self):
        return self.module

    def autocast_context(self):
        return nullcontext()


def _config(directory, source="data"):
    return {
        "training": {
            "method": "fastwam_action_tbsm",
            "output_dir": str(directory),
            "max_train_iters": 3,
            "gradient_accumulation_iters": 2,
            "save_every_iters": 2,
            "save_final": True,
            "lr_scheduler": "constant",
            "student": {"train_type": "full", "optimizer": {"learning_rate": 1e-3}},
            "action_tbsm": {"positive_source": source, "teacher_steps": 3, "ema_decay": 0.5},
        },
        "data": {"train": {"batch_size": 2}},
        "inference": {"infer_every_iters": 1, "num_samples": 2, "seed": 42},
        "logging": {"train_log_every_iters": 1},
        "resume": {},
    }


def _sample():
    return {
        "video": torch.zeros(2, 3, 1, 2, 2),
        "action": torch.randn(2, 3, 2),
        "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
        "context": torch.randn(2, 1, 2),
        "context_mask": torch.ones(2, 1, dtype=torch.bool),
        "first_frame_latents": torch.randn(2, 1, 2),
    }


def _trainer(config, model=None):
    from lightx2v_train.trainers import build_trainer

    trainer = build_trainer(config)
    trainer.set_model(_Model() if model is None else model)
    return trainer


@pytest.mark.parametrize("source", ["teacher", "data"])
def test_training_sources_independent_noise_and_frozen_cache(source, tmp_path):
    with _cpu_training_environment():
        torch.manual_seed(4)
        trainer = _trainer(_config(tmp_path, source))
        trainer.setup()
        sample = _sample()
        sample["action"].requires_grad_()
        inputs, condition, valid = trainer._prepare_batch(sample)
        loss, metrics = trainer._loss(inputs, condition, valid)
        loss.backward()
        assert set(metrics) == {"scattering"}
        student_calls = trainer.roles.student.calls
        assert [call[0] for call in student_calls] == [False, True]
        assert not torch.equal(student_calls[0][1], student_calls[1][1])
        assert all(torch.all(call[2] == 1000) for call in student_calls)
        teacher_calls = trainer.roles.teacher.calls
        assert len(teacher_calls) == (3 if source == "teacher" else 0)
        if teacher_calls:
            assert all(not call[0] for call in teacher_calls)
            assert all(not torch.equal(teacher_calls[0][1], call[1]) for call in student_calls)
        assert all(call[3] is condition.context for call in student_calls + teacher_calls)
        module = trainer.model.unwrap_module()
        assert module.mot.prefill_count == 1
        assert all(cache is condition.video_kv_cache for cache in module.mot.caches)
        assert not condition.video_kv_cache[0]["v"].requires_grad
        assert trainer.roles.student.q.weight.grad.abs().sum() > 0
        assert all(p.grad is None for role in (trainer.roles.teacher, trainer.roles.target, module.video_expert) for p in role.parameters())
        assert sample["action"].grad is None
        assert trainer.roles.target.calls == []  # EMA is not a training target.


def test_lora_updates_only_adapter_parameters(tmp_path):
    with _cpu_training_environment():
        config = _config(tmp_path)
        config["training"]["student"].update(train_type="lora", lora={"rank": 2, "alpha": 2, "target_modules": ["q"]})
        trainer = _trainer(config)
        trainer.setup()
        inputs, condition, valid = trainer._prepare_batch(_sample())
        trainer._loss(inputs, condition, valid)[0].backward()
        grads = {name for name, p in trainer.roles.student.named_parameters() if p.grad is not None}
        assert grads and all("lora_" in name for name in grads)
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.roles.update_target(trainer.parsed.ema_decay)
        expected = [copy.deepcopy(role.state_dict()) for role in (trainer.roles.student, trainer.roles.target)]
        trainer.checkpoints.save(1)
        with torch.no_grad():
            for target, student in trainer.roles.ema_pairs:
                target.zero_()
                student.zero_()
        assert trainer.checkpoints.load(tmp_path / "checkpoint-000000001") == 1
        for role, state in zip((trainer.roles.student, trainer.roles.target), expected):
            _assert_state_equal(role.state_dict(), state)


@pytest.mark.parametrize("source", ["teacher", "data"])
def test_real_action_dit_lora_keeps_gradients_under_bf16_autocast(source, tmp_path):
    with _cpu_training_environment():
        from lightx2v_train.model_zoo.native.wan.fastwam.action_dit import ActionDiT
        from lightx2v_train.model_zoo.native.wan.fastwam.mot import MoT
        from lightx2v_train.model_zoo.native.wan.fastwam.video_dit import FastWAMVideoDiT

        torch.manual_seed(17)
        model = _Model()
        module = model.unwrap_module()
        common = dict(hidden_dim=8, ffn_dim=16, text_dim=8, freq_dim=8, eps=1e-6, num_heads=1, attn_head_dim=8, num_layers=2)
        module.action_expert = ActionDiT(action_dim=2, **common).to(torch.bfloat16)
        module.video_expert = FastWAMVideoDiT(in_dim=2, out_dim=2, patch_size=(1, 1, 1), has_image_input=False, seperated_timestep=True, **common).to(torch.bfloat16)
        module.mot = MoT({"video": module.video_expert, "action": module.action_expert}, mot_checkpoint_mixed_attn=False)
        model.autocast_context = lambda: torch.autocast("cpu", dtype=torch.bfloat16)
        config = _config(tmp_path, source)
        config["training"]["student"].update(train_type="lora", lora={"rank": 2, "alpha": 2})
        trainer = _trainer(config, model)
        trainer.setup()
        assert all(parameter.dtype == torch.float32 for parameter in trainer.student_params)
        sample = _sample()
        sample.update(
            action=sample["action"].to(torch.bfloat16),
            first_frame_latents=torch.randn(2, 2, 1, 2, 2, dtype=torch.bfloat16),
            context=torch.randn(2, 2, 8, dtype=torch.bfloat16),
            context_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        for _ in range(2):
            trainer.optimizer.zero_grad(set_to_none=True)
            inputs, condition, valid = trainer._prepare_batch(sample)
            with model.autocast_context():
                loss, _ = trainer._loss(inputs, condition, valid)
                assert torch.is_autocast_cache_enabled()
                assert torch.get_autocast_dtype("cpu") == torch.bfloat16
            assert loss.requires_grad and loss.grad_fn is not None
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainer.student_params)
            assert any(parameter.grad.abs().sum() > 0 for parameter in trainer.student_params)
            assert all(parameter.grad is None for role in (trainer.roles.teacher, trainer.roles.target, module.video_expert) for parameter in role.parameters())
            trainer.optimizer.step()


def _assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_state_equal(a, b)
    else:
        assert left == right


def test_train_evaluate_ema_checkpoint_and_resume(tmp_path):
    with _cpu_training_environment():
        torch.manual_seed(12)
        config = _config(tmp_path, "teacher")
        trainer = _trainer(config)
        initial = copy.deepcopy(trainer.model.unwrap_module().state_dict())
        sample = _sample()
        trainer.set_data([sample], [sample])
        logs = []
        trainer.monitor = SimpleNamespace(log_metrics=lambda values, step: logs.append((step, values)), finish=lambda: None)
        trainer.train()
        assert [step for step, values in logs if "train/scattering_loss" in values] == [1, 2, 3]
        assert all("train/consistency_loss" not in values and "train/flow_loss" not in values for _, values in logs)
        assert len([values for _, values in logs if "eval/ema_teacher_l1" in values]) == 3
        assert all(torch.isfinite(torch.tensor(list(values.values()))).all() for _, values in logs)
        assert not torch.equal(trainer.roles.student.q.weight, initial["action_expert.q.weight"])
        torch.testing.assert_close(trainer.roles.teacher.q.weight, initial["action_expert.q.weight"])
        assert not torch.equal(trainer.roles.target.q.weight, initial["action_expert.q.weight"])

        checkpoint = tmp_path / "checkpoint-000000003"
        saved_rng = torch.load(checkpoint / "rng-rank-00000.pt", weights_only=True)["cpu"]
        expected_noise = torch.randn(5, generator=torch.Generator().set_state(saved_rng))
        resumed_config = copy.deepcopy(config)
        resumed_config["resume"] = {"auto_resume": True}
        resumed_model = _Model()
        resumed_model.unwrap_module().load_state_dict(initial)
        resumed = _trainer(resumed_config, resumed_model)
        assert resumed.setup() == 3
        torch.testing.assert_close(torch.randn(5), expected_noise)
        for name in ("student", "target", "teacher"):
            _assert_state_equal(getattr(trainer.roles, name).state_dict(), getattr(resumed.roles, name).state_dict())
        _assert_state_equal(trainer.optimizer.state_dict(), resumed.optimizer.state_dict())
        _assert_state_equal(trainer.scheduler.state_dict(), resumed.scheduler.state_dict())


def test_consistency_loss_and_metric_names_are_unchanged(tmp_path):
    with _cpu_training_environment():
        config = _config(tmp_path)
        training = config["training"]
        training["method"] = "fastwam_action_consistency"
        del training["action_tbsm"]
        training["action_consistency"] = {"target_steps": 10, "teacher_reference_steps": 3, "flow_loss_weight": 0.2}
        trainer = _trainer(config)
        trainer.set_data([_sample()])
        logs = []
        trainer.monitor = SimpleNamespace(log_metrics=lambda values, step: logs.append(values), finish=lambda: None)
        trainer.train()
        assert trainer.parsed.target_steps == 10
        assert all("train/consistency_loss" in values and "train/flow_loss" in values for values in logs)
        assert all("train/scattering_loss" not in values for values in logs)

        # A constant velocity field gives a closed-form consistency discrepancy.
        class ConstantVelocity(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = nn.Parameter(torch.tensor(value))

            def forward(self, action, timestep, condition):
                return torch.ones_like(action) * self.value

        trainer.student_denoiser = ConstantVelocity(0.2)
        trainer.teacher_denoiser = ConstantVelocity(0.3)
        trainer.target_denoiser = ConstantVelocity(0.5)
        t, s = torch.tensor([0.8, 0.4]), torch.tensor([0.3, 0.0])
        trainer._sigma_pair = lambda action: (t, s)
        action = torch.zeros(2, 3, 2)
        torch.manual_seed(9)
        noise = torch.randn_like(action)
        torch.manual_seed(9)
        loss, terms = trainer._loss({"action": action}, None, None)
        discrepancy = 0.1 * t + 0.2 * s
        expected_consistency = ((discrepancy.square() + 0.001**2).sqrt() - 0.001).mean()
        expected_flow = (0.2 - noise).square().mean()
        torch.testing.assert_close(terms["consistency"], expected_consistency)
        torch.testing.assert_close(terms["flow"], expected_flow)
        torch.testing.assert_close(loss, expected_consistency + 0.2 * expected_flow)


def _ddp_worker(rank, directory, bf16_lora):
    torch.set_num_threads(1)
    with _cpu_training_environment():
        dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=60))
        try:
            for source in ("data", "teacher"):
                torch.manual_seed(12)  # The frozen base checkpoint is identical on both ranks.
                config = _config(Path(directory) / source, source)
                model = _Model()
                if bf16_lora:
                    config["training"]["student"].update(train_type="lora", lora={"rank": 2, "alpha": 2, "target_modules": ["q"]})
                    model.autocast_context = lambda: torch.autocast("cpu", dtype=torch.bfloat16)
                trainer = _trainer(config, model)
                torch.manual_seed(100 + rank)
                sample = _sample()
                trainer.set_data([sample], [sample])
                trainer.train()
                for target, student in trainer.roles.ema_pairs:
                    for tensor in (student, target, student.grad):
                        assert tensor is not None and torch.isfinite(tensor).all()
                        copies = [torch.empty_like(tensor) for _ in range(2)]
                        dist.all_gather(copies, tensor.detach())
                        torch.testing.assert_close(copies[0], copies[1])
                checkpoint = Path(directory) / source / "checkpoint-000000003"
                assert (checkpoint / f"rng-rank-{rank:05d}.pt").exists()
                assert trainer.checkpoints.load(str(checkpoint)) == 3
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize("bf16_lora", [False, True])
def test_two_process_training_gradient_sync_and_resume(tmp_path, bf16_lora):
    mp.spawn(_ddp_worker, args=(str(tmp_path), bf16_lora), nprocs=2, join=True)
