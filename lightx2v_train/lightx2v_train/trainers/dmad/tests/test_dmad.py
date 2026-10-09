"""CPU gradient and packing checks for the differentiable H3 DMAD critic."""

import copy
import os
import tempfile
import time
import unittest
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import MiniMaxH3JointLatents, MiniMaxH3LatentShape
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability import MiniMaxH3DistributionMatchingCapability
from lightx2v_train.model_zoo.native.minimax_h3.modeling import _transformer_class, install_minimax_h3_dmad_features
from lightx2v_train.model_zoo.native.minimax_h3.packing import build_row_timesteps
from lightx2v_train.model_zoo.native.minimax_h3.packing_ref2av import MiniMaxH3ReferenceGeometry
from lightx2v_train.trainers.dmd.dmad import DmadCheckpointManager, DmadTrainer
from lightx2v_train.trainers.dmd.dmad_math import DmadConfig, DualHeadCritic, GapRouter, PowerEMA, generator_loss
from lightx2v_train.trainers.dmd.roles import DmdRoleRegistry


def _fsdp_feature_worker(rank, world_size, init_file):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=timedelta(seconds=30))
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        transformer, capability, latents, condition = DmadH3FeaturesTest().fixture(checkpointing=True)
        install_minimax_h3_dmad_features(transformer)
        for module in (transformer.norm_out, transformer.proj_out, transformer.audio_proj_out):
            module.requires_grad_(False)
        mesh = init_device_mesh("cpu", (world_size,))
        for block in [*transformer.token_refiner.refiner_blocks, *transformer.transformer_blocks]:
            fully_shard(block, mesh=mesh)
        fully_shard(transformer, mesh=mesh, reshard_after_forward=False)
        parameters = [p for p in transformer.parameters() if p.requires_grad]
        for parameter in parameters:
            parameter.requires_grad_(False)
        heads = DualHeadCritic(16)
        for parameter in heads.parameters():
            dist.broadcast(parameter.data, src=0)
        heads.requires_grad_(False)
        features = capability.predict_dmad_features(latents, torch.tensor(0.4), condition, feature_block=1)
        generator_loss(*heads(features["video"], features["audio"])).backward()
        for latent in (latents.video, latents.audio):
            assert latent.grad is not None and torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in transformer.parameters())
        assert all(parameter.grad is None for parameter in heads.parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        heads.requires_grad_(True)
        features = capability.predict_dmad_features(capability.detach(latents), torch.tensor(0.4), condition, feature_block=1)
        real, teacher = heads(features["video"], features["audio"])
        (torch.nn.functional.softplus(real) + torch.nn.functional.softplus(teacher)).mean().backward()
        assert all(parameter.grad is not None for parameter in parameters)
        for parameter in heads.parameters():
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(world_size)
        assert all(torch.isfinite(parameter.grad).all() for parameter in heads.parameters())

        # Power EMA must preserve distributed shards through DCP, never save
        # just rank zero's local tensor as if it were the complete state.
        ema = PowerEMA(transformer, 6.94)
        ema.update()
        with torch.no_grad():
            for parameter in transformer.parameters():
                if parameter.requires_grad:
                    parameter.add_(0.25)
        ema.update()
        expected = {name: value.to_local().clone() for name, value in ema.shadow.items()}
        checkpoint = str(Path(init_file).parent / "ema")
        metadata = {key: value for key, value in ema.state_dict().items() if key != "shadow"}
        dcp.save({"ema": ema.shadow}, checkpoint_id=checkpoint)
        restored = PowerEMA(transformer, 6.94)
        state = {"ema": restored.shadow}
        dcp.load(state, checkpoint_id=checkpoint)
        restored.load_state_dict({**metadata, "shadow": state["ema"]})
        assert restored.num_updates == 2
        for name, value in restored.shadow.items():
            torch.testing.assert_close(value.to_local(), expected[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class _TinyModel(SimpleNamespace):
    """Weak-referenceable owner for the real model capability."""


class DmadH3FeaturesTest(unittest.TestCase):
    def fixture(self, *, references=True, checkpointing=False):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(842)
            transformer = _transformer_class()(
                num_attention_heads=2,
                attention_head_dim=8,
                hidden_size=16,
                num_layers=2,
                num_refiner_layers=1,
                ffn_dim=32,
                in_channels=2,
                audio_in_channels=3,
                patch_size=(1, 2, 2),
                text_dim=6,
                freq_dim=8,
                time_embed_hidden_dim=16,
                time_embed_dim=8,
                rope_freq_dim=1,
            )
            shape = MiniMaxH3LatentShape(4, 1, 4, 4, 2, (1, 4, 8), (1, 4, 3))
            latents = MiniMaxH3JointLatents(
                torch.randn(shape.video_tokens, requires_grad=True),
                torch.randn(shape.audio_tokens, requires_grad=True),
                shape,
            )
            condition = {
                "prompt_embeds": torch.randn(1, 3, 6),
                "text_token_tags": torch.tensor([1, 0, 1]),
            }
            if references:
                condition.update(
                    references=(MiniMaxH3ReferenceGeometry("image", 1, 2, 4), MiniMaxH3ReferenceGeometry("audio", num_audio_latents=1)),
                    noised_condition_video_latents=torch.randn(2, 8),
                    condition_audio_latents=torch.randn(2, 3),
                )
        if checkpointing:
            transformer.enable_gradient_checkpointing()
        transformer.eval()
        model = _TinyModel(
            transformer_component="transformer_ref",
            transformer=transformer,
            device=torch.device("cpu"),
            patch_size=(1, 2, 2),
            vae_spatial_scale_factor=16,
            config={"training": {"dmd": {"num_inference_steps": 4}}},
            denoiser_module=lambda: transformer,
            transformer_forward_context=nullcontext,
        )
        capability = MiniMaxH3DistributionMatchingCapability(model, {"video_flow_shift": 12.0, "audio_flow_shift": 2.0})
        return transformer, capability, latents, condition

    def test_opt_in_preserves_denoising_outputs_and_is_idempotent(self):
        transformer, capability, latents, condition = self.fixture()
        expected = capability.predict_velocity(latents, torch.tensor(0.4), condition)
        install_minimax_h3_dmad_features(transformer)
        forward = transformer.forward
        install_minimax_h3_dmad_features(transformer)
        self.assertEqual(transformer.forward, forward)
        actual = capability.predict_velocity(latents, torch.tensor(0.4), condition)
        torch.testing.assert_close(actual.video, expected.video, rtol=0, atol=0)
        torch.testing.assert_close(actual.audio, expected.audio, rtol=0, atol=0)
        self.assertIn("dmad", capability.profile.supported_training_methods)

    def test_features_equal_raw_block_outputs_and_exclude_text_and_references(self):
        transformer, capability, latents, condition = self.fixture()
        captured = []
        handle = transformer.transformer_blocks[1].register_forward_hook(lambda _module, _inputs, output: captured.append(output))
        try:
            capability.predict_velocity(latents, torch.tensor(0.4), condition)
        finally:
            handle.remove()
        layout = capability._layout(condition, latents.shape)
        self.assertEqual(layout.num_condition_video_rows, 2)
        self.assertEqual(layout.num_condition_audio_rows, 2)
        install_minimax_h3_dmad_features(transformer)
        features = capability.predict_dmad_features(latents, torch.tensor(0.4), condition, feature_block=1)
        for name in ("video", "audio"):
            indices = getattr(layout, f"{name}_indices")[getattr(layout, f"num_condition_{name}_rows") :]
            expected = captured[0].index_select(1, indices)
            torch.testing.assert_close(features[name], expected, rtol=0, atol=0)
            self.assertEqual(features[name].shape, (1, 4, 16))
            self.assertTrue(features[name].requires_grad)

    def test_frozen_critic_preserves_generator_input_gradients_with_checkpointing(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                transformer, capability, latents, condition = self.fixture(checkpointing=checkpointing)
                install_minimax_h3_dmad_features(transformer)
                transformer.requires_grad_(False)
                heads = DualHeadCritic(16).requires_grad_(False)
                features = capability.predict_dmad_features(latents, torch.tensor(0.25), condition, feature_block=1)
                real, teacher = heads(features["video"], features["audio"])
                loss = generator_loss(real, teacher, weight=0.7)
                loss.backward()
                for latent in (latents.video, latents.audio):
                    self.assertIsNotNone(latent.grad)
                    self.assertTrue(torch.isfinite(latent.grad).all())
                    self.assertGreater(latent.grad.abs().sum().item(), 0)
                self.assertTrue(all(parameter.grad is None for parameter in transformer.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in heads.parameters()))

    def test_critic_features_train_used_backbone_but_skip_later_blocks_and_outputs(self):
        transformer, capability, latents, condition = self.fixture()
        install_minimax_h3_dmad_features(transformer)
        later = Mock()
        projection = Mock()
        normal_root_output = []
        handles = [
            transformer.transformer_blocks[1].register_forward_hook(later),
            transformer.proj_out.register_forward_hook(projection),
            transformer.audio_proj_out.register_forward_hook(projection),
            transformer.norm_out.register_forward_hook(projection),
            transformer.register_forward_hook(lambda _module, _args, output: normal_root_output.append(output)),
        ]
        try:
            features = capability.predict_dmad_features(latents, torch.tensor(0.3), condition, feature_block=0)
            (features["video"].square().mean() + features["audio"].square().mean()).backward()
        finally:
            for handle in handles:
                handle.remove()
        later.assert_not_called()
        projection.assert_not_called()
        self.assertEqual(len(normal_root_output), 1)
        self.assertIs(normal_root_output[0][0], features["video"])
        self.assertGreater(transformer.proj_in.weight.grad.abs().sum().item(), 0)
        self.assertGreater(transformer.audio_proj_in.weight.grad.abs().sum().item(), 0)
        self.assertTrue(any(parameter.grad is not None for parameter in transformer.transformer_blocks[0].parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in transformer.transformer_blocks[1].parameters()))
        self.assertIsNone(transformer.proj_out.weight.grad)

    def test_invalid_block_and_sequence_parallel_fail_before_forward(self):
        transformer, capability, latents, condition = self.fixture()
        install_minimax_h3_dmad_features(transformer)
        for index in (-1, 2, 0.5):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "feature block"):
                capability.predict_dmad_features(latents, torch.tensor(0.5), condition, feature_block=index)
        with (
            patch(
                "lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability.get_sequence_parallel_world_size",
                return_value=2,
            ),
            self.assertRaisesRegex(ValueError, "sequence_parallel.size=1"),
        ):
            capability.predict_dmad_features(latents, torch.tensor(0.5), condition, feature_block=1)

    def test_native_default_return_dict_is_unchanged(self):
        transformer, capability, latents, condition = self.fixture(references=False)
        # No reference rows are present, so ordinary native kwargs are direct.
        layout = capability._layout(condition, latents.shape)
        timestep, indices = build_row_timesteps(layout, torch.tensor(0.5), torch.tensor(0.2))
        kwargs = dict(
            hidden_states=latents.video,
            audio_hidden_states=latents.audio,
            encoder_hidden_states=condition["prompt_embeds"],
            timestep=timestep,
            timestep_indices=indices,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
        )
        expected = transformer(**kwargs)
        install_minimax_h3_dmad_features(transformer)
        actual = transformer(**kwargs)
        self.assertEqual(type(actual), type(expected))
        torch.testing.assert_close(actual.sample, expected.sample, rtol=0, atol=0)
        torch.testing.assert_close(actual.audio_sample, expected.audio_sample, rtol=0, atol=0)

    def test_two_rank_cpu_fsdp_frozen_input_grad_and_critic_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.spawn(_fsdp_feature_worker, args=(2, str(Path(directory) / "init")), nprocs=2, join=False)
            deadline = time.monotonic() + 50
            try:
                while time.monotonic() < deadline:
                    if context.join(timeout=1):
                        return
                self.fail("Two-rank CPU FSDP feature-gradient test timed out.")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=2)


class DmadTrainerStagesTest(unittest.TestCase):
    def fixture(self):
        student_model, student, latents, condition = DmadH3FeaturesTest().fixture(checkpointing=True)
        critic_model, fake, _, _ = DmadH3FeaturesTest().fixture(checkpointing=True)
        install_minimax_h3_dmad_features(critic_model)
        for module in (critic_model.norm_out, critic_model.proj_out, critic_model.audio_proj_out):
            module.requires_grad_(False)
        trainer = DmadTrainer.__new__(DmadTrainer)
        trainer.student, trainer.fake = student, fake
        trainer.dmad_config = DmadConfig(feature_block=1, gap_ready_bands=1, gap_min_count=1)
        trainer.critic_heads = DualHeadCritic(16)
        trainer.gap_router = GapRouter(trainer.dmad_config)
        trainer.trainable_params = list(student_model.parameters())
        trainer.fake_trainable_params = [p for p in critic_model.parameters() if p.requires_grad]
        trainer.optimizer = torch.optim.AdamW(trainer.trainable_params, lr=1e-3)
        trainer.fake_optimizer = torch.optim.AdamW(trainer.fake_trainable_params, lr=1e-3)
        trainer.head_optimizer = torch.optim.AdamW(trainer.critic_heads.parameters(), lr=1e-3)
        trainer.lr_scheduler = Mock()
        trainer.fake_lr_scheduler = Mock()
        trainer.dmad_emas = [PowerEMA(student_model, 6.94)]
        trainer.max_grad_norm = 0
        trainer.num_inference_steps = 4
        trainer.latent_dtype = torch.float32
        trainer._set_student_gradient_sync = Mock()
        trainer._set_fake_gradient_sync = Mock()
        trainer._encode_conditions = Mock(return_value=(condition, None))
        trainer._latent_shape = Mock(return_value=latents.shape)
        trainer.teacher = Mock(name="must_never_query_online_teacher")
        trainer._sample_synced_int = Mock(return_value=2)
        sample = {
            "dmad_real": {"video": latents.video.detach() + 0.2, "audio": latents.audio.detach() - 0.1},
            "dmad_teacher": {"video": latents.video.detach() - 0.3, "audio": latents.audio.detach() + 0.4},
        }
        return trainer, sample

    @staticmethod
    def snapshot(parameters):
        return [p.detach().clone() for p in parameters]

    @staticmethod
    def changed(parameters, before):
        return any(not torch.equal(p.detach(), original) for p, original in zip(parameters, before))

    def test_actual_student_and_critic_stages_update_only_their_own_parameters(self):
        trainer, sample = self.fixture()
        student_before = self.snapshot(trainer.trainable_params)
        critic_before = self.snapshot(trainer.fake_trainable_params)
        heads_before = self.snapshot(trainer.critic_heads.parameters())
        student_result = trainer._train_one_stage(iter([sample, sample]), "student", 2)
        self.assertTrue(torch.isfinite(torch.tensor(student_result["loss"])))
        self.assertTrue(self.changed(trainer.trainable_params, student_before))
        self.assertFalse(self.changed(trainer.fake_trainable_params, critic_before))
        self.assertFalse(self.changed(trainer.critic_heads.parameters(), heads_before))
        self.assertTrue(all(p.requires_grad for p in trainer.fake_trainable_params))
        self.assertEqual(trainer.dmad_emas[0].num_updates, 1)
        self.assertEqual([call.args[0] for call in trainer._set_student_gradient_sync.call_args_list], [False, True])
        trainer.lr_scheduler.step.assert_called_once()

        student_after = self.snapshot(trainer.trainable_params)
        trainer._critic_logits = Mock(wraps=trainer._critic_logits)
        critic_result = trainer._train_one_stage(iter([sample]), "fake", 1)
        self.assertTrue(torch.isfinite(torch.tensor(critic_result["loss"])))
        self.assertFalse(self.changed(trainer.trainable_params, student_after))
        self.assertTrue(self.changed(trainer.fake_trainable_params, critic_before))
        self.assertTrue(self.changed(trainer.critic_heads.parameters(), heads_before))
        self.assertEqual(trainer.dmad_emas[0].num_updates, 1)
        self.assertEqual(int(trainer.gap_router.counts.sum()), 1)
        self.assertEqual(trainer._critic_logits.call_count, 3)
        calls = trainer._critic_logits.call_args_list
        for call in calls[1:]:
            self.assertIs(call.args[1], calls[0].args[1])
            self.assertIs(call.args[2], calls[0].args[2])
        self.assertEqual(trainer.teacher.mock_calls, [])
        trainer.fake_lr_scheduler.step.assert_called_once()

    def test_checkpointed_last_only_rollout_does_not_use_euler_step_or_online_teacher(self):
        trainer, _ = self.fixture()
        condition, _ = trainer._encode_conditions({})
        shape = trainer._latent_shape({})
        trainer.student.step = Mock(side_effect=AssertionError("Euler is not the DMAD rollout"))
        trainer._noise = Mock(wraps=trainer._noise)
        states = []
        original = trainer.student.predict_velocity

        def tracked(*args):
            states.append(torch.is_grad_enabled())
            return original(*args)

        trainer.student.predict_velocity = tracked
        result = trainer.run_dmad_rollout(condition, shape, grad_enabled=True)
        self.assertEqual(states, [False, True])
        self.assertEqual(trainer._noise.call_count, 1)
        self.assertTrue(result.video.requires_grad)
        self.assertTrue(result.audio.requires_grad)
        trainer.student.step.assert_not_called()
        self.assertEqual(trainer.teacher.mock_calls, [])

    def test_local_checkpoint_restores_models_optimizers_heads_router_and_power_ema(self):
        trainer, sample = self.fixture()
        trainer.model, trainer.fake_model = trainer.student.model, trainer.fake.model
        trainer.config, trainer.training_config = {}, {}
        trainer.dataloader_train = SimpleNamespace(sampler=None, dataset=SimpleNamespace(manifest_digest="test-only-data"))
        trainer.gradient_accumulation_iters, trainer.fake_update_ratio = 1, 1
        trainer.student_train_type = trainer.fake_train_type = "full"
        trainer.parallel = SimpleNamespace(is_fsdp=lambda: False, state_module=lambda: trainer.model.transformer)
        trainer.role_registry = DmdRoleRegistry(trainer)
        trainer.checkpoint_manager = DmadCheckpointManager(trainer)
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
        trainer.fake_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.fake_optimizer, lambda _: 1.0)

        def save_weights(model, directory, *, role):
            del role
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(model.transformer.state_dict(), path / "weights.pt")

        def load_weights(model, directory, *, role):
            del role
            model.transformer.load_state_dict(torch.load(Path(directory) / "weights.pt", weights_only=True))

        trainer._save_model_weights = save_weights
        trainer._load_model_weights = load_weights
        trainer._train_one_stage(iter([sample]), "student", 1)
        trainer._train_one_stage(iter([sample]), "fake", 1)
        trainer._train_one_stage(iter([sample]), "student", 1)
        modules = [trainer.model.transformer, trainer.fake_model.transformer, trainer.critic_heads, trainer.gap_router]
        before = [copy.deepcopy(module.state_dict()) for module in modules]
        ema_before = copy.deepcopy(trainer.dmad_emas[0].state_dict())
        head_state = copy.deepcopy(trainer.head_optimizer.state_dict())
        student_step = next(iter(trainer.optimizer.state.values()))["step"].clone()
        fake_step = next(iter(trainer.fake_optimizer.state.values()))["step"].clone()
        with tempfile.TemporaryDirectory() as directory, patch.object(DmadCheckpointManager, "_parallel", return_value=trainer.parallel), patch("torch.cuda.is_available", return_value=False):
            trainer.output_train_dir = directory
            trainer.save_checkpoint(7, 2)
            checkpoint = Path(directory) / "checkpoint-000000007"
            self.assertTrue((checkpoint / "dmad_complete").is_file())
            self.assertFalse(any(path.name.startswith(".dmad-save-") for path in Path(directory).iterdir()))
            saved_state = torch.load(checkpoint / "training_state.pt", weights_only=False)
            for key in ("dmad_heads", "dmad_head_optimizer", "dmad_gap", "dmad_ema"):
                malformed = dict(saved_state)
                del malformed[key]
                with self.subTest(missing=key), self.assertRaisesRegex(RuntimeError, "missing required state"):
                    trainer.checkpoint_manager._validate_checkpoint_state(malformed, "training_state.pt", str(checkpoint))
            for key, value in (("steps", 8), ("data_digest", "different-pairs"), ("rollout", "euler")):
                malformed = {**saved_state, "dmad": {**saved_state["dmad"], key: value}}
                with self.subTest(changed=key), self.assertRaisesRegex(ValueError, "objective/data/rollout differs"):
                    trainer.checkpoint_manager._validate_checkpoint_state(malformed, "training_state.pt", str(checkpoint))
            with self.assertRaisesRegex(ValueError, "Not a complete DMAD checkpoint"):
                trainer._load_resume_state(str(Path(directory) / "checkpoint-000000006"))
            averaged = torch.load(checkpoint / "ema1_student" / "weights.pt", weights_only=True)
            for name, value in ema_before["shadow"].items():
                torch.testing.assert_close(averaged[name], value, rtol=0, atol=0)
            with torch.no_grad():
                for module in modules:
                    for value in module.state_dict().values():
                        value.zero_()
                for value in trainer.dmad_emas[0].shadow.values():
                    value.zero_()
            trainer.dmad_emas[0].num_updates = 0
            trainer.optimizer.state.clear()
            trainer.fake_optimizer.state.clear()
            trainer.head_optimizer.state.clear()
            trainer._load_resume_state(str(checkpoint))
        for module, expected in zip(modules, before):
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        for name, value in trainer.dmad_emas[0].shadow.items():
            torch.testing.assert_close(value, ema_before["shadow"][name], rtol=0, atol=0)
        self.assertEqual(trainer.dmad_emas[0].num_updates, 2)
        torch.testing.assert_close(next(iter(trainer.optimizer.state.values()))["step"], student_step)
        torch.testing.assert_close(next(iter(trainer.fake_optimizer.state.values()))["step"], fake_step)
        for key, expected in head_state["state"].items():
            for name, value in trainer.head_optimizer.state_dict()["state"][key].items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        self.assertIn("torch", trainer._resume_rng)


if __name__ == "__main__":
    unittest.main()
