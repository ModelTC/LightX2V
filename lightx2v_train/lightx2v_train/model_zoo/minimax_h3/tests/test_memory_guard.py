import unittest

from lightx2v_train.model_zoo.minimax_h3.memory_guard import (
    MiniMaxH3MemoryGuardConfig,
    _validate_h3_checkpoint_topology,
    _validate_h3_memory_topology,
    _validate_h3_peak_memory,
)


class MiniMaxH3MemoryGuardTest(unittest.TestCase):
    @staticmethod
    def _guard():
        return MiniMaxH3MemoryGuardConfig.from_mapping(
            {
                "enabled": True,
                "device_limit_gib": 80,
                "safety_margin_gib": 2,
                "min_fsdp_size": 4,
            }
        )

    def test_guard_has_78_gib_allocated_threshold_and_80_gib_reserved_limit(self):
        guard = self._guard()
        self.assertEqual(guard.peak_threshold_gib, 78)
        self.assertEqual(_validate_h3_peak_memory([(77.9, 79.9)], guard), (77.9, 79.9))
        with self.assertRaisesRegex(RuntimeError, "allocated <= 78.00"):
            _validate_h3_peak_memory([(78.1, 79.0)], guard)
        with self.assertRaisesRegex(RuntimeError, "reserved < 80.00"):
            _validate_h3_peak_memory([(77.0, 80.0)], guard)

    def test_guard_rejects_invalid_margin(self):
        with self.assertRaisesRegex(ValueError, "smaller than"):
            MiniMaxH3MemoryGuardConfig.from_mapping(
                {
                    "enabled": True,
                    "device_limit_gib": 80,
                    "safety_margin_gib": 80,
                }
            )

    def test_recommended_sp2_fsdp4_topology_is_valid(self):
        _validate_h3_memory_topology(
            self._guard(),
            configured_sp_size=2,
            configured_fsdp_size=4,
            fsdp_enabled=True,
            ddp_enabled=False,
            stream_load_pretrained=True,
            runtime_sp_size=2,
            runtime_fsdp_size=4,
            num_attention_heads=56,
            fsdp_wrapped_roles={"student": True, "fake": True, "teacher": True},
            gradient_checkpointing=True,
            adaptive_video_regularization=False,
            student_sla=False,
        )

    def test_smaller_weight_shard_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fsdp2.size >= 4"):
            _validate_h3_memory_topology(
                self._guard(),
                configured_sp_size=2,
                configured_fsdp_size=2,
                fsdp_enabled=True,
                ddp_enabled=False,
                stream_load_pretrained=True,
                runtime_sp_size=2,
                runtime_fsdp_size=2,
                num_attention_heads=56,
                fsdp_wrapped_roles={},
                gradient_checkpointing=True,
                adaptive_video_regularization=False,
                student_sla=False,
            )

    def test_streamed_post_fsdp_load_is_required(self):
        with self.assertRaisesRegex(ValueError, "stream_load_pretrained=true"):
            _validate_h3_memory_topology(
                self._guard(),
                configured_sp_size=2,
                configured_fsdp_size=4,
                fsdp_enabled=True,
                ddp_enabled=False,
                stream_load_pretrained=False,
                runtime_sp_size=2,
                runtime_fsdp_size=4,
                num_attention_heads=56,
                fsdp_wrapped_roles={},
                gradient_checkpointing=True,
                adaptive_video_regularization=False,
                student_sla=False,
            )

    def test_attention_heads_must_divide_sp_size(self):
        with self.assertRaisesRegex(ValueError, "num_attention_heads=56"):
            _validate_h3_memory_topology(
                self._guard(),
                configured_sp_size=3,
                configured_fsdp_size=4,
                fsdp_enabled=True,
                ddp_enabled=False,
                stream_load_pretrained=True,
                runtime_sp_size=3,
                runtime_fsdp_size=4,
                num_attention_heads=56,
                fsdp_wrapped_roles={},
                gradient_checkpointing=True,
                adaptive_video_regularization=False,
                student_sla=False,
            )

    def test_checkpoint_topology_must_match_exactly(self):
        current = {
            "world_size": 8,
            "sequence_parallel_size": 2,
            "fsdp2_size": 4,
            "all_denoisers_fsdp2": True,
        }
        self.assertTrue(_validate_h3_checkpoint_topology(dict(current), current, "state.pt"))
        saved = dict(current)
        saved["sequence_parallel_size"] = 1
        with self.assertRaisesRegex(RuntimeError, "parallel-topology mismatch"):
            _validate_h3_checkpoint_topology(saved, current, "state.pt")
        with self.assertRaisesRegex(RuntimeError, "without MiniMax-H3 parallel-topology"):
            _validate_h3_checkpoint_topology(None, current, "state.pt")
