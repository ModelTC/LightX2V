import unittest

from tools.profile.region_event_trace import (
    GpuEvent,
    OpEntry,
    RegionTraceConfig,
    collect_gpu_events,
    format_evt_line,
    is_core_kernel,
)
from tools.profile.profiler_step_gap import analyze_events


class RegionEventTraceTest(unittest.TestCase):
    def setUp(self):
        self.config = RegionTraceConfig(
            region_order=("self_attn",),
            peak_tflops_bf16=None,
            peak_tflops_fp8=None,
        )

    def test_uncorrelated_event_uses_matching_neighbor_anchors(self):
        events = [
            {"ph": "X", "cat": "kernel", "name": f"kernel_{index}", "ts": float(index), "dur": 0.1}
            for index in (1, 2, 3)
        ]
        gpu_annotations = [(0.5, 1.5, "self_attn"), (2.5, 3.5, "self_attn")]

        gpu_events = collect_gpu_events(events, 0.0, 4.0, [], gpu_annotations, self.config)

        self.assertEqual([event.region for event in gpu_events], ["self_attn"] * 3)

    def test_new_wan_kernels_are_classified(self):
        self.assertEqual(is_core_kernel("cutlass_scaled_fp4_mm", "kernel"), (True, "GEMM"))
        self.assertEqual(is_core_kernel("qk_int_sv_f8_block_sparse_attn_kernel", "kernel"), (True, "ATTN"))
        self.assertEqual(is_core_kernel("_attn_fwd", "kernel"), (True, "ATTN"))

    def test_dense_equivalent_flops_are_labeled(self):
        event = GpuEvent(
            idx=1,
            ts=0.0,
            dur_ms=2.0,
            cat="kernel",
            kernel="sparse_attn",
            kernel_raw="sparse_attn",
            region="self_attn",
            is_core=True,
            core_kind="ATTN",
        )
        op = OpEntry(
            seq=1,
            region="self_attn",
            kind="ATTN",
            tag="self_sdpa",
            flops=2e12,
            extra={"B": 1, "H": 1, "Sq": 1, "Sk": 1, "D": 1, "flops_semantics": "dense-equivalent"},
        )

        line = format_evt_line(event, op, peak_tflops=None)

        self.assertIn("GF (dense-equivalent)", line)
        self.assertIn("dense-equivalent TFLOPS", line)

    def test_step_gap_falls_back_to_widest_gpu_annotation(self):
        events = [
            {"ph": "X", "cat": "user_annotation", "name": "ProfilerStep#0", "ts": 0.0, "dur": 10.0},
            {"ph": "X", "cat": "gpu_user_annotation", "name": "transformer_step", "ts": 2.0, "dur": 7.0},
            {"ph": "X", "cat": "gpu_user_annotation", "name": "nested", "ts": 3.0, "dur": 2.0},
            {"ph": "X", "cat": "kernel", "name": "kernel", "ts": 3.0, "dur": 1.0},
        ]

        stats = analyze_events(events)

        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].window_category, "gpu_user_annotation")
        self.assertEqual(stats[0].window_name, "transformer_step")
        self.assertEqual(stats[0].wall_us, 7.0)
        self.assertEqual(stats[0].gpu_active_us, 1.0)


if __name__ == "__main__":
    unittest.main()
