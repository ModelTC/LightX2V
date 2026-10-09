#!/usr/bin/env python3
"""CLI regressions and optional CUDA tests for the read-only cache scanner."""

import datetime
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch

SCRIPT = Path(__file__).with_name("scan_ref2av_cache_finite.py").resolve()
SPEC = importlib.util.spec_from_file_location("ref2av_finite_scanner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
SUMMARY_COUNTS = (
    "complete",
    "scanned_rows",
    "clean_rows",
    "nonfinite_rows",
    "error_rows",
    "nan_count",
    "posinf_count",
    "neginf_count",
    "float_tensors_checked",
    "float_elements_checked",
    "finite_abs_max",
)


class Ref2AVCacheFiniteScanTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "input"
        self.data.mkdir()
        self.metadata = self.data / "metadata.jsonl"

    def _save(self, name, payload, *, absolute=False, source_index=None):
        path = self.data / "cache" / f"{name}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save(payload, path)
        row = {
            "condition_path": str(path if absolute else path.relative_to(self.data)),
            "source_id": name,
        }
        if source_index is not None:
            row["source_index"] = source_index
        return row

    def _write_metadata(self, rows):
        self.metadata.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _run(self, output="reports", *extra):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--metadata",
                str(self.metadata),
                "--output-dir",
                str(self.root / output),
                "--workers",
                "1",
                "--chunk-elements",
                "3",
                "--log-every",
                "1",
                "--log-interval",
                "1",
                *map(str, extra),
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def _argv(self, output="reports", *extra):
        return [
            "--metadata",
            str(self.metadata),
            "--output-dir",
            str(self.root / output),
            *map(str, extra),
        ]

    def _assert_exit(self, result, code):
        self.assertEqual(
            result.returncode,
            code,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    @staticmethod
    def _reject_nonstandard_json(value):
        raise ValueError(f"Reports must not contain a JSON {value} constant")

    def _reports(self, output="reports"):
        directory = self.root / output
        summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8"),
            parse_constant=self._reject_nonstandard_json,
        )
        bad = [json.loads(line, parse_constant=self._reject_nonstandard_json) for line in (directory / "bad.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertIs(summary["complete"], True)
        self.assertEqual(
            summary["scanned_rows"],
            summary["clean_rows"] + summary["nonfinite_rows"] + summary["error_rows"],
        )
        self.assertTrue(all(row["status"] in {"nonfinite", "error"} for row in bad))
        self.assertEqual(len(bad), summary["nonfinite_rows"] + summary["error_rows"])
        self.assertEqual(set(summary["worker_rows_by_device"]), set(summary["devices"]))
        self.assertEqual(sum(summary["worker_rows_by_device"].values()), summary["scanned_rows"])
        self.assertTrue(all(row["scan_device"] in summary["devices"] for row in bad))
        return summary, bad

    def test_clean_nested_floats_and_integer_tensors(self):
        row = self._save(
            "clean",
            {
                "float32": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                "nested": [
                    {"bf16": torch.tensor([1, -7, 2], dtype=torch.bfloat16)},
                    torch.tensor([2**62], dtype=torch.int64),
                    torch.tensor([True, False]),
                ],
                "scalar": torch.tensor(-1e100, dtype=torch.float64),
                "description": "Non-tensor metadata is ignored",
            },
            source_index=17,
        )
        self._write_metadata([row])
        cache = self.data / row["condition_path"]
        before = cache.read_bytes()

        self._assert_exit(self._run(), 0)
        summary, bad = self._reports()

        self.assertEqual(bad, [])
        self.assertEqual(summary["scanned_rows"], 1)
        self.assertEqual(summary["clean_rows"], 1)
        self.assertEqual(summary["float_tensors_checked"], 3)
        self.assertEqual(summary["float_elements_checked"], 10)
        self.assertEqual(summary["finite_abs_max"], 1e100)
        self.assertEqual(
            [summary[key] for key in ("nan_count", "posinf_count", "neginf_count")],
            [0, 0, 0],
        )
        self.assertEqual(cache.read_bytes(), before)

    def test_nonfinite_nested_noncontiguous_and_bfloat16_across_chunks(self):
        noncontiguous = torch.tensor(
            [
                [0.0, 1.0, float("nan"), 3.0],
                [float("inf"), -6.0, 7.0, -float("inf")],
                [8.0, float("nan"), -11.0, 12.0],
            ],
            dtype=torch.float32,
        ).t()
        self.assertFalse(noncontiguous.is_contiguous())
        row = self._save(
            "nonfinite",
            {
                "nested": [
                    {"noncontiguous": noncontiguous},
                    {
                        "bf16": torch.tensor(
                            [float("nan"), float("inf"), -float("inf"), -17.0],
                            dtype=torch.bfloat16,
                        )
                    },
                ],
                "clean": torch.tensor([21.0]),
            },
            absolute=True,
            source_index=42,
        )
        self._write_metadata([row])

        self._assert_exit(self._run(), 2)
        summary, bad = self._reports()

        self.assertEqual(summary["scanned_rows"], 1)
        self.assertEqual(summary["nonfinite_rows"], 1)
        self.assertEqual(summary["error_rows"], 0)
        self.assertEqual(summary["float_tensors_checked"], 3)
        self.assertEqual(summary["float_elements_checked"], 17)
        self.assertEqual(summary["nan_count"], 3)
        self.assertEqual(summary["posinf_count"], 2)
        self.assertEqual(summary["neginf_count"], 2)
        self.assertEqual(summary["finite_abs_max"], 21.0)
        record = bad[0]
        self.assertEqual(record["status"], "nonfinite")
        self.assertEqual(record["metadata_line"], 1)
        self.assertEqual(record["condition_path"], row["condition_path"])
        self.assertEqual(record["source_id"], "nonfinite")
        self.assertEqual(record["source_index"], 42)
        self.assertEqual(len(record["tensors"]), 2)
        fp32 = next(item for item in record["tensors"] if "noncontiguous" in item["field"])
        bf16 = next(item for item in record["tensors"] if "bf16" in item["field"])
        for item, shape, dtype, numel, counts, maximum in (
            (fp32, [4, 3], "float32", 12, (2, 1, 1), 12.0),
            (bf16, [4], "bfloat16", 4, (1, 1, 1), 17.0),
        ):
            self.assertIn("nested", item["field"])
            self.assertEqual(item["shape"], shape)
            self.assertIn(dtype, item["dtype"])
            self.assertEqual(item["numel"], numel)
            self.assertEqual(
                tuple(item[key] for key in ("nan_count", "posinf_count", "neginf_count")),
                counts,
            )
            self.assertEqual(item["finite_abs_max"], maximum)

    def test_bad_rows_and_empty_tensor_are_errors_and_scanning_continues(self):
        corrupt = self.data / "corrupt.pt"
        corrupt.write_bytes(b"This is not a torch checkpoint.\n")
        empty = self._save("empty", {"nested": [torch.empty(0, dtype=torch.float32)]})
        clean = self._save("after-errors", {"value": torch.tensor([2.0])})
        lines = [
            json.dumps({"condition_path": corrupt.name, "source_id": "corrupt"}),
            "{broken JSON",
            json.dumps({"source_id": "missing-field"}),
            json.dumps({"condition_path": "missing.pt", "source_id": "missing-file"}),
            json.dumps(empty),
            json.dumps(clean),
        ]
        self.metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")

        self._assert_exit(self._run(), 2)
        summary, bad = self._reports()

        self.assertEqual(summary["scanned_rows"], 6)
        self.assertEqual(summary["clean_rows"], 1)
        self.assertEqual(summary["nonfinite_rows"], 0)
        self.assertEqual(summary["error_rows"], 5)
        self.assertEqual(sorted(record["metadata_line"] for record in bad), [1, 2, 3, 4, 5])
        self.assertTrue(all(record["status"] == "error" for record in bad))
        self.assertTrue(all(record.get("error") for record in bad))
        self.assertNotIn("after-errors", [record.get("source_id") for record in bad])

    def test_weights_only_rejects_unsupported_pickle_and_continues(self):
        unsupported = self._save("unsupported", {"date": datetime.datetime(2026, 1, 1)})
        clean = self._save("clean", {"value": torch.tensor([1.0])})
        self._write_metadata([unsupported, clean])

        self._assert_exit(self._run(), 2)
        summary, bad = self._reports()

        self.assertEqual(summary["scanned_rows"], 2)
        self.assertEqual(summary["error_rows"], 1)
        self.assertEqual(summary["clean_rows"], 1)
        self.assertEqual(bad[0]["source_id"], "unsupported")
        self.assertTrue(bad[0]["error"])

    def test_blank_lines_do_not_affect_shards_but_malformed_rows_do(self):
        rows = [self._save(str(index), {"value": torch.tensor([float("nan") if index % 2 else 1.0])}) for index in range(6)]
        self.metadata.write_text(
            "\n" + json.dumps(rows[0]) + "\n \t\n" + json.dumps(rows[1]) + "\n\n" + "{bad JSON\n" + json.dumps(rows[3]) + "\n" + json.dumps(rows[4]) + "\n\t\n" + json.dumps(rows[5]) + "\n",
            encoding="utf-8",
        )

        self._assert_exit(self._run("even", "--num-shards", 2, "--shard-index", 0), 2)
        self._assert_exit(self._run("odd", "--num-shards", 2, "--shard-index", 1), 2)
        even_summary, even_bad = self._reports("even")
        odd_summary, odd_bad = self._reports("odd")

        self.assertEqual(even_summary["scanned_rows"], 3)
        self.assertEqual(even_summary["clean_rows"], 2)
        self.assertEqual(even_summary["error_rows"], 1)
        self.assertEqual(even_bad[0]["metadata_line"], 6)
        self.assertEqual(odd_summary["scanned_rows"], 3)
        self.assertEqual(odd_summary["nonfinite_rows"], 3)
        self.assertEqual({row["source_id"] for row in odd_bad}, {"1", "3", "5"})
        self.assertEqual(sorted(row["metadata_line"] for row in odd_bad), [4, 7, 10])

    def test_max_samples_applies_after_shard_selection(self):
        rows = [self._save(str(index), {"value": torch.tensor([float("nan")])}) for index in range(8)]
        self._write_metadata(rows)

        self._assert_exit(
            self._run("limited", "--num-shards", 2, "--shard-index", 1, "--max-samples", 2),
            2,
        )
        summary, bad = self._reports("limited")

        self.assertEqual(summary["scanned_rows"], 2)
        self.assertEqual({record["source_id"] for record in bad}, {"1", "3"})
        self.assertEqual(summary["nan_count"], 2)
        self.assertIsNone(summary["finite_abs_max"])
        self.assertTrue(all(record["tensors"][0]["finite_abs_max"] is None for record in bad))

    def test_multiple_workers_and_chunk_sizes_give_identical_findings(self):
        rows = [
            self._save("clean", {"value": torch.arange(11, dtype=torch.float32)}),
            self._save("nan", {"value": torch.tensor([1.0, float("nan"), -4.0])}),
            {"condition_path": "absent.pt", "source_id": "missing"},
            self._save("inf", {"value": torch.tensor([float("inf"), -float("inf"), 8.0])}),
        ]
        self._write_metadata(rows)

        self._assert_exit(self._run("serial", "--chunk-elements", 1), 2)
        self._assert_exit(self._run("parallel", "--workers", 2, "--chunk-elements", 4), 2)
        serial_summary, serial_bad = self._reports("serial")
        parallel_summary, parallel_bad = self._reports("parallel")

        self.assertEqual(
            {key: serial_summary[key] for key in SUMMARY_COUNTS},
            {key: parallel_summary[key] for key in SUMMARY_COUNTS},
        )
        self.assertEqual(
            sorted(serial_bad, key=lambda row: row["metadata_line"]),
            sorted(parallel_bad, key=lambda row: row["metadata_line"]),
        )

    def test_existing_output_directory_is_never_overwritten(self):
        self._write_metadata([self._save("clean", {"value": torch.tensor([1.0])})])
        output = self.root / "reports"
        output.mkdir()
        original = {"summary.json": b"original summary\n", "bad.jsonl": b"original bad\n", "keep": b"keep"}
        for name, contents in original.items():
            (output / name).write_bytes(contents)

        self._assert_exit(self._run(), 1)

        self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, original)

    def test_existing_empty_output_directory_is_also_refused(self):
        self._write_metadata([self._save("clean", {"value": torch.tensor([1.0])})])
        output = self.root / "reports"
        output.mkdir()

        self._assert_exit(self._run(), 1)

        self.assertEqual(list(output.iterdir()), [])

    def test_missing_metadata_and_invalid_cli_have_failure_exit_code(self):
        self._assert_exit(self._run("missing-metadata"), 1)
        self._write_metadata([self._save("clean", {"value": torch.tensor([1.0])})])
        for index, flags in enumerate(
            (
                ("--workers", "0"),
                ("--chunk-elements", "0"),
                ("--num-shards", "2", "--shard-index", "2"),
            )
        ):
            with self.subTest(flags=flags):
                self._assert_exit(self._run(f"invalid-{index}", *flags), 2)

    def test_gpu_worker_defaults_and_explicit_matching_worker_count(self):
        cpu = MODULE.parse_args(self._argv())
        self.assertEqual(cpu.workers, 4)
        self.assertEqual(MODULE.resolve_devices(cpu), ["cpu"])
        gpu = MODULE.parse_args(self._argv("gpu", "--gpus", "0,2"))
        self.assertEqual(gpu.gpus, [0, 2])
        self.assertEqual(gpu.workers, 2)
        with patch.object(MODULE.torch.cuda, "is_available", return_value=True), patch.object(MODULE.torch.cuda, "device_count", return_value=3):
            self.assertEqual(MODULE.resolve_devices(gpu), ["cuda:0", "cuda:2"])
        explicit = MODULE.parse_args(self._argv("explicit", "--gpus", "0,1", "--workers", 2))
        self.assertEqual(explicit.workers, 2)
        eight = MODULE.parse_args(self._argv("eight", "--gpus", "0,1,2,3,4,5,6,7"))
        self.assertEqual(eight.workers, 8)
        with patch.object(MODULE.torch.cuda, "is_available", return_value=True), patch.object(MODULE.torch.cuda, "device_count", return_value=8):
            self.assertEqual(MODULE.resolve_devices(eight), [f"cuda:{index}" for index in range(8)])

    def test_invalid_gpu_lists_and_mismatched_workers_are_cli_errors(self):
        cases = (
            ("--gpus", ""),
            ("--gpus", "0,"),
            ("--gpus", "0,,1"),
            ("--gpus", "one"),
            ("--gpus", "1.5"),
            ("--gpus=-1",),
            ("--gpus", "0,0"),
            ("--gpus", "0,1", "--workers", "1"),
        )
        for flags in cases:
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    MODULE.parse_args(self._argv("invalid-gpu", *flags))
                self.assertEqual(raised.exception.code, 2)

    def test_unavailable_gpu_fails_before_output_creation_without_cpu_fallback(self):
        self._write_metadata([self._save("clean", {"value": torch.tensor([1.0])})])
        stderr = io.StringIO()
        with (
            patch.object(MODULE.torch.cuda, "is_available", return_value=False),
            patch.object(MODULE.torch.cuda, "device_count", return_value=0),
            patch.object(MODULE, "ProcessPoolExecutor") as executor,
            redirect_stderr(stderr),
        ):
            result = MODULE.main(self._argv("no-gpu", "--gpus", "0"))

        self.assertEqual(result, 1)
        self.assertIn("cuda", stderr.getvalue().lower())
        self.assertFalse((self.root / "no-gpu").exists())
        executor.assert_not_called()

    def test_out_of_range_gpu_fails_before_output_creation(self):
        self._write_metadata([self._save("clean", {"value": torch.tensor([1.0])})])
        stderr = io.StringIO()
        with (
            patch.object(MODULE.torch.cuda, "is_available", return_value=True),
            patch.object(MODULE.torch.cuda, "device_count", return_value=2),
            patch.object(MODULE, "ProcessPoolExecutor") as executor,
            redirect_stderr(stderr),
        ):
            result = MODULE.main(self._argv("out-of-range", "--gpus", "0,2"))

        self.assertEqual(result, 1)
        self.assertFalse((self.root / "out-of-range").exists())
        self.assertTrue(stderr.getvalue())
        executor.assert_not_called()

    def test_device_reduction_algorithm_matches_reference_without_cuda(self):
        cases = [
            torch.tensor(
                [0.0, float("nan"), float("inf"), -17.0, -float("inf"), 2.0, float("nan")],
                dtype=dtype,
            )
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        ]
        cases.extend(
            [
                torch.tensor([float("nan"), float("nan")], dtype=torch.bfloat16),
                torch.tensor([float("nan"), float("inf"), -float("inf")]),
                torch.tensor([-1e100, float("nan"), 1e99, float("inf")], dtype=torch.float64),
                torch.zeros(5),
            ]
        )
        for index, value in enumerate(cases):
            before = value.clone()
            for chunk_elements in (1, 2, 4, 100):
                with self.subTest(case=index, dtype=value.dtype, chunk_elements=chunk_elements):
                    reference = MODULE.tensor_stats("$.value", value, chunk_elements, device="cpu")
                    counts, maximum = MODULE.chunked_device_reductions(value, chunk_elements, "cpu")
                    self.assertEqual(
                        list(counts),
                        [reference[key] for key in ("nan_count", "posinf_count", "neginf_count")],
                    )
                    self.assertEqual(maximum, reference["finite_abs_max"])
                    torch.testing.assert_close(value, before, equal_nan=True)

    def test_gpu_tensor_statistics_match_cpu_and_preserve_fp64(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable; actual GPU tensor execution was not tested")
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                magnitude = 1e100 if dtype == torch.float64 else 17.0
                value = torch.tensor(
                    [
                        [float("nan"), 1.0, float("inf")],
                        [-float("inf"), -magnitude, 3.0],
                        [4.0, float("nan"), -5.0],
                    ],
                    dtype=dtype,
                ).t()
                self.assertFalse(value.is_contiguous())
                cpu = MODULE.tensor_stats("$.value", value, 2, device="cpu")
                gpu = MODULE.tensor_stats("$.value", value, 2, device="cuda:0")
                self.assertEqual(cpu, gpu)
                self.assertEqual(gpu["nan_count"], 2)
                self.assertEqual(gpu["posinf_count"], 1)
                self.assertEqual(gpu["neginf_count"], 1)
                self.assertEqual(gpu["finite_abs_max"], magnitude)
                self.assertEqual(value.device.type, "cpu")
        all_nonfinite = torch.tensor([float("nan"), float("inf"), -float("inf")], dtype=torch.bfloat16)
        cpu = MODULE.tensor_stats("$.all_nonfinite", all_nonfinite, 1, device="cpu")
        gpu = MODULE.tensor_stats("$.all_nonfinite", all_nonfinite, 1, device="cuda:0")
        self.assertEqual(cpu, gpu)
        self.assertIsNone(gpu["finite_abs_max"])

    def test_gpu_tensor_statistics_require_dense_cpu_input(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable; actual GPU tensor execution was not tested")
        gpu_input = torch.tensor([1.0], device="cuda:0")
        with self.assertRaises(ValueError):
            MODULE.tensor_stats("$.gpu_input", gpu_input, 2, device="cuda:0")
        sparse_input = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse()
        with self.assertRaises(ValueError):
            MODULE.tensor_stats("$.sparse_input", sparse_input, 2, device="cuda:0")

    def test_two_gpu_cli_matches_cpu_findings_and_reports_worker_devices(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("Two CUDA devices are required; actual multi-GPU execution was not tested")
        rows = [
            self._save("clean", {"value": torch.arange(11, dtype=torch.float32)}),
            self._save("nan", {"value": torch.tensor([1.0, float("nan"), -4.0])}),
            {"condition_path": "absent.pt", "source_id": "missing"},
            self._save("inf", {"value": torch.tensor([float("inf"), -float("inf"), 8.0])}),
            self._save("bf16", {"value": torch.tensor([2.0, float("nan")], dtype=torch.bfloat16)}),
            self._save("fp64", {"value": torch.tensor([-1e100, float("inf")], dtype=torch.float64)}),
        ]
        self._write_metadata(rows)

        self._assert_exit(self._run("cpu", "--chunk-elements", 2), 2)
        self._assert_exit(self._run("gpu", "--gpus", "0,1", "--workers", 2), 2)
        cpu_summary, cpu_bad = self._reports("cpu")
        gpu_summary, gpu_bad = self._reports("gpu")

        self.assertEqual(
            {key: cpu_summary[key] for key in SUMMARY_COUNTS},
            {key: gpu_summary[key] for key in SUMMARY_COUNTS},
        )
        self.assertEqual(cpu_summary["devices"], ["cpu"])
        self.assertEqual(gpu_summary["devices"], ["cuda:0", "cuda:1"])
        self.assertEqual(set(gpu_summary["worker_rows_by_device"]), {"cuda:0", "cuda:1"})
        self.assertEqual(sum(gpu_summary["worker_rows_by_device"].values()), len(rows))
        self.assertTrue(all(record["scan_device"] == "cpu" for record in cpu_bad))
        self.assertTrue(all(record["scan_device"] in {"cuda:0", "cuda:1"} for record in gpu_bad))

        def findings(records):
            return sorted(
                [{key: value for key, value in record.items() if key != "scan_device"} for record in records],
                key=lambda record: record["metadata_line"],
            )

        self.assertEqual(findings(cpu_bad), findings(gpu_bad))


if __name__ == "__main__":
    unittest.main()
