#!/usr/bin/env python3
"""Fixed-rank partition tests and real, CPU-only torchrun integration tests."""

import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch

SCRIPT = Path(__file__).with_name("scan_ref2av_cache_finite.py").resolve()
SPEC = importlib.util.spec_from_file_location("ref2av_finite_scanner_ddp_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
COUNT_FIELDS = (
    "scanned_rows",
    "clean_rows",
    "nonfinite_rows",
    "error_rows",
    "float_tensors_checked",
    "float_elements_checked",
    "nan_count",
    "posinf_count",
    "neginf_count",
)


class Ref2AVCacheFiniteDDPTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "input"
        self.data.mkdir()
        self.metadata = self.data / "metadata.jsonl"

    def _save(self, index, values):
        path = self.data / f"cache-{index}.pt"
        torch.save({"nested": [{"value": torch.tensor(values, dtype=torch.float64)}]}, path)
        return {"condition_path": path.name, "source_id": f"sample-{index}", "source_index": index}

    def _write_with_blanks(self, rows):
        lines = [""]
        physical_lines = []
        for index, row in enumerate(rows):
            if index % 4 == 0:
                lines.append(" \t")
            physical_lines.append(len(lines) + 1)
            lines.append(row if isinstance(row, str) else json.dumps(row))
        self.metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return physical_lines

    def _argv(self, output="reports", *extra):
        return [
            "--metadata",
            str(self.metadata),
            "--output-dir",
            str(self.root / output),
            "--ddp",
            "--ddp-device",
            "cpu",
            "--log-every",
            "1",
            "--log-interval",
            "1",
            *map(str, extra),
        ]

    def _torchrun(self, output="reports", *extra):
        # Use the same interpreter/torch installation as unittest. Do not let
        # an enclosing launcher accidentally turn this into a multi-node run.
        environment = os.environ.copy()
        for key in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            environment.pop(key, None)
        environment["OMP_NUM_THREADS"] = "1"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(SCRIPT),
            *self._argv(output, *extra),
        ]
        with subprocess.Popen(
            command,
            cwd=self.root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=90)
            except subprocess.TimeoutExpired:
                # A synchronization regression must not leave peer processes
                # running after unittest fails. This group belongs to this run.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate()
                self.fail(f"torchrun timed out; its process group was stopped.\n{stdout}\n{stderr}")
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    def _assert_exit(self, result, expected):
        self.assertEqual(
            result.returncode,
            expected,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    @staticmethod
    def _reject_nonstandard_json(value):
        raise ValueError(f"Report contains nonstandard JSON constant: {value}")

    def _read_reports(self, output="reports", rank=None):
        directory = self.root / output
        if rank is not None:
            directory = directory / f"rank-{rank:05d}"
        summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8"),
            parse_constant=self._reject_nonstandard_json,
        )
        bad = [json.loads(line, parse_constant=self._reject_nonstandard_json) for line in (directory / "bad.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(
            summary["scanned_rows"],
            sum(summary[key] for key in ("clean_rows", "nonfinite_rows", "error_rows")),
        )
        self.assertEqual(len(bad), summary["nonfinite_rows"] + summary["error_rows"])
        return summary, bad

    def test_eight_rank_partition_has_no_padding_drops_or_overlap(self):
        rows = [{"condition_path": f"cache-{index}.pt", "source_index": index} for index in range(19)]
        rows[9] = "{malformed row still occupies one nonempty index"
        physical_lines = self._write_with_blanks(rows)
        args = MODULE.parse_args(self._argv())
        assigned_lines = []

        for rank in range(8):
            with self.subTest(rank=rank):
                selected = list(MODULE.ddp_selected_rows(args, rank, 8))
                lines = [line for line, _ in selected]
                self.assertEqual(lines, physical_lines[rank::8])
                assigned_lines.extend(lines)

        self.assertEqual(sorted(assigned_lines), sorted(physical_lines))
        self.assertEqual(len(assigned_lines), len(set(assigned_lines)))

    def test_ddp_max_samples_is_a_global_prefix_before_eight_rank_partition(self):
        rows = [{"condition_path": f"cache-{index}.pt"} for index in range(19)]
        physical_lines = self._write_with_blanks(rows)
        args = MODULE.parse_args(self._argv("limited", "--max-samples", 10))
        assigned_lines = []

        for rank in range(8):
            with self.subTest(rank=rank):
                selected = list(MODULE.ddp_selected_rows(args, rank, 8))
                lines = [line for line, _ in selected]
                self.assertEqual(lines, physical_lines[:10][rank::8])
                assigned_lines.extend(lines)

        self.assertEqual(sorted(assigned_lines), sorted(physical_lines[:10]))
        self.assertEqual(len(assigned_lines), 10)

    def test_ddp_rejects_worker_pool_chunk_and_external_shard_options(self):
        for flags in (
            ("--gpus", "0"),
            ("--workers", "1"),
            ("--num-shards", "1"),
            ("--shard-index", "0"),
            ("--chunk-elements", "1024"),
        ):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    MODULE.parse_args(self._argv("invalid", *flags))
                self.assertEqual(raised.exception.code, 2)

    def test_rank_scans_full_tensors_in_its_own_process_without_a_worker_pool(self):
        values = [1.0, float("nan"), float("inf"), -float("inf"), -7.0]
        self._write_with_blanks([self._save(0, values)])
        args = MODULE.parse_args(self._argv())
        args.output_dir.mkdir()
        # DDP must ignore the legacy chunk setting even if args is constructed
        # programmatically; the complete tensor is checked in one operation.
        args.chunk_elements = 1
        with patch.object(MODULE, "ProcessPoolExecutor") as executor, patch.object(MODULE, "scan_record", wraps=MODULE.scan_record) as scan, redirect_stderr(io.StringIO()):
            summary = MODULE.scan_ddp_rank(args, 0, 1, 0, "cpu")

        executor.assert_not_called()
        scan.assert_called_once()
        self.assertGreaterEqual(scan.call_args.args[3], len(values))
        self.assertIs(summary["complete"], True)
        self.assertEqual(summary["scanned_rows"], 1)
        self.assertEqual(summary["nonfinite_rows"], 1)
        self.assertEqual(summary["float_elements_checked"], len(values))
        self.assertEqual([summary[key] for key in ("nan_count", "posinf_count", "neginf_count")], [1, 1, 1])

    def test_two_rank_scan_finishes_with_bad_rows_and_merges_exact_partition(self):
        rows = []
        for index in range(18):
            anomaly = (float("nan"), float("inf"), -float("inf"))[index % 3]
            rows.append(self._save(index, [float(index), anomaly]))
        rows[5] = {"condition_path": "missing.pt", "source_id": "missing", "source_index": 5}
        rows[8] = "{bad JSON"
        physical_lines = self._write_with_blanks(rows)
        before = {path.name: path.read_bytes() for path in self.data.glob("*.pt")}

        # Finding corrupt/nonfinite data is an audit result, not a torchrun
        # worker failure: peers must finish and the launcher must return zero.
        self._assert_exit(self._torchrun(), 0)
        summary, bad = self._read_reports()
        rank_reports = [self._read_reports(rank=rank) for rank in range(2)]

        self.assertIs(summary["complete"], True)
        self.assertEqual(summary["audit_exit_code"], 2)
        self.assertEqual(summary["scanned_rows"], 18)
        self.assertEqual(summary["clean_rows"], 0)
        self.assertEqual(summary["nonfinite_rows"], 16)
        self.assertEqual(summary["error_rows"], 2)
        self.assertEqual(summary["nan_count"], 6)
        self.assertEqual(summary["posinf_count"], 6)
        self.assertEqual(summary["neginf_count"], 4)
        self.assertEqual(summary["float_tensors_checked"], 16)
        self.assertEqual(summary["float_elements_checked"], 32)
        self.assertEqual(summary["finite_abs_max"], 17.0)
        self.assertEqual(sorted(record["metadata_line"] for record in bad), physical_lines)
        seen = set()
        merged = []
        for rank, (rank_summary, rank_bad) in enumerate(rank_reports):
            self.assertIs(rank_summary["complete"], True)
            self.assertEqual(rank_summary["scanned_rows"], 9)
            lines = {record["metadata_line"] for record in rank_bad}
            self.assertEqual(lines, set(physical_lines[rank::2]))
            self.assertTrue(seen.isdisjoint(lines))
            seen.update(lines)
            merged.extend(rank_bad)
        self.assertEqual(seen, set(physical_lines))
        self.assertEqual(
            sorted(merged, key=lambda row: row["metadata_line"]),
            sorted(bad, key=lambda row: row["metadata_line"]),
        )
        for key in COUNT_FIELDS:
            self.assertEqual(summary[key], sum(rank_summary[key] for rank_summary, _ in rank_reports))
        self.assertEqual({path.name: path.read_bytes() for path in self.data.glob("*.pt")}, before)

    def test_global_limit_can_leave_an_empty_rank_and_still_complete_cleanly(self):
        rows = [self._save(0, [21.0])]
        rows.extend(self._save(index, [float("nan")]) for index in range(1, 5))
        self._write_with_blanks(rows)

        self._assert_exit(self._torchrun("limited", "--max-samples", 1), 0)
        summary, bad = self._read_reports("limited")
        first_summary, first_bad = self._read_reports("limited", rank=0)
        empty_summary, empty_bad = self._read_reports("limited", rank=1)

        self.assertIs(summary["complete"], True)
        self.assertEqual(summary["audit_exit_code"], 0)
        self.assertEqual(summary["scanned_rows"], 1)
        self.assertEqual(summary["clean_rows"], 1)
        self.assertEqual(summary["finite_abs_max"], 21.0)
        self.assertIs(first_summary["complete"], True)
        self.assertEqual(first_summary["scanned_rows"], 1)
        self.assertIs(empty_summary["complete"], True)
        for key in COUNT_FIELDS:
            self.assertEqual(empty_summary[key], 0)
        self.assertIsNone(empty_summary["finite_abs_max"])
        self.assertEqual(bad + first_bad + empty_bad, [])

    def test_existing_output_is_refused_without_overwriting_or_hanging_peers(self):
        self._write_with_blanks([self._save(0, [1.0])])
        output = self.root / "reports"
        output.mkdir()
        originals = {
            "summary.json": b"preserve prior summary\n",
            "bad.jsonl": b"preserve prior audit\n",
            "sentinel": b"preserve unrelated content\n",
        }
        for name, contents in originals.items():
            (output / name).write_bytes(contents)

        self._assert_exit(self._torchrun(), 1)

        self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, originals)

    def test_all_blank_manifest_fails_without_a_complete_root_summary(self):
        self.metadata.write_text("\n \t\n\n", encoding="utf-8")

        self._assert_exit(self._torchrun(), 1)
        summary = json.loads((self.root / "reports" / "summary.json").read_text(encoding="utf-8"))

        self.assertIs(summary["complete"], False)
        self.assertEqual(summary["scanned_rows"], 0)


if __name__ == "__main__":
    unittest.main()
