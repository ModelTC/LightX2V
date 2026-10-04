#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("filter_minimax_h3_ref2av_context_ir.py")
SPEC = importlib.util.spec_from_file_location("ref2av_context_ir_filter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _row(sample_id, image_count, *, enhanced=False, orientation="landscape"):
    height, width = (768, 1344) if orientation == "landscape" else (1344, 768)
    row = {
        "sample_id": sample_id,
        "task": "ref2av",
        "duration": 5,
        "target_num_frames": 124,
        "target_height": height,
        "target_width": width,
        "target_orientation": orientation,
        "reference_images": [f"image_{index}.jpg" for index in range(image_count)],
        "reference_videos": [],
        "reference_audios": [],
        "references": [{"order": index + 1, "kind": "image", "path": f"image_{index}.jpg"} for index in range(image_count)],
    }
    if enhanced:
        row["enhanced_prompt"] = f"enhanced {sample_id}"
    return row


class Ref2AVContextIRFilterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write(path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def _args(self):
        source = self.root / "raw_stories.jsonl"
        success = self.root / "context_ir.jsonl"
        failed = self.root / "context_ir.failed.jsonl"
        output = self.root / "context_ir_image_counts_1_2_3.jsonl"
        self._write(
            source,
            [
                _row("one", 1),
                _row("two", 2, orientation="portrait"),
                _row("three", 3),
                _row("four", 4),
            ],
        )
        self._write(success, [_row("three", 3, enhanced=True), _row("four", 4, enhanced=True), _row("one", 1, enhanced=True)])
        self._write(failed, [_row("two", 2, orientation="portrait")])
        return MODULE.parse_args(
            [
                "--source",
                str(source),
                "--success",
                str(success),
                "--failed",
                str(failed),
                "--output",
                str(output),
                "--image-counts",
                "1",
                "2",
                "3",
                "--expected-source-selected-count",
                "3",
                "--expected-source-count",
                "1=1",
                "--expected-source-count",
                "2=1",
                "--expected-source-count",
                "3=1",
            ]
        )

    def test_terminal_join_preserves_source_order_and_excludes_failures(self):
        args = self._args()
        manifest = MODULE.build_selection(args)
        output = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["sample_id"] for row in output], ["one", "three"])
        self.assertTrue(all("_audit_line_number" not in row for row in output))
        self.assertEqual(manifest["source_selected_rows"], 3)
        self.assertEqual(manifest["successful_selected_rows"], 2)
        self.assertEqual(manifest["terminal_failed_selected_rows"], 1)
        self.assertEqual(manifest["success_image_counts"], {1: 1, 3: 1})
        self.assertEqual(manifest["terminal_failed_sample_ids"], ["two"])

    def test_refuses_nonterminal_selected_story(self):
        args = self._args()
        args.failed.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not terminal"):
            MODULE.build_selection(args)

    def test_refuses_duplicate_success_identity(self):
        args = self._args()
        rows = MODULE.read_jsonl(args.success)
        MODULE.atomic_write_jsonl(args.success, [MODULE.clean_row(row) for row in rows] + [_row("one", 1, enhanced=True)], overwrite=True)
        with self.assertRaisesRegex(ValueError, "Duplicate selected sample_id"):
            MODULE.build_selection(args)

    def test_unrelated_later_rows_do_not_change_immutable_selection(self):
        args = self._args()
        first = MODULE.build_selection(args)
        with args.success.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_row("later-nine", 9, enhanced=True)) + "\n")
        second = MODULE.build_selection(args)
        self.assertEqual(first["selection_fingerprint"], second["selection_fingerprint"])
        stored = json.loads(args.output.with_suffix(args.output.suffix + ".manifest.json").read_text())
        self.assertEqual(stored["selection_fingerprint"], first["selection_fingerprint"])


if __name__ == "__main__":
    unittest.main()
