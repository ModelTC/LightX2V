#!/usr/bin/env python3
"""Offline scanner tests: mocked ffprobe only, compatible with Python 3.8."""

import contextlib
import copy
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT = Path(__file__).with_name("scan_audio_cases.py")
SPEC = importlib.util.spec_from_file_location("ref2av_audio_scan_under_test", SCRIPT)
SCAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCAN
SPEC.loader.exec_module(SCAN)


def probe_result(path, status="ok", has_audio=False):
    result = {"path": str(path), "status": status}
    if status == "ok":
        result.update(has_audio=has_audio, audio_streams=[{"codec_type": "audio"}] if has_audio else [])
    else:
        result["error"] = "mock " + status
    return result


class ReferenceSchemaTest(unittest.TestCase):
    def test_seko_input_media_preserves_audio_absent_from_inputs(self):
        row = {
            "sample_id": "seko_minimax_1",
            "inputs": [{"role": "reference_image", "assets": [{"modality": "image", "rel_path": "image.jpg"}]}],
            "seko": {
                "input_media": [
                    {"modality": "image", "rel_path": "image.jpg"},
                    {"modality": "video", "rel_path": "reference.mp4"},
                    {"modality": "audio", "rel_path": "voice.wav"},
                ]
            },
            "target": "target.mp4",
        }
        original = copy.deepcopy(row)
        self.assertEqual(
            SCAN.extract_references(row),
            [
                {"kind": "image", "path": "image.jpg"},
                {"kind": "video", "path": "reference.mp4"},
                {"kind": "audio", "path": "voice.wav"},
            ],
        )
        self.assertEqual(row, original)

    def test_authoritative_references_are_not_merged_with_raw_aliases(self):
        row = {
            "references": [{"kind": "video", "path": "chosen.mp4"}],
            "actual_ordered_references": [{"kind": "audio", "path": "old.wav"}],
            "seko": {"input_media": [{"modality": "audio", "rel_path": "raw.wav"}]},
            "reference_audios": ["alias.wav"],
        }
        self.assertEqual(SCAN.extract_references(row), [{"kind": "video", "path": "chosen.mp4"}])
        row["references"] = []
        self.assertEqual(SCAN.extract_references(row), [])

    def test_kind_path_aliases_and_reference_arrays(self):
        self.assertEqual(
            SCAN.extract_references(
                {
                    "ordered_references": [
                        {"type": "image_url", "url": "one.jpg"},
                        {"role": "reference_video", "local_path": "two.mp4"},
                        {"modality": "audio_url", "media_path": "three.wav"},
                    ]
                }
            ),
            [
                {"kind": "image", "path": "one.jpg"},
                {"kind": "video", "path": "two.mp4"},
                {"kind": "audio", "path": "three.wav"},
            ],
        )
        self.assertEqual(
            SCAN.extract_references(
                {
                    "reference_images": ["one.jpg"],
                    "reference_video_url": "two.mp4",
                    "reference_audios": ["three.wav", "four.wav"],
                }
            ),
            [
                {"kind": "image", "path": "one.jpg"},
                {"kind": "video", "path": "two.mp4"},
                {"kind": "audio", "path": "three.wav"},
                {"kind": "audio", "path": "four.wav"},
            ],
        )

    def test_malformed_authoritative_container_does_not_fall_back(self):
        for value in ({"kind": "video"}, ["video.mp4"], [{"kind": "unsupported"}]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SCAN.extract_references({"references": value, "reference_videos": ["fallback.mp4"]})

    def test_source_classification_uses_provenance(self):
        cases = [
            ({"sample_id": "seko_minimax_1"}, "seko_minimax"),
            ({"sample_id": "seko_volc_2"}, "seko_volc"),
            ({"source_name": "omini-r2v"}, "omni"),
            ({"sample_id": "r2v_3"}, "omni"),
            ({"source_name": "another_source", "prompt": "seko_minimax omni"}, "another_source"),
        ]
        for row, expected in cases:
            with self.subTest(row=row):
                self.assertEqual(SCAN.source_name(row), expected)


class ClassificationTest(unittest.TestCase):
    def test_positive_and_failure_is_affected_but_mapping_incomplete(self):
        result = SCAN.classify_case([probe_result("yes.mp4", has_audio=True), probe_result("missing.mp4", "missing")], 2)
        self.assertEqual(result["classification"], "affected")
        self.assertEqual(result["embedded_audio_count_lower_bound"], 1)
        self.assertFalse(result["mapping_complete"])
        self.assertIsNone(result["audio_index_shift"])

    def test_known_no_audio_is_distinct_from_unknown(self):
        result = SCAN.classify_case([probe_result("silent.mp4")], 1)
        self.assertEqual(result["classification"], "no_embedded_audio")
        self.assertEqual(result["audio_index_shift"], 0)
        self.assertTrue(result["mapping_complete"])
        for status in ("missing", "error", "unsupported"):
            with self.subTest(status=status):
                result = SCAN.classify_case([probe_result("unknown.mp4", status)], 1)
                self.assertEqual(result["classification"], "unknown")
                self.assertEqual(result["embedded_audio_count_lower_bound"], 0)
                self.assertIsNone(result["audio_index_shift"])
                self.assertFalse(result["mapping_complete"])

    def test_video_without_standalone_audio_still_needs_review(self):
        result = SCAN.classify_case([probe_result("video.mp4", has_audio=True)], 0)
        self.assertEqual(result["classification"], "affected")
        self.assertEqual(result["case_type"], "video_without_standalone_audio")
        self.assertEqual(result["audio_index_shift"], 1)

    def test_shift_counts_reference_occurrences_not_tracks_or_unique_files(self):
        video = probe_result("same.mp4", has_audio=True)
        video["audio_streams"] *= 2
        result = SCAN.classify_case([video, video], 1)
        self.assertEqual(result["embedded_audio_count_lower_bound"], 2)
        self.assertEqual(result["audio_index_shift"], 2)
        self.assertTrue(result["mapping_complete"])


class FilesystemFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ref2av-audio-audit-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = self.root / "video.mp4"
        self.video.write_bytes(b"mock local video; never decoded")
        self.input = self.root / "input.jsonl"
        self.input.write_text("", encoding="utf-8")

    def args(self, output="reports"):
        return SCAN.parse_args(
            [
                "--input",
                str(self.input),
                "--output-dir",
                str(self.root / output),
                "--media-root",
                str(self.root),
                "--workers",
                "2",
                "--probe-timeout",
                "1",
            ]
        )


class ProbeTest(FilesystemFixture):
    def test_success_requires_video_and_distinguishes_audio(self):
        for with_audio in (False, True):
            streams = [{"codec_type": "video", "index": 0}]
            if with_audio:
                streams.append({"codec_type": "audio", "index": 1, "channels": 2})
            completed = SimpleNamespace(stdout=json.dumps({"streams": streams}))
            with self.subTest(with_audio=with_audio):
                with mock.patch.object(SCAN.subprocess, "run", return_value=completed) as command:
                    result = SCAN.probe_video(str(self.video), "mock-ffprobe", 1)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["has_audio"], with_audio)
                self.assertEqual(len(result["audio_streams"]), int(with_audio))
                self.assertEqual(command.call_args.kwargs["timeout"], 1)

    def test_missing_and_remote_paths_never_launch_probe(self):
        with mock.patch.object(SCAN.subprocess, "run") as command:
            missing = SCAN.probe_video(str(self.root / "absent.mp4"), "ffprobe", 1)
            remote = SCAN.probe_video("https://example.invalid/video.mp4", "ffprobe", 1)
            absent = SCAN.probe_video(None, "ffprobe", 1)
        command.assert_not_called()
        self.assertEqual(missing["status"], "missing")
        self.assertEqual(remote["status"], "unsupported")
        self.assertEqual(absent["status"], "unsupported")
        for result in (missing, remote, absent):
            self.assertNotIn("has_audio", result)

    def test_timeouts_failed_decodes_and_invalid_streams_are_unknown_not_silent(self):
        failures = [
            subprocess.TimeoutExpired("ffprobe", 1),
            subprocess.CalledProcessError(1, "ffprobe", stderr="bad media"),
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                with mock.patch.object(SCAN.subprocess, "run", side_effect=failure):
                    result = SCAN.probe_video(str(self.video), "ffprobe", 1)
                self.assertEqual(result["status"], "error")
                self.assertNotIn("has_audio", result)
                self.assertEqual(SCAN.classify_case([result], 0)["classification"], "unknown")
        for stdout in ('{"streams": [{"codec_type": "audio"}]}', '{"streams": null}', "not json"):
            with self.subTest(stdout=stdout):
                with mock.patch.object(SCAN.subprocess, "run", return_value=SimpleNamespace(stdout=stdout)):
                    result = SCAN.probe_video(str(self.video), "ffprobe", 1)
                self.assertEqual(result["status"], "error")
                self.assertNotIn("has_audio", result)


class ScanIntegrationTest(FilesystemFixture):
    def test_reports_all_video_cases_caches_paths_and_preserves_inputs(self):
        rows = [
            {
                "sample_id": "seko_minimax_raw",
                "ir_status": "succeeded",
                "inputs": [{"assets": [{"modality": "image", "rel_path": "image.jpg"}]}],
                "seko": {"input_media": [{"modality": "video", "rel_path": "video.mp4"}, {"modality": "video", "rel_path": "missing.mp4"}, {"modality": "audio", "rel_path": "voice.wav"}]},
            },
            {"sample_id": "r2v_video_only", "ir_status": "succeeded", "references": [{"kind": "image", "path": "image.jpg"}, {"kind": "video", "path": "video.mp4"}]},
            {"sample_id": "seko_volc_silent", "ir_status": "failed", "references": [{"kind": "video", "path": "silent.mp4"}, {"kind": "audio", "path": "voice.wav"}]},
            {"source_name": "another_source", "references": [{"kind": "video", "path": "timeout.mp4"}]},
            {"source_name": "omni", "references": [{"kind": "image", "path": "image.jpg"}], "target": "target_must_not_be_probed.mp4"},
            {
                "sample_id": "seko_minimax_repeated",
                "ir_status": "succeeded",
                "references": [{"kind": "video", "path": "video.mp4"}, {"kind": "video", "path": "video.mp4"}, {"kind": "audio", "path": "voice.wav"}],
            },
        ]
        self.input.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        original_input, original_video = self.input.read_bytes(), self.video.read_bytes()

        def fake_probe(path, ffprobe, timeout):
            name = Path(path).name
            return probe_result(path, {"missing.mp4": "missing", "timeout.mp4": "error"}.get(name, "ok"), has_audio=name == "video.mp4")

        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            stack.enter_context(mock.patch.object(SCAN.shutil, "which", return_value="/mock/ffprobe"))
            probe = stack.enter_context(mock.patch.object(SCAN, "probe_video", side_effect=fake_probe))
            summary = SCAN.scan(self.args())
        self.assertTrue(summary["complete"])
        for metric, expected in {"rows": 6, "candidates": 5, "affected": 3, "affected_incomplete": 1, "no_embedded_audio": 1, "unknown": 1}.items():
            self.assertEqual(summary["totals"][metric], expected, metric)
        self.assertEqual(probe.call_count, 4)
        self.assertEqual(summary["unique_video_paths"], 4)
        self.assertEqual(summary["probe_status_counts"], {"ok": 2, "missing": 1, "error": 1})
        self.assertEqual(summary["by_source"]["seko_minimax"]["affected"], 2)
        self.assertEqual(summary["by_ir_status"]["succeeded"]["affected"], 3)
        self.assertEqual(summary["by_case_type"]["video_with_standalone_audio"]["candidates"], 3)
        self.assertEqual(summary["by_case_type"]["video_without_standalone_audio"]["candidates"], 2)
        cases = [json.loads(line) for line in (self.root / "reports/cases.jsonl").read_text().splitlines()]
        self.assertEqual([case["input_line"] for case in cases], [1, 2, 3, 4, 6])
        self.assertEqual(cases[0]["explicit_audio_count"], 1)
        self.assertFalse(cases[0]["mapping_complete"])
        self.assertIsNone(cases[0]["audio_index_shift"])
        self.assertEqual(cases[-1]["audio_index_shift"], 2)
        self.assertEqual(self.input.read_bytes(), original_input)
        self.assertEqual(self.video.read_bytes(), original_video)
        self.assertEqual(json.loads((self.root / "reports/summary.json").read_text()), summary)

    def test_existing_output_directory_is_protected(self):
        output = self.root / "reports"
        output.mkdir()
        sentinel = output / "summary.json"
        sentinel.write_text("pre-existing report", encoding="utf-8")
        with mock.patch.object(SCAN.shutil, "which", return_value="/mock/ffprobe"):
            with self.assertRaises(FileExistsError):
                SCAN.scan(self.args())
        self.assertEqual(sentinel.read_text(), "pre-existing report")
        self.assertFalse((output / "cases.jsonl").exists())

    def test_duplicate_input_and_nonfinite_timeouts_are_rejected(self):
        args = self.args()
        args.input.append(self.input)
        with self.assertRaisesRegex(ValueError, "Duplicate --input"):
            SCAN.scan(args)
        self.assertFalse(args.output_dir.exists())
        for value in ("nan", "inf", "0", "-1"):
            with self.subTest(timeout=value):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        SCAN.parse_args(["--probe-timeout", value])


if __name__ == "__main__":
    unittest.main()
