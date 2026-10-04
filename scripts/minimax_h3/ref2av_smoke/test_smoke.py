#!/usr/bin/env python3
"""CPU-only sampler/client contract tests; no model, GPU, or HTTP service needed."""

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"ref2av_smoke_test_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


POST = load_script("post")
SAMPLE = load_script("sample")
SERVICE_METADATA = {"model_cls": "minimax_h3", "nproc_per_node": 8}


class SmokeFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ref2av-smoke-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "samples.jsonl"
        self.image = self.make_file("image.jpg", b"reference image")
        self.video = self.make_file("video.mp4", b"reference video")
        self.audio = self.make_file("audio.wav", b"reference audio")
        self.target = self.make_file("target.mp4", b"original target video")
        self.target_copy = self.make_file("targets/case.mp4", self.target.read_bytes())

    def make_file(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def write_jsonl(self, rows, path=None):
        path = path or self.manifest
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def client_row(self):
        return {
            "sample_id": "seko_minimax_case",
            "ir_status": "succeeded",
            "enhanced_prompt": "<Picture 1> 角色说：你好。",
            "smoke": {
                "id": "case",
                "index": 0,
                "group": "seko",
                "provider": "h3",
                "references": [{"kind": "image", "path": str(self.image)}],
                "target_source": str(self.target),
                "target_copy": str(self.target_copy),
                "generated_path": str(self.root / "generated" / "case.mp4"),
                "seed": 42,
                "num_frames": 124,
                "size": [768, 1344],
                "fps": 24,
            },
        }

    def args(self):
        return SimpleNamespace(
            manifest=self.manifest,
            base_url="http://127.0.0.1:8000",
            http_timeout=1,
            health_timeout=1,
            task_timeout=1,
            poll_interval=0.001,
        )


class PostContractTest(SmokeFixture):
    def test_payload_preserves_enhanced_prompt_and_never_conditions_on_target(self):
        row = self.client_row()
        row["duration"] = 29
        row["target_num_frames"] = 697
        row["smoke"]["references"].append({"kind": "audio", "path": str(self.audio)})
        self.write_jsonl([row])
        loaded = POST.load_manifest(self.manifest, expected_count=1)
        payload = POST.build_payload(loaded[0], self.manifest)
        self.assertEqual(payload["prompt"], row["enhanced_prompt"])
        self.assertEqual(payload["task"], "ref2av")
        self.assertEqual(payload["num_frames"], 124)
        self.assertEqual(payload["size"], [768, 1344])
        self.assertEqual(payload["image_path"], str(self.image))
        self.assertEqual(payload["audio_path"], str(self.audio))
        self.assertEqual(payload["video_path"], "")
        self.assertNotIn(str(self.target), json.dumps(payload))
        self.assertNotIn(str(self.target_copy), json.dumps(payload))

    def test_manifest_rejects_missing_reference_and_input_output_overlap(self):
        row = self.client_row()
        row["smoke"]["references"][0]["path"] = str(self.root / "missing.jpg")
        self.write_jsonl([row])
        with self.assertRaises(POST.ClientError):
            POST.load_manifest(self.manifest, expected_count=1)
        row = self.client_row()
        row["smoke"]["generated_path"] = str(self.target)
        self.write_jsonl([row])
        with self.assertRaisesRegex(POST.ClientError, "overlap"):
            POST.load_manifest(self.manifest, expected_count=1)

    def test_video_soundtrack_plus_explicit_audio_is_rejected(self):
        row = self.client_row()
        row["smoke"]["references"] += [
            {"kind": "video", "path": str(self.video)},
            {"kind": "audio", "path": str(self.audio)},
        ]
        self.write_jsonl([row])
        with mock.patch.object(POST, "probe_media", return_value=[{"codec_type": "video"}, {"codec_type": "audio"}]):
            with self.assertRaisesRegex(POST.ClientError, "Audio N|soundtrack"):
                POST.load_manifest(self.manifest, expected_count=1)
        with mock.patch.object(POST, "probe_media", return_value=[{"codec_type": "video"}]):
            self.assertEqual(len(POST.load_manifest(self.manifest, expected_count=1)), 1)

    def test_rejects_silent_reordering_audio_only_and_excess_images(self):
        examples = [
            [{"kind": "audio", "path": str(self.audio)}],
            [{"kind": "audio", "path": str(self.audio)}, {"kind": "image", "path": str(self.image)}],
            [{"kind": "image", "path": str(self.image)} for _ in range(10)],
        ]
        for refs in examples:
            with self.subTest(refs=refs):
                row = self.client_row()
                row["smoke"]["references"] = refs
                self.write_jsonl([row])
                with self.assertRaises(POST.ClientError):
                    POST.load_manifest(self.manifest, expected_count=1)

    def test_dry_run_does_not_submit_or_create_state(self):
        self.write_jsonl([self.client_row()])
        with mock.patch.object(POST, "HttpClient") as client, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(POST.main(["--manifest", str(self.manifest), "--expected-count", "1", "--dry-run"]), 0)
        client.assert_not_called()
        self.assertFalse((self.root / ".post.state.json").exists())
        self.assertFalse((self.root / "results.jsonl").exists())

    def test_async_submission_polls_and_completed_resume_never_resubmits(self):
        row = self.client_row()
        self.write_jsonl([row])
        task_id = POST.stable_task_id(self.manifest, "case")
        client = mock.Mock()
        client.json_request.side_effect = [
            SERVICE_METADATA,
            {"task_id": task_id, "task_status": "pending"},
        ]
        client.task_status.side_effect = [None, {"task_id": task_id, "status": "processing"}, {"task_id": task_id, "status": "completed", "save_result_path": row["smoke"]["generated_path"]}]
        with mock.patch.object(POST, "HttpClient", return_value=client), mock.patch.object(POST, "wait_healthy"), mock.patch.object(POST, "verify_output"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(POST.run(self.args(), [row]), 0)
        submissions = [call for call in client.json_request.call_args_list if call.args[0] == "/v1/tasks/video/"]
        self.assertEqual(len(submissions), 1)
        self.assertEqual(json.loads((self.root / "results.jsonl").read_text())["status"], "completed")
        resumed_client = mock.Mock()
        with (
            mock.patch.object(POST, "HttpClient", return_value=resumed_client),
            mock.patch.object(POST, "wait_healthy"),
            mock.patch.object(POST, "verify_output"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(POST.run(self.args(), [row]), 0)
        resumed_client.task_status.assert_not_called()
        resumed_client.json_request.assert_not_called()

    def test_failed_task_is_recorded_and_returns_failure(self):
        row = self.client_row()
        self.write_jsonl([row])
        task_id = POST.stable_task_id(self.manifest, "case")
        client = mock.Mock()
        client.json_request.side_effect = [SERVICE_METADATA, {"task_id": task_id, "task_status": "pending"}]
        client.task_status.side_effect = [None, {"task_id": task_id, "status": "failed", "error": "mock inference failure", "error_type": "MockError"}]
        with mock.patch.object(POST, "HttpClient", return_value=client), mock.patch.object(POST, "wait_healthy"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(POST.run(self.args(), [row]), 1)
        result = json.loads((self.root / "results.jsonl").read_text())
        self.assertEqual(result["error_type"], "MockError")
        self.assertEqual(result["error"], "mock inference failure")

    def test_wrong_service_and_unverified_existing_task_never_submit(self):
        row = self.client_row()
        self.write_jsonl([row])
        for metadata, status in [
            ({"model_cls": "minimax_h3", "nproc_per_node": 1}, None),
            (SERVICE_METADATA, {"task_id": POST.stable_task_id(self.manifest, "case"), "status": "processing"}),
        ]:
            client = mock.Mock()
            client.json_request.return_value = metadata
            client.task_status.return_value = status
            with self.subTest(metadata=metadata), mock.patch.object(POST, "HttpClient", return_value=client), mock.patch.object(POST, "wait_healthy"), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(POST.ClientError):
                    POST.run(self.args(), [row])
            self.assertEqual([call.args[0] for call in client.json_request.call_args_list], ["/v1/service/metadata"])

    def test_untracked_existing_output_is_not_overwritten(self):
        row = self.client_row()
        output = self.make_file("generated/case.mp4", b"pre-existing user result")
        self.write_jsonl([row])
        with mock.patch.object(POST, "HttpClient") as client:
            with self.assertRaisesRegex(POST.ClientError, "overwrite"):
                POST.run(self.args(), [row])
        client.assert_not_called()
        self.assertEqual(output.read_bytes(), b"pre-existing user result")

    def test_submission_timeout_is_not_automatically_retried(self):
        row = self.client_row()
        self.write_jsonl([row])
        client = mock.Mock()
        client.json_request.side_effect = [SERVICE_METADATA, TimeoutError("mock timeout")]
        client.task_status.return_value = None
        with mock.patch.object(POST, "HttpClient", return_value=client), mock.patch.object(POST, "wait_healthy"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(POST.ClientError, "refusing automatic resubmission"):
                POST.run(self.args(), [row])
        saved = json.loads((self.root / ".post.state.json").read_text())
        self.assertEqual(saved["samples"]["case"]["status"], "submission_uncertain")
        client.reset_mock()
        client.json_request.side_effect = [SERVICE_METADATA]
        with mock.patch.object(POST, "HttpClient", return_value=client), mock.patch.object(POST, "wait_healthy"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(POST.ClientError, "Refusing automatic resubmission"):
                POST.run(self.args(), [row])
        self.assertEqual([call.args[0] for call in client.json_request.call_args_list], ["/v1/service/metadata"])

    def test_output_verification_checks_audio_geometry_fps_and_frame_count(self):
        row = self.client_row()
        output = self.make_file("generated/case.mp4", b"mock generated video")
        status = {"save_result_path": str(output)}
        streams = [{"codec_type": "video", "height": 768, "width": 1344, "avg_frame_rate": "24/1", "nb_frames": "124"}, {"codec_type": "audio"}]
        with mock.patch.object(POST, "probe_media", return_value=streams):
            POST.verify_output(row, status)
        for key, value in [("height", 544), ("avg_frame_rate", "25/1"), ("nb_frames", "123")]:
            broken = copy.deepcopy(streams)
            broken[0][key] = value
            with self.subTest(key=key), mock.patch.object(POST, "probe_media", return_value=broken):
                with self.assertRaises(POST.ClientError):
                    POST.verify_output(row, status)
        with mock.patch.object(POST, "probe_media", return_value=streams[:1]):
            with self.assertRaisesRegex(POST.ClientError, "audio"):
                POST.verify_output(row, status)


class SamplerContractTest(SmokeFixture):
    def source_row(self, index=0, provider="h3"):
        prefix = {"h3": "seko_minimax", "seedance": "seko_volc", "omni": "r2v"}[provider]
        return {
            "sample_id": f"{prefix}_{index}",
            "source_name": "omni" if provider == "omni" else prefix,
            "task": "ref2av",
            "ir_status": "succeeded",
            "enhanced_prompt": f"<Picture 1> 增强后的原始提示 {index}。",
            "duration": 29,
            "target_num_frames": 697,
            "target_height": 854,
            "target_width": 480,
            "references": [{"order": 1, "kind": "image", "path": self.image.name, "canonical_label": "<Picture 1>"}],
            "target": self.target.name,
        }

    def sampler_args(self, output="selection", seko=10, omni=10):
        return SAMPLE.parse_args(
            [
                "--input",
                str(self.manifest),
                "--output-dir",
                str(self.root / output),
                "--seko-count",
                str(seko),
                "--omni-count",
                str(omni),
                "--seed",
                "42",
                "--seko-media-root",
                str(self.root),
                "--omni-media-root",
                str(self.root),
            ]
        )

    def test_classification_uses_provenance_not_prompt_words(self):
        for provider in ("h3", "seedance", "omni"):
            row = self.source_row(provider=provider)
            self.assertEqual(SAMPLE.classify(row), ("omni" if provider == "omni" else "seko", provider))
        self.assertIsNone(SAMPLE.classify({"sample_id": "unrelated", "enhanced_prompt": "H3 Seedance Omni seko_minimax"}))

    def test_selects_ten_combined_seko_and_ten_omni_and_client_accepts(self):
        original = [self.source_row(index, provider) for provider, count in (("h3", 5), ("seedance", 5), ("omni", 10)) for index in range(count)]
        invalid = copy.deepcopy(original[0])
        invalid.update(sample_id="seko_minimax_missing_reference")
        invalid["references"][0]["path"] = "does-not-exist.jpg"
        raw = {"sample_id": "seko_volc_raw", "prompt_cn": "raw is not enhanced", "target": self.target.name}
        self.write_jsonl(original + [invalid, raw, original[0]])
        input_before = self.manifest.read_bytes()
        args = self.sampler_args()
        with mock.patch.object(SAMPLE, "ffprobe_video") as probe, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(SAMPLE.run(args), 0)
        probe.assert_not_called()
        selected_path = args.output_dir / "samples.jsonl"
        selected = POST.load_manifest(selected_path)
        self.assertEqual(Counter(row["smoke"]["group"] for row in selected), {"seko": 10, "omni": 10})
        self.assertEqual(Counter(row["smoke"]["provider"] for row in selected), {"h3": 5, "seedance": 5, "omni": 10})
        source_by_id = {row["sample_id"]: row for row in original}
        for row in selected:
            self.assertEqual(row["enhanced_prompt"], source_by_id[row["sample_id"]]["enhanced_prompt"])
            self.assertEqual(row["references"], source_by_id[row["sample_id"]]["references"])
            self.assertEqual(Path(row["smoke"]["target_copy"]).read_bytes(), self.target.read_bytes())
            self.assertFalse(Path(row["smoke"]["target_copy"]).is_symlink())
            self.assertEqual(POST.build_payload(row, selected_path)["num_frames"], 124)
            self.assertEqual(row["smoke"]["size"], [768, 1344])
        self.assertEqual(self.manifest.read_bytes(), input_before)
        with mock.patch.object(POST, "HttpClient") as client, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(POST.main(["--manifest", str(selected_path), "--dry-run"]), 0)
        client.assert_not_called()

    def test_same_seed_is_reproducible_and_manifest_is_not_overwritten(self):
        rows = [self.source_row(index, provider) for provider in ("h3", "seedance", "omni") for index in range(15)]
        self.write_jsonl(rows)
        args_a, args_b = self.sampler_args("first"), self.sampler_args("second")
        with contextlib.redirect_stdout(io.StringIO()):
            SAMPLE.run(args_a)
            SAMPLE.run(args_b)
        first = [json.loads(line) for line in (args_a.output_dir / "samples.jsonl").read_text().splitlines()]
        second = [json.loads(line) for line in (args_b.output_dir / "samples.jsonl").read_text().splitlines()]
        self.assertEqual([row["sample_id"] for row in first], [row["sample_id"] for row in second])
        snapshot = (args_a.output_dir / "samples.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            SAMPLE.run(args_a)
        self.assertEqual((args_a.output_dir / "samples.jsonl").read_bytes(), snapshot)

    def test_downloads_path_repair_is_recorded_without_changing_source(self):
        row = self.source_row()
        row["references"][0]["path"] = "downloads/image.jpg"
        row["target"] = "downloads/target.mp4"
        original = copy.deepcopy(row)
        refs, target, repairs = SAMPLE.validate_candidate(row, "seko", self.sampler_args(seko=1, omni=0))
        self.assertEqual(refs[0]["path"], str(self.image))
        self.assertEqual(target, self.target)
        self.assertEqual(len(repairs), 2)
        self.assertEqual(row, original)

    def test_incomplete_final_line_and_raw_rows_do_not_fill_quota(self):
        self.write_jsonl([self.source_row()])
        with self.manifest.open("ab") as stream:
            stream.write(json.dumps(self.source_row(1)).encode())
        args = self.sampler_args(seko=2, omni=0)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Not enough valid"):
            SAMPLE.run(args)
        self.assertFalse((args.output_dir / "samples.jsonl").exists())
        self.assertFalse((args.output_dir / "targets").exists())

    def test_reference_validation_rejects_bad_order_labels_and_limits(self):
        row = self.source_row()
        cases = []
        wrong_label = copy.deepcopy(row)
        wrong_label["references"][0]["canonical_label"] = "<Picture 2>"
        cases.append(wrong_label)
        wrong_order = copy.deepcopy(row)
        wrong_order["references"][0]["order"] = 2
        cases.append(wrong_order)
        too_many = copy.deepcopy(row)
        too_many["references"] = [{"kind": "image", "path": self.image.name} for _ in range(10)]
        cases.append(too_many)
        audio_only = copy.deepcopy(row)
        audio_only["references"] = [{"kind": "audio", "path": self.audio.name}]
        cases.append(audio_only)
        missing_target = copy.deepcopy(row)
        missing_target["target"] = "missing.mp4"
        cases.append(missing_target)
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises((ValueError, FileNotFoundError)):
                SAMPLE.validate_candidate(candidate, "seko", self.sampler_args())

    def test_audio_bearing_video_and_explicit_audio_are_not_silently_remapped(self):
        row = self.source_row()
        row["references"] += [{"order": 2, "kind": "video", "path": self.video.name}, {"order": 3, "kind": "audio", "path": self.audio.name}]
        with mock.patch.object(SAMPLE, "ffprobe_video", return_value=True):
            with self.assertRaisesRegex(ValueError, "audio_numbering"):
                SAMPLE.validate_candidate(row, "seko", self.sampler_args())
        with mock.patch.object(SAMPLE, "ffprobe_video", return_value=False):
            refs, _, _ = SAMPLE.validate_candidate(row, "seko", self.sampler_args())
        self.assertEqual([ref["canonical_label"] for ref in refs], ["<Picture 1>", "<Video 1>", "<Audio 1>"])

    def test_target_cannot_be_used_as_reference_before_selection(self):
        row = self.source_row()
        row["references"] = [{"order": 1, "kind": "video", "path": self.target.name}]
        with mock.patch.object(SAMPLE, "ffprobe_video", return_value=False):
            with self.assertRaisesRegex(ValueError, "target_conditioning"):
                SAMPLE.validate_candidate(row, "seko", self.sampler_args())
        self.write_jsonl([row])
        args = self.sampler_args(seko=1, omni=0)
        with mock.patch.object(SAMPLE, "ffprobe_video", return_value=False), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Not enough valid"):
                SAMPLE.run(args)
        self.assertFalse((args.output_dir / "samples.jsonl").exists())
        self.assertFalse((args.output_dir / "targets").exists())


if __name__ == "__main__":
    unittest.main()
