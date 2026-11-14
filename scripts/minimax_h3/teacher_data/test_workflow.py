"""CPU-only selection, request, resume, and launch contract tests."""

import argparse
import ast
import contextlib
import io
import json
import select
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch
from collect import api, completed, request_fingerprint, request_for, validate_export, wait_ready
from common import atomic_json, atomic_jsonl, digest, file_digest, file_identity, records
from launch import smoke_command
from prepare_selection import ordered_images, prepare


class TeacherWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw, self.cache, self.output = self.root / "raw.jsonl", self.root / "cache.jsonl", self.root / "output"
        self.image = self.root / "reference.jpg"
        self.image.write_bytes(b"fixture-image")
        self.raw_rows, self.cache_rows = [], []
        for count in range(2, 7):
            for item in range(6):
                index = len(self.raw_rows)
                source_id = f"source-{count}-{item}"
                self.raw_rows.append(
                    {
                        "sample_id": source_id,
                        "task": "ref2v",
                        "enhanced_prompt": f"  prompt {source_id}  ",
                        "target_height": 3840,
                        "target_width": 2160,
                        "ref_image_count": 99,
                        "references": [{"modality": "image", "rel_path": self.image.name, "order": i + 1} for i in range(count)],
                    }
                )
                self.cache_rows.append(
                    {
                        "source_index": index,
                        "source_id": source_id,
                        "task": "ref2av",
                        "target_height": 768,
                        "target_width": 1344,
                        "num_frames": 124,
                        "condition_path": f"conditions/{index}.pt",
                        "reference_image_count": count,
                        "ref_image_count": count,
                        "prompt_source": "enhanced_prompt",
                    }
                )
        self.write_fixture()

    def write_fixture(self):
        atomic_jsonl(self.raw, self.raw_rows)
        atomic_jsonl(self.cache, self.cache_rows)

    def test_local_http_ignores_environment_and_global_proxy_for_all_requests(self):
        local_requests, proxy_requests = [], []

        def start_server(requests):
            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.reply()

                def do_POST(self):
                    self.reply()

                def reply(self):
                    payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    requests.append((self.command, self.path, payload))
                    body = json.dumps({"status": "ok", "task_id": "local-task"}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            def close():
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.addCleanup(close)
            return server, f"http://127.0.0.1:{server.server_port}"

        server, local_url = start_server(local_requests)
        _, proxy_url = start_server(proxy_requests)
        with patch.dict("os.environ", {"http_proxy": proxy_url, "HTTP_PROXY": proxy_url, "all_proxy": proxy_url, "no_proxy": "", "NO_PROXY": ""}):
            proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy_url}))
            with patch.object(urllib.request, "_opener", proxy_opener):
                # Reproduce the old false-ready behavior: a proxy can return another server's health.
                with urllib.request.urlopen(local_url + "/health") as response:
                    self.assertEqual(json.load(response)["status"], "ok")
                self.assertEqual(len(proxy_requests), 1)
                proxy_requests.clear()

                wait_ready(local_url, timeout=1)
                self.assertEqual(api(local_url + "/v1/tasks/video/", {"prompt": "test"})["task_id"], "local-task")
                api(local_url + "/v1/tasks/local-task/status")
                self.assertEqual([row[:2] for row in local_requests], [("GET", "/health"), ("POST", "/v1/tasks/video/"), ("GET", "/v1/tasks/local-task/status")])
                self.assertEqual(json.loads(local_requests[1][2]), {"prompt": "test"})
                # With no local listener, the proxy must not make readiness succeed.
                server.shutdown()
                server.server_close()
                with patch("collect.time.sleep"), self.assertRaises(TimeoutError):
                    wait_ready(local_url, timeout=0.05)
                self.assertEqual(proxy_requests, [])

    def test_gpu_lock_contention_and_release_across_processes(self):
        worker = """
import sys
from launch import gpu_locks
try:
    with gpu_locks(sys.argv[2:], sys.argv[1]):
        print('locked', flush=True)
        if sys.stdin.readline().strip() == 'fail':
            raise ValueError('fixture failure')
except ValueError:
    pass
print('released', flush=True)
sys.stdin.readline()
"""
        command = [sys.executable, "-c", worker, str(self.root)]
        options = dict(cwd=Path(__file__).resolve().parent, text=True)

        def attempt(devices):
            return subprocess.run(command + devices, input="", capture_output=True, timeout=10, **options)

        def read_line(process):
            self.assertTrue(select.select([process.stdout], [], [], 10)[0], "Lock worker timed out")
            return process.stdout.readline().strip()

        for action in ("finish", "fail"):
            with self.subTest(action=action):
                with subprocess.Popen(command + ["C", "B"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options) as owner:
                    try:
                        self.assertEqual(read_line(owner), "locked")
                        # A is acquired before B conflicts; that partial reservation must release.
                        blocked = attempt(["B", "A"])
                        self.assertNotEqual(blocked.returncode, 0)
                        self.assertIn(f"launcher PID {owner.pid}", blocked.stderr)
                        self.assertEqual(attempt(["A"]).returncode, 0)
                        owner.stdin.write(action + "\n")
                        owner.stdin.flush()
                        self.assertEqual(read_line(owner), "released")
                        self.assertIsNone(owner.poll())
                        # Reacquire while the original process is still alive, including on failure.
                        released = attempt(["B", "C"])
                        self.assertEqual(released.returncode, 0, released.stderr)
                        self.assertTrue((self.root / "lightx2v-h3-teacher-gpu-B.lock").exists())
                    finally:
                        owner.communicate("\n", timeout=10)

    def test_duplicate_node_and_smoke_stop_before_preparation(self):
        for mode in ("node", "smoke"):
            with self.subTest(mode=mode), patch.object(sys, "argv", ["launch.py", mode]), patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}):
                with patch.object(launch, "gpu_locks", side_effect=RuntimeError("GPU busy")), patch.object(launch, "run") as run:
                    with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                        launch.main()
                    run.assert_not_called()

    def select(self, node=0):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(self.raw, self.cache, self.root, self.output, node, quota=4)

    def test_selection_balanced_stable_cached_geometry_and_no_stale_count(self):
        first, second = self.select(0), self.select(1)
        self.assertEqual(first, second)
        self.assertEqual(first["total_selected"], 20)
        for node in (0, 1):
            rows = [row for _, row in records(self.output / "selection" / f"node{node}" / f"node{node}.jsonl")]
            self.assertEqual(len(rows), 10)
            self.assertEqual({count: sum(row["ref_image_count"] == count for row in rows) for count in range(2, 7)}, dict.fromkeys(range(2, 7), 2))
            self.assertTrue(all((row["target_height"], row["target_width"]) == (768, 1344) for row in rows))
        self.assertEqual(self.select(0)["selection_sha256"], first["selection_sha256"])

    def test_missing_enhanced_prompt_and_mixed_modalities_cannot_fill_quota(self):
        for row in self.raw_rows[:3]:
            row["enhanced_prompt"] = " "
            row["prompt_en"] = "must not fall back"
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Insufficient eligible"):
            self.select()

    def test_exact_source_index_and_identity_join(self):
        for row in self.cache_rows[:3]:
            row["source_id"] = "wrong-id"
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Insufficient eligible"):
            self.select()

    def test_missing_selected_image_is_fatal(self):
        self.image.unlink()
        with self.assertRaises(FileNotFoundError):
            self.select()

    def test_wrong_cache_reference_count_is_fatal(self):
        self.cache_rows[0]["reference_image_count"] = 99
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "actual ordered references"):
            self.select()

    def test_node_selection_mismatch_is_fatal(self):
        self.select(0)
        self.raw_rows[0]["enhanced_prompt"] = "changed prompt"
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Nodes disagree"):
            self.select(1)

    def test_reference_order_aliases_fallback_and_blank_lines(self):
        row = {"actual_ordered_references": [{"kind": "reference_image", "path": "b.jpg", "order": 0}, {"kind": "image_url", "path": "a.jpg", "order": 1}]}
        self.assertEqual(ordered_images(row, self.root), [str(self.root / "b.jpg"), str(self.root / "a.jpg")])
        row["actual_ordered_references"].reverse()
        with self.assertRaisesRegex(ValueError, "refusing to reorder"):
            ordered_images(row, self.root)
        self.assertEqual(ordered_images({"reference_images": ["b.jpg", "a.jpg"]}, self.root), [str(self.root / "b.jpg"), str(self.root / "a.jpg")])
        with self.assertRaisesRegex(ValueError, "non-image"):
            ordered_images({"references": [{"modality": "video", "path": "a.mp4"}]}, self.root)
        self.raw.write_text('\n{"id": 1}\n\n{"id": 2}\n')
        self.assertEqual([index for index, _ in records(self.raw)], [0, 1])

    def example(self):
        self.select()
        return next(records(self.output / "selection/node0/node0.jsonl"))[1]

    def test_http_request_matches_actual_server_schema(self):
        # AST inspection uses the actual schema without importing GPU/platform dependencies.
        repo = Path(__file__).resolve().parents[3]
        tree = ast.parse((repo / "lightx2v/server/schema.py").read_text())
        fields = {
            statement.target.id
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name in {"BaseTaskRequest", "VideoTaskRequest"}
            for statement in cls.body
            if isinstance(statement, ast.AnnAssign)
        }
        payload = request_for(self.example(), self.root / "final_latents")
        self.assertFalse(set(payload) - fields)
        self.assertTrue(payload["teacher_latents_only"])
        self.assertNotIn("fps", payload)

    def write_export(self, row):
        directory = self.root / "export"
        directory.mkdir()
        manifest = {
            "schema": "h3_teacher_final_v1",
            "export_mode": "final",
            "task": "ref2av",
            "dtype": "float32",
            "normalized": True,
            "num_steps": 29,
            "num_states": 1,
            "final_state_index": 29,
            "final_video_sigma": 0.0,
            "final_audio_sigma": 0.0,
            "video_sigmas": [(29 - index) / 29 for index in range(30)],
            "audio_sigmas": [(29 - index) / 29 for index in range(30)],
            "num_frames": 124,
            "height": row["target_height"],
            "width": row["target_width"],
            "fps": 24,
            "seed": row["seed"],
            "prompt": row["prompt"],
            "source_id": row["source_id"],
            "image_paths": row["reference_images"],
            "scheduler_type": "reference_blend",
            "model_variant": "ref2av",
            "cache_fingerprint": "runtime-fingerprint",
            "state_paths": [],
            "condition_path": "condition.pt",
            "video_latents_path": "video_latents.pt",
            "audio_latents_path": "audio_latents.pt",
        }
        for name in ["condition.pt", "video_latents.pt", "audio_latents.pt"]:
            (directory / name).write_bytes(b"tensor-fixture")
        atomic_json(directory / "manifest.json", manifest)
        return directory, manifest

    def write_receipt(self, directory, manifest, fingerprint="request-A"):
        atomic_json(
            directory / "completion.json", {"request_fingerprint": fingerprint, "cache_fingerprint": manifest["cache_fingerprint"], "manifest_sha256": file_digest(directory / "manifest.json")}
        )

    def test_resume_requires_success_manifest_matching_fingerprint_and_three_final_files(self):
        row = self.example()
        directory, manifest = self.write_export(row)
        self.assertEqual({path.name for path in directory.iterdir()}, {"condition.pt", "video_latents.pt", "audio_latents.pt", "manifest.json"})
        self.assertEqual(validate_export(directory, row), manifest)
        self.assertIsNone(completed(directory, row, "request-A"))
        self.write_receipt(directory, manifest)
        self.assertEqual(completed(directory, row, "request-A"), manifest)
        self.assertIsNone(completed(directory, row, "request-B"))
        atomic_json(directory / "manifest.json", {**manifest, "cache_fingerprint": "changed"})
        self.assertIsNone(completed(directory, row, "request-A"))
        atomic_json(directory / "manifest.json", manifest)
        for name in ("condition.pt", "video_latents.pt", "audio_latents.pt"):
            with self.subTest(name=name):
                path = directory / name
                path.unlink()
                self.assertIsNone(completed(directory, row, "request-A"))
                path.touch()
                self.assertIsNone(completed(directory, row, "request-A"))
                path.write_bytes(b"tensor-fixture")
                self.assertEqual(completed(directory, row, "request-A"), manifest)

    def test_resume_rejects_invalid_final_export_metadata_and_stray_states(self):
        row = self.example()
        directory, manifest = self.write_export(row)
        for key, value in (
            ("schema", "h3_teacher_trajectory_v1"),
            ("export_mode", "trajectory"),
            ("export_mode", None),
            ("num_steps", 28),
            ("num_states", 30),
            ("num_states", True),
            ("state_paths", ["state_029.pt"]),
            ("state_paths", None),
            ("final_state_index", 28),
            ("final_video_sigma", 0.1),
            ("final_audio_sigma", 0.1),
            ("final_audio_sigma", False),
        ):
            with self.subTest(key=key, value=value):
                invalid = {**manifest, key: value}
                atomic_json(directory / "manifest.json", invalid)
                self.write_receipt(directory, invalid)
                self.assertIsNone(completed(directory, row, "request-A"))
        atomic_json(directory / "manifest.json", manifest)
        self.write_receipt(directory, manifest)
        (directory / "state_000.pt").write_bytes(b"obsolete-trajectory")
        self.assertIsNone(completed(directory, row, "request-A"))

    def test_resume_rejects_invalid_sigma_metadata(self):
        row = self.example()
        directory, manifest = self.write_export(row)
        for key in ("video_sigmas", "audio_sigmas"):
            schedule = manifest[key]
            for value in (None, schedule[:-1], [*schedule[:-1], 0.01], [1.0] * 29 + [0.0], [float("nan"), *schedule[1:]], [True, *schedule[1:]]):
                with self.subTest(key=key, value=value):
                    invalid = {**manifest, key: value}
                    atomic_json(directory / "manifest.json", invalid)
                    self.write_receipt(directory, invalid)
                    self.assertIsNone(completed(directory, row, "request-A"))

    def test_request_fingerprint_separates_final_exports_from_old_trajectory_runs(self):
        row = self.example()
        config, model, selection_hash = {"infer_steps": 29}, {"model_path": "fixture"}, "selection"
        old_fingerprint = digest(
            {"schema": "h3_teacher_request_v1", "sample": row, "config": config, "model": model, "selection_sha256": selection_hash, "images": [file_identity(path) for path in row["reference_images"]]}
        )
        fingerprint = request_fingerprint(row, config, model, selection_hash)
        self.assertNotEqual(fingerprint, old_fingerprint)
        directory, manifest = self.write_export(row)
        self.write_receipt(directory, manifest, old_fingerprint)
        self.assertIsNone(completed(directory, row, fingerprint))

    def test_resume_rejects_old_trajectory_receipt_even_with_matching_fingerprint(self):
        row = self.example()
        directory, manifest = self.write_export(row)
        manifest.update(schema="h3_teacher_trajectory_v1", num_states=30, state_paths=[f"state_{index:03d}.pt" for index in range(30)])
        for key in ("export_mode", "final_state_index", "final_video_sigma", "final_audio_sigma"):
            manifest.pop(key)
        for name in manifest["state_paths"]:
            (directory / name).write_bytes(b"old-state")
        atomic_json(directory / "manifest.json", manifest)
        self.write_receipt(directory, manifest)
        self.assertIsNone(completed(directory, row, "request-A"))

    def test_smoke_is_offline_eight_gpus_with_decode_and_source_identity(self):
        row = self.example()
        args = types.SimpleNamespace(model_path="merged", config_json="teacher.json", no_export=False)
        command = smoke_command(row, args, self.root / "final_latents", self.root / "sample.mp4")
        self.assertIn("--standalone", command)
        self.assertIn("--nproc_per_node=8", command)
        self.assertIn("lightx2v.infer", command)
        self.assertNotIn("--teacher_latents_only", command)
        self.assertNotIn("--no-teacher_latents_only", command)
        self.assertEqual(command[command.index("--teacher_sample_id") + 1], row["source_id"])
        self.assertNotIn("lightx2v.server", command)
        tree = ast.parse((Path(__file__).resolve().parents[3] / "lightx2v/infer.py").read_text())
        parse_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_parser")
        namespace = {"argparse": argparse, "json": json, "RUNNER_MODULES": {"minimax_h3": None}}
        exec(compile(ast.Module(body=[parse_function], type_ignores=[]), "infer.py", "exec"), namespace)
        from unittest.mock import patch

        inference_args = command[command.index("lightx2v.infer") + 1 :]
        with patch.object(sys, "argv", ["lightx2v.infer", *inference_args]):
            parsed = namespace["build_parser"]().parse_args()
        self.assertFalse(parsed.teacher_latents_only)
        self.assertEqual(parsed.teacher_sample_id, row["source_id"])


if __name__ == "__main__":
    unittest.main()
