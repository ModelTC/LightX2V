"""Real subprocess consumer dispatch; no model/API requests."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.exploration_stop_hook import hook


class StopHookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.entry = self.root / "consumer.py"
        self.entry.write_text('import sys\nfrom pathlib import Path\nPath(__file__).with_suffix(".called").write_text("verified")\nsys.exit(0)\n')
        self.policy = self.root / "policy.json"
        self.policy.write_text("{}")
        self.work = self.root / "work.json"
        self.work.write_text(json.dumps({"task_id": "task", "status": "completed", "next_actions": []}))
        self.binding = self.root / "binding.json"
        self.write_binding()

    def sha(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write_binding(self):
        self.binding.write_text(
            json.dumps(
                {
                    "kind": "exploration_consumer_binding_v1",
                    "task_id": "task",
                    "entry": str(self.entry),
                    "entry_sha256": self.sha(self.entry),
                    "checkers": {str(self.entry): self.sha(self.entry)},
                    "policy": str(self.policy),
                    "policy_sha256": self.sha(self.policy),
                    "work_package": str(self.work),
                    "receipt": str(self.root / "receipt.json"),
                    "parent_completion_allowed": True,
                }
            )
        )
        self.expected = self.sha(self.binding)

    def call(self, event="Stop"):
        return hook({"hook_event_name": event}, self.binding, self.expected)

    def test_valid_completion_dispatches_actual_verifier(self):
        self.assertEqual(self.call(), {})
        self.assertTrue(self.entry.with_suffix(".called").exists())

    def test_failed_consumer_creates_actionable_continuation(self):
        self.entry.write_text('raise SystemExit("missing real evidence")\n')
        self.write_binding()
        result = self.call()
        self.assertEqual(result["decision"], "block")
        self.assertIn("missing real evidence", result["reason"])

    def test_deleted_config_or_replaced_checker_never_falls_back(self):
        self.binding.unlink()
        self.assertEqual(self.call()["decision"], "block")
        self.write_binding()
        self.entry.write_text("raise SystemExit(0)\n")
        self.assertEqual(self.call()["decision"], "block")
        self.assertFalse(self.entry.with_suffix(".called").exists())

    def test_forged_completed_does_not_override_failure(self):
        self.entry.write_text("raise SystemExit(1)\n")
        self.write_binding()
        self.assertEqual(self.call()["decision"], "block")

    def test_partial_scope_and_pending_action_cannot_stop_parent(self):
        binding = json.loads(self.binding.read_text())
        binding["parent_completion_allowed"] = False
        self.binding.write_text(json.dumps(binding))
        self.expected = self.sha(self.binding)
        self.assertEqual(self.call()["decision"], "block")
        self.write_binding()
        self.work.write_text(json.dumps({"task_id": "task", "status": "in_progress", "next_actions": [{"id": "repair", "execution_state": "pending"}]}))
        self.assertIn("repair", self.call()["reason"])

    def test_resume_restores_failure_context_and_interrupt_never_restarts(self):
        self.binding.unlink()
        result = self.call("SessionStart")
        self.assertIn("additionalContext", result["hookSpecificOutput"])
        self.assertEqual(self.call("Interrupt"), {})

    def test_incomplete_stop_preserves_status_without_granting_completion(self):
        self.entry.write_text('raise SystemExit("completion is not accepted")\n')
        self.write_binding()
        for status in ("blocked", "paused", "budget_exhausted"):
            with self.subTest(status=status):
                work = {"task_id": "task", "status": status, "stop_reason": "External resource unavailable or explicit user limit.", "next_actions": []}
                self.work.write_text(json.dumps(work))
                before = self.work.read_bytes()
                self.assertEqual(self.call(), {})
                self.assertEqual(self.work.read_bytes(), before)
                self.assertFalse(self.entry.with_suffix(".called").exists())
                self.assertIn("incomplete", self.call("SessionStart")["hookSpecificOutput"]["additionalContext"])
        # A blocked branch cannot stop independent executable work.
        work["status"] = "blocked"
        work["next_actions"] = [{"id": "independent", "execution_state": "pending"}]
        self.work.write_text(json.dumps(work))
        self.assertEqual(self.call()["decision"], "block")
        work["next_actions"] = [{"id": "independent", "execution_state": "blocked", "reason": "Needs unavailable GPU."}]
        self.work.write_text(json.dumps(work))
        self.assertEqual(self.call(), {})
        del work["stop_reason"]
        self.work.write_text(json.dumps(work))
        self.assertEqual(self.call()["decision"], "block")

    def test_unconsumed_auxiliary_actions_do_not_change_stop_contract(self):
        called = self.entry.with_suffix(".called")
        for status in ("paused", "budget_exhausted", "completed", "passed", "passed_with_scope"):
            with self.subTest(status=status):
                called.unlink(missing_ok=True)
                work = {"task_id": "task", "status": status, "stop_reason": "Explicit user limit", "next_actions": [{"id": "auxiliary_note", "note": "Not a required completion action"}]}
                self.work.write_text(json.dumps(work))
                before = self.work.read_bytes()
                self.assertEqual(self.call(), {})
                self.assertEqual(self.work.read_bytes(), before)
                self.assertEqual(called.exists(), status in {"completed", "passed", "passed_with_scope"})

    def test_malformed_state_returns_protocol_rejection_instead_of_crashing(self):
        baseline_work = {"task_id": "task", "status": "blocked", "stop_reason": "GPU unavailable", "next_actions": []}
        variants = [
            {"next_actions": [{"execution_state": "blocked", "reason": None}]},
            {"next_actions": [None]},
            {"next_actions": None},
            {"next_actions": [{"execution_state": []}]},
            {"status": []},
        ]
        baseline_binding = json.loads(self.binding.read_text())
        cases = [(dict(baseline_work, **changes), baseline_binding) for changes in variants]
        cases += [(baseline_work, dict(baseline_binding, **changes)) for changes in ({"checkers": ["invalid"]}, {"entry": None})]
        cases.append((dict(baseline_work, status="completed"), dict(baseline_binding, receipt=[])))
        for work, binding in cases:
            with self.subTest(work=work, binding=binding):
                self.work.write_text(json.dumps(work))
                self.binding.write_text(json.dumps(binding))
                self.expected = self.sha(self.binding)
                for event in ("Stop", "SessionStart"):
                    result = subprocess.run(
                        [sys.executable, "-m", "tools.exploration_stop_hook", "--binding", str(self.binding), "--binding-sha256", self.expected],
                        input=json.dumps({"hook_event_name": event}),
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    response = json.loads(result.stdout)
                    if event == "Stop":
                        self.assertEqual(response["decision"], "block")
                    else:
                        self.assertIn("additionalContext", response["hookSpecificOutput"])
                    self.assertEqual(result.stderr, "")
                self.assertFalse(self.entry.with_suffix(".called").exists())


if __name__ == "__main__":
    unittest.main()
