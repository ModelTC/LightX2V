"""CPU action/closure and dependency-propagation contracts, not AI-quality tests."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_exploration_acceptance import fixture, write_json

from tools import exploration_acceptance as acceptance
from tools import exploration_lifecycle as lifecycle


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.policy, self.work = fixture(self.root)
        self.context = self.root / "model.py"
        self.marker = self.root / "expensive-started"
        action = {
            "id": "candidate-action",
            "validation_scope": "request",
            "command": [
                sys.executable,
                "-c",
                f"import os\nfrom pathlib import Path\nassert Path.cwd() == Path({str(self.root)!r})\n"
                f'assert os.environ.get("EXPLORATION_TEST_CONTEXT") == "reviewed"\nPath({str(self.marker)!r}).write_text("started")',
            ],
            "cwd": str(self.root),
            "environment": {"EXPLORATION_TEST_CONTEXT": "reviewed"},
            "subject": {"model": "fixture", "candidate": "candidate"},
            "inputs": [str(self.context)],
            "expected_cost": {"basis": "existing complete block negative result", "seconds": 60},
            "execution_state": "pending",
        }
        self.work["next_actions"].append(action)
        execution = self.root / "review-execution.json"
        write_json(
            execution, {"kind": "exploration_review_execution_v1", "reviewer": "independent-CPU-contract-fixture", "scope_id": self.policy["scope_id"], "commands": [["read", str(self.context)]]}
        )
        self.review = {
            "kind": "exploration_semantic_review_v1",
            "task_id": self.policy["task_id"],
            "scope_id": self.policy["scope_id"],
            "inventory_sha256": lifecycle.inventory_digest(self.policy),
            "reviewer": "independent-CPU-contract-fixture",
            "execution_receipt": str(execution),
            "verdict": "accepted",
            "observed_inputs": {str(self.context): acceptance.digest(self.context)},
            "topics": {key: {"assessment": "Actual source and cost support the scoped choice.", "inputs": [str(self.context)]} for key in lifecycle.REVIEW_TOPICS},
            "findings": [],
            "item_decisions": {key: lifecycle.decision_digest(value) for key, value in self.work["exploration_fulfillment"]["items"].items()},
            "actions": {
                "candidate-action": {
                    "decision": "reject",
                    "scope": "request",
                    "action_sha256": lifecycle.action_digest(action),
                    "rationale": "The local negative already answers this question; inspect source rather than run a request.",
                    "inputs": [str(self.context)],
                }
            },
        }
        self.review_path = self.root / "review.json"
        write_json(self.review_path, self.review)
        self.policy["lifecycle"] = {
            "review": {"path": str(self.review_path)},
            "dependencies": [{"path": str(self.context), "kind": "source", "items": ["compute"], "candidates": ["base"]}],
            "candidates": {"base": {"parents": [], "items": ["compute"]}, "combo": {"parents": ["base"], "items": ["vision"]}, "unrelated": {"parents": [], "items": ["later"]}},
        }
        self.policy["artifacts"].update({str(self.review_path): acceptance.digest(self.review_path), str(execution): acceptance.digest(execution)})
        self.review["inventory_sha256"] = lifecycle.inventory_digest(self.policy)
        self.resign_review()

    def save(self):
        write_json(self.root / "policy.json", self.policy)

    def inputs(self):
        inputs = acceptance.Inputs()
        inputs.read(Path(lifecycle.__file__).resolve())
        acceptance.load_policy(inputs, self.root / "policy.json", acceptance.digest(self.root / "policy.json"))
        return inputs

    def resign_review(self):
        write_json(self.review_path, self.review)
        self.policy["artifacts"][str(self.review_path)] = acceptance.digest(self.review_path)
        self.save()

    def scoped_action(self):
        other = self.root / "later-only.py"
        other.write_text("# initial later path\n")
        self.policy["lifecycle"]["dependencies"].append({"kind": "source", "path": str(other), "items": ["later"]})
        self.policy["artifacts"][str(other)] = acceptance.digest(other)
        self.review["observed_inputs"][str(other)] = acceptance.digest(other)
        self.review["input_items"] = {str(self.context): ["compute", "vision"], str(other): ["later"]}
        self.review["actions"]["candidate-action"].update(decision="allow", items=["compute"])
        self.review["inventory_sha256"] = lifecycle.inventory_digest(self.policy)
        self.resign_review()
        return other

    def test_review_scoped_action_runs_while_unrelated_changed_branch_stays_unaccepted(self):
        other = self.scoped_action()
        other.write_text("# changed only later path\n")
        action = self.work["next_actions"][-1]
        lifecycle.authorize_action(self.policy, self.work, action["id"], self.inputs())
        subprocess.run(action["command"], cwd=action["cwd"], env={**os.environ, **action["environment"]}, check=True)
        self.assertTrue(self.marker.exists())
        with self.assertRaisesRegex(ValueError, "pinned input changed"):
            lifecycle.check_review(self.policy, self.inputs(), work_package=self.work)

    def test_action_cannot_opt_itself_into_narrower_freshness(self):
        other = self.scoped_action()
        del self.review["actions"]["candidate-action"]["items"]
        self.work["next_actions"][-1]["items"] = ["compute"]  # Mutable activity has no authority.
        self.resign_review()
        other.write_text("# changed only later path\n")
        with self.assertRaisesRegex(ValueError, "pinned input changed"):
            lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())

    def test_scoped_action_keeps_all_actual_basis_inputs_even_if_owned_by_other_items(self):
        other = self.scoped_action()
        action = self.work["next_actions"][-1]
        action["inputs"].append(str(other))
        decision = self.review["actions"][action["id"]]
        decision["inputs"].append(str(other))
        decision["action_sha256"] = lifecycle.action_digest(action)
        self.resign_review()
        lifecycle.authorize_action(self.policy, self.work, action["id"], self.inputs())
        other.write_text("# changed input used by the scoped command\n")
        with self.assertRaisesRegex(ValueError, "pinned input changed"):
            lifecycle.authorize_action(self.policy, self.work, action["id"], self.inputs())

    def test_scoped_action_keeps_unowned_global_cost_basis(self):
        self.scoped_action()
        cost = self.root / "global-cost.json"
        write_json(cost, {"available_budget": 60})
        self.review["observed_inputs"][str(cost)] = acceptance.digest(cost)
        self.review["topics"]["priority"]["inputs"].append(str(cost))
        self.policy["artifacts"][str(cost)] = acceptance.digest(cost)
        self.policy["lifecycle"]["dependencies"].append({"kind": "cost", "path": str(cost)})
        self.review["inventory_sha256"] = lifecycle.inventory_digest(self.policy)
        self.resign_review()
        write_json(cost, {"available_budget": 1})
        with self.assertRaisesRegex(ValueError, "pinned input changed"):
            lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())

    def test_input_item_label_without_fixed_dependency_cannot_remove_freshness(self):
        other = self.scoped_action()
        self.policy["lifecycle"]["dependencies"].pop()  # input_items alone must not establish locality.
        self.review["inventory_sha256"] = lifecycle.inventory_digest(self.policy)
        self.resign_review()
        other.write_text("# changed input without a fixed dependency scope\n")
        with self.assertRaisesRegex(ValueError, "pinned input changed"):
            lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())

    def test_global_input_mapped_to_all_items_preserves_partial_delivery_and_staleness(self):
        from test_exploration_acceptance import collect

        self.scoped_action()
        original_policy, original_review = copy.deepcopy(self.policy), copy.deepcopy(self.review)
        global_path = self.root / "global-basis.json"

        def partial():
            inputs = self.inputs()
            facts = collect(inputs, self.policy)
            return acceptance.check(self.policy, acceptance.catalog_families((self.root / "MAP.md").read_text()), self.work, facts, inputs, mode="delivery", selected=["compute"])

        for kind in ("report", "checker", "catalog", "cost"):
            with self.subTest(kind=kind):
                self.policy, self.review = copy.deepcopy(original_policy), copy.deepcopy(original_review)
                write_json(global_path, {"value": 1})
                self.policy["lifecycle"]["dependencies"].append({"kind": kind, "path": str(global_path)})
                self.policy["artifacts"][str(global_path)] = acceptance.digest(global_path)
                self.review["observed_inputs"][str(global_path)] = acceptance.digest(global_path)
                self.review["input_items"][str(global_path)] = list(self.policy["items"])
                self.review["inventory_sha256"] = lifecycle.inventory_digest(self.policy)
                self.resign_review()
                result = partial()
                self.assertTrue(result["passed"], result["errors"])
                self.assertFalse(result["task_complete"])
                lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())
                write_json(global_path, {"value": 2})
                with self.assertRaisesRegex(ValueError, "pinned input changed"):
                    partial()
                with self.assertRaisesRegex(ValueError, "pinned input changed"):
                    lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())
                write_json(global_path, {"value": 1})
                self.review["input_items"][str(global_path)] = ["later"]
                self.resign_review()
                self.assertIn("review_inputs", {error["code"] for error in partial()["errors"]})

    def test_scope_must_be_registered_and_input_ownership_cover_candidate_closure(self):
        self.scoped_action()
        for items in ([], ["unknown"]):
            self.review["actions"]["candidate-action"]["items"] = items
            self.resign_review()
            with self.subTest(items=items), self.assertRaisesRegex(ValueError, "registered review items"):
                lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())
        self.review["actions"]["candidate-action"]["items"] = ["compute"]
        self.review["input_items"][str(self.context)] = ["compute"]  # Fixed dependency also reaches vision through combo.
        self.resign_review()
        with self.assertRaisesRegex(ValueError, "ownership omits"):
            lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())

    def test_review_binding_cannot_be_replaced_by_passed_label(self):
        self.review["scope_id"] = "other"
        self.resign_review()
        _, errors = lifecycle.check_review(self.policy, self.inputs(), work_package=self.work)
        self.assertIn("review_scope", {x["code"] for x in errors})

    def test_hidden_scope_item_or_changed_method_map_invalidates_review(self):
        self.policy["items"]["new-variant"] = {**self.policy["items"]["compute"], "label": "additional real branch"}
        self.save()
        _, errors = lifecycle.check_review(self.policy, self.inputs(), work_package=self.work)
        self.assertIn("review_scope", {x["code"] for x in errors})

    def test_required_finding_needs_reviewed_resolution_and_real_input(self):
        self.policy["lifecycle"]["required_finding_ids"] = ["missed-variant"]
        self.review["findings"] = [{"id": "missed-variant", "items": ["compute"], "disposition": "open"}]
        self.resign_review()
        _, errors = lifecycle.check_review(self.policy, self.inputs(), work_package=self.work)
        self.assertIn("review_unclosed", {x["code"] for x in errors})
        self.review["findings"][0]["required"] = False  # An obsolete row flag has no authority.
        self.resign_review()
        _, errors = lifecycle.check_review(self.policy, self.inputs(), work_package=self.work)
        self.assertIn("review_unclosed", {x["code"] for x in errors})
        del self.review["findings"][0]["required"]
        self.review["findings"][0].update(disposition="fixed", resolution="Actual branch examined and added.", resolution_inputs=[str(self.context)])
        self.resign_review()
        _, errors = lifecycle.check_review(self.policy, self.inputs(), work_package=self.work)
        self.assertEqual(errors, [])

    def test_execution_evidence_is_pinned_before_any_costly_action(self):
        self.review["actions"]["candidate-action"]["decision"] = "allow"
        self.resign_review()
        lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())
        path = Path(self.review["execution_receipt"])
        execution = json.loads(path.read_text())
        execution["commands"].append(["different-review-execution"])
        write_json(path, execution)
        action = self.work["next_actions"][-1]
        with self.assertRaisesRegex(ValueError, "pinned input changed"):
            lifecycle.authorize_action(self.policy, self.work, action["id"], self.inputs())
            subprocess.run(action["command"], cwd=action["cwd"], env={**os.environ, **action["environment"]}, check=True)
        self.assertFalse(self.marker.exists())

    def test_action_basis_uses_the_same_fixed_artifacts_as_completion(self):
        note = self.root / "cost-note.json"
        write_json(note, {"measured_cost_ms": 3})
        self.review["observed_inputs"][str(note)] = acceptance.digest(note)
        self.review["topics"]["priority"]["inputs"].append(str(note))
        self.policy["artifacts"][str(note)] = acceptance.digest(note)
        self.review["actions"]["candidate-action"]["decision"] = "allow"
        self.resign_review()
        lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())
        # A review can bind current bytes while the consumer still expects a
        # different registered revision. Both entrypoints must reject that gap.
        self.policy["artifacts"][str(note)] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "artifact_identity"):
            lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())
        self.assertFalse(self.marker.exists())
        del self.policy["artifacts"][str(note)]
        self.save()
        with self.assertRaisesRegex(ValueError, "unregistered_source"):
            lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())

    def test_dependency_version_in_artifacts_invalidates_old_review_inventory(self):
        self.context.write_text("# new dependency revision\n")
        self.policy["artifacts"][str(self.context)] = acceptance.digest(self.context)
        self.review["observed_inputs"][str(self.context)] = acceptance.digest(self.context)
        self.resign_review()
        _, errors = lifecycle.check_review(self.policy, self.inputs(), work_package=self.work)
        self.assertIn("review_scope", {row["code"] for row in errors})

    def test_new_blocking_finding_must_be_registered_and_partial_delivery_is_scoped(self):
        self.review["findings"] = [{"id": "later-blocker", "items": ["later"], "disposition": "open"}]
        self.resign_review()
        _, errors = lifecycle.check_review(self.policy, self.inputs(), purpose="action")
        self.assertIn("review_findings", {row["code"] for row in errors})
        self.policy["lifecycle"]["required_finding_ids"] = ["later-blocker"]
        self.review["verdict"] = "changes_required"
        self.resign_review()
        _, errors = lifecycle.check_review(self.policy, self.inputs(), selected=["compute"], work_package=self.work)
        self.assertEqual(errors, [])
        _, errors = lifecycle.check_review(self.policy, self.inputs(), work_package=self.work)
        self.assertIn("review_unclosed", {row["code"] for row in errors})

    def test_corrective_action_does_not_wait_for_its_own_finding_to_close(self):
        self.review["verdict"] = "changes_required"
        self.policy["lifecycle"]["required_finding_ids"] = ["needs-input"]
        self.review["findings"] = [{"id": "needs-input", "items": ["compute"], "disposition": "open"}]
        self.review["actions"]["candidate-action"]["decision"] = "allow"
        self.resign_review()
        lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())
        action = self.work["next_actions"][-1]
        subprocess.run(action["command"], cwd=action["cwd"], env={**os.environ, **action["environment"]}, check=True)
        self.assertTrue(self.marker.exists())
        _, errors = lifecycle.check_review(self.policy, self.inputs(), work_package=self.work)
        self.assertIn("review_unclosed", {row["code"] for row in errors})

    def test_unrelated_dependency_does_not_invalidate_partial_delivery(self):
        from test_exploration_acceptance import collect

        other = self.root / "later-only.py"
        other.write_text("# initial later path")
        self.policy["lifecycle"]["dependencies"].append({"kind": "source", "path": str(other), "items": ["later"]})
        self.policy["artifacts"][str(other)] = acceptance.digest(other)
        self.review["observed_inputs"][str(other)] = acceptance.digest(other)
        self.review["input_items"] = {str(other): ["later"]}
        self.review["inventory_sha256"] = lifecycle.inventory_digest(self.policy)
        self.resign_review()
        other.write_text("# changed only later path")
        inputs = self.inputs()
        facts = collect(inputs, self.policy)
        result = acceptance.check(self.policy, acceptance.catalog_families((self.root / "MAP.md").read_text()), self.work, facts, inputs, mode="delivery", selected=["compute"])
        self.assertTrue(result["passed"], result["errors"])
        self.assertFalse(result["task_complete"])

    def test_parent_approval_change_reopens_descendants_and_preserves_history(self):
        self.work["exploration_fulfillment"]["candidate_approvals"] = {"base": {"status": "rejected"}, "combo": {"status": "approved"}}
        self.policy["lifecycle"]["dependencies"][0]["kind"] = "approval"
        self.context.write_text("# actual parent decision changed")
        impact = lifecycle.assess_changes(self.policy, acceptance.Inputs())
        lifecycle.apply_changes(self.work, impact)
        self.assertEqual(self.work["exploration_fulfillment"]["candidate_approvals"]["combo"]["status"], "needs_review")
        self.assertEqual(self.work["exploration_fulfillment"]["candidate_approvals"]["combo"]["historical_decision"], {"status": "approved"})
        self.assertEqual(self.work["exploration_fulfillment"]["items"]["compute"]["disposition"], "retain")

    def test_registered_source_and_reconfirmed_labels_cannot_grant_approval(self):
        from test_exploration_acceptance import collect

        output = {"sha256": "fixture-output-pinned-by-consumer"}
        self.policy["lifecycle"]["candidates"]["combo"]["output_identity"] = output
        parent = {"status": "rejected"}
        parent_identity = hashlib.sha256(json.dumps(parent, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        self.work["exploration_fulfillment"]["candidate_approvals"] = {
            "base": parent,
            "combo": {"status": "approved", "output_identity": output, "decision_evidence": str(self.context), "reconfirmed_dependencies": {"base": parent_identity}},
        }
        self.review["inventory_sha256"] = lifecycle.inventory_digest(self.policy)
        self.resign_review()
        inputs = self.inputs()
        result = acceptance.check(self.policy, acceptance.catalog_families((self.root / "MAP.md").read_text()), self.work, collect(inputs, self.policy), inputs)
        self.assertFalse(result["passed"])
        self.assertIn("approval_unsupported", {row["code"] for row in result["errors"]})

    def test_research_delivery_accepts_pending_rejected_and_reopened_quality(self):
        from test_exploration_acceptance import collect

        for status in ["research", "pending_review", "rejected", "needs_review"]:
            with self.subTest(status=status):
                self.work["exploration_fulfillment"]["candidate_approvals"] = {"base": {"status": status}}
                inputs = self.inputs()
                result = acceptance.check(self.policy, acceptance.catalog_families((self.root / "MAP.md").read_text()), self.work, collect(inputs, self.policy), inputs)
                self.assertTrue(result["passed"], result["errors"])

    def test_disposition_basis_cannot_change_under_an_old_semantic_verdict(self):
        self.work["exploration_fulfillment"]["items"]["later"]["rationale"] = "Uninvestigated cost is assumed zero."
        _, errors = lifecycle.check_review(self.policy, self.inputs(), work_package=self.work)
        self.assertIn("review_decision", {row["code"] for row in errors})

    def test_negative_scope_rejects_actual_costly_action_before_subprocess(self):
        action = self.work["next_actions"][-1]
        with self.assertRaisesRegex(ValueError, "action_choice"):
            lifecycle.authorize_action(self.policy, self.work, action["id"], self.inputs())
            subprocess.run(action["command"], cwd=action["cwd"], env={**os.environ, **action["environment"]}, check=True)
        self.assertFalse(self.marker.exists())

    def test_allowed_scoped_action_executes_but_command_swap_is_rejected(self):
        self.review["actions"]["candidate-action"]["decision"] = "allow"
        self.resign_review()
        lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())
        action = self.work["next_actions"][-1]
        subprocess.run(action["command"], cwd=action["cwd"], env={**os.environ, **action["environment"]}, check=True)
        self.assertTrue(self.marker.exists())
        self.marker.unlink()
        self.work["next_actions"][-1]["command"][-1] += '; print("changed action")'
        with self.assertRaisesRegex(ValueError, "action_choice"):
            lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())
        self.assertFalse(self.marker.exists())

    def test_action_inputs_need_reviewed_content_and_task_scope_binding(self):
        request = self.root / "request.json"
        request.write_text('{"frames": 1}\n')
        action = self.work["next_actions"][-1]
        action["cwd"] = str(self.root)
        action["inputs"].append("request.json")
        decision = self.review["actions"]["candidate-action"]
        decision.update(decision="allow", action_sha256=lifecycle.action_digest(action))
        self.resign_review()
        with self.assertRaisesRegex(ValueError, "action_inputs"):
            lifecycle.authorize_action(self.policy, self.work, action["id"], self.inputs())
        self.review["observed_inputs"][str(request)] = acceptance.digest(request)
        self.policy["artifacts"][str(request)] = acceptance.digest(request)
        decision["inputs"].append(str(request))
        self.resign_review()
        lifecycle.authorize_action(self.policy, self.work, action["id"], self.inputs())
        request.write_text('{"frames": 999999}\n')
        with self.assertRaisesRegex(ValueError, "pinned input changed"):
            lifecycle.authorize_action(self.policy, self.work, action["id"], self.inputs())

    def test_action_cannot_use_foreign_work_package(self):
        self.review["actions"]["candidate-action"]["decision"] = "allow"
        self.resign_review()
        for target in (self.work, self.work["exploration_fulfillment"]):
            field = "task_id" if target is self.work else "scope_id"
            old = target[field]
            target[field] = "another-activity"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "another task/scope"):
                lifecycle.authorize_action(self.policy, self.work, "candidate-action", self.inputs())
            target[field] = old

    def test_action_requires_resolved_cwd_and_declared_environment(self):
        original = self.work["next_actions"][-1]
        for key, value in [("cwd", None), ("cwd", "relative"), ("environment", None), ("environment", {"OMP_NUM_THREADS": 2})]:
            action = copy.deepcopy(original)
            action[key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "absolute cwd, declared environment"):
                lifecycle.action_digest(action)

    def test_changed_source_reopens_descendants_not_unrelated_or_old_raw(self):
        self.context.write_text("# changed implementation\n")
        before = copy.deepcopy(self.work)
        change = lifecycle.assess_changes(self.policy, acceptance.Inputs())
        self.assertTrue(change["changed"])
        self.assertEqual(change["changes"][0]["items"], ["compute", "vision"])
        self.assertEqual(self.work, before)
        lifecycle.apply_changes(self.work, change)
        self.assertEqual(self.work["exploration_fulfillment"]["items"]["compute"]["disposition"], "open")
        self.assertEqual(self.work["exploration_fulfillment"]["items"]["later"], before["exploration_fulfillment"]["items"]["later"])
        self.assertEqual(self.work["exploration_fulfillment"]["items"]["compute"]["evidence"], before["exploration_fulfillment"]["items"]["compute"]["evidence"])
        after = copy.deepcopy(self.work)
        lifecycle.apply_changes(self.work, change)
        self.assertEqual(self.work, after)

    def test_approval_change_reopens_adoption_and_preserves_measurement_disposition(self):
        dependency = self.policy["lifecycle"]["dependencies"][0]
        dependency["kind"] = "approval"
        self.context.write_text("# parent rejected\n")
        change = lifecycle.assess_changes(self.policy, acceptance.Inputs())
        lifecycle.apply_changes(self.work, change)
        state = self.work["exploration_fulfillment"]["items"]["compute"]
        self.assertEqual(state["disposition"], "retain")
        self.assertEqual(self.work["exploration_fulfillment"]["candidate_approvals"]["base"]["status"], "needs_review")
        self.assertEqual(state["adoption"], "research")
        self.assertEqual(change["changes"][0]["candidates"], ["base", "combo"])

    def test_report_or_checker_change_requires_delivery_only_no_model_reopen(self):
        for kind in ["report", "checker"]:
            with self.subTest(kind=kind):
                self.policy["lifecycle"]["dependencies"][0]["kind"] = kind
                self.context.write_text("# editorial or checker update\n")
                change = lifecycle.assess_changes(self.policy, acceptance.Inputs())
                work = copy.deepcopy(self.work)
                lifecycle.apply_changes(work, change)
                self.assertEqual(change["changes"][0]["domains"], ["delivery"])
                self.assertEqual(work["exploration_fulfillment"]["items"], self.work["exploration_fulfillment"]["items"])

    def test_catalog_change_triggers_discovery_without_global_measurement_rerun(self):
        self.policy["lifecycle"]["dependencies"][0]["kind"] = "catalog"
        self.context.write_text("# new method family\n")
        impact = lifecycle.assess_changes(self.policy, acceptance.Inputs())
        self.assertEqual(impact["changes"][0]["items"], sorted(self.policy["items"]))
        self.assertEqual(impact["changes"][0]["domains"], ["discovery_decision"])

    def test_cycle_and_unknown_dependency_cannot_hide_items(self):
        self.policy["lifecycle"]["candidates"]["base"]["parents"] = ["combo"]
        with self.assertRaisesRegex(ValueError, "cyclic"):
            lifecycle.assess_changes(self.policy, acceptance.Inputs())

    def test_unchanged_resume_reuses_current_evidence_without_reopening(self):
        before = copy.deepcopy(self.work)
        impact = lifecycle.assess_changes(self.policy, acceptance.Inputs())
        self.assertFalse(impact["changed"])
        lifecycle.apply_changes(self.work, impact)
        self.assertEqual(self.work, before)
        _, errors = lifecycle.check_review(self.policy, self.inputs(), work_package=self.work)
        self.assertEqual(errors, [])

    def test_full_acceptance_really_consumes_required_semantic_review(self):
        from test_exploration_acceptance import collect

        self.policy["lifecycle"]["required_finding_ids"] = ["not-fixed"]
        self.review["findings"] = [{"id": "not-fixed", "items": ["compute"], "disposition": "open"}]
        self.resign_review()
        inputs = self.inputs()
        facts = collect(inputs, self.policy)
        result = acceptance.check(self.policy, acceptance.catalog_families((self.root / "MAP.md").read_text()), self.work, facts, inputs)
        self.assertIn("review_unclosed", {x["code"] for x in result["errors"]})
        self.assertFalse(result["task_complete"])

    def test_sparse_shared_ancestors_complete_with_a_bounded_cpu_entry(self):
        program = """
from tools.exploration_acceptance import Inputs
from tools.exploration_lifecycle import assess_changes, _descendants
candidates = {"root": {"parents": [], "items": []}}
previous = ["root"]
for i in range(22):
    current = [f"{i}a", f"{i}b"]
    for name in current:
        candidates[name] = {"parents": previous, "items": []}
    previous = current
policy = {"task_id": "fixture", "scope_id": "fixture", "items": {}, "lifecycle": {"candidates": candidates}}
assert not assess_changes(policy, Inputs())["changed"]
print(len(_descendants(candidates, {"root"})))
"""
        result = subprocess.run([sys.executable, "-B", "-c", program], capture_output=True, text=True, timeout=3, check=True)
        self.assertEqual(result.stdout.strip(), "45")

    def test_shared_graph_validation_keeps_cycle_and_unknown_parent_rejection(self):
        for candidates, roots, message in [
            ({"a": {"parents": ["missing"]}}, set(), "unknown parent"),
            ({"a": {"parents": []}}, {"missing"}, "unknown dependent"),
            ({"a": {"parents": ["b"]}, "b": {"parents": ["a"]}}, set(), "cyclic"),
        ]:
            with self.subTest(candidates=candidates, roots=roots), self.assertRaisesRegex(ValueError, message):
                lifecycle._descendants(candidates, roots)


if __name__ == "__main__":
    unittest.main()
