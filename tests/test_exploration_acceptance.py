"""CPU contracts for scope closure, original evidence, and published claims.

The native-format fixtures below are test data, not measured GPU results.
"""

import copy
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools import exploration_acceptance as acceptance
from tools import exploration_evidence as evidence


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_module(directory, name, values):
    identity = {"target": {"completion_boundary": "one block"}, "input": {"kind": "real", "sha256": "fixture-input"}, "source": {"revision": "fixture-revision"}}
    parameters = dict(warmup=1, repeats=2, iterations=2)
    raw = dict(
        rank=0,
        identity=identity,
        parameters=parameters,
        groups={"measurement_ranks": [0], "tp_ranks": [0], "sp_ranks": [0]},
        correctness={"status": "strict_equal", "scope": "one block output", "input_unchanged": True},
        samples=[dict(repeat=i // 2, iteration=i % 2, latency_ms=value) for i, value in enumerate(values)],
    )
    report = dict(
        kind="module_benchmark_v1",
        identity=identity,
        parameters=parameters,
        factory={"config": {"implementation": name}},
        ranks=[{k: raw[k] for k in ("rank", "groups", "correctness")}],
        max_rank_latency_ms=values,
        latency_ms={"median": statistics.median(values), "mean": statistics.mean(values)},
    )
    report_path, raw_path = directory / (name + ".json"), directory / (name + ".jsonl")
    write_json(report_path, report)
    raw_path.write_text(json.dumps(raw) + "\n")
    return [str(report_path), str(raw_path)]


def collect(inputs, policy):
    # This fixture certifies fake runs only to isolate acceptance contracts.
    # Real task collectors must verify original run evidence before setting True.
    facts = {
        key: evidence.source_reference(inputs, spec["paths"]) if spec["reader"] == "source" else evidence.module_comparison(inputs, spec["parents"], spec["candidates"])
        for key, spec in policy["evidence"].items()
    }
    for fact in facts.values():
        if fact["kind"] == "measurement":
            fact["valid"] = True
    return facts


def fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    job = root / "job"
    parent = write_module(job, "dense", [10.0, 12.0, 14.0, 16.0])
    candidate = write_module(job, "candidate", [8.0, 8.0, 8.0, 8.0])
    code = root / "model.py"
    code.write_text("# fixture: no vision branch in this task\n")
    catalog = root / "MAP.md"
    catalog.write_text("### Compute\n\n### Vision\n\n### Deferred\n")
    policy = dict(
        kind="exploration_policy_v1",
        task_id="optimization-refactor-20260925",
        scope_id="fixture_v1",
        scope_label="CPU fixture scope",
        task_root=str(root),
        report=str(root / "REPORT.md"),
        catalog={"path": str(catalog), "sha256": acceptance.digest(catalog)},
        evidence={"speed": dict(reader="module", kind="measurement", parents=[parent], candidates=[candidate]), "structure": dict(reader="source", kind="source", paths=[str(code)])},
        items={
            "compute": dict(family="compute", label="Block", question="Does this block improve?", locations=["block"], evidence=["speed"]),
            "vision": dict(family="vision", label="Vision", question="Is this path used?", locations=["input"], evidence=["structure"]),
            "later": dict(family="deferred", label="Later", question="Is additional integration worth its cost?", locations=["output"], evidence=["structure"]),
        },
        claims=[dict(id="block-speed", block="results", item="compute", evidence="speed", label="Block candidate", reference="job/candidate.json")],
        report_blocks={"results": "| item | scope | time | decision |\n|---|---|---|---|"},
        required_action_ids=["deliver"],
    )
    activity = dict(
        scope_id="fixture_v1",
        items={
            "compute": dict(disposition="retain", rationale="Native block comparison improves the target boundary.", evidence=["speed"], adoption="research"),
            "vision": dict(disposition="not_applicable", rationale="The actual task code has no vision branch.", evidence=["structure"]),
            "later": dict(
                disposition="defer",
                rationale="Current output cost is below integration and validation cost.",
                reopen_if="Output becomes a bottleneck or a matching implementation is available.",
                evidence=["structure"],
            ),
        },
    )
    inputs = acceptance.Inputs()
    facts = collect(inputs, policy)
    for key, fact in facts.items():
        policy["evidence"][key]["subject"] = fact["subject"]
    policy["artifacts"] = {str(path): sha for path, sha in inputs.hashes.items()}
    families = acceptance.catalog_families(catalog.read_text())
    blocks = acceptance.rendered_blocks(policy, families, activity, facts)
    text = (
        "\n\n".join(
            f"# {i}. section"
            + (
                "\n\n<!-- exploration:coverage -->\n" + blocks["coverage"] + "\n<!-- /exploration:coverage -->"
                if i == 2
                else "\n\n<!-- exploration:results -->\n" + blocks["results"] + "\n<!-- /exploration:results -->"
                if i == 3
                else ""
            )
            for i in range(1, 8)
        )
        + "\n"
    )
    Path(policy["report"]).write_text(text)
    policy["report_context"] = acceptance.report_context(text, {"coverage", *policy["report_blocks"]})
    work = dict(task_id=policy["task_id"], status="in_progress", exploration_fulfillment=activity, next_actions=[dict(id="deliver", execution_state="completed")])
    write_json(root / "policy.json", policy)
    write_json(root / "work_package.json", work)
    return policy, work


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.policy, self.work = fixture(self.root)

    def check(self, mode="task", selected=None, facts=None):
        inputs = acceptance.Inputs()
        policy, families = acceptance.load_policy(inputs, self.root / "policy.json", acceptance.digest(self.root / "policy.json"))
        actual = collect(inputs, policy)
        result = acceptance.check(policy, families, self.work, actual if facts is None else facts, inputs, mode, selected)
        return result, inputs

    def codes(self):
        return {row["code"] for row in self.check()[0]["errors"]}

    def test_valid_scope_has_source_negative_or_deferred_branches(self):
        result, _ = self.check()
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["task_complete"])
        self.work["exploration_fulfillment"]["items"]["compute"]["disposition"] = "reject"
        report = Path(self.policy["report"])
        text = report.read_text().replace("保留；数值严格相同", "不选；数值严格相同").replace("：保留；", "：不选；")
        report.write_text(text)
        self.assertTrue(self.check()[0]["passed"])

    def test_missing_necessary_child_cannot_hide_behind_parent(self):
        del self.work["exploration_fulfillment"]["items"]["vision"]
        self.assertIn("inventory", self.codes())

    def test_empty_deferral_and_missing_reopen_refused(self):
        item = self.work["exploration_fulfillment"]["items"]["later"]
        item["rationale"] = "未接线"
        item.pop("reopen_if")
        self.assertTrue({"rationale", "reopen"} <= self.codes())

    def test_native_object_not_mutable_activity_mapping_decides_identity(self):
        candidate = Path(self.policy["evidence"]["speed"]["candidates"][0][0])
        report = json.loads(candidate.read_text())
        report["factory"]["config"]["implementation"] = "other-real-candidate"
        write_json(candidate, report)
        self.work["exploration_fulfillment"]["items"]["compute"]["candidate"] = "other-real-candidate"
        self.assertTrue({"evidence_identity", "artifact_identity"} <= self.codes())

    def test_changed_timing_scope_or_parent_cannot_be_resigned_by_structure_checker(self):
        report = Path(self.policy["report"])
        original = report.read_text()
        for before, after in [("13.000", "999.000"), ("模块/block/阶段", "完整请求"), ("父方案", "另一个父方案"), ("真实输入", "合成输入")]:
            with self.subTest(after=after):
                report.write_text(original.replace(before, after))
                self.assertTrue(acceptance.report_check.check(report)["passed"])
                self.assertIn("report_fact", self.codes())
        report.write_text(original)

    def test_extra_unbound_summary_cannot_escape_protected_row_check(self):
        report = Path(self.policy["report"])
        report.write_text(report.read_text().replace("# 1. section", "# 1. section\n\nRequest is 100x faster."))
        self.assertIn("unbound_declaration", self.codes())

    def test_template_heading_levels_and_hidden_examples_share_context_parser(self):
        template = Path(acceptance.__file__).parent / "EXPLORATION_REPORT_TEMPLATE.md"
        self.assertEqual(set(acceptance.report_context(template.read_text())), {"1", "3", "4", "6"})
        report = Path(self.policy["report"])
        text = report.read_text().replace("# ", "## ")
        text = "<!--\n# 1. hidden\n-->\n```md\n# 1. example\n```\n" + text
        report.write_text(text)
        self.policy["report_context"] = acceptance.report_context(text, {"coverage", *self.policy["report_blocks"]})
        write_json(self.root / "policy.json", self.policy)
        self.assertTrue(self.check()[0]["passed"])

    def test_unregistered_block_body_cannot_bypass_critical_context(self):
        report = Path(self.policy["report"])
        text = report.read_text().replace("# 1. section", "# 1. section\n<!-- exploration:unknown -->\nOriginal summary.\n<!-- /exploration:unknown -->")
        report.write_text(text)
        self.policy["report_context"] = acceptance.report_context(text, {"coverage", *self.policy["report_blocks"]})
        write_json(self.root / "policy.json", self.policy)
        self.assertTrue(self.check()[0]["passed"])
        report.write_text(text.replace("Original summary.", "Full request 100x faster; production quality approved."))
        self.assertIn("unbound_declaration", self.codes())
        self.policy["report_blocks"]["unknown"] = "Unused table"
        write_json(self.root / "policy.json", self.policy)
        with self.assertRaisesRegex(ValueError, "must consume registered claims"):
            self.check()

    def test_visible_timing_has_an_independent_arithmetic_and_unit_oracle(self):
        fact = collect(acceptance.Inputs(), self.policy)["speed"]
        spec = self.policy["claims"][0]
        state = self.work["exploration_fulfillment"]["items"]["compute"]
        self.assertIn("13.000 → 8.000ms；延迟降低38.46%", acceptance.render_claim(spec, fact, state))
        fact["values"]["candidate"] = 16.0
        fact["unit"] = "us"
        self.assertIn("13.000 → 16.000µs；延迟增加23.08%", acceptance.render_claim(spec, fact, state))
        fact["unit"] = "minutes"
        result, _ = self.check(facts={**collect(acceptance.Inputs(), self.policy), "speed": fact})
        self.assertIn("measurement_unit", {row["code"] for row in result["errors"]})

    def test_formal_results_layout_places_parent_change_scope_and_timing_in_their_columns(self):
        template = Path(acceptance.__file__).parent / "EXPLORATION_REPORT_TEMPLATE.md"
        header = next(line for line in template.read_text().splitlines() if line.startswith("| 优化项及改动方式 |"))
        self.policy["report_blocks"]["results"] = header + "\n|---|---|---|---|---|"
        claim = self.policy["claims"][0]
        claim.update(layout="results", parent_label="Dense block", candidate_label="Fused block", change="代码：仅合并 projection | output；其余条件固定")
        facts = collect(acceptance.Inputs(), self.policy)
        families = acceptance.catalog_families((self.root / "MAP.md").read_text())
        blocks = acceptance.rendered_blocks(self.policy, families, self.work["exploration_fulfillment"], facts)
        row = next(line for line in blocks["results"].splitlines() if line.startswith("| <a"))
        # The renderer's spaced delimiters distinguish cells from escaped pipes in descriptions.
        cells = [cell.strip().replace(r"\|", "|") for cell in row.strip("| ").split(" | ")]
        self.assertEqual(len(cells), 5)
        self.assertIn("Block candidate", cells[0])
        self.assertEqual(cells[1], "Dense block → Fused block；" + claim["change"])
        self.assertIn("模块/block/阶段（one block），真实输入", cells[2])
        self.assertIn(facts["speed"]["statistic"], cells[2])
        self.assertIn("首筛", cells[2])
        self.assertNotIn("Dense block", cells[2])
        self.assertEqual(cells[3], "13.000 → 8.000ms；延迟降低38.46%")
        self.assertIn("保留；数值严格相同", cells[4])
        self.assertIn("job/candidate.json", cells[4])
        report = Path(self.policy["report"])
        report.write_text(acceptance.replace_blocks(report.read_text(), blocks))
        write_json(self.root / "policy.json", self.policy)
        result, _ = self.check()
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["task_complete"])

    def test_formal_results_layout_requires_explicit_parent_and_change(self):
        fact = collect(acceptance.Inputs(), self.policy)["speed"]
        claim = dict(self.policy["claims"][0], layout="results", parent_label="Dense block", change="Only fuse projection")
        state = self.work["exploration_fulfillment"]["items"]["compute"]
        for key in ("parent_label", "change"):
            for value in (None, "", "  ", 1):
                invalid = {**claim, key: value}
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "explicit parent_label and change"):
                    acceptance.render_claim(invalid, fact, state)
            invalid = dict(claim)
            del invalid[key]
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "explicit parent_label and change"):
                acceptance.render_claim(invalid, fact, state)

    def test_report_table_rejects_layout_header_and_separator_column_mismatches(self):
        template = Path(acceptance.__file__).parent / "EXPLORATION_REPORT_TEMPLATE.md"
        formal_header = next(line for line in template.read_text().splitlines() if line.startswith("| 优化项及改动方式 |"))
        legacy_header = self.policy["report_blocks"]["results"]
        claim = self.policy["claims"][0]
        for layout, header in (
            (None, formal_header + "\n|---|---|---|---|---|"),
            ("results", legacy_header),
            (None, legacy_header.splitlines()[0] + "\n|---|---|---|---|---|"),
        ):
            claim.update(layout=layout, parent_label="Dense block", change="Only fuse projection")
            self.policy["report_blocks"]["results"] = header
            write_json(self.root / "policy.json", self.policy)
            with self.subTest(layout=layout, header=header), self.assertRaisesRegex(ValueError, "report table column mismatch"):
                self.check()
        claim.pop("layout")
        claim.pop("parent_label")
        claim.pop("change")
        self.policy["report_blocks"]["results"] = legacy_header
        write_json(self.root / "policy.json", self.policy)
        result, _ = self.check()
        self.assertTrue(result["passed"], result["errors"])

    def test_report_table_allows_blank_lines_and_empty_edge_header_cells(self):
        facts = collect(acceptance.Inputs(), self.policy)
        families = acceptance.catalog_families((self.root / "MAP.md").read_text())
        report = Path(self.policy["report"])
        for heading in (
            "| item | scope | time | decision |",
            "|| scope | time | decision |",
            "| item | scope | time ||",
            "|| scope | time ||",
        ):
            self.policy["report_blocks"]["results"] = "\n \t\n" + heading + "\n|---|---|---|---|\n\n"
            with self.subTest(heading=heading):
                blocks = acceptance.rendered_blocks(self.policy, families, self.work["exploration_fulfillment"], facts)
                report.write_text(acceptance.replace_blocks(report.read_text(), blocks))
                write_json(self.root / "policy.json", self.policy)
                result, _ = self.check()
                self.assertTrue(result["passed"], result["errors"])

    def test_report_table_rejects_extra_empty_edge_cells(self):
        for heading in ("|| item | scope | time | decision |", "| item | scope | time | decision ||"):
            self.policy["report_blocks"]["results"] = heading + "\n|---|---|---|---|"
            write_json(self.root / "policy.json", self.policy)
            with self.subTest(heading=heading), self.assertRaisesRegex(ValueError, "report table column mismatch"):
                self.check()

    def test_report_table_counts_escaped_pipes_and_real_separators(self):
        facts = collect(acceptance.Inputs(), self.policy)
        families = acceptance.catalog_families((self.root / "MAP.md").read_text())
        report = Path(self.policy["report"])
        for heading in (
            r"| item \| label | scope | time | decision |",
            r"| item | scope | time | decision\|",
            r"| item \\| scope | time | decision |",
            r"| item \\\| label | scope | time | decision |",
        ):
            self.policy["report_blocks"]["results"] = heading + "\n|---|---|---|---|"
            with self.subTest(heading=heading):
                blocks = acceptance.rendered_blocks(self.policy, families, self.work["exploration_fulfillment"], facts)
                report.write_text(acceptance.replace_blocks(report.read_text(), blocks))
                write_json(self.root / "policy.json", self.policy)
                result, _ = self.check()
                self.assertTrue(result["passed"], result["errors"])
        self.policy["report_blocks"]["results"] = r"| item \\| extra | scope | time | decision |" + "\n|---|---|---|---|"
        write_json(self.root / "policy.json", self.policy)
        with self.assertRaisesRegex(ValueError, "report table column mismatch"):
            self.check()

    def test_report_table_requires_header_and_separator_after_ignoring_blank_lines(self):
        for header in ("", "\n \t\n", "| item | scope | time | decision |"):
            self.policy["report_blocks"]["results"] = header
            write_json(self.root / "policy.json", self.policy)
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, "header and separator"):
                self.check()

    def test_native_samples_need_explicit_run_verification_before_completion(self):
        facts = collect(acceptance.Inputs(), self.policy)
        spec = self.policy["evidence"]["speed"]
        facts["speed"] = evidence.module_comparison(acceptance.Inputs(), spec["parents"], spec["candidates"])
        self.assertIs(facts["speed"]["valid"], False)
        for value in (False, None, "verified", 1):
            facts["speed"]["valid"] = value
            with self.subTest(value=value):
                result, _ = self.check(facts=facts)
                self.assertIn("measurement", {row["code"] for row in result["errors"]})
                self.assertFalse(result["task_complete"])

    def test_metric_first_claim_keeps_coverage_and_alias_links_reachable(self):
        facts = collect(acceptance.Inputs(), self.policy)
        facts["speed"]["kind"] = "metric"
        self.policy["evidence"]["speed"]["kind"] = "metric"
        self.policy["claims"][0]["aliases"] = ["metric-alias"]
        state = self.work["exploration_fulfillment"]["items"]["compute"]
        state.update(disposition="defer", reopen_if="A controlled timing comparison is available.")
        report = Path(self.policy["report"])
        families = acceptance.catalog_families((self.root / "MAP.md").read_text())
        text = acceptance.replace_blocks(report.read_text(), acceptance.rendered_blocks(self.policy, families, self.work["exploration_fulfillment"], facts))
        report.write_text(text + "\n[Metric detail](#metric-alias)\n")
        self.policy["report_context"] = acceptance.report_context(report.read_text(), {"coverage", *self.policy["report_blocks"]})
        write_json(self.root / "policy.json", self.policy)
        result, _ = self.check(facts=facts)
        self.assertTrue(result["passed"], result["errors"])

    def test_wrong_catalog_or_scope_invalidates_fixed_policy(self):
        inputs = acceptance.Inputs()
        old = acceptance.digest(self.root / "policy.json")
        (self.root / "MAP.md").write_text("### Changed\n")
        with self.assertRaisesRegex(ValueError, "pinned input changed"):
            acceptance.load_policy(inputs, self.root / "policy.json", old)

    def test_partial_delivery_cannot_close_parent_with_other_action(self):
        self.work["next_actions"].append(dict(id="other", required_for_scope=self.policy["scope_id"], execution_state="pending"))
        result, _ = self.check(mode="delivery", selected=["compute"])
        self.assertTrue(result["passed"], result["errors"])
        self.assertFalse(result["task_complete"])
        for state in acceptance.COMPLETE_STATES:
            with self.subTest(state=state):
                self.work["status"] = state
                with self.assertRaisesRegex(ValueError, "parent task"):
                    acceptance.completion_state(self.work, result)
        self.assertIn("next_action", self.codes())

    def test_blocked_branch_does_not_complete_remaining_scope(self):
        self.work["exploration_fulfillment"]["items"]["later"]["disposition"] = "blocked"
        self.assertIn("unclosed", self.codes())
        result, _ = self.check("delivery", ["compute"])
        # Coverage itself must be regenerated from the current activity.
        self.assertFalse(result["passed"])
        self.assertIn("later", result["remaining_items"])

    def test_originals_and_report_cannot_be_receipt_output(self):
        result, inputs = self.check()
        for path in [self.root / "model.py", self.root / "REPORT.md", self.root / "policy.json"]:
            with self.subTest(path=path):
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    acceptance.write_receipt(result, path, inputs)
                self.assertEqual(path.read_bytes(), before)

    def test_current_receipt_does_not_accept_changed_file_or_other_task(self):
        result, inputs = self.check()
        receipt = self.root / "receipt.json"
        acceptance.write_receipt(result, receipt, inputs)
        acceptance.verify_receipt(receipt, result)
        changed = copy.deepcopy(result)
        changed["task_id"] = "other"
        with self.assertRaisesRegex(ValueError, "different scope"):
            acceptance.verify_receipt(receipt, changed)
        (self.root / "model.py").write_text("changed\n")
        current, _ = self.check()
        with self.assertRaisesRegex(ValueError, "passing"):
            acceptance.verify_receipt(receipt, current)

    def test_auxiliary_originals_cannot_be_consumed_without_registration(self):
        inputs = acceptance.Inputs()
        policy, families = acceptance.load_policy(inputs, self.root / "policy.json", acceptance.digest(self.root / "policy.json"))
        facts = collect(inputs, policy)
        extra = self.root / "unregistered.json"
        extra.write_text("{}")
        inputs.read(extra)
        result = acceptance.check(policy, families, self.work, facts, inputs)
        self.assertIn("unregistered_source", {row["code"] for row in result["errors"]})

    def test_unknown_status_needs_explicit_consumer_mapping(self):
        result, _ = self.check()
        self.work["status"] = "complete"
        with self.assertRaisesRegex(ValueError, "unknown task state"):
            acceptance.completion_state(self.work, result)

    def test_completion_result_cannot_close_a_foreign_task_or_scope(self):
        result, _ = self.check()
        self.work["status"] = "completed"
        self.assertEqual(acceptance.completion_state(self.work, result), "completed")
        for target, key in ((self.work, "task_id"), (self.work["exploration_fulfillment"], "scope_id")):
            original = target[key]
            target[key] = "foreign"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "another task/scope"):
                acceptance.completion_state(self.work, result)
            target[key] = original

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        for value in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}']:
            with self.subTest(value=value):
                path = self.root / "bad.jsonl"
                path.write_text(value + "\n")
                with self.assertRaises(ValueError):
                    acceptance.Inputs().jsonl(path)
                with self.assertRaises(ValueError):
                    acceptance.Inputs().json(path)

    def test_generated_reference_with_parentheses_passes_full_acceptance(self):
        code = self.root / "model (v2).py"
        code.write_bytes((self.root / "model.py").read_bytes())
        self.policy["evidence"]["structure"]["paths"] = [str(code)]
        facts = collect(acceptance.Inputs(), self.policy)
        self.policy["evidence"]["structure"]["subject"] = facts["structure"]["subject"]
        self.policy["artifacts"][str(code)] = acceptance.digest(code)
        self.policy["claims"].append(dict(id="vision-basis", block="results", item="vision", evidence="structure", label="Source basis", reference=code.name))
        report = Path(self.policy["report"])
        families = acceptance.catalog_families((self.root / "MAP.md").read_text())
        blocks = acceptance.rendered_blocks(self.policy, families, self.work["exploration_fulfillment"], facts)
        report.write_text(acceptance.replace_blocks(report.read_text(), blocks))
        write_json(self.root / "policy.json", self.policy)
        result, _ = self.check()
        self.assertTrue(result["passed"], result["errors"])

    def test_generated_review_targets_with_parentheses_remain_reachable(self):
        (self.root / "review (v2).html").write_text('<html><body id="pair">comparison</body></html>')
        observed = collect(acceptance.Inputs(), self.policy)["speed"]
        spec = dict(id="quality", label="Candidate", reference="job/candidate.json")
        for kind, layout, scope in [("quality", None, "module"), ("measurement", "quality", "module"), ("measurement", None, "request")]:
            with self.subTest(kind=kind, layout=layout, scope=scope):
                fact = copy.deepcopy(observed)
                fact.update(kind=kind, quality="pending_review", review="review (v2).html#pair")
                fact["subject"]["scope"] = scope
                row = acceptance.render_claim({**spec, "layout": layout}, fact, dict(disposition="retain"))
                report = self.root / "GENERATED.md"
                report.write_text("\n\n".join(f"# {i}. section" for i in range(1, 8)) + "\n\n| item | state | review |\n|---|---|---|\n" + row + "\n")
                result = acceptance.report_check.check(report)
                self.assertTrue(result["passed"], result["errors"])

    def test_blocked_unwired_branch_does_not_reject_unrelated_partial_delivery(self):
        self.work["exploration_fulfillment"]["items"]["later"].update(disposition="blocked", rationale="backend 未接线，目标设备由外部提供")
        facts = collect(acceptance.Inputs(), self.policy)
        families = acceptance.catalog_families((self.root / "MAP.md").read_text())
        report = Path(self.policy["report"])
        report.write_text(acceptance.replace_blocks(report.read_text(), acceptance.rendered_blocks(self.policy, families, self.work["exploration_fulfillment"], facts)))
        partial, _ = self.check(mode="delivery", selected=["compute"])
        self.assertTrue(partial["passed"], partial["errors"])
        self.assertFalse(partial["task_complete"])
        self.assertEqual(partial["remaining_items"], ["later"])
        whole, _ = self.check()
        self.assertFalse(whole["passed"])
        self.assertFalse(whole["task_complete"])
        self.assertIn("unclosed", {row["code"] for row in whole["errors"]})
        self.work["status"] = "blocked"
        self.assertEqual(acceptance.completion_state(self.work, partial), "blocked")
        self.work["status"] = "completed"
        with self.assertRaisesRegex(ValueError, "parent task"):
            acceptance.completion_state(self.work, partial)

    def test_combination_missing_comparisons_are_displayed_without_filling_them(self):
        facts = collect(acceptance.Inputs(), self.policy)
        observed_parent = facts["speed"]["values"]["parent"]
        self.policy["claims"][0]["layout"] = "combination"
        self.policy["report_blocks"]["results"] = "| combination | ms | step | cumulative | decision |\n|---|---|---|---|---|"
        families = acceptance.catalog_families((self.root / "MAP.md").read_text())
        for parent, original, expected_step, expected_total in [
            (None, observed_parent, "未测单步", "38.46%"),
            (observed_parent, None, "+38.46%", "未测直接累计"),
            (observed_parent, observed_parent, "+38.46%", "38.46%"),
            (None, None, "未测单步", "未测直接累计"),
        ]:
            with self.subTest(parent=parent, original=original):
                facts["speed"]["values"].update(parent=parent, original=original)
                report = Path(self.policy["report"])
                blocks = acceptance.rendered_blocks(self.policy, families, self.work["exploration_fulfillment"], facts)
                report.write_text(acceptance.replace_blocks(report.read_text(), blocks))
                write_json(self.root / "policy.json", self.policy)
                result, _ = self.check(facts=facts)
                self.assertTrue(result["passed"], result["errors"])
                self.assertIn(expected_step, blocks["results"])
                self.assertIn(expected_total, blocks["results"])
                self.assertIn("8.000", blocks["results"])


class SourceExampleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = Path(acceptance.__file__).resolve().parents[1]
        self.example = self.repository / "tools/examples/exploration_source_task.py"

    def command(self, script, task, action, *arguments):
        return subprocess.run(
            [sys.executable, "-B", str(script), action, str(task), *arguments],
            capture_output=True, text=True, timeout=5,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
        )

    def test_example_is_readonly_on_verify_and_cannot_close_parent_or_reuse_changed_evidence(self):
        task = self.root / "task"
        initialized = self.command(self.example, task, "init")
        self.assertEqual(initialized.returncode, 0, initialized.stderr)
        result = json.loads(initialized.stdout)
        self.assertTrue(result["passed"])
        self.assertFalse(result["task_complete"])
        arguments = ["--binding-sha256", result["binding_sha256"]]
        originals = {path: path.read_bytes() for path in task.iterdir()}
        verified = self.command(self.example, task, "verify", *arguments)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        self.assertEqual(originals, {path: path.read_bytes() for path in task.iterdir()})
        self.assertNotEqual(self.command(self.example, task, "init").returncode, 0)
        self.assertEqual(originals, {path: path.read_bytes() for path in task.iterdir()})
        for name in ("INPUT.py", "policy.json", "REPORT.md"):
            path = task / name
            try:
                path.write_bytes(originals[path] + b"\n")
                self.assertNotEqual(self.command(self.example, task, "verify", *arguments).returncode, 0, name)
            finally:
                path.write_bytes(originals[path])
        work_path = task / "work_package.json"
        work = json.loads(originals[work_path])
        work["status"] = "completed"
        try:
            write_json(work_path, work)
            rejected = self.command(self.example, task, "verify", *arguments)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("parent task", rejected.stderr)
        finally:
            work_path.write_bytes(originals[work_path])
        self.assertEqual(self.command(self.example, task, "verify", *arguments).returncode, 0)
        self.assertEqual(originals, {path: path.read_bytes() for path in task.iterdir()})

    def test_example_rejects_changed_dependency_before_import(self):
        replica = self.root / "replica"
        relative = Path("tools/examples/exploration_source_task.py")
        for name in [relative, *[Path("tools") / filename for filename in (
            "__init__.py", "exploration_acceptance.py", "exploration_evidence.py", "check_exploration_report.py",
        )]]:
            destination = replica / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.repository / name, destination)
        task = self.root / "replica-task"
        initialized = self.command(replica / relative, task, "init")
        self.assertEqual(initialized.returncode, 0, initialized.stderr)
        binding_sha256 = json.loads(initialized.stdout)["binding_sha256"]
        (replica / "tools/exploration_evidence.py").write_text("raise RuntimeError('UNPINNED_CODE_EXECUTED')\n")
        rejected = self.command(replica / relative, task, "verify", "--binding-sha256", binding_sha256)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("pinned consumer dependency changed", rejected.stderr)
        self.assertNotIn("UNPINNED_CODE_EXECUTED", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
