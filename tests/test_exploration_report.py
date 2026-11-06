"""CPU regressions for the report-delivery consumer, including stale receipts."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tools/check_exploration_report.py"
spec = importlib.util.spec_from_file_location("check_exploration_report", SOURCE)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ReportDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.report = self.root / "REPORT.md"
        self.page = self.root / "review.html"
        self.page.write_text('<html><body id="pair">review</body></html>', encoding="utf-8")

    def write_report(self, row="| candidate | 待审 | [对比](review.html) |", extra=""):
        text = "\n\n".join(f"# {number}. section" for number in range(1, 8))
        self.report.write_text(text + "\n\n| 方案 | 状态 | 审核 |\n|---|---|---|\n" + row + "\n" + extra, encoding="utf-8")

    def codes(self):
        return {error["code"] for error in checker.check(self.report)["errors"]}

    def test_pending_without_entry_rejected(self):
        self.write_report("| candidate | 待审 | 以后审核 |")
        self.assertIn("missing_review_entry", self.codes())

    def test_unrelated_method_link_does_not_count_as_review(self):
        (self.root / "method.md").write_text("# method\n", encoding="utf-8")
        self.write_report("| candidate | 待审 | [方法](method.md) |")
        self.assertIn("missing_review_entry", self.codes())

    def test_static_markdown_review_reaches_original_images(self):
        image = self.root / "candidate.png"
        image.write_bytes(b"saved media payload")
        page = self.root / "review.md"
        page.write_text('# <a id="pair"></a>Image comparison\n\nSame input.\n\n![candidate](candidate.png)\n', encoding="utf-8")
        for target in ("review.md", "review.md#pair"):
            with self.subTest(target=target):
                self.write_report(f"| candidate | 待审 | [对比]({target}) |")
                self.assertTrue(checker.check(self.report)["passed"])
        receipt = self.root / "CHECKS.json"
        checker.write_receipt(checker.check(self.report), receipt)
        page.write_text(page.read_text() + "\nChanged comparison.\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            checker.verify(receipt)

    def test_markdown_review_rejects_missing_empty_and_external_media(self):
        page = self.root / "review.md"
        self.write_report("| candidate | 待审 | [对比](review.md) |")
        (self.root / "empty.png").write_bytes(b"")
        for target in ("missing.png", "empty.png", "https://example.test/image.png"):
            with self.subTest(target=target):
                page.write_text(f"# Comparison\n\n![candidate]({target})\n")
                self.assertIn("missing_review_entry", self.codes())

    def test_markdown_review_cannot_borrow_another_sections_image(self):
        (self.root / "other.png").write_bytes(b"other candidate media")
        self.write_report("| candidate | 待审 | [对比](review.md#candidate) |")
        for boundary in ("## other", '<a id="other"></a>'):
            with self.subTest(boundary=boundary):
                (self.root / "review.md").write_text(f"## candidate\n\nNo saved output.\n\n{boundary}\n\n![other](other.png)\n")
                self.assertIn("missing_review_entry", self.codes())

    def test_direct_local_media_review_entry(self):
        image = self.root / "candidate.png"
        self.write_report("| candidate | 待审 | [原图](candidate.png) |")
        image.write_bytes(b"saved media payload")
        self.assertTrue(checker.check(self.report)["passed"])
        image.write_bytes(b"")
        self.assertIn("missing_review_entry", self.codes())

    def test_candidate_anchor_reaches_review(self):
        self.write_report("| candidate | 待审 | [审核](#quality-candidate) |", extra='<a id="quality-candidate"></a>\n[对比](review.html#pair)\n')
        self.assertTrue(checker.check(self.report)["passed"])

    def test_other_candidates_review_is_not_borrowed(self):
        self.write_report("| candidate | 待审 | [审核](#candidate) |", extra='\n| item | entry |\n|---|---|\n| <a id="candidate"></a>candidate | missing |\n| other | [对比](review.html) |\n')
        self.assertIn("missing_review_entry", self.codes())

    def test_circular_anchor_does_not_pass(self):
        self.write_report('| <a id="candidate"></a>candidate | 待审 | [审核](#candidate) |')
        self.assertIn("missing_review_entry", self.codes())

    def test_heading_cannot_override_another_candidates_explicit_anchor(self):
        for name, headings in (("candidate", "## candidate"), ("candidate-1", "## candidate\n\n## candidate")):
            self.write_report(f'| <a id="{name}"></a>candidate | 待审 | [审核](#{name}) |', extra=f"\n{headings}\n\n[其他候选](review.html)\n")
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, str(SOURCE), "--report", str(self.report)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn("ambiguous Markdown anchor", result.stderr)
        self.write_report("| candidate | 待审 | [审核](#candidate-1) |", extra="\n## candidate\n\nNo review here.\n\n## candidate\n\n[对比](review.html)\n")
        self.assertTrue(checker.check(self.report)["passed"])
        self.write_report("| candidate | 待审 | [审核](#candidate) |", extra='\n## <a id="candidate" name="candidate"></a>candidate\n\n[对比](review.html)\n')
        self.assertTrue(checker.check(self.report)["passed"])

    def test_heading_anchor_with_blank_line_reaches_review(self):
        self.write_report("| candidate | 待审 | [审核](#candidate) |", extra="\n### candidate\n\n[对比](review.html)\n")
        self.assertTrue(checker.check(self.report)["passed"])

    def test_angle_reference_targets_with_fragment_and_spaces(self):
        spaced = self.root / "review pair.html"
        spaced.write_text('<html><body id="pair">review</body></html>', encoding="utf-8")
        for target in ("<review.html>", "<review.html#pair>", "<review pair.html#pair>"):
            with self.subTest(target=target):
                self.write_report("| candidate | 待审 | [审核][pair] |", extra="\n[pair]: " + target + "\n")
                self.assertTrue(checker.check(self.report)["passed"])

    def test_html_anchor_attributes_are_parsed_not_searched(self):
        self.write_report("| candidate | 待审 | [审核](review.html#pair) |")
        for html in ('<!-- <div id="pair"> --><html>no anchor</html>', '<html><body data-id="pair">no anchor</body></html>', "<html><script>const text='<a id=\"pair\">'</script></html>"):
            with self.subTest(html=html):
                self.page.write_text(html, encoding="utf-8")
                self.assertIn("local_link", self.codes())
        self.page.write_text('<HTML><BODY ID="pair">real anchor</BODY></HTML>', encoding="utf-8")
        self.assertTrue(checker.check(self.report)["passed"])

    def test_id_and_name_on_one_element_share_a_target(self):
        self.page.write_text('<html><a id="pair" name="pair">real anchor</a></html>', encoding="utf-8")
        self.write_report("| candidate | 待审 | [审核](review.html#pair) |")
        self.assertTrue(checker.check(self.report)["passed"])
        self.write_report("| candidate | 待审 | [审核](#pair) |", extra='\n<a id="pair" name="pair"></a>\n\n[对比](review.html)\n')
        self.assertTrue(checker.check(self.report)["passed"])
        self.page.write_text('<html><a id="pair"></a><a name="pair"></a></html>', encoding="utf-8")
        self.write_report("| candidate | 待审 | [审核](review.html#pair) |")
        self.assertIn("local_link", self.codes())

    def test_markdown_data_id_is_not_an_anchor(self):
        self.write_report("| candidate | 待审 | [审核](#pair) |", extra='\n<a data-id="pair"></a>\n[对比](review.html)\n')
        self.assertIn("local_link", self.codes())

    def test_receipt_output_cannot_replace_linked_evidence(self):
        payload = self.root / "evidence.json"
        original = '{"measured_ms": 1.2}\n'
        for existing in (True, False):
            with self.subTest(existing=existing):
                if existing:
                    payload.write_text(original, encoding="utf-8")
                elif payload.exists():
                    payload.unlink()
                self.write_report(extra="\n[测量](evidence.json)\n")
                result = subprocess.run([sys.executable, str(SOURCE), "--report", str(self.report), "--output", str(payload)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                if existing:
                    self.assertEqual(payload.read_text(), original)
                else:
                    self.assertFalse(payload.exists())

    def test_receipt_hardlink_cannot_overwrite_report_review_or_payload(self):
        self.write_report(extra="\n[原始数据](raw.json)\n")
        payload = self.root / "raw.json"
        payload.write_text('{"latency_ms": 1.2}\n')
        alias = self.root / "receipt.json"
        for original in (self.report, self.page, payload):
            with self.subTest(original=original.name):
                before = original.read_bytes()
                os.link(original, alias)
                result = subprocess.run([sys.executable, str(SOURCE), "--report", str(self.report), "--output", str(alias)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertEqual(original.read_bytes(), before)
                alias.unlink()

    def test_reference_links_and_url_query(self):
        self.write_report("| candidate | 待审 | [审核][pair] |", extra="\n[pair]: review.html?left=A&right=B\n")
        self.assertTrue(checker.check(self.report)["passed"])

    def test_repeated_reference_labels_use_the_first_visible_definition(self):
        for label, duplicate in (("pair", "PAIR"), ("pair key", "PAIR  KEY")):
            with self.subTest(label=label):
                self.write_report(f"| candidate | 待审 | [审核][{label}] |", extra=f"\n[{label}]: missing.html\n[{duplicate}]: review.html\n")
                self.assertIn("local_link", self.codes())
                self.assertIn("missing_review_entry", self.codes())
                self.write_report(f"| candidate | 待审 | [审核][{label}] |", extra=f"\n    [{label}]: missing.html\n[{label}]: review.html\n[{duplicate}]: missing.html\n")
                self.assertTrue(checker.check(self.report)["passed"])

    def test_broken_or_empty_review_page_rejected(self):
        self.write_report()
        self.page.unlink()
        self.assertIn("local_link", self.codes())
        self.page.write_text("", encoding="utf-8")
        self.assertIn("missing_review_entry", self.codes())

    def test_external_link_not_certified_offline(self):
        self.write_report("| candidate | 待审 | [审核](https://example.test/review) |")
        self.assertIn("missing_review_entry", self.codes())

    def test_unresolved_applicability_and_unwired_only_rejected(self):
        self.write_report("| method | 关联待核 | 未接线 |")
        self.assertEqual(self.codes(), {"unresolved_relation", "unwired_disposition"})

    def test_evidenced_deferment_not_mistaken_for_pending_quality(self):
        self.write_report("| method | 未接线，暂缓 | [依据](#basis) |", extra="\n# basis\n已有证据及重开条件。\n")
        self.assertTrue(checker.check(self.report)["passed"])

    def test_examples_and_headers_not_pending_candidates(self):
        self.write_report("| method | 不选 | 无待审输出 |", extra="\n```md\n| fake | 待审 | 无入口 |\n```\n<!-- | fake | 关联待核 | -->\n")
        self.assertTrue(checker.check(self.report)["passed"])

    def test_missing_section_rejected(self):
        self.write_report()
        self.report.write_text(self.report.read_text().replace("# 4. section", "# section"), encoding="utf-8")
        self.assertIn("report_sections", self.codes())

    def test_receipt_revalidated_and_report_edit_invalidates_it(self):
        self.write_report()
        receipt = self.root / "CHECKS.json"
        checker.write_receipt(checker.check(self.report), receipt)
        self.assertTrue(checker.verify(receipt)["passed"])
        self.report.write_text(self.report.read_text() + "\nchanged\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            checker.verify(receipt)

    def test_review_page_edit_and_receipt_hash_tampering_rejected(self):
        self.write_report()
        receipt = self.root / "CHECKS.json"
        checker.write_receipt(checker.check(self.report), receipt)
        self.page.write_text("changed page", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            checker.verify(receipt)
        saved = json.loads(receipt.read_text())
        saved["inputs"] = {}
        receipt.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            checker.verify(receipt)

    def test_delivery_consumer_binds_expected_task_report(self):
        self.write_report()
        receipt = self.root / "CHECKS.json"
        checker.write_receipt(checker.check(self.report), receipt)
        self.assertTrue(checker.verify(receipt, self.report)["passed"])
        other = self.root / "OTHER.md"
        other.write_text(self.report.read_text(), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "another report"):
            checker.verify(receipt, other)
        result = subprocess.run([sys.executable, str(SOURCE), "--verify", str(receipt), "--expected-report", str(other)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)

    def test_failure_receipt_cannot_be_verified(self):
        self.write_report("| candidate | 待审 | missing |")
        receipt = self.root / "CHECKS.json"
        checker.write_receipt(checker.check(self.report), receipt)
        with self.assertRaisesRegex(ValueError, "no passing"):
            checker.verify(receipt)

    def test_cli_exit_status_and_verify(self):
        self.write_report("| candidate | 待审 | missing |")
        receipt = self.root / "CHECKS.json"
        failed = subprocess.run([sys.executable, str(SOURCE), "--report", str(self.report), "--output", str(receipt)], capture_output=True, text=True)
        self.assertEqual(failed.returncode, 1)
        self.write_report()
        passed = subprocess.run([sys.executable, str(SOURCE), "--report", str(self.report), "--output", str(receipt)], capture_output=True, text=True)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        verified = subprocess.run([sys.executable, str(SOURCE), "--verify", str(receipt)], capture_output=True, text=True)
        self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_literal_code_links_do_not_satisfy_pending_review(self):
        for entry in ("`[审核](review.html)`", "``[审核](review.html) `literal` ``"):
            with self.subTest(entry=entry):
                self.write_report(f"| candidate | 待审 | {entry} |")
                self.assertIn("missing_review_entry", self.codes())
        self.write_report("| candidate | 待审 | `[ignored](missing.html)` [审核](review.html) |", extra="\n`[标签][不存在]`\n")
        self.assertTrue(checker.check(self.report)["passed"])

    def test_code_anchors_do_not_hide_real_adjacent_anchors_or_heading_text(self):
        self.write_report("| candidate | 待审 | [审核](#fake) |", extra='\n`<a id="fake"></a>`\n[对比](review.html)\n')
        self.assertIn("local_link", self.codes())
        self.write_report(
            "| candidate | 待审 | [审核](#real) |",
            extra='\n`<a id="fake"></a>` <a id="real"></a> [对比](review.html)\n\n# Use `compile`\n\n[标题](#use-compile)\n',
        )
        self.assertTrue(checker.check(self.report)["passed"])

    def test_escaped_and_unclosed_backticks_leave_real_links_visible(self):
        for entry in (r"\`[审核](review.html)\`", "`unclosed [审核](review.html)"):
            with self.subTest(entry=entry):
                self.write_report(f"| candidate | 待审 | {entry} |")
                self.assertTrue(checker.check(self.report)["passed"])


if __name__ == "__main__":
    unittest.main()
