"""Native-format CPU fixtures; these are not performance measurements."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from test_exploration_acceptance import write_json, write_module

from tools.benchmarks import operator_bench as producer
from tools.benchmarks.operator_backends import load_registry
from tools.exploration_acceptance import Inputs, render_claim
from tools.exploration_evidence import module_comparison, module_observation, operator_comparison


class NativeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def operator_rows(self):
        case = producer.generate_sweep("gemm", {"m": [2], "n": [4], "k": [8]}, "fixture", "bf16")["cases"][0]
        case["case_id"] = "ffn"
        registry = load_registry()
        options = dict(warmup=3, iterations=10, prewarm_seconds=0.0, device="cuda:0", moe_routing="balanced", seed=42, timing_mode=producer.TIMING_MODE, rate_schema=producer.RATE_SCHEMA)
        rows = []
        for backend, values in [("torch_linear", [10.0, 12.0, 14.0]), ("Default", [7.0, 8.0, 9.0])]:
            for i, value in enumerate(values):
                rows.append(
                    dict(
                        status="ok",
                        run_id=f"repeat-{i:03d}",
                        case=producer._runtime_case(case, registry.require(backend), options),
                        environment={"device": "cuda:0", "git_commit": "fixture"},
                        correctness={"checked": False},
                        metrics={"latency_ms_mean": value},
                    )
                )
        return rows

    def operator(self, rows, backends=("torch_linear", "Default")):
        path = self.root / "operator.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return operator_comparison(Inputs(), path, *backends, "ffn")

    def test_operator_uses_native_mean_and_preserves_synthetic_unchecked(self):
        fact = self.operator(self.operator_rows())
        self.assertEqual(fact["values"], dict(parent=12.0, candidate=8.0))
        self.assertEqual(fact["subject"]["input_kind"], "synthetic")
        self.assertEqual(fact["quality"], "unavailable")
        self.assertIs(fact["valid"], False)
        self.assertEqual(fact["subject"]["source"]["kind"], "synthetic_sweep")
        row = render_claim(dict(id="op", label="GEMM", reference="operator.jsonl"), fact, dict(disposition="retain"))
        self.assertIn("合成输入", row)
        self.assertIn("12.000000 → 8.000000ms；延迟降低33.33%", row)

    def test_operator_executable_input_identity_is_compared_and_published(self):
        original = self.operator_rows()
        for key, value in [("operator_family", "moe"), ("routing", {"expert_counts": [2, 2]}), ("replay", {"manifest": "B.json", "sha256": "b" * 64}), ("sparse", {"keep_ratio": 0.5})]:
            rows = copy.deepcopy(original)
            rows[-1]["case"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "operator input"):
                self.operator(rows)
        # Use a real accepted attention-suite representation for replay metadata;
        # the scalar raw samples below are still CPU fixtures, never GPU results.
        suite = producer.generate_sweep("dense_attention", {"batch": [1], "seq_q": [16], "seq_kv": [16], "heads": [1], "head_dim": [64]}, "replay", "bf16")
        suite["cases"][0]["replay"] = {"manifest": "A.json", "sha256": "a" * 64}
        producer.validate_suite(suite)
        replay_case = suite["cases"][0]
        replay_case["case_id"] = "ffn"
        registry = load_registry()
        backends = ("torch_sdpa", "flash_attn2")
        for row in original:
            backend = {"torch_linear": "torch_sdpa", "Default": "flash_attn2"}[row["case"]["backend"]]
            row["case"] = producer._runtime_case(replay_case, registry.require(backend), row["case"]["run"])
        fact = self.operator(original, backends)
        self.assertEqual(fact["subject"]["input_kind"], "real")
        self.assertEqual(fact["subject"]["input_identity"]["replay"], replay_case["replay"])
        original[-1]["case"]["replay"] = {"manifest": "B.json", "sha256": "b" * 64}
        with self.assertRaisesRegex(ValueError, "operator input"):
            self.operator(original, backends)

    def test_operator_source_annotations_may_change_without_changing_inputs(self):
        rows = self.operator_rows()
        original_source = copy.deepcopy(rows[0]["case"]["source"])
        for row in rows:
            if row["run_id"] == "repeat-002":
                row["case"]["source"]["note"] = "Updated provenance annotation, same operands."
        fact = self.operator(rows)
        self.assertEqual(fact["values"], {"parent": 12.0, "candidate": 8.0})
        self.assertEqual(len(fact["subject"]["provenance"]), 2)
        self.assertIn(original_source, fact["subject"]["provenance"])
        self.assertIn("note", fact["subject"]["provenance"][1])

    def test_operator_changed_seed_missing_or_duplicate_repetition_rejected(self):
        for mutation in ["seed", "missing", "duplicate"]:
            rows = self.operator_rows()
            if mutation == "seed":
                rows[-1]["case"]["run"]["seed"] = 43
            if mutation == "missing":
                rows.pop()
            if mutation == "duplicate":
                rows[-1]["run_id"] = rows[-2]["run_id"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.operator(rows)

    def test_operator_precision_environment_and_nonfinite_metric_rejected(self):
        original = self.operator_rows()
        for section, key, value in [("precision", "implementation", "other"), ("environment", "device", "cuda:1"), ("metrics", "latency_ms_mean", float("nan"))]:
            rows = copy.deepcopy(original)
            if section == "precision":
                rows[-1]["case"][section][key] = value
            else:
                rows[-1][section][key] = value
            with self.subTest(section=section), self.assertRaises(ValueError):
                self.operator(rows)

    def test_module_aggregate_is_recomputed_from_raw(self):
        args = write_module(self.root, "dense", [10.0, 12.0, 14.0, 16.0])
        fact = module_observation(Inputs(), *args)
        self.assertEqual(fact["median"], 13.0)
        report = Path(args[0])
        row = json.loads(report.read_text())
        row["latency_ms"]["median"] = 99.0
        write_json(report, row)
        with self.assertRaisesRegex(ValueError, "aggregate"):
            module_observation(Inputs(), *args)

    def test_module_factory_preserves_task_owned_config_and_rejects_mixed_repeats(self):
        parent = write_module(self.root, "parent", [10.0, 12.0, 14.0, 16.0])
        first = write_module(self.root, "first", [8.0, 8.0, 8.0, 8.0])
        second = write_module(self.root, "second", [4.0, 4.0, 4.0, 4.0])
        for key in ("selected", "candidate_id", "status", "reason"):
            for args, value in ((first, "A"), (second, "B")):
                report = json.loads(Path(args[0]).read_text())
                report["factory"] = {"config": {key: value}}
                write_json(Path(args[0]), report)
            with self.subTest(key=key):
                fact = module_comparison(Inputs(), [parent], [first])
                self.assertEqual(fact["subject"]["candidate"], {"config": {key: "A"}})
                self.assertIs(fact["valid"], False)
                with self.assertRaisesRegex(ValueError, "different actual candidates"):
                    module_comparison(Inputs(), [parent], [first, second])

    def test_module_inconsistent_rank_samples_and_input_mutation_rejected(self):
        args = write_module(self.root, "dense", [10.0, 12.0, 14.0, 16.0])
        path = Path(args[1])
        original = path.read_text()
        for mutation in ["count", "duplicate"]:
            row = json.loads(original)
            if mutation == "count":
                row["samples"].pop()
            if mutation == "duplicate":
                row["samples"][1]["iteration"] = 0
            path.write_text(json.dumps(row) + "\n")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                module_observation(Inputs(), *args)
        row = json.loads(original)
        row["correctness"]["input_unchanged"] = False
        path.write_text(json.dumps(row) + "\n")
        report_path = Path(args[0])
        report = json.loads(report_path.read_text())
        report["ranks"][0]["correctness"] = row["correctness"]
        write_json(report_path, report)
        with self.assertRaisesRegex(ValueError, "input was modified"):
            module_observation(Inputs(), *args)

    def test_module_requires_recorded_factory_and_complete_native_rank_group(self):
        args = write_module(self.root, "native", [10.0, 12.0, 14.0, 16.0])
        report_path, raw_path = map(Path, args)
        original = json.loads(report_path.read_text())
        report = copy.deepcopy(original)
        del report["factory"]
        report["candidate"] = {"name": "undocumented-alias"}
        write_json(report_path, report)
        with self.assertRaisesRegex(ValueError, "factory context"):
            module_observation(Inputs(), *args)
        report = copy.deepcopy(original)
        raw = json.loads(raw_path.read_text())
        raw["groups"]["measurement_ranks"] = [0, 1]
        report["ranks"][0]["groups"] = raw["groups"]
        write_json(report_path, report)
        raw_path.write_text(json.dumps(raw) + "\n")
        with self.assertRaisesRegex(ValueError, "complete measurement group"):
            module_observation(Inputs(), *args)

    def test_module_repeated_original_and_file_aliases_are_not_new_groups(self):
        parent = write_module(self.root, "parent", [10.0] * 4)
        candidate = write_module(self.root, "candidate", [5.0] * 4)
        aliases = [parent]
        for name, hardlink in [("symlink.jsonl", False), ("hardlink.jsonl", True)]:
            raw = self.root / name
            if hardlink:
                os.link(parent[1], raw)
            else:
                raw.symlink_to(parent[1])
            aliases.append([parent[0], str(raw)])
        for duplicate in aliases:
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(ValueError, "duplicate module run"):
                module_comparison(Inputs(), [parent, duplicate], [candidate])
        with self.assertRaisesRegex(ValueError, "duplicate module run"):
            module_comparison(Inputs(), [parent], [parent])

    def test_module_distinct_experiments_with_equal_contents_are_retained(self):
        first = write_module(self.root, "first", [10.0] * 4)
        second = write_module(self.root, "second", [10.0] * 4)
        report = json.loads(Path(first[0]).read_text())
        write_json(Path(second[0]), report)
        self.assertEqual(Path(first[1]).read_bytes(), Path(second[1]).read_bytes())
        candidate = write_module(self.root, "candidate", [5.0] * 4)
        fact = module_comparison(Inputs(), [first, second], [candidate])
        self.assertEqual(fact["values"], dict(parent=10.0, candidate=5.0))
        self.assertIn("2/1组", fact["statistic"])
        self.assertIs(fact["valid"], False)


if __name__ == "__main__":
    unittest.main()
