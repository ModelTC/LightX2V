"""Validate a declared exploration scope against observed evidence and report facts.

The task consumer owns the pinned policy and evidence collector. This module does
not execute models, infer discovery completeness, or approve perceptual quality.
Pins describe one accepted task revision. Model/platform/project changes may
require a revised policy, collector or checker; validate the affected contract
before rebinding, while retaining compatible historical evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from tools import check_exploration_report as report_check

DISPOSITIONS = {"retain": "保留", "reject": "不选", "defer": "暂缓", "not_applicable": "不适用", "outside_scope": "范围外", "blocked": "外部阻塞", "open": "待核", "running": "执行中"}
COMPLETE_STATES = {"completed", "passed", "passed_with_scope"}
INCOMPLETE_STATES = {"in_progress", "pending", "failed", "blocked", "paused", "budget_exhausted", "partial", "deferred"}
LIMITS = [
    "Checks the pinned discovered scope, observed evidence bindings and protected report declarations; does not prove discovery completeness or deferral sufficiency.",
    "The pinned task collector must read native production records; an Agent-authored mapping alone is not execution evidence.",
    "Only protected facts/coverage blocks are mechanically checked; explanatory prose still needs semantic review.",
    "Same-permission edits to consumer policy/code and host final messages are outside this consumer's enforcement boundary.",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _unique_pairs(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"duplicate JSON key: {key}")
        data[key] = value
    return data


def _nonfinite(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        _nonfinite(value)
    return parsed


class Inputs:
    """Record files actually read, independently of the mutable activity ledger."""

    def __init__(self):
        self.hashes = {}
        self.paths = set()
        self.bytes_read = 0
        self.evidence_baseline = None

    def read(self, path):
        path = Path(path).resolve()
        self.paths.add(path)
        data = path.read_bytes()
        self.bytes_read += len(data)
        value = hashlib.sha256(data).hexdigest()
        if path in self.hashes and self.hashes[path] != value:
            raise ValueError(f"input changed while checking: {path}")
        self.hashes[path] = value
        return data

    def json(self, path, expected_sha256=None):
        data = self.read(path) if expected_sha256 is None else self.pinned(path, expected_sha256)
        return json.loads(data, object_pairs_hook=_unique_pairs, parse_float=_finite_float, parse_constant=_nonfinite)

    def jsonl(self, path):
        return [json.loads(line, object_pairs_hook=_unique_pairs, parse_float=_finite_float, parse_constant=_nonfinite) for line in self.read(path).decode().splitlines() if line.strip()]

    def pinned(self, path, expected):
        data = self.read(path)
        if self.hashes[Path(path).resolve()] != expected:
            raise ValueError(f"pinned input changed: {path}")
        return data

    def unchanged(self):
        for path, expected in self.hashes.items():
            if digest(path) != expected:
                raise ValueError(f"input changed while checking: {path}")

    def mismatched_artifacts(self, artifacts):
        return [path for path, actual in self.hashes.items() if str(path) in artifacts and actual != artifacts[str(path)]]

    def unregistered_sources(self, artifacts):
        if self.evidence_baseline is None:
            raise ValueError("load the fixed policy before collecting native evidence")
        return [path for path in set(self.hashes) - self.evidence_baseline if str(path) not in artifacts]


def catalog_families(text):
    titles = re.findall(r"^###\s+(.+)$", text, re.M)
    ids = [next(iter(report_check.anchors(["# " + title]))) for title in titles]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("method catalog needs distinct family headings")
    return dict(zip(ids, titles))


def load_policy(inputs, path, expected_sha256):
    """Validate this task revision, not a permanent cross-model inventory.

    Callers adapt the contract to current producers/consumers before supplying
    new bindings; changing hashes alone does not validate a semantic migration.
    """
    path = Path(path).resolve()
    policy = inputs.json(path, expected_sha256)
    if policy.get("kind") != "exploration_policy_v1":
        raise ValueError("unsupported exploration policy")
    for field in ("task_id", "scope_id", "scope_label", "items", "evidence", "claims", "artifacts"):
        if not policy.get(field):
            raise ValueError(f"policy missing {field}")
    inputs.paths.add(path)
    catalog = policy["catalog"]
    text = inputs.pinned(catalog["path"], catalog["sha256"]).decode()
    # Protect every registered original even when a partial check does not read it.
    inputs.paths.update(Path(value).resolve() for value in policy["artifacts"])
    inputs.paths.add(Path(policy["report"]).resolve())
    families = catalog_families(text)
    covered = {item["family"] for item in policy["items"].values()}
    if covered != set(families):
        raise ValueError(f"policy/catalog family mismatch: missing={set(families) - covered}, unknown={covered - set(families)}")
    for item_id, item in policy["items"].items():
        if not item.get("question") or not item.get("locations"):
            raise ValueError(f"item lacks question/location: {item_id}")
        if item.get("parent") and item["parent"] not in policy["items"]:
            raise ValueError(f"unknown parent item: {item_id}")
        chain = {item_id}
        parent = item.get("parent")
        while parent:
            if parent in chain:
                raise ValueError(f"cyclic item parent: {item_id}")
            chain.add(parent)
            parent = policy["items"][parent].get("parent")
        if not item.get("evidence") or len(item["evidence"]) != len(set(item["evidence"])) or not set(item["evidence"]) <= set(policy["evidence"]):
            raise ValueError(f"item evidence absent from pinned policy: {item_id}")
    claim_ids = set()
    anchors = {"coverage", *policy.get("report_blocks", {})}
    for claim in policy["claims"]:
        if not re.fullmatch(r"[a-z0-9_-]+", claim["id"]) or claim["id"] in claim_ids or claim["id"] == "coverage":
            raise ValueError("invalid/duplicate report claim id")
        claim_ids.add(claim["id"])
        for anchor in [claim["id"], *claim.get("aliases", [])]:
            if not re.fullmatch(r"[a-z0-9_-]+", anchor) or anchor in anchors:
                raise ValueError("invalid/duplicate report anchor")
            anchors.add(anchor)
        if claim.get("block") not in policy.get("report_blocks", {}):
            raise ValueError("report claim has no registered table block")
        if claim["item"] not in policy["items"] or claim["evidence"] not in policy["items"][claim["item"]]["evidence"]:
            raise ValueError("report claim is not bound to its discovered item")
    if set(policy["report_blocks"]) != {claim["block"] for claim in policy["claims"]} or "coverage" in policy["report_blocks"]:
        raise ValueError("every report table block must consume registered claims; coverage is reserved")
    inputs.evidence_baseline = set(inputs.hashes)
    return policy, families


def subset(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and subset(actual[key], value) for key, value in expected.items())
    return actual == expected and type(actual) is type(expected)


def block_text(text, block_id):
    pattern = re.compile(r"<!-- exploration:" + re.escape(block_id) + r" -->\n([\s\S]*?)\n<!-- /exploration:" + re.escape(block_id) + r" -->")
    found = pattern.findall(text)
    if len(found) != 1:
        raise ValueError(f"expected exactly one protected report block: {block_id}")
    return found[0]


def replace_blocks(text, blocks):
    for block_id, value in blocks.items():
        pattern = re.compile(r"(<!-- exploration:" + re.escape(block_id) + r" -->\n)[\s\S]*?(\n<!-- /exploration:" + re.escape(block_id) + r" -->)")
        if len(pattern.findall(text)) != 1:
            raise ValueError(f"expected exactly one protected report block: {block_id}")
        text = pattern.sub(lambda match: match.group(1) + value + match.group(2), text)
    return text


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_claim(spec, fact, activity):
    """A small fact row, not a full report renderer or prose interpreter."""
    label = _cell(spec["label"])
    anchor = f'<a id="{spec["id"]}"></a>' + "".join(f'<a id="{value}"></a>' for value in spec.get("aliases", []))
    subject = fact["subject"]
    scope = {"operator": "算子", "module": "模块/block/阶段", "request": "完整请求", "source": "源码/前提"}[subject["scope"]]
    if subject["scope"] == "module":
        target = subject.get("identity", {}).get("target", {})
        boundary = target.get("completion_boundary", target.get("boundary", target.get("scope", target.get("module", ""))))
        if boundary:
            scope += f"（{_cell(boundary)}）"
    input_kind = {"real": "真实输入", "synthetic": "合成输入", "source": "源码依据"}[subject["input_kind"]]
    decision = DISPOSITIONS[activity["disposition"]]
    reference = f"<{spec['reference']}>"
    if fact["kind"] == "quality":
        qa = {"strict_equal": "数值严格相同", "pending_review": "输出有变化，待审", "unavailable": "数值未闭合"}[fact["quality"]]
        return f"| {anchor}{label} | {scope}；{qa}；生产采用未批准 | [审核](<{fact['review']}>) |"
    if fact["kind"] == "metric":
        values = fact["values"]
        return f"| {anchor}{label} | {values['parent']:.3f} | {values['candidate']:.3f} | {values['candidate'] - values['parent']:+.3f} |"
    if fact["kind"] == "measurement":
        values = fact["values"]
        precision = 6 if subject["scope"] == "operator" else 3
        unit = {"s": "秒", "ms": "ms", "us": "µs", "ns": "ns"}[fact["unit"]]
        if values.get("parent") is None:
            timing = f"{values['candidate']:.{precision}f}{unit}；无同组父对照"
        else:
            ratio = 100 * (1 - values["candidate"] / values["parent"])
            timing = f"{values['parent']:.{precision}f} → {values['candidate']:.{precision}f}{unit}；延迟{'降低' if ratio >= 0 else '增加'}{abs(ratio):.2f}%"
        ratio = None if values.get("parent") is None else 100 * (1 - values["candidate"] / values["parent"])
        names = f"{_cell(spec.get('parent_label', '父方案'))} → {_cell(spec.get('candidate_label', spec['label']))}"
        qa = {"strict_equal": "数值严格相同", "pending_review": "数值有变化", "unavailable": "数值验证未闭合"}[fact["quality"]]
        if subject["scope"] == "request" and fact["quality"] == "pending_review" and activity["disposition"] == "retain":
            qa += f"；[待审](<{fact['review']}>)"
        if spec.get("layout") == "combination":
            original = values.get("original")
            cumulative = "未测直接累计" if original is None else f"{100 * (1 - values['candidate'] / original):.2f}%"
            step = f"父对照未测 → {values['candidate']:.3f}" if ratio is None else f"{values['parent']:.3f} → {values['candidate']:.3f}"
            increment = "未测单步" if ratio is None else f"{ratio:+.2f}%"
            return f"| {anchor}{label} | {step} | {increment} | {cumulative} | {decision}；{qa}；[证据]({reference}) |"
        if spec.get("layout") == "quality":
            entry = f"[审核](<{fact['review']}>)" if fact.get("review") else f"[数值原记录]({reference})"
            return f"| {anchor}{label} | {scope}；{qa}；生产采用未批准 | {entry} |"
        if spec.get("layout") == "results":
            if any(not isinstance(spec.get(key), str) or not spec[key].strip() for key in ("parent_label", "change")):
                raise ValueError("results layout needs an explicit parent_label and change description")
            return f"| {anchor}{label} | {names}；{_cell(spec['change'])} | {scope}，{input_kind}；{_cell(fact['statistic'])} | {timing} | {decision}；{qa}；[证据]({reference}) |"
        return f"| {anchor}{label} | {scope}，{input_kind}；{names}；{_cell(fact['statistic'])} | {timing} | {decision}；{qa}；[证据]({reference}) |"
    return f"| {anchor}{label} | {scope}，{input_kind} | {_cell(activity['rationale'])} | {decision}；[依据]({reference}) |"


def render_coverage(policy, families, ledger):
    rows = ["| 方法族 | 实际位置／必要子项 | 当前处置 |", "|---|---|---|"]
    for family, title in families.items():
        selected = [(key, item) for key, item in policy["items"].items() if item["family"] == family]
        parts = []
        for key, item in selected:
            state = ledger.get(key, {})
            status = DISPOSITIONS.get(state.get("disposition"), "缺记录")
            targets = [claim for claim in policy["claims"] if claim["item"] == key]
            name = _cell(item["label"])
            if targets:
                name = f"[{name}](#{targets[0]['id']})"
            reason = _cell(state.get("rationale", "尚无处置记录"))
            parts.append(f"{name}：{status}；{reason}")
        positions = sorted({location for _, item in selected for location in item["locations"]})
        rows.append(
            f"| {_cell(title)} | {_cell('、'.join(positions))}<br>{'<br>'.join(parts)} | "
            + (
                "已逐项处置"
                if all(ledger.get(key, {}).get("disposition") in DISPOSITIONS and ledger[key]["disposition"] not in {"open", "running", "blocked"} for key, _ in selected)
                else "仍有未闭合项"
            )
            + " |"
        )
    return "\n".join(rows)


def rendered_blocks(policy, families, activity, facts, selected=None):
    ledger = activity["items"]
    blocks = {"coverage": "\n" + render_coverage(policy, families, ledger) + "\n"}
    groups = {}
    for spec in policy["claims"]:
        if selected is None or spec["item"] in selected:
            groups.setdefault(spec["block"], []).append(render_claim(spec, facts[spec["evidence"]], ledger[spec["item"]]))
    for name, rows in groups.items():
        header = policy["report_blocks"][name]
        header_lines = [line for line in header.splitlines() if line.strip()]
        if len(header_lines) != 2:
            raise ValueError(f"report table needs a header and separator: {name}")
        columns = set()
        for line in [*header_lines, *rows]:
            line = line.strip()
            if not line:
                continue
            # A separator follows an even number of backslashes; remove only actual outer separators.
            separators = [match.end() - 1 for match in re.finditer(r"(?<!\\)(?:\\\\)*\|", line)]
            count = len(separators) + 1
            if separators and separators[0] == 0:
                count -= 1
            if separators and separators[-1] == len(line) - 1:
                count -= 1
            columns.add(count)
        if len(columns) != 1:
            raise ValueError(f"report table column mismatch: {name}")
        blocks[name] = "\n" + header + "\n" + "\n".join(rows) + "\n"
    return blocks


def report_context(text, protected_blocks=()):
    """Freeze critical-section context; table facts are validated separately.

    Analysis sections remain editable and need semantic review. This prevents a
    second unbound result table from being added beside the verified table.
    """
    raw = text.splitlines(keepends=True)
    # Use the same visible headings as the structural checker and formal template.
    starts = {}
    for index, line in enumerate(report_check.visible_markdown_lines(text)):
        match = re.match(r"^#{1,6}\s+([1-7])\.\s", line)
        if match:
            number = int(match.group(1))
            if number in starts:
                raise ValueError("duplicate numbered report section")
            starts[number] = index
    if list(starts) != list(range(1, 8)):
        raise ValueError("critical surface requires seven numbered report sections")
    result = {}
    for number in (1, 3, 4, 6):
        end = starts.get(number + 1, len(raw))
        body = "".join(raw[starts[number] : end])
        for block_id in protected_blocks:
            name = re.escape(block_id)
            body = re.sub(r"(<!-- exploration:" + name + r" -->\n)[\s\S]*?(\n<!-- /exploration:" + name + r" -->)", r"\1\2", body)
        result[str(number)] = hashlib.sha256(body.strip().encode()).hexdigest()
    return result


def check(policy, families, work_package, facts, inputs, mode="task", selected=None):
    errors = []
    activity = work_package.get("exploration_fulfillment", {})
    expected = set(policy["items"])
    ledger = activity.get("items", {})
    chosen = expected if mode == "task" else set(selected or [])
    if mode not in {"task", "delivery"} or not chosen or not chosen <= expected:
        raise ValueError("unknown/empty completion scope")

    def error(item, code, message):
        errors.append({"item": item, "code": code, "message": message})

    if work_package.get("task_id") != policy["task_id"] or activity.get("scope_id") != policy["scope_id"]:
        error("scope", "identity", "activity belongs to a different task/scope")
    if set(ledger) - expected or (mode == "task" and expected - set(ledger)):
        error("scope", "inventory", f"missing={sorted(expected - set(ledger))}; unknown={sorted(set(ledger) - expected)}")
    for item_id in sorted(chosen):
        item = policy["items"][item_id]
        state = ledger.get(item_id, {})
        disposition = state.get("disposition")
        if disposition not in DISPOSITIONS or disposition in {"open", "running", "blocked"}:
            error(item_id, "unclosed", "necessary item remains open/running/blocked")
            continue
        if not state.get("rationale", "").strip() or state["rationale"].strip() in {"暂缓", "未接线", "defer", "not wired"}:
            error(item_id, "rationale", "give the actual basis, not the disposition label")
        if disposition in {"defer", "outside_scope"} and not state.get("reopen_if", "").strip():
            error(item_id, "reopen", "deferred/scope-limited item needs a reopening condition")
        refs = state.get("evidence", [])
        if not refs or len(refs) != len(set(refs)) or not set(refs) <= set(item["evidence"]):
            error(item_id, "evidence_refs", "evidence absent, duplicated, or not bound to this discovered item")
            continue
        if disposition in {"retain", "reject"} and not any(facts.get(key, {}).get("kind") == "measurement" for key in refs):
            error(item_id, "evidence_strength", "a measured disposition cannot be supported only by references/metrics")
        for evidence_id in refs:
            spec = policy["evidence"][evidence_id]
            fact = facts.get(evidence_id)
            if fact is None or fact.get("kind") != spec["kind"] or not subset(fact.get("subject"), spec["subject"]):
                error(item_id, "evidence_identity", f"native source does not match expected object: {evidence_id}")
                continue
            if not fact.get("sources") or not all(Path(path).resolve() in inputs.hashes and str(Path(path).resolve()) in policy["artifacts"] for path in fact["sources"]):
                error(item_id, "unread_source", f"collector did not read the cited original sources: {evidence_id}")
            if fact["kind"] in {"measurement", "quality"} and fact.get("quality") not in {"strict_equal", "pending_review", "unavailable"}:
                error(item_id, "quality", "unknown numerical verification state")
            if fact["kind"] == "measurement":
                if fact.get("unit") not in {"s", "ms", "us", "ns"}:
                    error(item_id, "measurement_unit", "unsupported measurement time unit")
                values = fact.get("values", {})
                if fact.get("valid") is not True or not all(
                    type(v) in {int, float} and math.isfinite(v) and v > 0 for v in [values.get("candidate"), *([] if values.get("parent") is None else [values["parent"]])]
                ):
                    error(item_id, "measurement", "measurement/normal samples/resource checks are not valid")
                if disposition == "retain" and fact["subject"]["scope"] == "request" and fact.get("quality") == "pending_review" and not fact.get("review"):
                    error(item_id, "review_binding", "request research output needs a bound parent/candidate review entry")
        if state.get("adoption", "research") != "research":
            error(item_id, "adoption", "this research acceptance does not grant production or perceptual approval")
    for claim in policy["claims"]:
        if claim["item"] in chosen and claim["evidence"] not in ledger.get(claim["item"], {}).get("evidence", []):
            error(claim["id"], "claim_evidence", "report claim uses evidence not consumed by its item")
        if claim["item"] in chosen and claim["evidence"] in facts:
            target = (Path(policy["report"]).parent / claim["reference"]).resolve()
            if str(target) not in facts[claim["evidence"]].get("sources", []):
                error(claim["id"], "claim_reference", "displayed evidence link is not an observed source for this fact")
    if policy.get("lifecycle") is not None:
        from tools import exploration_lifecycle

        inputs.read(Path(exploration_lifecycle.__file__).resolve())
        _, review_errors = exploration_lifecycle.check_review(policy, inputs, None if mode == "task" else chosen, work_package=work_package)
        errors.extend(review_errors)
        errors.extend(exploration_lifecycle.check_adoption(policy, work_package, None if mode == "task" else chosen))
        impact = exploration_lifecycle.assess_changes(policy, inputs, None if mode == "task" else chosen)
        for change in impact["changes"]:
            errors.append(
                {
                    "item": ",".join(change["items"]) or "delivery",
                    "code": "dependency_changed",
                    "message": f"{change['path']}: reassess {','.join(change['domains'])}; historical measurements remain valid only in their old scope",
                }
            )
    remaining = sorted(key for key in expected if ledger.get(key, {}).get("disposition") not in DISPOSITIONS or ledger[key]["disposition"] in {"open", "running", "blocked"})
    if mode == "task":
        action_rows = work_package.get("next_actions", [])
        action_ids = [action["id"] for action in action_rows if action.get("id")]
        if len(action_ids) != len(set(action_ids)):
            error("scope", "next_action", "duplicate next-action ID cannot shadow unfinished work")
        actions = {action.get("id"): action for action in action_rows}
        for action_id in policy.get("required_action_ids", []):
            if actions.get(action_id, {}).get("execution_state") not in {"completed", "complete_in_declared_scope"}:
                error("scope", "next_action", f"required executable action is missing or unfinished: {action_id}")
        for action in work_package.get("next_actions", []):
            if action.get("required_for_scope") == policy["scope_id"] and action.get("execution_state") not in {"completed", "complete_in_declared_scope"}:
                error("scope", "next_action", "required executable action remains")
    for path in inputs.unregistered_sources(policy["artifacts"]):
        error("evidence", "unregistered_source", f"collector consumed an unregistered original input: {path}")
    report = Path(policy["report"]).resolve()
    text = inputs.read(report).decode()
    if report_context(text, {"coverage", *policy["report_blocks"]}) != policy.get("report_context"):
        error("report", "unbound_declaration", "critical report context changed or an unbound result declaration was added")
    if not errors:
        rendered = rendered_blocks(policy, families, activity, facts) if mode == "task" else {"coverage": "\n" + render_coverage(policy, families, ledger) + "\n"}
        for block_id, expected_text in rendered.items():
            try:
                if block_text(text, block_id) != expected_text:
                    error(block_id, "report_fact", "published declaration/coverage differs from observed facts")
            except ValueError as exc:
                error(block_id, "report_fact", str(exc))
        if mode == "delivery":
            for spec in policy["claims"]:
                if spec["item"] in chosen:
                    row = render_claim(spec, facts[spec["evidence"]], ledger[spec["item"]])
                    if row not in block_text(text, spec["block"]).splitlines():
                        error(spec["id"], "report_fact", "partial-delivery declaration differs from its observed fact")
    structure = report_check.check(report)
    for path, value in structure["inputs"].items():
        if path in inputs.hashes and inputs.hashes[path] != value:
            raise ValueError(f"input changed during report checking: {path}")
        inputs.hashes[path] = value
    inputs.paths.update(structure["local_targets"])
    errors.extend({"item": "report", **row} for row in structure["errors"])
    inputs.read(Path(__file__).resolve())
    # The existing structural checker reads additional navigation targets. They
    # are bound in the receipt, but only native collector reads may supply facts.
    for path in inputs.mismatched_artifacts(policy["artifacts"]):
        error("evidence", "artifact_identity", f"original evidence changed: {path}")
    inputs.unchanged()
    return {
        "kind": "exploration_acceptance_v1",
        "passed": not errors,
        "task_complete": not errors and mode == "task" and not remaining,
        "mode": mode,
        "task_id": policy["task_id"],
        "scope_id": policy["scope_id"],
        "scope_label": policy["scope_label"],
        "checked_items": sorted(chosen),
        "remaining_items": remaining,
        "errors": errors,
        "inputs": {str(path): value for path, value in inputs.hashes.items()},
        "read_bytes": inputs.bytes_read,
        "limitations": LIMITS,
        "evidence_limits": {key: fact.get("limits", []) for key, fact in facts.items() if any(key in ledger.get(item, {}).get("evidence", []) for item in chosen)},
    }


def write_receipt(result, path, inputs):
    path = Path(path).resolve()
    if path in inputs.paths or path in inputs.hashes:
        raise ValueError("acceptance receipt cannot overwrite referenced evidence/report")
    if path.exists():
        identity = path.stat()
        for original in inputs.paths | set(inputs.hashes):
            if original.exists():
                info = original.stat()
                if (identity.st_dev, identity.st_ino) == (info.st_dev, info.st_ino):
                    raise ValueError("acceptance receipt cannot overwrite a hard-linked original")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_receipt(receipt, current):
    old = json.loads(Path(receipt).read_text(), object_pairs_hook=_unique_pairs, parse_float=_finite_float, parse_constant=_nonfinite)
    if old.get("kind") != "exploration_acceptance_v1" or old.get("passed") is not True or not current["passed"]:
        raise ValueError("no current passing exploration acceptance")
    for key in ("task_id", "scope_id", "mode", "checked_items", "task_complete", "inputs"):
        if old.get(key) != current[key]:
            raise ValueError(f"acceptance is stale or belongs to a different scope: {key}")
    return current


def completion_state(work_package, result):
    """A caller's successful partial delivery cannot finish the parent task."""
    if work_package.get("task_id") != result["task_id"] or work_package.get("exploration_fulfillment", {}).get("scope_id") != result["scope_id"]:
        raise ValueError("completion result belongs to another task/scope")
    state = work_package["status"]
    if state not in COMPLETE_STATES | INCOMPLETE_STATES:
        raise ValueError(f"unknown task state needs explicit consumer mapping: {state}")
    if state in COMPLETE_STATES and (not result["passed"] or not result["task_complete"] or result["mode"] != "task"):
        raise ValueError("partial/invalid acceptance cannot close the parent task")
    return state
