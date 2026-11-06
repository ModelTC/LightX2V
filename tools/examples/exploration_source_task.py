"""Runnable CPU example of a pinned, partial exploration consumer.

Creates demonstration source records, not model measurements or a complete
optimization. Run directly from the repository; see EXPLORATION_REPORT_CHECKS.md.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
CODE = [Path(__file__).resolve(), *[REPOSITORY / "tools" / name for name in (
    "__init__.py", "exploration_acceptance.py", "exploration_evidence.py", "check_exploration_report.py",
)]]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def initialize(root, acceptance, evidence):
    root.mkdir(parents=True, exist_ok=False)  # Do not overwrite an existing task.
    source, catalog, report = root / "INPUT.py", root / "MAP.md", root / "REPORT.md"
    source.write_text("# Demonstration source; read for identity, never executed.\n", encoding="utf-8")
    catalog.write_text("### Source identity\n", encoding="utf-8")
    inputs = acceptance.Inputs()
    facts = {"basis": evidence.source_reference(inputs, [source])}
    policy = dict(
        kind="exploration_policy_v1", task_id="source-consumer-example", scope_id="source-only-v1",
        scope_label="CPU接线示例的局部源码依据", report=str(report),
        catalog=dict(path=str(catalog), sha256=acceptance.digest(catalog)),
        items={"source-basis": dict(family="source-identity", label="源码依据", question="Which source bytes were inspected?", locations=["INPUT.py"], evidence=["basis"])},
        evidence={"basis": dict(kind="source", paths=[str(source)], subject=facts["basis"]["subject"])},
        artifacts={str(path): value for path, value in inputs.hashes.items()},
        claims=[dict(id="source-basis", block="basis", item="source-basis", evidence="basis", label="源码依据", reference="INPUT.py")],
        report_blocks={"basis": "| 项目 | 范围 | 判断依据 | 处置 |\n|---|---|---|---|"},
    )
    activity = dict(scope_id=policy["scope_id"], items={"source-basis": dict(
        disposition="defer", rationale="只核源码身份，尚无性能或适用性验证。",
        reopen_if="接入真实任务输入、执行记录和完成边界后重新评价。", evidence=["basis"],
    )})
    families = acceptance.catalog_families(catalog.read_text())
    blocks = acceptance.rendered_blocks(policy, families, activity, facts)
    text = "\n\n".join([
        "# 1. 任务与当前结论\n\n本例只演示局部证据验收；没有执行模型，也没有性能结论。",
        "# 2. 任务范围\n\n本地MAP仅为示例范围，不代替项目方法库或完整模型探索。\n\n"
        "<!-- exploration:coverage -->\n" + blocks["coverage"] + "\n<!-- /exploration:coverage -->\n\n"
        "<!-- exploration:basis -->\n" + blocks["basis"] + "\n<!-- /exploration:basis -->",
        "# 3. 优化结果\n\n未测。",
        "# 4. 组合结果\n\n未测。",
        "# 5. 必要分析\n\n源码身份检查不证明方法适用或暂缓理由充分。",
        "# 6. 质量、资源与有效性\n\n未运行模型，未采集质量、性能或显存。",
        "# 7. 交付与剩余工作\n\n局部验收不关闭父任务；真实模型的collector和完成入口仍须接线。",
    ]) + "\n"
    report.write_text(text, encoding="utf-8")
    policy["report_context"] = acceptance.report_context(text, {"coverage", *policy["report_blocks"]})
    write_json(root / "policy.json", policy)
    write_json(root / "work_package.json", dict(task_id=policy["task_id"], status="in_progress", exploration_fulfillment=activity))
    write_json(root / "binding.json", dict(policy_sha256=sha256(root / "policy.json"), code={str(path): sha256(path) for path in CODE}))


def current(root, binding_sha256, acceptance, evidence):
    inputs = acceptance.Inputs()
    binding = inputs.json(root / "binding.json", binding_sha256)
    for path, expected in binding["code"].items():
        inputs.pinned(path, expected)
    work = inputs.json(root / "work_package.json")
    policy, families = acceptance.load_policy(inputs, root / "policy.json", binding["policy_sha256"])
    facts = {"basis": evidence.source_reference(inputs, policy["evidence"]["basis"]["paths"])}
    result = acceptance.check(policy, families, work, facts, inputs, mode="delivery", selected=["source-basis"])
    acceptance.completion_state(work, result)
    if not result["passed"]:
        raise ValueError(json.dumps(result["errors"], ensure_ascii=False))
    return result, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "verify"))
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--binding-sha256")
    args = parser.parse_args()
    root = args.task_dir.resolve()
    if args.command == "verify":
        if not args.binding_sha256:
            parser.error("verify requires the binding SHA256 printed by init")
        binding_path = root / "binding.json"
        if sha256(binding_path) != args.binding_sha256:
            raise ValueError("pinned binding changed")
        binding = json.loads(binding_path.read_text())
        if set(binding["code"]) != {str(path) for path in CODE}:
            raise ValueError("consumer code inventory changed")
        for path in CODE:
            if sha256(path) != binding["code"][str(path)]:
                raise ValueError(f"pinned consumer dependency changed: {path}")

    # Verify code before importing the task's execution dependencies.
    sys.path.insert(0, str(REPOSITORY))
    from tools import exploration_acceptance as acceptance
    from tools import exploration_evidence as evidence

    if args.command == "init":
        initialize(root, acceptance, evidence)
        args.binding_sha256 = sha256(root / "binding.json")
    result, inputs = current(root, args.binding_sha256, acceptance, evidence)
    receipt = root / "ACCEPTANCE.json"
    if args.command == "init":
        acceptance.write_receipt(result, receipt, inputs)
    else:
        acceptance.verify_receipt(receipt, result)
    print(json.dumps(dict(passed=result["passed"], task_complete=result["task_complete"], binding_sha256=args.binding_sha256, report=str(root / "REPORT.md")), ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(1)
