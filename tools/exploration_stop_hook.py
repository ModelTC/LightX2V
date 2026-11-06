"""Codex hook for a fixed completion consumer; install/trust belongs to its host.

Continuation does not erase a visible reply or isolate code running as this UID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invoke(binding_path, expected):
    if digest(binding_path) != expected:
        raise ValueError("fixed completion binding changed")
    binding = json.loads(Path(binding_path).read_text())
    if not isinstance(binding, dict) or binding.get("kind") != "exploration_consumer_binding_v1":
        raise ValueError("unknown completion binding")
    if digest(binding["entry"]) != binding["entry_sha256"]:
        raise ValueError("registered completion entry changed")
    if not binding.get("checkers"):
        raise ValueError("mandatory completion checkers missing")
    for path, sha in binding["checkers"].items():
        if digest(path) != sha:
            raise ValueError("registered checker changed: " + path)
    if digest(binding["policy"]) != binding["policy_sha256"]:
        raise ValueError("registered scope policy changed")
    work = json.loads(Path(binding["work_package"]).read_text())
    if not isinstance(work, dict) or work.get("task_id") != binding["task_id"]:
        raise ValueError("bound task identity changed")
    if work.get("status") in {"blocked", "paused", "budget_exhausted"}:
        if not isinstance(work.get("stop_reason"), str) or not work["stop_reason"].strip():
            raise ValueError("incomplete stop needs its actual blocker or user limit")
        if work["status"] == "blocked":
            unfinished = [row for row in work.get("next_actions", []) if row.get("execution_state") not in {"completed", "complete_in_declared_scope"}]
            if any(row.get("execution_state") != "blocked" or not row.get("reason", "").strip() for row in unfinished):
                raise ValueError("blocked task still has an executable or unexplained next action")
        return work  # Incomplete stop grants neither an acceptance receipt nor completion.
    result = subprocess.run(
        [sys.executable, binding["entry"], "--binding", str(binding_path), "--work-package", binding["work_package"], "--verify", binding["receipt"]], capture_output=True, text=True, timeout=60
    )
    if result.returncode:
        raise ValueError((result.stderr + result.stdout)[-12000:])
    if binding.get("parent_completion_allowed") is not True:
        raise ValueError("local-scope pass cannot close the broader task")
    if work.get("status") not in {"completed", "passed", "passed_with_scope"}:
        pending = [row.get("action", row.get("id")) for row in work.get("next_actions", []) if row.get("execution_state") not in {"completed", "complete_in_declared_scope"}]
        raise ValueError("continue authorized parent actions: " + json.dumps(pending, ensure_ascii=False))
    return work


def hook(event, binding_path, expected):
    if not isinstance(event, dict):
        raise ValueError("hook event must be a JSON object")
    name = event.get("hook_event_name")
    if name == "Interrupt":
        return {}
    if name not in {"Stop", "SessionStart"}:
        raise ValueError("unsupported exploration lifecycle event")
    try:
        work = invoke(binding_path, expected)
    except (OSError, ValueError, TypeError, AttributeError, KeyError, subprocess.TimeoutExpired) as error:
        reason = "Exploration completion was not accepted. Diagnose and continue authorized work; preserve raw evidence.\n" + str(error)
        if name == "SessionStart":
            return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": reason}}
        return {"decision": "block", "reason": reason}
    if name == "SessionStart" and work.get("status") in {"blocked", "paused", "budget_exhausted"}:
        return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "Exploration remains incomplete (" + work["status"] + "): " + work["stop_reason"]}}
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    args = parser.parse_args()
    try:
        result = hook(json.load(sys.stdin), args.binding, args.binding_sha256)
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
        result = {"decision": "block", "reason": "Exploration lifecycle input/configuration is invalid: " + str(error)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
