"""Scoped semantic review and dependency changes for exploration acceptance.

No model runner or state database: the consumer pins review and dependencies;
explicit reconciliation updates the caller's existing activity record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REVIEW_TOPICS = {"discovery", "priority", "granularity", "stop_or_expand"}
KINDS = {"source", "config", "input", "parent", "approval", "catalog", "report", "checker", "cost", "evidence"}


def inventory_digest(policy):
    value = {key: policy[key] for key in ("task_id", "scope_id", "items", "catalog", "evidence")}
    value["candidate_dependencies"] = policy.get("lifecycle", {}).get("candidates", {})
    value["input_dependencies"] = [{**dependency, "sha256": policy["artifacts"][str(Path(dependency["path"]).resolve())]} for dependency in policy.get("lifecycle", {}).get("dependencies", [])]
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _descendants(candidates, roots):
    selected = set(roots)
    unknown = selected - set(candidates)
    if unknown:
        raise ValueError(f"unknown dependent candidate: {sorted(unknown)}")
    for name, candidate in candidates.items():
        if not set(candidate.get("parents", [])) <= set(candidates):
            raise ValueError(f"candidate has unknown parent: {name}")

    # Validate the dependency direction as well as finding its affected closure.
    finished = set()

    def visit(name, chain):
        if name in chain:
            raise ValueError("cyclic candidate dependency")
        if name in finished:
            return
        for parent in candidates[name].get("parents", []):
            visit(parent, chain | {name})
        finished.add(name)

    for name in candidates:
        visit(name, set())
    while True:
        expanded = selected | {name for name, value in candidates.items() if set(value.get("parents", [])) & selected}
        if expanded == selected:
            return selected
        selected = expanded


def _dependency_scope(policy, value):
    if value["kind"] not in KINDS or not set(value.get("items", [])) <= set(policy["items"]):
        raise ValueError("invalid dependency kind/item binding")
    candidates = policy.get("lifecycle", {}).get("candidates", {})
    affected_candidates = _descendants(candidates, value.get("candidates", []))
    chosen = set(value.get("items", []))
    for name in affected_candidates:
        chosen.update(candidates[name].get("items", []))
    if not chosen <= set(policy["items"]):
        raise ValueError("candidate refers to unknown affected item")
    return chosen, affected_candidates


def assess_changes(policy, inputs, selected=None):
    """Report changed dependencies for reassessment; retain historical facts.

    A mismatch does not reject a new capability or mandate a model rerun. The
    caller adapts the affected task contract, validates it and refreshes bindings.
    """
    contract = policy.get("lifecycle", {})
    candidates = contract.get("candidates", {})
    _descendants(candidates, set())
    changes = []
    for value in contract.get("dependencies", []):
        kind = value["kind"]
        chosen, affected_candidates = _dependency_scope(policy, value)
        if selected is not None and kind not in {"report", "checker", "catalog"} and chosen and not chosen & set(selected):
            continue
        path = Path(value["path"]).resolve()
        expected = policy["artifacts"][str(path)]
        try:
            inputs.read(path)
            actual = inputs.hashes[path]
        except FileNotFoundError:
            actual = None
        if actual == expected:
            continue
        selected_items = chosen
        if kind in {"report", "checker"}:
            domains = ["delivery"]
        elif kind == "approval":
            domains = ["quality_adoption"]
        elif kind in {"catalog", "cost"}:
            domains = ["discovery_decision"]
            if kind == "catalog":
                selected_items.update(policy["items"])
        else:
            domains = ["current_evidence", "applicability", "quality_adoption"]
        changes.append(
            {
                "path": str(path),
                "kind": kind,
                "before": expected,
                "after": actual,
                "items": sorted(selected_items),
                "candidates": sorted(affected_candidates),
                "domains": domains,
            }
        )
    return {"kind": "exploration_change_assessment_v1", "task_id": policy["task_id"], "scope_id": policy["scope_id"], "changed": bool(changes), "changes": changes}


def apply_changes(work_package, assessment):
    """Explicit state mutation; checking itself never changes the work package."""
    activity = work_package["exploration_fulfillment"]
    if work_package["task_id"] != assessment["task_id"] or activity["scope_id"] != assessment["scope_id"]:
        raise ValueError("change assessment belongs to another activity")
    for change in assessment["changes"]:
        if "quality_adoption" in change["domains"]:
            approvals = activity.setdefault("candidate_approvals", {})
            for name in change["candidates"]:
                previous = approvals.get(name, {"status": "research"})
                if previous.get("status") != "needs_review":
                    approvals[name] = {"status": "needs_review", "historical_decision": previous, "reason": "Changed dependency: " + change["path"]}
        for key in change["items"]:
            state = activity["items"][key]
            if set(change["domains"]) & {"current_evidence", "applicability", "discovery_decision"}:
                state["disposition"] = "open"
                state["rationale"] = "Reassess changed dependency: " + change["path"]
            if "quality_adoption" in change["domains"]:
                state["adoption"] = "research"
    return work_package


def check_adoption(policy, work_package, selected=None):
    """Check research states; positive approval needs a native decision reader."""
    candidates = policy.get("lifecycle", {}).get("candidates", {})
    approvals = work_package["exploration_fulfillment"].get("candidate_approvals", {})
    errors = []
    if not set(approvals) <= set(candidates):
        errors.append({"item": "adoption", "code": "approval_identity", "message": "approval refers to an unknown candidate"})
    for name, candidate in candidates.items():
        if selected is not None and not set(candidate.get("items", [])) & set(selected):
            continue
        status = approvals.get(name, {"status": "research"}).get("status")
        if status == "approved":
            errors.append({"item": name, "code": "approval_unsupported", "message": "research acceptance has no native positive-approval reader; approved quality/adoption cannot be accepted"})
        elif status not in {"research", "pending_review", "rejected", "needs_review"}:
            errors.append({"item": name, "code": "approval_state", "message": "unknown candidate approval state"})
    return errors


def decision_digest(state):
    value = {key: state.get(key) for key in ("disposition", "rationale", "reopen_if", "adoption")}
    value["evidence"] = sorted(state.get("evidence", []))
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _action_items(policy, review, action_id):
    items = review.get("actions", {}).get(action_id, {}).get("items")
    if items is None:
        return None  # Existing reviews retain task-wide freshness.
    if not isinstance(items, list) or not items or not all(isinstance(item, str) for item in items) or not set(items) <= set(policy["items"]):
        raise ValueError("scoped action needs nonempty registered review items")
    return set(items)


def check_review(policy, inputs, selected=None, purpose="completion", work_package=None, action_id=None):
    """Validate registered review/input bindings and required finding closure."""
    contract = policy.get("lifecycle")
    if contract is None:
        return None, []
    errors = []

    def error(code, message):
        errors.append({"item": "semantic_review", "code": code, "message": message})

    spec = contract["review"]
    review = inputs.json(spec["path"], policy["artifacts"][str(Path(spec["path"]).resolve())])
    if purpose == "action" and action_id is not None:
        selected = _action_items(policy, review, action_id)
    if (
        review.get("kind") != "exploration_semantic_review_v1"
        or review.get("task_id") != policy["task_id"]
        or review.get("scope_id") != policy["scope_id"]
        or review.get("inventory_sha256") != inventory_digest(policy)
    ):
        error("review_scope", "review belongs to a different discovered scope")
    if not review.get("reviewer") or not review.get("execution_receipt"):
        error("review_producer", "review needs an identifiable actual review execution")
    else:
        execution = inputs.json(review["execution_receipt"], policy["artifacts"][str(Path(review["execution_receipt"]).resolve())])
        if (
            execution.get("kind") != "exploration_review_execution_v1"
            or execution.get("reviewer") != review["reviewer"]
            or execution.get("scope_id") != policy["scope_id"]
            or not execution.get("commands")
        ):
            error("review_producer", "semantic review execution does not belong to this reviewer/scope")
    observed = review.get("observed_inputs", {})
    if not observed:
        error("review_inputs", "semantic review did not bind actual inputs")
    input_items = review.get("input_items", {})
    if set(input_items) - set(observed) or any(not set(items) <= set(policy["items"]) for items in input_items.values()):
        error("review_inputs", "review input/item mapping is outside the discovered scope")
    owned_inputs = set()
    if selected is not None:
        for path, items in input_items.items():
            if not items:
                continue  # Unowned/shared inputs remain global.
            for dependency in contract.get("dependencies", []):
                if Path(path).resolve() != Path(dependency["path"]).resolve():
                    continue
                affected, _ = _dependency_scope(policy, dependency)
                global_dependency = dependency["kind"] in {"report", "checker", "catalog"} or not affected
                required_items = set(policy["items"]) if global_dependency else affected
                if not required_items <= set(items):
                    error("review_inputs", "input ownership omits global or affected dependency items: " + path)
                elif not global_dependency:
                    owned_inputs.add(Path(path).resolve())
    action_basis = {Path(path).resolve() for path in review.get("actions", {}).get(action_id, {}).get("inputs", [])} if action_id is not None else set()
    for path, expected in observed.items():
        affected = set(input_items.get(path, []))
        if selected is None or not affected or affected & set(selected) or Path(path).resolve() not in owned_inputs or Path(path).resolve() in action_basis:
            inputs.pinned(path, expected)
    if set(review.get("topics", {})) != REVIEW_TOPICS:
        error("review_topics", "review must cover discovery, priority, granularity, and stop/expand")
    for topic, value in review.get("topics", {}).items():
        if not value.get("assessment", "").strip() or not value.get("inputs") or not set(value["inputs"]) <= set(observed):
            error("review_basis", "review judgment lacks actual input references: " + topic)
    finding_ids = set()
    chosen = set(policy["items"]) if selected is None else set(selected)
    if purpose == "completion":
        if work_package is None:
            raise ValueError("completion review must consume the current activity decisions")
        decisions = work_package.get("exploration_fulfillment", {}).get("items", {})
        for key in chosen:
            if review.get("item_decisions", {}).get(key) != decision_digest(decisions.get(key, {})):
                error("review_decision", "item disposition/basis changed since semantic review: " + key)
    for finding in review.get("findings", []):
        key = finding["id"]
        if key in finding_ids:
            error("review_findings", "duplicate semantic finding ID")
        finding_ids.add(key)
        if not set(finding.get("items", [])) <= set(policy["items"]):
            error("review_findings", "finding refers to unknown items")
        required_here = not finding.get("items") or bool(set(finding["items"]) & chosen)
        if purpose == "completion" and required_here:
            if (
                finding.get("disposition") not in {"fixed", "rebutted"}
                or not finding.get("resolution", "").strip()
                or not finding.get("resolution_inputs")
                or not set(finding["resolution_inputs"]) <= set(observed)
            ):
                error("review_unclosed", "required semantic finding is not verified closed: " + key)
    if set(contract.get("required_finding_ids", [])) != finding_ids:
        error("review_findings", "blocking findings must match the fixed required IDs; register scope changes before acceptance")
    if purpose not in {"completion", "action"}:
        raise ValueError("unknown semantic review purpose")
    if purpose == "action":
        if review.get("verdict") not in {"accepted", "changes_required"}:
            error("review_unclosed", "action has no actual review decision")
        return review, errors
    limited = selected is not None and chosen < set(policy["items"])
    if review.get("verdict") != "accepted" and not (limited and review.get("verdict") == "changes_required" and not errors):
        error("review_unclosed", "semantic review has not accepted the declared scope")
    return review, errors


def action_digest(action):
    fields = {key: action.get(key) for key in ("id", "validation_scope", "command", "cwd", "environment", "resource_kind", "subject", "inputs", "expected_cost")}
    if (
        not isinstance(fields["command"], list)
        or not fields["command"]
        or not all(isinstance(value, str) for value in fields["command"])
        or not isinstance(fields["cwd"], str)
        or not Path(fields["cwd"]).is_absolute()
        or not isinstance(fields["environment"], dict)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in fields["environment"].items())
        or not fields["subject"]
        or not isinstance(fields["inputs"], list)
        or not fields["inputs"]
        or not all(isinstance(value, str) and value for value in fields["inputs"])
        or not isinstance(fields["validation_scope"], str)
        or not fields["validation_scope"].strip()
        or not fields["expected_cost"]
    ):
        raise ValueError("executable action needs exact command, absolute cwd, declared environment, subject, inputs, scope and cost basis")
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def authorize_action(policy, work_package, action_id, inputs):
    """Preflight one declared costly action, before its runner starts work."""
    if work_package.get("task_id") != policy["task_id"] or work_package.get("exploration_fulfillment", {}).get("scope_id") != policy["scope_id"]:
        raise ValueError("action work package belongs to another task/scope")
    review, errors = check_review(policy, inputs, purpose="action", action_id=action_id)
    if review is None:
        raise ValueError("costly action has no registered semantic review")
    actions = [row for row in work_package.get("next_actions", []) if row.get("id") == action_id]
    if len(actions) != 1:
        raise ValueError("action must have one current declaration")
    action = actions[0]
    signature = action_digest(action)
    decision = review.get("actions", {}).get(action_id)
    if (
        not decision
        or decision.get("decision") != "allow"
        or decision.get("scope") != action.get("validation_scope")
        or decision.get("action_sha256") != signature
        or not decision.get("rationale", "").strip()
        or not decision.get("inputs")
        or not set(decision["inputs"]) <= set(review.get("observed_inputs", {}))
    ):
        errors.append({"item": action_id, "code": "action_choice", "message": "action/scope is unreviewed or was replaced/rejected; follow the reviewed minimum sufficient action"})
    base = Path(action["cwd"]).resolve()
    action_inputs = {(base / path).resolve() for path in action["inputs"]}
    reviewed_inputs = {Path(path).resolve() for path in (decision or {}).get("inputs", [])}
    observed_inputs = {Path(path).resolve() for path in review.get("observed_inputs", {})}
    if not action_inputs <= reviewed_inputs & observed_inputs:
        errors.append({"item": action_id, "code": "action_inputs", "message": "every declared action input must have a reviewed current content identity"})
    changes = assess_changes(policy, inputs, _action_items(policy, review, action_id))
    if changes["changed"]:
        errors.append({"item": action_id, "code": "action_stale", "message": "review dependencies changed; reassess affected choice before execution"})
    for path in inputs.mismatched_artifacts(policy["artifacts"]):
        errors.append({"item": action_id, "code": "artifact_identity", "message": "action basis differs from the fixed artifact: " + str(path)})
    for path in inputs.unregistered_sources(policy["artifacts"]):
        errors.append({"item": action_id, "code": "unregistered_source", "message": "action basis is not registered: " + str(path)})
    if errors:
        raise ValueError(json.dumps(errors, ensure_ascii=False))
    inputs.unchanged()
    return decision
