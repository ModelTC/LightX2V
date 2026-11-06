"""Require complete, successful on-policy groups before training or validation."""

from collections import Counter
from collections.abc import Mapping


def validate_rollout_groups(*, keys, tags, expected_uids, group_statuses, num_candidates, global_step):
    """Validate a submitted batch against its returned single-output candidates.

    Expected IDs come from submission, not from the surviving trajectories.
    This function never drops, pads, retries, or replaces samples. In particular,
    a failed prompt remains invalid even if all its trajectory rows are present.
    """
    if isinstance(num_candidates, bool) or not isinstance(num_candidates, int) or num_candidates < 1:
        raise ValueError("num_candidates must be a positive integer.")
    expected_uids = tuple(expected_uids)
    if not expected_uids or any(not isinstance(uid, str) or not uid for uid in expected_uids):
        raise ValueError("Expected rollout prompt IDs must be nonempty strings.")
    if len(set(expected_uids)) != len(expected_uids):
        raise ValueError("Expected rollout prompt IDs must be unique.")

    sessions = {uid: Counter() for uid in expected_uids}
    row_errors = []
    if len(keys) != len(tags):
        row_errors.append(f"keys/tags length mismatch: {len(keys)}/{len(tags)}")
    for key, tag in zip(keys, tags, strict=False):
        parts = key.rsplit("_", 2) if isinstance(key, str) else []
        if len(parts) != 3 or not parts[0] or not parts[1].isascii() or not parts[1].isdigit() or parts[2] != "0":
            row_errors.append(f"invalid trajectory key {key!r}; expected <uid>_<session>_0")
            continue
        uid, session_text, _ = parts
        session = int(session_text)
        if uid not in sessions:
            row_errors.append(f"unexpected prompt uid={uid!r} in {key!r}")
            continue
        if session_text != str(session) or session >= num_candidates:
            row_errors.append(f"uid={uid!r}: invalid candidate session={session_text!r}, expected 0..{num_candidates - 1}")
            continue
        sessions[uid][session] += 1
        if not isinstance(tag, Mapping):
            row_errors.append(f"uid={uid!r} session={session}: missing trajectory metadata")
            continue
        if tag.get("status") != "success":
            row_errors.append(f"uid={uid!r} session={session}: trajectory status={tag.get('status')!r}")
        for field in ("global_steps", "min_global_steps", "max_global_steps"):
            if tag.get(field) != global_step:
                row_errors.append(f"uid={uid!r} session={session}: {field}={tag.get(field)!r}, expected {global_step}")

    group_errors = []
    expected_sessions = set(range(num_candidates))
    for uid, counts in sessions.items():
        missing = sorted(expected_sessions - counts.keys())
        duplicates = sorted(session for session, count in counts.items() if count != 1)
        status = group_statuses.get(uid)
        if status != "finished" or missing or duplicates:
            group_errors.append(f"uid={uid!r}: status={status!r}, missing_sessions={missing}, duplicate_sessions={duplicates}")
    errors = group_errors + row_errors
    if errors:
        details = "; ".join(errors[:16])
        if len(errors) > 16:
            details += f"; ... {len(errors) - 16} more issue(s)"
        raise RuntimeError(
            f"Rollout completeness check failed at step={global_step}: "
            f"expected {len(expected_uids)} groups x {num_candidates} candidates "
            f"({len(expected_uids) * num_candidates} trajectories), received {len(keys)}. "
            f"{details}. Stopping before batch use; inspect rollout/reward worker errors."
        )
