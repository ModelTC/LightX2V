import hashlib
import json
import os
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def result_path(output, task):
    digest = hashlib.sha256(task["key"].encode()).hexdigest()[:16]
    return Path(output) / "tasks" / f"{digest}.json"


def score(episodes):
    successes = sum(bool(e["success"]) for e in episodes)
    return {"successes": successes, "trials": len(episodes), "success_rate": successes / len(episodes) if episodes else None}


def summarize(output, tasks, trials):
    episodes, groups, errors = [], {"categories": {}, "phases": {}, "suites": {}}, []
    completed_tasks = 0
    for task in tasks:
        path = result_path(output, task)
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        rows = result["episodes"]
        episodes.extend(rows)
        completed_tasks += result["status"] == "complete"
        if result.get("error"):
            errors.append({"task": task["key"], "error": result["error"]})
        for name, key in (("categories", "category"), ("phases", "phase"), ("suites", "suite")):
            if task.get(key):
                groups[name].setdefault(task[key], []).extend(rows)
    return {
        "aggregation": "micro: total successes / completed trials; not mean of category percentages",
        "complete": completed_tasks == len(tasks) and not errors,
        "planned_tasks": len(tasks),
        "completed_tasks": completed_tasks,
        "planned_trials": len(tasks) * trials,
        "completed_trials": len(episodes),
        "overall": score(episodes),
        **{name: {k: score(v) for k, v in group.items()} for name, group in groups.items()},
        "errors": errors,
    }
