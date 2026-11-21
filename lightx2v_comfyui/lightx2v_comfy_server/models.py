"""Discover model/task choices from LightX2V source without importing its runtime."""

from __future__ import annotations

import ast
import hashlib
import os
from functools import lru_cache
from importlib.util import resolve_name
from pathlib import Path

from .files import local_path

# Initial widget choices when the copied plugin has no LightX2V checkout yet.
DEFAULT_TASKS = {
    "minimax_h3": ["t2av", "i2av", "l2av", "fl2av", "ref2av"],
    "relayvsr_x4": ["sr"],
    "relayvsr_x2": ["sr"],
}


def default_lightx2v_path() -> str:
    checkout = Path(__file__).resolve().parents[2]
    return os.environ.get("LIGHTX2V_PATH", str(checkout))


def inferred_variant(model_cls: str, task: str) -> str:
    if model_cls == "minimax_h3":
        return "ref2av" if task == "ref2av" else "fl2av"
    if model_cls == "taomate_h3":
        return "fl2av"
    return ""  # Other model-specific variants can remain in config JSON.


def validate_selection(catalog: dict, model_cls: str, task: str):
    tasks = catalog["models"].get(model_cls)
    if not tasks:
        raise ValueError(f"Model {model_cls!r} has no discoverable task declaration in this LightX2V directory")
    if task not in tasks:
        raise ValueError(f"{model_cls} does not support task {task!r}; choose from: {', '.join(tasks)}")


class _SourceIndex:
    def __init__(self, root: Path):
        self.root = root
        self.modules = {}

    def module(self, name):
        if name not in self.modules:
            if not name.startswith("lightx2v.models.runners."):
                raise ValueError("Not a runner module: " + name)
            path = self.root.joinpath(*name.split(".")).with_suffix(".py")
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            assignments = {}
            imports = {}
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    assignments.update((target.id, node.value) for target in node.targets if isinstance(target, ast.Name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level:
                        module = resolve_name("." * node.level + module, name.rpartition(".")[0])
                    for alias in node.names:
                        imports[alias.asname or alias.name] = (module, alias.name)
            self.modules[name] = (assignments, imports)
        return self.modules[name]

    def assignment(self, module, name):
        assignments, _ = self.module(module)
        return assignments[name]

    def spec_tasks(self, module, model):
        specs = self.assignment(module, "SPECS")
        declarations = {ast.literal_eval(key): value for key, value in zip(specs.keys, specs.values)}
        return self.tasks(module, declarations[model])

    def tasks(self, module, value):
        while isinstance(value, ast.Name):
            _, imports = self.module(module)
            module, name = imports.get(value.id, (module, value.id))
            value = self.assignment(module, name)
        # RequestSpec and its subclasses all take the task mapping first.
        if isinstance(value, ast.Call) and value.args:
            return self.tasks(module, value.args[0])
        if isinstance(value, ast.Dict):
            tasks = []
            for key, child in zip(value.keys, value.values):
                if key is None:
                    tasks.extend(self.tasks(module, child))
                elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                    tasks.append(key.value)
                else:
                    raise ValueError("Task names must be string constants")
            return list(dict.fromkeys(tasks))
        if isinstance(value, ast.Attribute) and value.attr == "tasks":
            return self.tasks(module, value.value)
        if isinstance(value, ast.DictComp) and len(value.generators) == 1:
            generator = value.generators[0]
            if generator.ifs or generator.is_async:
                raise ValueError("Conditional task declarations are not supported")
            iterator = generator.iter
            if isinstance(iterator, (ast.Tuple, ast.List)) and isinstance(generator.target, ast.Name) and isinstance(value.key, ast.Name) and value.key.id == generator.target.id:
                tasks = ast.literal_eval(iterator)
                if all(isinstance(task, str) for task in tasks):
                    return list(tasks)
            if (
                isinstance(iterator, ast.Call)
                and isinstance(iterator.func, ast.Attribute)
                and iterator.func.attr == "items"
                and not iterator.args
                and not iterator.keywords
                and isinstance(generator.target, ast.Tuple)
                and len(generator.target.elts) == 2
                and isinstance(generator.target.elts[0], ast.Name)
                and isinstance(value.key, ast.Name)
                and value.key.id == generator.target.elts[0].id
            ):
                return self.tasks(module, iterator.func.value)
        raise ValueError("Unsupported task declaration")


@lru_cache(maxsize=8)
def _discover(root: str, signature: tuple) -> tuple:
    index = _SourceIndex(Path(root))
    declaration = index.assignment("lightx2v.models.runners.registry", "RUNNER_MODULES")
    registry = {ast.literal_eval(key): "lightx2v.models.runners." + ast.literal_eval(value.args[0]).rpartition(".")[0] + ".request_spec" for key, value in zip(declaration.keys, declaration.values)}
    models, unavailable = {}, {}
    for model, module in registry.items():
        try:
            tasks = index.spec_tasks(module, model)
            if not tasks:
                raise ValueError("No task declaration")
            models[model] = tuple(tasks)
        except (KeyError, OSError, SyntaxError, ValueError) as exc:
            unavailable[model] = str(exc)
    # Keep the demonstrated models near the top; other registered models follow alphabetically.
    order = [m for m in DEFAULT_TASKS if m in models] + sorted(set(models) - DEFAULT_TASKS.keys())
    return tuple((m, models[m]) for m in order), tuple(unavailable.items())


def model_catalog(lightx2v_path: str = "") -> dict:
    root = local_path(lightx2v_path or default_lightx2v_path())
    runners = root / "lightx2v/models/runners"
    if not runners.is_dir():
        return {"models": {key: list(value) for key, value in DEFAULT_TASKS.items()}, "source": "builtin", "unavailable": {}}
    declarations = [runners / "registry.py", *sorted(runners.rglob("request_spec.py"))]
    signature = tuple((str(p.relative_to(runners)), hashlib.sha256(p.read_bytes()).digest()) for p in declarations)
    models, unavailable = _discover(str(root), signature)
    if not models:
        raise ValueError("No model/task declarations found in this LightX2V directory")
    return {"models": {key: list(value) for key, value in models}, "source": "lightx2v", "unavailable": dict(unavailable)}
