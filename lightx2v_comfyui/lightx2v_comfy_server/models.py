"""Discover model/task choices from LightX2V source without importing its runtime."""

from __future__ import annotations

import ast
import hashlib
import os
from functools import lru_cache
from importlib.util import resolve_name
from pathlib import Path

from .files import local_path

TASK_ATTRIBUTE = "supported_request_fields_by_task"
BUILTIN_TASKS = {
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
        self.visiting = set()

    def module(self, name):
        if name not in self.modules:
            if not name.startswith("lightx2v.models.runners."):
                raise ValueError("Not a runner module: " + name)
            path = self.root.joinpath(*name.split(".")).with_suffix(".py")
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
            imports = {}
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level:
                        module = resolve_name("." * node.level + module, name.rpartition(".")[0])
                    for alias in node.names:
                        imports[alias.asname or alias.name] = (module, alias.name)
            self.modules[name] = (tree, classes, imports)
        return self.modules[name]

    def reference(self, module, node):
        if not isinstance(node, ast.Name):
            raise ValueError("Unsupported runner class reference")  # noqa: TRY004 - invalid input value
        _, classes, imports = self.module(module)
        if node.id in classes:
            return module, node.id
        if node.id in imports:
            return imports[node.id]
        raise ValueError("Unknown runner class: " + node.id)

    def expression_tasks(self, module, value):
        if isinstance(value, ast.Dict):
            tasks = []
            for key, child in zip(value.keys, value.values):
                if key is None:
                    tasks.extend(self.expression_tasks(module, child))
                elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                    tasks.append(key.value)
                else:
                    raise ValueError("Task names must be string constants")
            return list(dict.fromkeys(tasks))
        if isinstance(value, ast.Attribute) and value.attr == TASK_ATTRIBUTE:
            return self.class_tasks(*self.reference(module, value.value))
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
                return self.expression_tasks(module, iterator.func.value)
        raise ValueError("Unsupported task declaration")

    def class_tasks(self, module, name):
        key = (module, name)
        if key in self.visiting:
            raise ValueError("Cyclic runner inheritance")
        self.visiting.add(key)
        try:
            _, classes, _ = self.module(module)
            cls = classes[name]
            for node in cls.body:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
                if any(isinstance(target, ast.Name) and target.id == TASK_ATTRIBUTE for target in targets):
                    return self.expression_tasks(module, node.value)
            for base in cls.bases:
                try:
                    reference = self.reference(module, base)
                except ValueError:
                    continue
                if not reference[0].startswith("lightx2v.models.runners."):
                    continue  # External mixins need not declare tasks.
                try:
                    tasks = self.class_tasks(*reference)
                except (KeyError, OSError):
                    continue
                if tasks:
                    return tasks
            return []
        finally:
            self.visiting.remove(key)

    def registered_class(self, module, model):
        _, classes, _ = self.module(module)
        for name, cls in classes.items():
            for decorator in cls.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "RUNNER_REGISTER"
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                    and decorator.args[0].value == model
                ):
                    return name
        raise ValueError("Missing runner registration for " + model)


@lru_cache(maxsize=8)
def _discover(root: str, signature: tuple) -> tuple:
    index = _SourceIndex(Path(root))
    tree, _, _ = index.module("lightx2v.models.runners.runner_factory")
    registry = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == "RUNNER_MODULES" for n in node.targets):
            registry = ast.literal_eval(node.value)
    if not isinstance(registry, dict):
        raise ValueError("RUNNER_MODULES must be a literal dictionary")  # noqa: TRY004 - invalid input value
    models, unavailable = {}, {}
    for model, module in registry.items():
        try:
            tasks = index.class_tasks(module, index.registered_class(module, model))
            if not tasks:
                raise ValueError("No task declaration")
            models[model] = tuple(tasks)
        except (KeyError, OSError, SyntaxError, ValueError) as exc:
            unavailable[model] = str(exc)
    # Keep the demonstrated models near the top; other registered models follow alphabetically.
    order = [m for m in BUILTIN_TASKS if m in models] + sorted(set(models) - BUILTIN_TASKS.keys())
    return tuple((m, models[m]) for m in order), tuple(unavailable.items())


def model_catalog(lightx2v_path: str = "") -> dict:
    root = local_path(lightx2v_path or default_lightx2v_path())
    runners = root / "lightx2v/models/runners"
    if not (runners / "runner_factory.py").is_file():
        return {"models": {key: list(value) for key, value in BUILTIN_TASKS.items()}, "source": "builtin", "unavailable": {}}
    signature = tuple((str(p.relative_to(runners)), hashlib.sha256(p.read_bytes()).digest()) for p in sorted(runners.rglob("*.py")))
    models, unavailable = _discover(str(root), signature)
    if not models:
        raise ValueError("No model/task declarations found in this LightX2V directory")
    return {"models": {key: list(value) for key, value in models}, "source": "lightx2v", "unavailable": dict(unavailable)}
