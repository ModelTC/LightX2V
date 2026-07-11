import importlib.util
import sys
import types
import unittest
from collections.abc import MutableMapping
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_target():
    return "first"


def second_target():
    return "second"


platform_registry = load_module("_platform_registry_for_test", REPO_ROOT / "lightx2v_platform" / "registry_factory.py")

platform_package = types.ModuleType("lightx2v_platform")
platform_package.registry_factory = platform_registry
with patch.dict(
    sys.modules,
    {
        "lightx2v_platform": platform_package,
        "lightx2v_platform.registry_factory": platform_registry,
    },
):
    runtime_registry = load_module("_runtime_registry_for_test", REPO_ROOT / "lightx2v" / "utils" / "registry_factory.py")

training_registry = load_module(
    "_training_registry_for_test",
    REPO_ROOT / "lightx2v_train" / "lightx2v_train" / "utils" / "registry.py",
)


REGISTER_CLASSES = {
    "runtime": runtime_registry.Register,
    "platform": platform_registry.Register,
    "training": training_registry.Register,
}


class RegisterTest(unittest.TestCase):
    def test_implements_mutable_mapping(self):
        for name, register_cls in REGISTER_CLASSES.items():
            with self.subTest(register=name):
                self.assertTrue(issubclass(register_cls, MutableMapping))
                self.assertFalse(issubclass(register_cls, dict))

    def test_initial_mapping_uses_registry_storage(self):
        def initial_target():
            return "initial"

        def keyword_target():
            return "keyword"

        for name, register_cls in REGISTER_CLASSES.items():
            with self.subTest(register=name):
                registry = register_cls({"initial": initial_target}, keyword=keyword_target)

                self.assertEqual(len(registry), 2)
                self.assertEqual(list(registry), ["initial", "keyword"])
                self.assertEqual(dict(registry), {"initial": initial_target, "keyword": keyword_target})
                self.assertEqual(str(registry), str({"initial": initial_target, "keyword": keyword_target}))

    def test_registers_named_and_unnamed_targets(self):
        for name, register_cls in REGISTER_CLASSES.items():
            with self.subTest(register=name):
                registry = register_cls()

                @registry("named")
                def named_target():
                    return "named"

                @registry
                def automatic_target():
                    return "automatic"

                self.assertIs(registry["named"], named_target)
                self.assertIs(registry.get("automatic_target"), automatic_target)
                self.assertEqual(set(registry.keys()), {"named", "automatic_target"})
                self.assertEqual(set(registry.values()), {named_target, automatic_target})
                self.assertEqual(set(registry.items()), {("named", named_target), ("automatic_target", automatic_target)})

    def test_rejects_invalid_and_duplicate_targets(self):
        for name, register_cls in REGISTER_CLASSES.items():
            with self.subTest(register=name):
                registry = register_cls()

                def original_target():
                    return "original"

                with self.assertRaisesRegex(Exception, "must be callable"):
                    registry.register("not callable")

                registry("target")(original_target)
                with self.assertRaisesRegex(Exception, "already exists"):
                    registry("target")(lambda: None)
                self.assertIs(registry["target"], original_target)

    def test_supports_mutable_mapping_operations(self):
        for name, register_cls in REGISTER_CLASSES.items():
            with self.subTest(register=name):
                registry = register_cls()

                registry.update({"first": first_target})
                registry.setdefault("second", second_target)
                self.assertEqual(len(registry), 2)
                self.assertIn("first", registry)

                self.assertIs(registry.pop("first"), first_target)
                del registry["second"]
                self.assertFalse(registry)

                registry["first"] = first_target
                registry["second"] = second_target
                registry.clear()
                self.assertEqual(dict(registry), {})

    def test_merges_registries_and_rejects_conflicts(self):
        for name, register_cls in REGISTER_CLASSES.items():
            with self.subTest(register=name):
                registry = register_cls({"first": first_target})
                other = register_cls({"second": second_target})

                registry.merge(other)
                self.assertEqual(dict(registry), {"first": first_target, "second": second_target})

                with self.assertRaisesRegex(Exception, "already exists"):
                    registry.merge(register_cls({"first": first_target}))
                self.assertIs(registry["first"], first_target)


if __name__ == "__main__":
    unittest.main()
