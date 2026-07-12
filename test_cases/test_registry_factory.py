import sys
import unittest
from collections.abc import MutableMapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lightx2v_train"))

import lightx2v.utils.registry_factory as runtime_registry
import lightx2v_platform.registry_factory as platform_registry
import lightx2v_train.utils.registry as training_registry


def first_target():
    return "first"


def second_target():
    return "second"


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
