import unittest
from types import SimpleNamespace

from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model


def _branch(strength=1.0):
    return SimpleNamespace(
        has_lora_branch=True,
        lora_strength=strength,
        _modules={},
        _parameters={},
    )


def _root(*children):
    return SimpleNamespace(
        _modules={str(index): child for index, child in enumerate(children)},
        _parameters={},
    )


class MiniMaxH3EmpurpleLoraTest(unittest.TestCase):
    def test_runtime_strength_updates_resident_and_offload_branches(self):
        pre_branch = _branch()
        source_branch = _branch()
        offload_buffer_branch = _branch()
        post_branch = _branch()
        clear_calls = []

        model = object.__new__(MiniMaxH3Model)
        model.config = {"lora_dynamic_apply": True}
        model.pre_weight = _root(pre_branch)
        model.transformer_weights = _root(source_branch, offload_buffer_branch)
        model.post_weight = _root(post_branch)
        model.transformer_infer = SimpleNamespace(_clear_adaln_cache=lambda: clear_calls.append(True))
        model._runtime_lora_strength = 1.0
        model.lora_strength = 1.0

        model.set_dynamic_lora_strength(0.0)

        self.assertEqual([pre_branch.lora_strength, source_branch.lora_strength, offload_buffer_branch.lora_strength, post_branch.lora_strength], [0.0] * 4)
        self.assertEqual(model.lora_strength, 0.0)
        self.assertEqual(len(clear_calls), 1)

        # Reapplying the current phase is intentionally a no-op.
        model.set_dynamic_lora_strength(0.0)
        self.assertEqual(len(clear_calls), 1)


if __name__ == "__main__":
    unittest.main()
