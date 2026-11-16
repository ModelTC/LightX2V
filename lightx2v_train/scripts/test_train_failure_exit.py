"""Exercise entry-point failures in real processes, without importing model stacks."""

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1] / "train.py"
CHILD = textwrap.dedent(
    """
    import ast
    import os
    import sys
    import time
    import traceback
    from pathlib import Path
    from types import SimpleNamespace

    source, mode = sys.argv[1:]
    tree = ast.parse(Path(source).read_text())
    entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    def cleanup_distributed():
        print("CLEANUP_ENTERED", flush=True)
        if mode == "distributed_failure":
            # A failed rank must never enter a collective teardown that waits
            # for peers still inside their training collectives.
            time.sleep(60)

    class Model:
        def load_components(self, **kwargs):
            pass

    class Trainer:
        def set_model(self, model):
            pass

        def set_data(self, *data):
            pass

        def train(self):
            if mode != "success":
                raise RuntimeError("sentinel failure on nonzero rank")
            print("TRAIN_COMPLETED", flush=True)

    torch = SimpleNamespace(backends=SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=None)),
        cudnn=SimpleNamespace(allow_tf32=None),
    ))
    logger = SimpleNamespace(exception=lambda *args: None)
    parse_args = lambda: SimpleNamespace(config="unused.yaml")
    load_config = lambda _: {"training": {}}
    init_distributed = lambda _: None
    get_rank = lambda: 6
    is_distributed = lambda: mode != "single_failure"
    setup_logger = lambda _: None
    is_train_cache_dataset = lambda _: True
    check_val_is_enabled = lambda _: False
    build_sample_processor = lambda _: None
    build_data = lambda *args, **kwargs: []
    build_model = lambda _: Model()
    build_trainer = lambda _: Trainer()

    exec(compile(ast.Module(body=[entry], type_ignores=[]), source, "exec"), globals())
    if mode == "single_failure":
        try:
            main()
        except RuntimeError as error:
            print("CALLER_CAUGHT: " + str(error), flush=True)
        else:
            raise AssertionError("Single-process errors must propagate to the caller")
    else:
        main()
    """
)


class TrainFailureExitTest(unittest.TestCase):
    def run_entry(self, mode):
        return subprocess.run(
            [sys.executable, "-c", CHILD, str(TRAIN), mode],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    def test_nonzero_rank_failure_is_visible_and_does_not_wait_in_cleanup(self):
        result = self.run_entry("distributed_failure")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("[rank6]", result.stderr)
        self.assertIn("Traceback (most recent call last):", result.stderr)
        self.assertIn("RuntimeError: sentinel failure on nonzero rank", result.stderr)
        self.assertNotIn("CLEANUP_ENTERED", result.stdout)

    def test_success_still_cleans_up(self):
        result = self.run_entry("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["TRAIN_COMPLETED", "CLEANUP_ENTERED"])
        self.assertEqual(result.stderr, "")

    def test_single_process_failure_can_be_caught_by_caller(self):
        result = self.run_entry("single_failure")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CALLER_CAUGHT: sentinel failure on nonzero rank", result.stdout)
        self.assertIn("RuntimeError: sentinel failure on nonzero rank", result.stderr)
        self.assertNotIn("CLEANUP_ENTERED", result.stdout)


if __name__ == "__main__":
    unittest.main()
