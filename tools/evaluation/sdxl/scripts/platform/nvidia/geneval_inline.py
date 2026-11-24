"""Run the pinned GenEval launcher in the held legacy-interpreter task PID."""

import ast
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path


def run_score_in_process(command):
    if len(command) < 2 or command[1] != "score" or "--dataset" not in command or command[command.index("--dataset") + 1] != "geneval":
        raise ValueError("Inline launcher is restricted to GenEval scoring")
    paths = json.loads(Path(command[command.index("--paths") + 1]).read_text())
    if os.path.abspath(sys.executable) != os.path.abspath(paths["geneval_python"]):
        raise ValueError("GenEval scoring must use its configured legacy interpreter")
    tree = ast.parse(Path(command[0]).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "score"]
    launchers = [
        ast.literal_eval(node.value)
        for function in functions
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "launcher" for target in node.targets)
    ]
    if len(launchers) != 1 or not isinstance(launchers[0], str):
        raise ValueError("Expected exactly one literal launcher in the pinned score function")
    launcher = launchers[0]
    original_run, original_argv = subprocess.run, sys.argv
    calls = 0

    def dispatch(arguments, *positional, **options):
        nonlocal calls
        if isinstance(arguments, (list, tuple)) and len(arguments) > 2 and os.path.abspath(str(arguments[0])) == os.path.abspath(paths["geneval_python"]) and arguments[1] == "-c":
            if arguments[2] != launcher or positional or options != {"check": True, "timeout": 3300}:
                raise ValueError("Unrecognized GenEval launcher or execution options")
            if len(arguments) < 5 or arguments[3] != paths["geneval_clip"] or Path(arguments[4]).resolve() != (Path(paths["geneval"]) / "evaluation/evaluate_images.py").resolve():
                raise ValueError("GenEval evaluator/cache paths changed")
            calls += 1
            if calls != 1:
                raise ValueError("Expected one GenEval evaluator invocation per score task")
            saved_argv = sys.argv
            sys.argv = ["-c", *arguments[3:]]
            print(json.dumps({"inline_geneval_launcher_sha256": hashlib.sha256(launcher.encode()).hexdigest(), "python": sys.executable}), flush=True)
            try:
                # Execute the original literal launcher and original CLI arguments.
                # The existing outer GNU timeout bounds this same-PID execution.
                try:
                    exec(compile(launcher, "<pinned-geneval-launcher>", "exec"), {"__name__": "__main__"})
                except SystemExit as error:
                    if error.code not in (None, 0):
                        raise subprocess.CalledProcessError(error.code, arguments) from error
            finally:
                sys.argv = saved_argv
            return subprocess.CompletedProcess(arguments, 0)
        return original_run(arguments, *positional, **options)

    subprocess.run = dispatch
    sys.argv = list(command)
    try:
        runpy.run_path(command[0], run_name="__main__")
        if calls != 1:
            raise ValueError("Pinned GenEval score task did not invoke its evaluator")
    finally:
        subprocess.run = original_run
        sys.argv = original_argv
