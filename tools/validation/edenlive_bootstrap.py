"""Resolve local source paths before importing validation dependencies."""

import sys
from pathlib import Path


def configure_import_paths():
    """Support direct scripts and an optional ``--upstream`` checkout path."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    arguments = sys.argv[1:]
    upstream = None
    for index, argument in enumerate(arguments):
        if argument.startswith("--upstream="):
            upstream = argument.partition("=")[2]
        elif argument == "--upstream" and index + 1 < len(arguments):
            upstream = arguments[index + 1]
    if upstream:
        sys.path.insert(0, str(Path(upstream).resolve()))
