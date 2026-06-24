import sys
from pathlib import Path


def _ensure_lightx2v_on_path():
    try:
        import lightx2v  # noqa: F401

        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_lightx2v_on_path()

from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy  # noqa: E402,F401
