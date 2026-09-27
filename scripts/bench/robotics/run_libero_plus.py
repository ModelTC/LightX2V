import sys
from pathlib import Path

# Do not expose this directory's adapter packages as top-level 'libero':
# the benchmark itself uses a namespace package with the same name.
_script_dir = Path(__file__).resolve().parent
sys.path[:] = [str(_script_dir.parents[2]), *(p for p in sys.path if Path(p or ".").resolve() != _script_dir)]
from scripts.bench.robotics.common.manager import main  # noqa: E402

if __name__ == "__main__":
    main("libero_plus")
