#!/usr/bin/env python3
"""Apply/check the small external-runtime patch without modifying its git pin."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lightx2v_rl.backends.verl_omni.runtime import DEFAULT_CHECKOUT, prepare_runtime  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--check", action="store_true", help="Verify without changing any files")
    args = parser.parse_args()
    try:
        checkout = prepare_runtime(args.checkout, check_only=args.check)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    print(f"verl-omni runtime extensions ready: {checkout}")


if __name__ == "__main__":
    main()
