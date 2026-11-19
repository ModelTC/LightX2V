"""Locate the verl-omni checkout pinned by LightX2V."""

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SUBMODULE_PATH = "third_party/verl-omni"
DEFAULT_CHECKOUT = REPOSITORY_ROOT / SUBMODULE_PATH


def validate_runtime(checkout: Path = DEFAULT_CHECKOUT) -> Path:
    checkout = checkout.resolve()
    update = "Run git submodule sync -- third_party/verl-omni and git submodule update --init third_party/verl-omni."
    if not (checkout / "verl_omni").is_dir():
        raise RuntimeError(f"Missing verl-omni checkout at {checkout}. {update}")

    def revision(directory: Path, ref: str) -> str:
        return subprocess.check_output(["git", "-C", str(directory), "rev-parse", ref], text=True).strip()

    try:
        expected = revision(REPOSITORY_ROOT, f"HEAD:{SUBMODULE_PATH}")
        current = revision(checkout, "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not verify the pinned verl-omni checkout. {update}") from exc
    if current != expected:
        raise RuntimeError(f"Expected verl-omni {expected} at {checkout}, found {current}. {update}")
    return checkout
