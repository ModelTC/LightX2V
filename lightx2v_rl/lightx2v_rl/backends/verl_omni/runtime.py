"""Reproducible, explicit preparation of the pinned external runtime."""

import shutil
import subprocess
import tempfile
from pathlib import Path

VERL_OMNI_COMMIT = "e09850c5abb919f9c1893dc72f9d623698742093"
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKOUT = PACKAGE_ROOT.parent / "third_party" / "verl-omni"
PATCH = PACKAGE_ROOT / "patches" / "0001-external-diffusion-extensions.patch"
MEDIA_PATCH = PACKAGE_ROOT / "patches" / "0002-rgba-validation-images.patch"
ENGINE_DEFAULTS_PATCH = PACKAGE_ROOT / "patches" / "0003-offline-engine-argument-defaults.patch"
ROLLOUT_PORT_PATCH = PACKAGE_ROOT / "patches" / "0004-reserve-rollout-master-ports.patch"
TRACKING_PATCH = PACKAGE_ROOT / "patches" / "0005-custom-tracking-class.patch"
VALIDATION_REWARD_PATCH = PACKAGE_ROOT / "patches" / "0006-validation-reward-details.patch"
ADVANTAGE_PATCH = PACKAGE_ROOT / "patches" / "0007-custom-advantage-function.patch"


def prepare_runtime(checkout: Path = DEFAULT_CHECKOUT, *, check_only: bool = False) -> Path:
    """Apply the reviewed patches once; never reset a checkout or fetch implicitly."""
    checkout = checkout.resolve()
    if not (checkout / "verl_omni").is_dir():
        raise RuntimeError("Initialize third_party/verl-omni with git submodule update --init first.")

    def git(*args):
        return subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True, check=False)

    def patch_paths(patch):
        stats = git("apply", "--numstat", "-z", str(patch))
        if stats.returncode:
            raise RuntimeError("Invalid runtime extension patch:\n" + stats.stderr)
        return [entry.split("\t", 2)[2] for entry in stats.stdout.split("\0") if entry]

    def only_prepared_changes(affected, applied):
        """Recognize earlier patches without modifying the real checkout.

        A later patch may extend a file already changed by an installed patch.
        Reverse the installed patches in a disposable copy, then require that
        the affected files match HEAD exactly. Unknown edits remain protected.
        """
        if not git("diff", "--quiet", "HEAD", "--", *affected).returncode:
            return True
        with tempfile.TemporaryDirectory(prefix="lightx2v-rl-patches-") as directory:
            snapshot = Path(directory)
            paths = set(affected)
            for patch in applied:
                paths.update(patch_paths(patch))
            for relative in paths:
                source = checkout / relative
                if source.is_file():
                    target = snapshot / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            for patch in reversed(applied):
                result = subprocess.run(
                    ["git", "-C", str(snapshot), "apply", "--reverse", str(patch)],
                    capture_output=True, text=True, check=False,
                )
                if result.returncode:
                    return False
            for relative in affected:
                baseline = git("show", f"HEAD:{relative}")
                target = snapshot / relative
                if baseline.returncode:
                    if target.exists():
                        return False
                elif not target.is_file() or target.read_text() != baseline.stdout:
                    return False
        return True

    revision = git("rev-parse", "HEAD")
    if revision.returncode or revision.stdout.strip() != VERL_OMNI_COMMIT:
        raise RuntimeError(f"Expected verl-omni commit {VERL_OMNI_COMMIT}; refusing to patch another revision.")
    pending = []
    applied_patches = []
    for patch in (PATCH, MEDIA_PATCH, ENGINE_DEFAULTS_PATCH, ROLLOUT_PORT_PATCH, TRACKING_PATCH,
                  VALIDATION_REWARD_PATCH, ADVANTAGE_PATCH):
        if not patch.is_file():
            raise RuntimeError(f"Missing runtime extension patch: {patch}")
        if git("apply", "--reverse", "--check", str(patch)).returncode == 0:
            applied_patches.append(patch)
            continue
        # Inspect every pending patch before mutating anything, including
        # upgrades that extend files already covered by installed patches.
        affected = patch_paths(patch)
        if not affected or not only_prepared_changes(affected, applied_patches):
            raise RuntimeError("Runtime patch is partially applied or conflicts with local edits to affected files.")
        applicable = git("apply", "--check", str(patch))
        if applicable.returncode:
            raise RuntimeError("Runtime patch is partially applied or conflicts with local edits:\n" + applicable.stderr)
        pending.append(patch)
    if check_only and pending:
        raise RuntimeError("Runtime extensions are not applied; run lightx2v_rl/scripts/setup/prepare_verl_omni.py.")
    for patch in pending:
        applied = git("apply", str(patch))
        if applied.returncode:
            raise RuntimeError(applied.stderr)
    return checkout
