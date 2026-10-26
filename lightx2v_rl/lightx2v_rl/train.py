"""Launch an external FewStep recipe through verl-omni's generic v1 trainer."""

import argparse
import importlib.util
import sys
from pathlib import Path

from omegaconf import OmegaConf

from lightx2v_rl.backends.verl_omni.runtime import DEFAULT_CHECKOUT, prepare_runtime
from lightx2v_rl.paths import resolve_runtime_paths
from lightx2v_rl.training.config import (
    REWARD_RECIPES,
    compose_runtime_config,
    load_recipe,
    load_training_recipe,
    validate_recipe,
)

# Re-export the recipe API for existing external callers of the CLI module.
__all__ = ["compose_runtime_config", "load_recipe", "load_training_recipe", "validate_recipe", "main"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="wan_geo", help="Bundled recipe name or YAML path")
    parser.add_argument(
        "--reward-config",
        choices=REWARD_RECIPES,
        help="Optional HPSv3/VideoAlign HTTP reward recipe; merged before key=value overrides",
    )
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--check-config", action="store_true", help="Validate recipe only; does not load models/runtime")
    parser.add_argument("--print-config", action="store_true", help="Print recipe overlay, without launching training")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotted key=value overrides")
    args = parser.parse_args(argv)
    try:
        for item in args.overrides:
            if "=" not in item or item.startswith(("+", "~")):
                raise ValueError("Use plain key=value overrides; Hydra +/~ syntax is not used by this entrypoint.")
        recipe = load_training_recipe(args.config, args.reward_config)
        overlay = OmegaConf.merge(recipe, OmegaConf.from_dotlist(args.overrides))
        fewstep, missing = validate_recipe(overlay, require_inputs=not (args.check_config or args.print_config))
        if args.print_config:
            print(OmegaConf.to_yaml(overlay, resolve=True))
            return
        if args.check_config:
            print(f"Recipe valid: {fewstep.bridge}; runtime/model loading not checked.")
            if missing:
                print("Required before training: " + ", ".join(missing))
            return
        if sys.version_info < (3, 11):
            raise RuntimeError("The pinned verl-omni training runtime requires Python >= 3.11; core tests support 3.10.")
        checkout = prepare_runtime(args.checkout, check_only=True)
        installed = importlib.util.find_spec("verl_omni")
        if installed is None or Path(installed.origin).resolve().parent != checkout / "verl_omni":
            raise RuntimeError("Install the pinned checkout editable in the training environment: pip install -e third_party/verl-omni")
        config = resolve_runtime_paths(compose_runtime_config(recipe, args.overrides, checkout))
        validate_recipe(config, require_inputs=True)
        from verl_omni.trainer.main_diffusion_v1 import run_diffusion_v1

        run_diffusion_v1(config)
    except (ValueError, RuntimeError, ImportError, FileNotFoundError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
