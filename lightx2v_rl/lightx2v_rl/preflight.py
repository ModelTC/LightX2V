"""Dispatch preflight checks to the selected model without loading other families."""

import argparse
import json

from omegaconf import OmegaConf

from lightx2v_rl.models.registry import get_model_spec, resolve_hook
from lightx2v_rl.training.config import load_recipe


def check(config_name, **kwargs):
    overrides = kwargs.get("overrides") or []
    if any("=" not in item or item.startswith(("+", "~")) for item in overrides):
        raise ValueError("Use plain key=value preflight overrides.")
    config = OmegaConf.merge(load_recipe(config_name), OmegaConf.from_dotlist(overrides))
    model = config.actor_rollout_ref.model
    spec = get_model_spec(model.architecture, model.get("rl_adapter"))
    if spec.preflight is None:
        raise ValueError(f"Model-specific preflight is not implemented for {spec.architecture}.")
    return resolve_hook(spec.preflight)(config_name, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen_image21_turbo_geo_40gpu")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--skip-services", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    try:
        result = check(args.config, skip_cuda=args.skip_cuda, skip_services=args.skip_services, overrides=args.overrides)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Preflight failed: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
