"""Portable recipe composition and shared learner validation."""

from pathlib import Path

from omegaconf import OmegaConf

from lightx2v_rl.algorithms.fewstep import FewStepConfig
from lightx2v_rl.models.registry import get_model_spec, validate_model_recipe
from lightx2v_rl.training.advantage import validate_advantage_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
REWARD_RECIPES = ("hpsv3", "videoalign", "hpsv3_videoalign")
RECIPE_ALIASES = {
    **{f"wan_{name}": f"wan/{name}" for name in ("base", "geo", "cps", "flow_grpo")},
    **{f"reward_{name}": f"rewards/{name}" for name in ("base", *REWARD_RECIPES)},
    **{
        f"qwen_image21_turbo_{bridge}{suffix}": f"qwen_image21_turbo/{bridge}{profile}"
        for bridge in ("geo", "cps", "flow_grpo")
        for suffix, profile in (("", ""), ("_32gpu", "_4nodes"), ("_40gpu", "_5nodes"))
    },
    "qwen_image21_agent": "qwen_image21_turbo/agent",
}


def load_recipe(name: str, _seen: frozenset[Path] = frozenset()):
    path = Path(name)
    if not path.is_file():
        key = str(name).removesuffix(".yaml")
        path = CONFIG_DIR / f"{RECIPE_ALIASES.get(key, key)}.yaml"
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Recursive recipe inheritance: {path}")
    config = OmegaConf.load(path)
    parents = config.pop("inherits", [])
    if isinstance(parents, str):
        parents = [parents]
    base = OmegaConf.create({})
    for parent in parents:
        sibling = path.parent / f"{str(parent).removesuffix('.yaml')}.yaml"
        inherited = load_recipe(str(sibling) if sibling.is_file() else parent, _seen | {path})
        base = OmegaConf.merge(base, inherited)
    return OmegaConf.merge(base, config)


def load_training_recipe(name: str, reward_config: str | None = None):
    """Merge algorithm, then optional reward recipe; caller applies CLI last."""
    recipe = load_recipe(name)
    if reward_config is not None:
        if reward_config not in REWARD_RECIPES:
            raise ValueError(f"Unknown reward recipe: {reward_config}; choose from {', '.join(REWARD_RECIPES)}")
        recipe = OmegaConf.merge(recipe, load_recipe(f"reward_{reward_config}"))
    return recipe


def validate_recipe(config, *, require_inputs: bool = False):
    model = config.actor_rollout_ref.model
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    fewstep = FewStepConfig(**OmegaConf.to_container(model.fewstep, resolve=True))
    if model.algorithm != f"fewstep_{fewstep.bridge}" or actor.diffusion_loss.loss_mode != model.algorithm:
        raise ValueError("Model algorithm, FewStep bridge and loss mode must agree.")
    get_model_spec(model.architecture, model.get("rl_adapter"))
    if model.get("lora_rank", 0) != 0:
        raise ValueError("FewStep recipes currently support full-weight training only.")
    if actor.get("enable_timestep_staging", False):
        raise ValueError("FewStep video transitions require actor.enable_timestep_staging=false.")
    if actor.strategy not in ("fsdp", "fsdp2"):
        raise ValueError("The initial FewStep adapter requires the FSDP training engine.")
    for role in (actor, config.actor_rollout_ref.ref):
        engine = role.fsdp_config
        param_dtype = (engine.get("mixed_precision") or {}).get("param_dtype", "bf16")
        if engine.model_dtype not in ("bf16", "bfloat16") or param_dtype not in ("bf16", "bfloat16"):
            raise ValueError("FewStep adapters require bf16 model weights and FSDP parameter dtype.")
    for algo in (rollout.algo, rollout.val_kwargs.algo):
        if OmegaConf.to_container(algo.fewstep, resolve=True) != OmegaConf.to_container(model.fewstep, resolve=True):
            raise ValueError("Rollout and learner FewStep options must match.")
    if actor.diffusion_loss.clip_ratio != fewstep.clip_range or actor.diffusion_loss.adv_clip_max != fewstep.adv_clip_max:
        raise ValueError("Loss clipping options must match model.fewstep.")
    if fewstep.kl_beta > 0 and not actor.use_kl_loss:
        raise ValueError("Positive FewStep kl_beta requires actor.use_kl_loss=true.")
    if not rollout.calculate_log_probs or rollout.n < 2:
        raise ValueError("FewStep GRPO requires calculate_log_probs=true and rollout.n >= 2.")
    if rollout.pipeline.guidance_scale != rollout.pipeline.true_cfg_scale:
        raise ValueError("Rollout guidance_scale must equal true_cfg_scale.")
    if config.algorithm.adv_estimator != "flow_grpo":
        raise ValueError("FewStep recipes require the flow_grpo advantage estimator.")
    advantage_weights = validate_advantage_config(config.algorithm)
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "sync":
        raise ValueError("The initial FewStep recipes require the synchronous v1 trainer.")
    validate_model_recipe(config, fewstep)
    # Custom reward functions and native multi-reward scorers are both provided
    # by upstream. Do not silently train against a placeholder reward.
    required = ["actor_rollout_ref.model.path", "data.train_files", "data.val_files"]
    if not OmegaConf.select(config, "reward.reward_functions", default={}):
        required.append("reward.custom_reward_function.path")
    missing = [key for key in required if OmegaConf.select(config, key, default=None) in (None, "")]
    if config.algorithm.get("weight_advantages", True) and not advantage_weights:
        missing.append("algorithm.reward_advantage.weights")
    if require_inputs and missing:
        raise ValueError("Set required runtime inputs: " + ", ".join(missing))
    return fewstep, missing


def compose_runtime_config(recipe, overrides: list[str], checkout: Path):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(checkout / "verl_omni/trainer/config"), version_base=None):
        base = compose(config_name="diffusion_trainer")
    OmegaConf.set_struct(base, False)
    return OmegaConf.merge(base, recipe, OmegaConf.from_dotlist(overrides))
