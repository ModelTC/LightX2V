"""Register FewStep objectives with verl-omni's worker-side loss dispatcher."""

from typing import Any

from verl_omni.trainer.diffusion.diffusion_algos import (
    DiffusionLossFn,
    DiffusionLossResult,
    register_diffusion_loss,
)

from lightx2v_rl.algorithms.fewstep import compute_policy_loss

from .config import get_fewstep_config


@register_diffusion_loss("fewstep_geo")
@register_diffusion_loss("fewstep_cps")
@register_diffusion_loss("fewstep_flow_grpo")
class FewStepLoss(DiffusionLossFn):
    """Shared dispatch for Geo-FSRL, CPS and Flow-GRPO's FewStep objectives."""

    required_model_output_keys = (
        "log_probs",
        "prev_sample_mean",
        "transition_std",
        "sigma",
        "next_sigma",
    )
    required_data_keys = ("old_log_probs", "advantages")

    @classmethod
    def compute_loss(cls, **kwargs: Any):
        result = compute_policy_loss(**kwargs)
        return result.loss, result.metrics

    def __call__(self, *, config, model_output, data):
        fewstep = get_fewstep_config(config.model_config)
        if config.diffusion_loss.loss_mode != f"fewstep_{fewstep.bridge}":
            raise ValueError("FewStep loss mode and model bridge must agree.")
        if config.diffusion_loss.clip_ratio != fewstep.clip_range:
            raise ValueError("diffusion_loss.clip_ratio must equal model.fewstep.clip_range.")
        if config.diffusion_loss.adv_clip_max != fewstep.adv_clip_max:
            raise ValueError("diffusion_loss.adv_clip_max must equal model.fewstep.adv_clip_max.")
        if fewstep.kl_beta > 0 and not config.use_kl_loss:
            raise ValueError("FewStep kl_beta > 0 requires actor.use_kl_loss=true to produce reference means.")

        loss, metrics = self.compute_loss(
            new_log_probs=model_output["log_probs"],
            old_log_probs=data["old_log_probs"],
            advantages=data["advantages"],
            sigma=model_output["sigma"],
            next_sigma=model_output["next_sigma"],
            config=fewstep,
            transition_mean=model_output["prev_sample_mean"],
            reference_mean=data.get("ref_prev_sample_mean", None),
            transition_std=model_output["transition_std"],
            rollout_is_weights=data.get("rollout_is_weights", None),
        )
        return DiffusionLossResult(
            loss=loss,
            metrics=metrics,
            add_loss_metric=True,
            includes_kl_loss=True,
        )
