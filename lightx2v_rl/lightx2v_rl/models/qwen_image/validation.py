"""Recipe constraints required for official Qwen-Image-2.1-Turbo parity."""

import ipaddress
from urllib.parse import urlsplit

from omegaconf import OmegaConf


def validate_recipe(config, fewstep):
    """Reject changes which silently break official sampling or task alternation."""
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    if (fewstep.min_steps, fewstep.max_steps) != (8, 8):
        raise ValueError("Qwen-Image-2.1-Turbo requires its saved eight-step schedule.")
    if fewstep.timestep_shift != 1 or fewstep.timestep_sampling_mode != "fixed_uniform":
        raise ValueError("Qwen Turbo preserves its checkpoint schedule without an additional timestep shift.")
    for pipeline in (rollout.pipeline, rollout.val_kwargs.pipeline):
        if pipeline.true_cfg_scale != 1 or pipeline.guidance_scale not in (None, 1):
            raise ValueError("Official Qwen-Image-2.1-Turbo sampling uses CFG=1.")
        if pipeline.num_inference_steps != 8:
            raise ValueError("Qwen Turbo advertises eight steps; its checkpoint grid is authoritative.")
        if pipeline.height < 32 or pipeline.width < 32 or pipeline.height % 32 or pipeline.width % 32:
            raise ValueError("Qwen canvas dimensions must be positive multiples of 32.")
    if rollout.pipeline.get("use_kv_cache", False):
        raise ValueError("Training disables KV caching for differentiable full-prefix replay.")
    if actor.ppo_micro_batch_size_per_gpu != 1 or rollout.log_prob_micro_batch_size_per_gpu != 1:
        raise ValueError("QwenImage21 requires training and logprob microbatches of one for variable references.")
    if config.actor_rollout_ref.ref.get("log_prob_micro_batch_size_per_gpu", 1) != 1:
        raise ValueError("QwenImage21 reference-policy microbatches must also be one.")
    if rollout.tensor_model_parallel_size != 1 or rollout.get("ulysses_degree", 1) != 1 or rollout.get("ring_degree", 1) != 1:
        raise ValueError("The official Diffusers Qwen rollout uses one GPU per replica.")
    if rollout.get("step_execution", False) or rollout.max_num_seqs != 1:
        raise ValueError("The official Qwen pipeline requires serial whole-request execution.")
    if rollout.max_prompt_embed_length != rollout.pipeline.replay_prompt_length:
        raise ValueError("Replay padding and rollout.max_prompt_embed_length must agree.")
    if config.data.get("gen_batch_size") not in (None, config.data.train_batch_size):
        raise ValueError("Task alternation requires gen_batch_size == train_batch_size.")
    if actor.ppo_mini_batch_size != config.data.train_batch_size:
        raise ValueError("Use one optimizer mini-batch per task batch for exact 1:1 updates.")
    if OmegaConf.select(config, "trainer.v1.sync.parameter_sync_step", default=1) != 1:
        raise ValueError("Task alternation requires parameter_sync_step=1.")
    if OmegaConf.select(config, "trainer.v1.sampler.drop_incomplete_groups", default=False):
        raise ValueError("Incomplete-group refill can cross task boundaries; disable it for exact alternation.")
    if OmegaConf.select(config, "data.sampler.class_name") != "AlternatingTaskSampler":
        raise ValueError("Qwen mixed training requires the alternating task sampler.")
    if not config.algorithm.get("weight_advantages", True) and config.algorithm.get("global_std", False):
        raise ValueError("Qwen combined-reward advantages require per-input-group normalization.")
    if config.data.val_batch_size < 1 or config.data.get("validation_shuffle", False):
        raise ValueError("Qwen validation requires a positive batch size and ordered per-task metrics.")
    nodes = int(config.trainer.get("nnodes", 1))
    gpus_per_node = int(config.trainer.n_gpus_per_node)
    if min(nodes, gpus_per_node, config.data.train_batch_size) < 1:
        raise ValueError("Training node, GPU and input batch counts must be positive.")
    world_size = nodes * gpus_per_node
    if config.data.train_batch_size * rollout.n % world_size:
        raise ValueError("train_batch_size * rollout.n must be divisible by the training GPU count.")
    if nodes > 1:
        from lightx2v_rl.rewards.endpoint_pool import normalize_endpoints

        settings = config.reward.custom_reward_function.reward_kwargs
        for name, weight in (("unifiedreward", "unifiedreward_weight"), ("hps", "hps_weight")):
            if settings.get(weight, 0) <= 0:
                continue
            for endpoint in normalize_endpoints(settings[name + "_endpoint"], name=name + "_endpoint"):
                host = urlsplit(endpoint).hostname
                try:
                    address = ipaddress.ip_address(host)
                    local = address.is_loopback or address.is_unspecified
                except ValueError:
                    local = host.lower() in {"localhost", "localhost.localdomain"}
                if local:
                    raise ValueError(f"Multi-node {name} reward needs a host reachable from every worker, got {endpoint}.")
