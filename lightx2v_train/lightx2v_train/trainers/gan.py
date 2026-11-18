"""Adversarial losses with a classifier on the fake-score model's features."""

from contextlib import contextmanager
from math import isfinite

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from lightx2v_train.runtime.distributed import get_rank, get_world_size
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value

from .pdmd_trainers.utils import fork_rng, noise_like, seed_generator, seed_rng, stream_seed


class GanHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim, bins):
        super().__init__()
        self.bins = bins
        self.classifier = nn.Sequential(nn.Linear(feature_dim * bins, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features):
        length = features.shape[1]
        with torch.autocast(device_type=features.device.type, enabled=False):
            pooled = torch.stack(
                [features[:, i * length // self.bins : ((i + 1) * length + self.bins - 1) // self.bins].mean(1, dtype=torch.float32) for i in range(self.bins)],
                dim=1,
            )
            pooled = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + 1e-5)
            return self.classifier(pooled.flatten(1)).squeeze(-1)


class GanObjective:
    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.dmd_config["gan"]
        self.seed = int(trainer.config.get("seed", 42))
        self.generator_weight = float(self.config.get("generator_weight", 0.005))
        self.discriminator_weight = float(self.config.get("discriminator_weight", 0.01))
        self.sigma_max = float(self.config.get("sigma_max", 0.999))
        self.feature_block = self.config["feature_block"]
        if not 0 < self.sigma_max < 1 or any(not isfinite(weight) or weight <= 0 for weight in (self.generator_weight, self.discriminator_weight)):
            raise ValueError("GAN requires positive finite loss weights and a sigma_max in (0, 1).")
        if not hasattr(trainer.fake, "predict_gan_features"):
            raise ValueError("GAN requires the fake model to expose differentiable target features.")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed + 20903)
            self.head = GanHead(trainer.fake.gan_feature_dim, self.config.get("head_hidden_dim", 1024), self.config.get("head_temporal_bins", 4))
        self.head.to(device=trainer.student.device, dtype=torch.float32)
        if dist.is_initialized():
            for parameter in self.head.parameters():
                dist.broadcast(parameter.detach(), src=0)
        defaults = trainer.fake_optimizer.defaults
        self.optimizer = trainer._build_optimizer(
            self.head.parameters(),
            dict(
                learning_rate=defaults["lr"],
                adam_beta1=defaults["betas"][0],
                adam_beta2=defaults["betas"][1],
                weight_decay=defaults["weight_decay"],
                adam_epsilon=defaults["eps"],
            ),
        )
        self.scheduler = trainer._build_lr_scheduler(self.optimizer, num_warmup_steps=0, num_training_steps=trainer._optimizer_steps()[1])

    @contextmanager
    def frozen_critic(self):
        params = [*self.trainer.fake_trainable_params, *self.head.parameters()]
        for param in params:
            param.requires_grad_(False)
        try:
            yield
        finally:
            for param in params:
                param.requires_grad_(True)
            # A resume can enter G before any D forward. Restore FSDP's cached
            # grad dtype after lazily initializing the group with frozen params.
            for module in self.trainer.fake.denoiser().modules():
                if isinstance(module, FSDPModule):
                    state = module._get_fsdp_state()
                    if state._is_root is not None and state._fsdp_param_group is not None:
                        state._fsdp_param_group._init_mp_dtypes()

    def _logits(self, clean, condition, iteration, micro, branch):
        trainer = self.trainer
        device = trainer.student.device
        seed = (self.seed, iteration, get_rank(), micro, branch)
        steps = trainer.num_train_timestep
        sigma = torch.randint(steps, (1,), generator=seed_generator(*seed, "gan_sigma")).float() * (self.sigma_max / max(1, steps - 1))
        sigma = sigma.to(device)
        if not torch.is_tensor(clean):
            sigma = sigma.expand(2).contiguous()
        noise = noise_like(clean, seed_generator(*seed, "gan_noise", device=device))
        query = trainer.student.add_noise(trainer.scheduler, clean, noise, sigma)
        features = self.trainer.fake.predict_gan_features(query, sigma, condition, self.feature_block)
        return self.head(features), sigma[0]

    def generator(self, clean, condition, iteration, micro):
        logits, sigma = self._logits(clean, condition, iteration, micro, "generator")
        loss = F.softplus(-logits).mean()
        return self.generator_weight * loss, {"gan_g": loss.detach(), "gan_g_logit": logits.detach().mean(), "gan_g_sigma": sigma}

    def backward_discriminator(self, clean, condition, sample, index, iteration, micro, scale):
        trainer = self.trainer
        logits, fake_sigma = self._logits(trainer.student.detach(clean), condition, iteration, micro, "fake")
        fake_loss = F.softplus(logits).mean()
        (self.discriminator_weight * fake_loss * scale).backward()
        metrics = {"gan_d_fake": fake_loss.detach(), "gan_d_fake_logit": logits.detach().mean(), "gan_d_fake_sigma": fake_sigma}
        del logits, fake_loss

        dataset = trainer.dataloader_train.dataset
        with torch.no_grad(), fork_rng(trainer.student.device) as devices:
            seed_rng(stream_seed(self.seed, "gan_condition", iteration, get_rank(), micro), devices)
            if hasattr(dataset, "gan_sample"):
                geometry = trainer.student.gan_geometry(clean)
                sample = dataset.gan_sample(index * get_world_size() + get_rank(), geometry)
            real = trainer.student.extract_real_latents(sample, trainer.latent_dtype, broadcast_sequence_parallel_value)
            real_condition, _ = trainer._encode_conditions(sample)
        logits, real_sigma = self._logits(real, real_condition, iteration, micro, "real")
        real_loss = F.softplus(-logits).mean()
        (self.discriminator_weight * real_loss * scale).backward()
        metrics.update(gan_d_real=real_loss.detach(), gan_d_real_logit=logits.detach().mean(), gan_d_real_sigma=real_sigma)
        return self.discriminator_weight * (metrics["gan_d_fake"] + metrics["gan_d_real"]), metrics

    def clip_discriminator_gradients(self, max_norm):
        head_params = list(self.head.parameters())
        if dist.is_initialized():
            for param in head_params:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(get_world_size())
        fake_params = self.trainer.fake_trainable_params
        fake_norm = torch.nn.utils.get_total_norm([param.grad for param in fake_params if param.grad is not None])
        if isinstance(fake_norm, DTensor):
            fake_norm = fake_norm.full_tensor()
        head_norm = torch.nn.utils.get_total_norm([param.grad for param in head_params])
        norm = torch.linalg.vector_norm(torch.stack((fake_norm.float(), head_norm.float())))
        if bool(torch.isfinite(norm)):
            coefficient = (max_norm / (norm + 1e-6)).clamp(max=1).item()
            for param in (*fake_params, *head_params):
                if param.grad is not None:
                    param.grad.mul_(coefficient)
        return norm

    def state_dict(self):
        return dict(config=self.config, head=self.head.state_dict(), optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict())

    def load_state_dict(self, state):
        if state["config"] != self.config:
            raise ValueError("Checkpoint GAN configuration differs from the current run.")
        self.head.load_state_dict(state["head"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])

    def step(self):
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
