"""DMD2 logistic GAN losses on a shared H3 fake-score backbone.

Source: https://github.com/tianweiy/DMD2/blob/main/main/sd_guidance.py
H3 adaptations: RF noising of image/joint AV latents, target-token classifier,
and the existing PDMD role schedule/trajectories (not DMD2's DDIM schedule).
Real samples keep their own text/reference conditioning; they need not match rollout prompts.
"""

import math

import torch
import torch.nn.functional as F

from .official_pdmd_core import joint_noise_like, positive_condition, stream_seed


def validate_gan_config(config):
    gan = config["training"].get("dmd", {}).get("gan", {})
    if not gan.get("enabled", False):
        return
    dmd = config["training"]["dmd"]
    model = config["model"]
    if not dmd.get("official_pdmd") or model["name"] not in {"minimax_h3_t2av", "minimax_h3_ref2av", "minimax_h3_image"}:
        raise ValueError("DMD2 GAN requires official PDMD with minimax_h3_t2av, minimax_h3_ref2av or minimax_h3_image")
    if config["training"]["fake"]["train_type"] != "full":
        raise ValueError("DMD2 GAN trains the shared full fake-score backbone")
    paired_manifest = config.get("data", {}).get("train", {}).get("name") == "h3_selfgen_manifest"
    objective = gan.get("objective", "dmd2")
    if objective not in {"dmd2", "dmad_single_head"}:
        raise ValueError(f"Unknown GAN objective: {objective}")
    if objective == "dmad_single_head" and not paired_manifest:
        raise ValueError("Single-head DMAD uses a paired H3 audio-video manifest")
    if not paired_manifest and not gan.get("real_cache"):
        raise ValueError("DMD2 GAN requires real_cache with clean target latents and their own conditions")
    for key, default in (("generator_weight", 0.005), ("discriminator_weight", 0.01)):
        value = float(gan.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.dmd.gan.{key} must be finite and positive")
    if not 0 < float(gan.get("sigma_max", 0.999)) < 1:
        raise ValueError("training.dmd.gan.sigma_max must be in (0, 1)")
    metadata_geometry = (
        (model["name"] == "minimax_h3_ref2av" or paired_manifest)
        and model.get("capabilities", {}).get("distribution_matching", {}).get("geometry_from_metadata") is True
        and not dmd.get("generation_shapes")
    )
    if len(dmd.get("generation_shapes") or ()) != 1 and not metadata_geometry:
        raise ValueError("DMD2 GAN requires one fixed generation geometry matching its real samples, or metadata-driven Ref2AV geometry")


def generator_logistic_loss(logits):
    return F.softplus(-logits.float()).mean()


def discriminator_logistic_loss(logits, *, real):
    return F.softplus(-logits.float() if real else logits.float()).mean()


def gan_sigmas(generator, device, maximum=0.999, *, image=False):
    """Uniform 1000 physical RF noise levels, never shifted a second time.

    DMD2 samples one diffusion timestep per image. For H3 the video and audio
    input use that same RF sigma; real/fake calls draw independently.
    """
    sigma = torch.randint(1000, (), generator=generator).float() * (maximum / 999)
    return (sigma if image else sigma.repeat(2)).to(device=device)


class Dmd2GanObjective:
    def __init__(self, trainer, real_stream):
        self.trainer = trainer
        self.real_stream = real_stream
        self.config = trainer.dmd_config["gan"]
        self.seed = int(trainer.config.get("seed", 42))
        self.generator_weight = float(self.config.get("generator_weight", 0.005))
        self.discriminator_weight = float(self.config.get("discriminator_weight", 0.01))

    def _sigmas(self, clean, iteration, slot, branch):
        t = self.trainer
        draws = torch.Generator().manual_seed(stream_seed(self.seed, "gan_sigma", branch, iteration, slot))
        image = torch.is_tensor(clean)
        return gan_sigmas(draws, t.student.device, float(self.config.get("sigma_max", 0.999)), image=image)

    def _head(self, features, image):
        return self.trainer.dmd2_gan_head(features["image" if image else "video"])

    def _generator_loss(self, logits):
        return generator_logistic_loss(logits)

    def _logits(self, clean, condition, iteration, slot, branch):
        t = self.trainer
        sigmas = self._sigmas(clean, iteration, slot, branch)
        generator = torch.Generator(device=t.student.device).manual_seed(stream_seed(self.seed, "gan_noise", branch, iteration, slot))
        noise = joint_noise_like(clean, generator)
        query = t.student.add_noise(None, clean, noise, sigmas)
        features = t.fake.predict_dmad_features(query, sigmas, positive_condition(condition), feature_block=int(self.config.get("feature_block", 49)))
        return self._head(features, torch.is_tensor(clean)), sigmas.reshape(-1)[0]

    def generator(self, clean, condition, iteration, slot):
        """Caller freezes fake/head through backward, without disabling autograd."""
        logits, sigma = self._logits(clean, condition, iteration, slot, "generator")
        raw = self._generator_loss(logits)
        return self.generator_weight * raw, {"gan_g": raw.detach(), "gan_g_logit": logits.detach().mean(), "gan_g_sigma": sigma.detach()}

    def backward_discriminator(self, trajectory, cursor, iteration, slot, backward_scale, *, real_sample=None):
        """Release each H3 graph before the next forward to limit peak memory."""
        t = self.trainer
        metrics = {}
        total = 0.0
        for real in (False, True):
            if real and self.real_stream is None:
                clean = t.student.extract_turbo_latents(real_sample, trajectory.states[-1].shape)
                condition = trajectory.condition
            elif real:
                real_options = {}
                if t.config.get("model", {}).get("name") == "minimax_h3_ref2av":
                    shape = trajectory.states[-1].shape
                    scale = t.student.model.vae_spatial_scale_factor
                    real_options["geometry"] = (shape.num_frames, shape.latent_height * scale, shape.latent_width * scale)
                clean, condition = self.real_stream.load(t.student, cursor, **real_options)
            else:
                clean = t.student.detach(trajectory.states[-1])
                condition = trajectory.condition
            branch = "real" if real else "fake"
            logits, sigma = self._logits(clean, condition, iteration, slot, branch)
            raw = discriminator_logistic_loss(logits, real=real)
            weighted = raw * self.discriminator_weight
            (weighted * backward_scale).backward()
            total += float(weighted.detach())
            metrics[f"gan_d_{branch}"] = float(raw.detach())
            metrics[f"gan_d_{branch}_logit"] = float(logits.detach().mean())
            metrics[f"gan_d_{branch}_sigma"] = float(sigma)
            del clean, condition, logits, raw, weighted
        return total, metrics

class SingleHeadDmadObjective(Dmd2GanObjective):
    """One teacher-positive classifier, sharing the PDMD fake-score backbone."""

    def _sigmas(self, clean, iteration, slot, branch):
        from .rollout_schedule import rollout_shifts

        # Both discriminator examples use the same time, with independent noise.
        draws = torch.Generator().manual_seed(stream_seed(self.seed, "dmad_sigma", iteration, slot))
        base = torch.empty(()).uniform_(0.02, 0.98, generator=draws)
        shifts = torch.tensor(rollout_shifts(self.trainer.dmd_config, iteration))
        return (shifts * base / (1 + (shifts - 1) * base)).to(self.trainer.student.device)

    def _head(self, features, image):
        return self.trainer.dmd2_gan_head(features["video"], features["audio"])

    def _generator_loss(self, logits):
        return -logits.float().mean()
