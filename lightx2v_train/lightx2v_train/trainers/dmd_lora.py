import os

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm

from lightx2v_train.runtime.checkpoint import prune_checkpoints
from lightx2v_train.utils.registry import TRAINER_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .base import BaseTrainer


def _linear_shift(mu, t):
    return mu / (mu + (1 / t - 1))


def _add_noise(x0, noise, sigma):
    sigma = _expand_to(sigma, x0).to(dtype=torch.float32)
    return ((1.0 - sigma) * x0.float() + sigma * noise.float()).to(dtype=x0.dtype)


def _euler_step(x, velocity, sigma, target_sigma):
    sigma = _expand_to(sigma, x).to(dtype=torch.float32)
    target_sigma = _expand_to(target_sigma, x).to(dtype=torch.float32)
    return x.float() + (target_sigma - sigma) * velocity.float()


def _expand_to(value, target):
    value = value.to(device=target.device)
    while value.ndim < target.ndim:
        value = value.view(*value.shape, 1)
    return value


def _do_cfg(cond_pred, uncond_pred, cfg_scale, cfg_norm):
    pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
    if cfg_norm in (None, "none"):
        return pred
    if cfg_norm == "layer_norm":
        cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True)
        pred_norm = torch.norm(pred, dim=-1, keepdim=True)
        return pred * (cond_norm / torch.clamp(pred_norm, min=1e-12))
    if cfg_norm == "scalar":
        cond_norm = torch.norm(cond_pred)
        pred_norm = torch.norm(pred)
        return pred * min(1.0, (cond_norm / torch.clamp(pred_norm, min=1e-12)).item())
    raise ValueError(f"Unsupported cfg_norm: {cfg_norm}")


def _dmd_loss(latents, x_pred_fake_flow, x_pred_teacher):
    with torch.no_grad():
        grad = x_pred_fake_flow - x_pred_teacher
        dims = tuple(range(1, latents.ndim))
        normalizer = torch.abs(latents - x_pred_teacher).mean(dim=dims, keepdim=True)
        grad = torch.nan_to_num(grad / normalizer)
    return 0.5 * F.mse_loss(latents.float(), (latents.float() - grad.float()).detach(), reduction="mean")


class _DMDEulerScheduler:
    def __init__(self, shift=3.0, device="cuda"):
        self.shift = float(shift)
        self.device = torch.device(device)
        self.num_train_timesteps = 1000

    def set_timesteps(self, num_inference_steps):
        timesteps = torch.linspace(
            1000,
            0,
            int(num_inference_steps) + 1,
            dtype=torch.float32,
            device=self.device,
        )
        self.sigmas = _linear_shift(self.shift, timesteps / self.num_train_timesteps)

    def step(self, model_output, step_idx, sample):
        sigma = self.sigmas[step_idx].expand(sample.shape[0]).to(sample.device)
        sigma_next = self.sigmas[step_idx + 1].expand(sample.shape[0]).to(sample.device)
        x0 = sample.float() - _expand_to(sigma, sample).float() * model_output.float()
        next_sample = _euler_step(sample, model_output, sigma, sigma_next)
        return next_sample.to(sample.dtype), x0.to(sample.dtype)


@TRAINER_REGISTER("dmd_lora")
class DmdLoraTrainer(BaseTrainer):
    def get_configs(self):
        model_config = self.config["model"]
        if model_config.get("name") != "qwen_image":
            raise ValueError("dmd_lora currently supports model.name: qwen_image only.")
        self.running_dtype = get_running_dtype(model_config["running_dtype"])

        training_config = self.config["training"]
        lora_config = training_config.get("lora", {})
        self.lora_rank = lora_config.get("rank", 16)
        self.lora_alpha = lora_config.get("alpha", self.lora_rank)
        self.lora_target_modules = lora_config.get("target_modules")

        fake_config = training_config.get("fake", {})
        fake_lora_config = fake_config.get("lora", lora_config)
        self.fake_lora_rank = fake_lora_config.get("rank", self.lora_rank)
        self.fake_lora_alpha = fake_lora_config.get("alpha", self.fake_lora_rank)
        self.fake_lora_target_modules = fake_lora_config.get("target_modules", self.lora_target_modules)

        self.gradient_checkpointing = training_config.get("gradient_checkpointing", True)

        optimizer_config = training_config.get("optimizer", {})
        self.optimizer_learning_rate = optimizer_config.get("learning_rate", 1e-4)
        self.optimizer_adam_beta1 = optimizer_config.get("adam_beta1", 0.9)
        self.optimizer_adam_beta2 = optimizer_config.get("adam_beta2", 0.999)
        self.optimizer_weight_decay = optimizer_config.get("weight_decay", 0.01)
        self.optimizer_adam_epsilon = optimizer_config.get("adam_epsilon", 1e-8)

        fake_optimizer_config = fake_config.get("optimizer", {})
        self.fake_optimizer_learning_rate = fake_optimizer_config.get("learning_rate", self.optimizer_learning_rate)
        self.fake_optimizer_adam_beta1 = fake_optimizer_config.get("adam_beta1", self.optimizer_adam_beta1)
        self.fake_optimizer_adam_beta2 = fake_optimizer_config.get("adam_beta2", self.optimizer_adam_beta2)
        self.fake_optimizer_weight_decay = fake_optimizer_config.get("weight_decay", self.optimizer_weight_decay)
        self.fake_optimizer_adam_epsilon = fake_optimizer_config.get("adam_epsilon", self.optimizer_adam_epsilon)

        self.lr_scheduler_name = training_config.get("lr_scheduler", "constant")
        self.lr_warmup_iters = training_config.get("lr_warmup_iters", 0)
        self.max_train_iters = training_config["max_train_iters"]

        self.output_dir = training_config["output_dir"]
        self.gradient_accumulation_iters = training_config.get("gradient_accumulation_iters", 1)
        self.max_grad_norm = training_config.get("max_grad_norm", 1.0)
        self.save_every_iters = training_config.get("save_every_iters", 0)
        self.save_total_limit = training_config.get("save_total_limit")
        self.save_fake_lora = fake_config.get("save_lora", False)

        dmd_config = training_config.get("dmd", {})
        self.num_inference_steps = int(dmd_config.get("num_inference_steps", 4))
        self.fake_update_ratio = int(dmd_config.get("fake_update_ratio", 1))
        self.guidance_scale = float(dmd_config.get("guidance_scale", 3.0))
        self.negative_prompt = dmd_config.get("negative_prompt", " ")
        self.cfg_norm = dmd_config.get("cfg_norm", "layer_norm")
        self.min_sigma = float(dmd_config.get("sigma_min", 0.02))
        self.max_sigma = float(dmd_config.get("sigma_max", 1.0))
        self.discrete_samples = int(dmd_config.get("discrete_samples", 1000))
        self.renoise_shift = float(dmd_config.get("renoise_shift", 5.0))
        self.inference_shift = float(dmd_config.get("inference_shift", 3.0))

    def setup(self):
        self.get_configs()
        print("[dmd_lora] single-GPU resident mode: student/fake/teacher transformers stay on CUDA")

        self.model.add_lora(self.lora_rank, self.lora_alpha, self.lora_target_modules)
        self.model.set_lora_trainable()
        if self.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        self.fake_transformer = self.model.load_transformer()
        self._add_lora_to_transformer(
            self.fake_transformer,
            self.fake_lora_rank,
            self.fake_lora_alpha,
            self.fake_lora_target_modules,
        )
        self._set_lora_trainable(self.fake_transformer)
        if self.gradient_checkpointing and hasattr(self.fake_transformer, "enable_gradient_checkpointing"):
            self.fake_transformer.enable_gradient_checkpointing()

        self.teacher_transformer = self.model.load_transformer()
        self.teacher_transformer.requires_grad_(False)
        self.teacher_transformer.eval()

        self.optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=self.optimizer_learning_rate,
            betas=(self.optimizer_adam_beta1, self.optimizer_adam_beta2),
            weight_decay=self.optimizer_weight_decay,
            eps=self.optimizer_adam_epsilon,
        )
        self.fake_optimizer = torch.optim.AdamW(
            (p for p in self.fake_transformer.parameters() if p.requires_grad),
            lr=self.fake_optimizer_learning_rate,
            betas=(self.fake_optimizer_adam_beta1, self.fake_optimizer_adam_beta2),
            weight_decay=self.fake_optimizer_weight_decay,
            eps=self.fake_optimizer_adam_epsilon,
        )
        self.lr_scheduler = get_scheduler(
            self.lr_scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=self.lr_warmup_iters,
            num_training_steps=self.max_train_iters,
        )
        self.fake_lr_scheduler = get_scheduler(
            self.lr_scheduler_name,
            optimizer=self.fake_optimizer,
            num_warmup_steps=0,
            num_training_steps=max(1, self.max_train_iters * self.fake_update_ratio),
        )
        self.scheduler = _DMDEulerScheduler(shift=self.inference_shift, device=self.model.device)

        print(f"[dmd_lora] student trainable params={self._count_trainable(self.model.transformer)}")
        print(f"[dmd_lora] fake trainable params={self._count_trainable(self.fake_transformer)}")

    @staticmethod
    def _add_lora_to_transformer(transformer, rank, alpha, target_modules):
        transformer.add_adapter(
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
        )

    @staticmethod
    def _set_lora_trainable(transformer):
        transformer.requires_grad_(False)
        transformer.train()
        for name, param in transformer.named_parameters():
            param.requires_grad = "lora" in name

    @staticmethod
    def _count_trainable(module):
        return sum(1 for param in module.parameters() if param.requires_grad)

    def _latent_shape(self, sample):
        image = sample["target_image"]
        batch_size = image.shape[0]
        latent_channels = getattr(self.model.vae.config, "z_dim", None)
        if latent_channels is None:
            latent_channels = self.model.transformer.config.in_channels // 4
        return (
            batch_size,
            1,
            int(latent_channels),
            image.shape[-2] // self.model.vae_scale_factor,
            image.shape[-1] // self.model.vae_scale_factor,
        )

    def _encode_conditions(self, sample):
        prompt = sample["prompt"]
        if isinstance(prompt, str):
            negative_prompt = self.negative_prompt
        else:
            negative_prompt = [self.negative_prompt] * len(prompt)
        with torch.no_grad():
            condition = self.model.encode_prompt_condition(prompt)
            negative_condition = self.model.encode_prompt_condition(negative_prompt)
        return condition, negative_condition

    def _predict_velocity(self, transformer, latents, sigma, condition):
        denoiser_input = self.model.prepare_denoiser_input(latents, {}, condition)
        prediction = self.model.denoise_with_transformer(transformer, denoiser_input, sigma, condition)
        prediction = self.model.postprocess_denoiser_output(prediction, denoiser_input)
        return self.model.prepare_flow_matching_target(prediction)

    def sample_initial_latents(self, latent_shape):
        return torch.randn(latent_shape, device=self.model.device, dtype=self.running_dtype)

    def sample_end_step(self):
        return int(torch.randint(0, self.num_inference_steps, (1,), device=self.model.device).item())

    def sample_renoise_sigma(self, batch_size):
        raw = torch.rand((batch_size,), device=self.model.device, dtype=torch.float32)
        if self.discrete_samples > 0:
            raw = torch.ceil(raw * self.discrete_samples) / self.discrete_samples
        raw = torch.clamp(raw, 1e-7, 1 - 1e-7)
        return torch.clamp(_linear_shift(self.renoise_shift, raw), self.min_sigma, self.max_sigma).to(self.running_dtype)

    def run_back_simulation(self, condition, latent_shape, end_step_idx, grad_enabled, xt=None):
        self.scheduler.set_timesteps(self.num_inference_steps)
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        x0 = None
        self.model.transformer.train()
        for idx in range(end_step_idx + 1):
            sigma = self.scheduler.sigmas[idx].expand(latent_shape[0]).to(self.model.device, self.running_dtype)
            context = torch.enable_grad if (grad_enabled and idx == end_step_idx) else torch.no_grad
            with context():
                velocity = self._predict_velocity(self.model.transformer, xt, sigma, condition)
            xt, x0 = self.scheduler.step(velocity, idx, xt)
        return x0

    def forward_loss(self, sample, stage):
        condition, negative_condition = self._encode_conditions(sample)
        latent_shape = self._latent_shape(sample)
        end_step_idx = self.sample_end_step()
        xt_start = self.sample_initial_latents(latent_shape)
        x0_ref = self.run_back_simulation(condition, latent_shape, end_step_idx, grad_enabled=False, xt=xt_start)

        sigma = self.sample_renoise_sigma(latent_shape[0])
        noise = torch.randn(latent_shape, device=self.model.device, dtype=torch.float32)
        renoised_xt = _add_noise(x0_ref, noise, sigma)
        velocity_gt = noise - x0_ref.float()

        if stage == "fake":
            self.fake_transformer.train()
            velocity_fake = self._predict_velocity(self.fake_transformer, renoised_xt, sigma, condition)
            return F.mse_loss(velocity_fake.float(), velocity_gt.float(), reduction="mean")

        with torch.no_grad():
            self.fake_transformer.eval()
            velocity_fake = self._predict_velocity(self.fake_transformer, renoised_xt, sigma, condition)
            velocity_teacher_cond = self._predict_velocity(self.teacher_transformer, renoised_xt, sigma, condition)
            velocity_teacher_uncond = self._predict_velocity(self.teacher_transformer, renoised_xt, sigma, negative_condition)
            velocity_teacher = _do_cfg(velocity_teacher_cond, velocity_teacher_uncond, self.guidance_scale, self.cfg_norm)

        zeros = torch.zeros_like(sigma)
        x_pred_fake = _euler_step(renoised_xt, velocity_fake, sigma, zeros)
        x_pred_teacher = _euler_step(renoised_xt, velocity_teacher, sigma, zeros)
        x0 = self.run_back_simulation(condition, latent_shape, end_step_idx, grad_enabled=True, xt=xt_start)
        return _dmd_loss(x0, x_pred_fake, x_pred_teacher)

    def train(self):
        self.setup()
        os.makedirs(self.output_dir, exist_ok=True)

        current_iter = 0
        running_dmd = 0.0
        running_fake = 0.0
        progress = tqdm(total=self.max_train_iters, desc="DMD-LoRA iterations")

        while current_iter < self.max_train_iters:
            for sample in self.dataloader:
                loss_dmd = self.forward_loss(sample, stage="generator")
                loss_dmd.backward()
                torch.nn.utils.clip_grad_norm_(self.model.transformer.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                running_dmd += loss_dmd.detach().float().item()

                fake_losses = []
                for _ in range(self.fake_update_ratio):
                    loss_fake = self.forward_loss(sample, stage="fake")
                    loss_fake.backward()
                    torch.nn.utils.clip_grad_norm_(self.fake_transformer.parameters(), self.max_grad_norm)
                    self.fake_optimizer.step()
                    self.fake_lr_scheduler.step()
                    self.fake_optimizer.zero_grad(set_to_none=True)
                    fake_losses.append(loss_fake.detach())
                if fake_losses:
                    running_fake += torch.stack(fake_losses).mean().float().item()

                current_iter += 1
                progress.update(1)
                progress.set_postfix(
                    dmd=running_dmd,
                    fake=running_fake,
                    lr=self.lr_scheduler.get_last_lr()[0],
                )
                running_dmd = 0.0
                running_fake = 0.0

                if self.save_every_iters and current_iter % self.save_every_iters == 0:
                    self.save_checkpoint(current_iter, self.save_total_limit)

                if current_iter >= self.max_train_iters:
                    break

        progress.close()

    def save_checkpoint(self, iteration, save_total_limit):
        prune_checkpoints(self.output_dir, save_total_limit)
        save_dir = os.path.join(self.output_dir, f"checkpoint-{iteration}")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_lora_weights(save_dir)
        if self.save_fake_lora:
            fake_dir = os.path.join(save_dir, "fake")
            os.makedirs(fake_dir, exist_ok=True)
            fake_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.fake_transformer))
            self.model.pipeline_cls.save_lora_weights(fake_dir, fake_state, safe_serialization=True)
