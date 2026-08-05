import torch
import torch.distributed as dist

from lightx2v_train.model_zoo.native.ltx2 import Modality
from lightx2v_train.runtime.distributed import get_sequence_parallel_group, get_sequence_parallel_world_size, is_sequence_parallel_enabled
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value, sequence_parallel_frame_slice, sync_sequence_parallel_gradients
from lightx2v_train.schedulers.flow_matching import CausalForcingFlowMatchScheduler
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .flow import FlowMatchingTrainer, LTX2T2AVFlowTrainer


@TRAINER_REGISTER("teacher_forcing")
class TFTrainer(FlowMatchingTrainer):
    trainer_name = "teacher_forcing"

    def __init__(self, config):
        super().__init__(config)
        if self.train_type != "full":
            raise ValueError("teacher_forcing trainer only supports training.train_type='full'.")

        self.teacher_forcing_config = self.training_config.get("teacher_forcing", {})
        self.teacher_forcing_mode = self.teacher_forcing_config.get("mode", "chunkwise")
        if self.teacher_forcing_mode != "chunkwise":
            raise ValueError(f"Unsupported teacher_forcing.mode={self.teacher_forcing_mode!r}; only 'chunkwise' is implemented.")

        self.num_frame_per_chunk = int(self.teacher_forcing_config.get("num_frame_per_chunk"))
        self.noise_augmentation_max_timestep = int(self.teacher_forcing_config.get("noise_augmentation_max_timestep", 0))
        time_shift_settings = self.config["scheduler"].get("time_shift_settings", {})
        self.teacher_forcing_scheduler = CausalForcingFlowMatchScheduler(
            num_train_timesteps=self.config["scheduler"].get("num_train_timesteps", 1000),
            time_shift_settings=time_shift_settings,
        )

    def compute_loss_on_sample(self, sample):
        if not hasattr(self.model, "denoise_teacher_forcing"):
            raise RuntimeError("teacher_forcing trainer currently requires the wan_t2v model.")

        with torch.no_grad():
            latent = self._teacher_forcing_latent(sample)
            batch_size, _, num_frames, _, _ = latent.shape
            sp_size = get_sequence_parallel_world_size() if is_sequence_parallel_enabled() else 1
            frame_multiple = sp_size * self.num_frame_per_chunk
            if num_frames % frame_multiple != 0:
                raise ValueError(f"Teacher forcing num_frames={num_frames} must be divisible by sp_size * num_frame_per_chunk = {sp_size} * {self.num_frame_per_chunk} = {frame_multiple}.")
            latent = broadcast_sequence_parallel_value(latent)

            noise = torch.randn_like(latent, dtype=self.running_dtype)
            noise = broadcast_sequence_parallel_value(noise)
            timestep_or_sigma, timestep_weights = self.teacher_forcing_scheduler.sample_chunkwise(
                batch_size=batch_size,
                num_frames=num_frames,
                num_frame_per_chunk=self.num_frame_per_chunk,
                device=latent.device,
                dtype=self.running_dtype,
            )
            timestep_or_sigma = broadcast_sequence_parallel_value(timestep_or_sigma)
            timestep_weights = broadcast_sequence_parallel_value(timestep_weights)
            noisy_latent = self.teacher_forcing_scheduler.add_noise(latent, noise, timestep_or_sigma)
            condition = self.model.encode_condition(sample)
            condition = broadcast_sequence_parallel_value(condition)

            clean_latent = latent
            aug_timestep_or_sigma = None
            if self.noise_augmentation_max_timestep > 0:
                aug_timestep_or_sigma = self.teacher_forcing_scheduler.sample_clean_augmentation(
                    batch_size=batch_size,
                    num_frames=num_frames,
                    num_frame_per_chunk=self.num_frame_per_chunk,
                    max_timestep=self.noise_augmentation_max_timestep,
                    device=latent.device,
                    dtype=self.running_dtype,
                )
                aug_timestep_or_sigma = broadcast_sequence_parallel_value(aug_timestep_or_sigma)
                clean_latent = self.teacher_forcing_scheduler.add_noise(latent, noise, aug_timestep_or_sigma)

            frame_start, frame_end, _ = sequence_parallel_frame_slice(num_frames, self.num_frame_per_chunk)
            latent = latent[:, :, frame_start:frame_end].contiguous()
            noise = noise[:, :, frame_start:frame_end].contiguous()
            noisy_latent = noisy_latent[:, :, frame_start:frame_end].contiguous()
            clean_latent = clean_latent[:, :, frame_start:frame_end].contiguous()
            timestep_or_sigma = timestep_or_sigma[:, frame_start:frame_end].contiguous()
            timestep_weights = timestep_weights[:, frame_start:frame_end].contiguous()
            if aug_timestep_or_sigma is not None:
                aug_timestep_or_sigma = aug_timestep_or_sigma[:, frame_start:frame_end].contiguous()

        prediction = self.model.denoise_teacher_forcing(
            noisy_latent,
            timestep_or_sigma,
            condition,
            clean_latent=clean_latent,
            aug_timestep_or_sigma=aug_timestep_or_sigma,
            frame_offset=frame_start,
            global_num_frames=num_frames,
        )

        target = self.noise_scheduler.build_train_gt(latent, noise)
        frame_loss = ((prediction.float() - target.float()) ** 2).mean(dim=(1, 3, 4))
        weighted_loss = (frame_loss * timestep_weights).sum()
        if is_sequence_parallel_enabled():
            dist.all_reduce(weighted_loss, op=dist.ReduceOp.SUM, group=get_sequence_parallel_group())
        return weighted_loss / (batch_size * num_frames)

    def _teacher_forcing_latent(self, sample):
        latent = sample["inputs"].get("latents")
        if latent is None:
            latent = self.model.encode_to_latent(sample)
        latent = latent.to(device=self.model.device, dtype=self.running_dtype)
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)

        latent_channels = self._latent_channels()
        if latent_channels is not None and latent.shape[1] != latent_channels and latent.shape[2] == latent_channels:
            latent = latent.permute(0, 2, 1, 3, 4).contiguous()
        if latent_channels is not None and latent.shape[1] != latent_channels:
            raise ValueError(f"Teacher forcing latent channels={latent.shape[1]} does not match model latent_channels={latent_channels}.")
        return latent

    def _latent_channels(self):
        return int(self.model_config.get("latent_channels", 16))

    def _after_backward(self):
        sync_sequence_parallel_gradients(self.trainable_params)


@TRAINER_REGISTER("ltx_t2av_teacher_forcing")
class LTX2T2AVTeacherForcingTrainer(LTX2T2AVFlowTrainer):
    trainer_name = "ltx_t2av_teacher_forcing"

    def __init__(self, config):
        super().__init__(config)
        if self.train_type != "full":
            raise ValueError("ltx_t2av_teacher_forcing currently supports training.train_type='full'.")
        self.teacher_forcing_config = self.training_config.get("teacher_forcing", {})
        self.num_frame_per_chunk = int(self.teacher_forcing_config.get("num_frame_per_chunk", 1))
        self.noise_augmentation_max_timestep = int(self.teacher_forcing_config.get("noise_augmentation_max_timestep", 0))
        time_shift_settings = self.config["scheduler"].get("time_shift_settings", {})
        self.teacher_forcing_scheduler = CausalForcingFlowMatchScheduler(
            num_train_timesteps=self.config["scheduler"].get("num_train_timesteps", 1000),
            time_shift_settings=time_shift_settings,
        )
        self._teacher_forcing_mask_cache = {}

    def compute_loss_on_sample(self, sample):
        if is_sequence_parallel_enabled():
            raise ValueError("ltx_t2av_teacher_forcing does not support sequence parallel yet.")

        inputs = sample["inputs"]
        video_data = inputs.get("video_latents")
        audio_data = inputs.get("audio_latents")
        conditions = sample["conditioning"].get("positive")
        if video_data is None or audio_data is None or conditions is None:
            raise KeyError("ltx_t2av_teacher_forcing expects inputs.video_latents, inputs.audio_latents and conditioning.positive.")

        with torch.no_grad():
            video_latents = self._video_latents(video_data)
            audio_latents = self._audio_latents(audio_data)
            video_latents, audio_latents = self._crop_latents_to_chunk_multiple(video_latents, audio_latents)
            video_num_frames = int(video_latents.shape[2])
            video_tokens = self.video_patchifier.patchify(video_latents)
            audio_tokens = self.audio_patchifier.patchify(audio_latents)
            video_context, audio_context, context_mask = self.model.prepare_text_condition(conditions)

            (
                video_input,
                video_target,
                video_timesteps,
                video_weights,
                video_sigmas,
                video_frame_weights,
                video_aug_sigmas,
                video_attention_mask,
                video_block_ids,
            ) = self._prepare_video_teacher_forcing_inputs(
                video_tokens,
                num_frames=video_num_frames,
                spatial_tokens=int(video_latents.shape[3] * video_latents.shape[4]),
            )
            (
                audio_input,
                audio_target,
                audio_timesteps,
                audio_weights,
                audio_sigmas,
                audio_attention_mask,
                audio_block_ids,
            ) = self._prepare_audio_teacher_forcing_inputs(
                audio_tokens,
                frame_sigmas=video_sigmas,
                frame_weights=video_frame_weights,
                aug_frame_sigmas=video_aug_sigmas,
                num_video_frames=video_num_frames,
            )

            video_positions = self._get_video_positions(
                num_frames=video_num_frames,
                height=int(video_data["height"][0].item()) if "height" in video_data else video_latents.shape[3],
                width=int(video_data["width"][0].item()) if "width" in video_data else video_latents.shape[4],
                batch_size=video_tokens.shape[0],
                fps=float(video_data["fps"][0].item()) if "fps" in video_data else self.default_fps,
                device=video_tokens.device,
            )
            video_positions = torch.cat([video_positions, video_positions], dim=2)
            audio_positions = self._get_audio_positions(
                num_time_steps=audio_tokens.shape[1],
                batch_size=audio_tokens.shape[0],
                device=audio_tokens.device,
            )
            audio_positions = torch.cat([audio_positions, audio_positions], dim=2)
            video_cross_attention_mask = self._teacher_forcing_cross_attention_mask(
                video_tokens.device,
                video_block_ids,
                audio_block_ids,
            )
            audio_cross_attention_mask = self._teacher_forcing_cross_attention_mask(
                audio_tokens.device,
                audio_block_ids,
                video_block_ids,
            )

        video_modality = Modality(
            enabled=True,
            latent=video_input,
            sigma=video_sigmas[:, 0],
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_context,
            context_mask=context_mask,
            attention_mask=video_attention_mask,
            cross_attention_mask=video_cross_attention_mask,
        )
        audio_modality = Modality(
            enabled=True,
            latent=audio_input,
            sigma=audio_sigmas[:, 0],
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_context,
            context_mask=context_mask,
            attention_mask=audio_attention_mask,
            cross_attention_mask=audio_cross_attention_mask,
        )

        with self.model.transformer_forward_context():
            video_pred, audio_pred = self.model.denoiser_module()(
                video=video_modality,
                audio=audio_modality,
                perturbations=None,
            )

        video_loss = self._weighted_token_mse(video_pred, video_target, video_weights)
        audio_loss = self._weighted_token_mse(audio_pred, audio_target, audio_weights)
        return (self.video_loss_weight * video_loss + self.audio_loss_weight * audio_loss).mean()

    def _crop_latents_to_chunk_multiple(self, video_latents, audio_latents):
        chunk = int(self.num_frame_per_chunk)
        if chunk <= 1:
            return video_latents, audio_latents

        num_frames = int(video_latents.shape[2])
        keep_frames = (num_frames // chunk) * chunk
        if keep_frames == num_frames:
            return video_latents, audio_latents
        if keep_frames <= 0:
            raise ValueError(f"LTX teacher forcing sample has only {num_frames} latent frames, smaller than num_frame_per_chunk={chunk}.")

        audio_frames = int(audio_latents.shape[2])
        keep_audio_frames = max(1, int(audio_frames * keep_frames // num_frames))
        video_latents = video_latents[:, :, :keep_frames].contiguous()
        audio_latents = audio_latents[:, :, :keep_audio_frames].contiguous()
        return video_latents, audio_latents

    def _prepare_video_teacher_forcing_inputs(self, tokens, num_frames, spatial_tokens):
        batch_size = tokens.shape[0]
        frame_sigmas, frame_weights = self.teacher_forcing_scheduler.sample_chunkwise(
            batch_size=batch_size,
            num_frames=num_frames,
            num_frame_per_chunk=self.num_frame_per_chunk,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        token_sigmas = frame_sigmas.repeat_interleave(spatial_tokens, dim=1)
        token_weights = frame_weights.repeat_interleave(spatial_tokens, dim=1)
        noisy, target, noise = self._add_token_noise(tokens, token_sigmas)
        clean = tokens
        clean_timesteps = torch.zeros_like(token_sigmas)
        aug_frame_sigmas = None
        if self.noise_augmentation_max_timestep > 0:
            aug_frame_sigmas = self.teacher_forcing_scheduler.sample_clean_augmentation(
                batch_size=batch_size,
                num_frames=num_frames,
                num_frame_per_chunk=self.num_frame_per_chunk,
                max_timestep=self.noise_augmentation_max_timestep,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            clean_timesteps = aug_frame_sigmas.repeat_interleave(spatial_tokens, dim=1)
            clean = (1.0 - clean_timesteps.unsqueeze(-1)) * tokens + clean_timesteps.unsqueeze(-1) * noise
        frame_ids = torch.arange(num_frames, device=tokens.device).repeat_interleave(spatial_tokens)
        block_ids = frame_ids // self.num_frame_per_chunk
        attention_mask = self._teacher_forcing_attention_mask(tokens.device, block_ids)
        return (
            torch.cat([clean, noisy], dim=1),
            target,
            torch.cat([clean_timesteps, token_sigmas], dim=1),
            token_weights,
            frame_sigmas,
            frame_weights,
            aug_frame_sigmas,
            attention_mask,
            block_ids,
        )

    def _prepare_audio_teacher_forcing_inputs(self, tokens, frame_sigmas, frame_weights, aug_frame_sigmas, num_video_frames):
        batch_size, seq_len, _ = tokens.shape
        frame_ids = torch.arange(seq_len, device=tokens.device) * max(1, int(num_video_frames)) // max(1, int(seq_len))
        frame_ids = frame_ids.clamp(max=frame_sigmas.shape[1] - 1)
        token_sigmas = frame_sigmas.gather(1, frame_ids.unsqueeze(0).expand(batch_size, -1))
        token_weights = frame_weights.gather(1, frame_ids.unsqueeze(0).expand(batch_size, -1))
        noisy, target, noise = self._add_token_noise(tokens, token_sigmas)
        clean = tokens
        clean_timesteps = torch.zeros_like(token_sigmas)
        if aug_frame_sigmas is not None:
            clean_timesteps = aug_frame_sigmas.gather(1, frame_ids.unsqueeze(0).expand(batch_size, -1))
            clean = (1.0 - clean_timesteps.unsqueeze(-1)) * tokens + clean_timesteps.unsqueeze(-1) * noise
        block_ids = frame_ids // self.num_frame_per_chunk
        attention_mask = self._teacher_forcing_attention_mask(tokens.device, block_ids)
        return (
            torch.cat([clean, noisy], dim=1),
            target,
            torch.cat([clean_timesteps, token_sigmas], dim=1),
            token_weights,
            token_sigmas,
            attention_mask,
            block_ids,
        )

    def _teacher_forcing_attention_mask(self, device, block_ids):
        cache_key = ("self", self._mask_device_key(device), self._block_ids_cache_key(block_ids))
        if cache_key in self._teacher_forcing_mask_cache:
            return self._teacher_forcing_mask_cache[cache_key]

        transformer = self.model.denoiser_module()
        for candidate in (
            transformer,
            getattr(transformer, "module", None),
            getattr(transformer, "_fsdp_wrapped_module", None),
        ):
            if candidate is None:
                continue
            mask_fn = getattr(candidate, "prepare_teacher_forcing_mask_from_block_ids", None)
            if mask_fn is not None:
                mask = mask_fn(device, block_ids)
                self._teacher_forcing_mask_cache[cache_key] = mask
                return mask
        raise RuntimeError("ltx_t2av_teacher_forcing requires prepare_teacher_forcing_mask_from_block_ids on the LTX transformer.")

    def _teacher_forcing_cross_attention_mask(self, device, query_block_ids, key_block_ids):
        cache_key = (
            "cross",
            self._mask_device_key(device),
            self._block_ids_cache_key(query_block_ids),
            self._block_ids_cache_key(key_block_ids),
        )
        if cache_key in self._teacher_forcing_mask_cache:
            return self._teacher_forcing_mask_cache[cache_key]

        transformer = self.model.denoiser_module()
        for candidate in (
            transformer,
            getattr(transformer, "module", None),
            getattr(transformer, "_fsdp_wrapped_module", None),
        ):
            if candidate is None:
                continue
            mask_fn = getattr(candidate, "prepare_teacher_forcing_cross_mask_from_block_ids", None)
            if mask_fn is not None:
                mask = mask_fn(device, query_block_ids, key_block_ids)
                self._teacher_forcing_mask_cache[cache_key] = mask
                return mask
        raise RuntimeError("ltx_t2av_teacher_forcing requires prepare_teacher_forcing_cross_mask_from_block_ids on the LTX transformer.")

    @staticmethod
    def _mask_device_key(device):
        device = torch.device(device)
        index = device.index
        if device.type == "cuda" and index is None and torch.cuda.is_available():
            index = torch.cuda.current_device()
        return device.type, index

    @staticmethod
    def _block_ids_cache_key(block_ids):
        block_ids = block_ids.detach()
        if block_ids.numel() == 0:
            return 0, None, None, None, None, None
        block_ids_long = block_ids.to(dtype=torch.long)
        positions = torch.arange(block_ids_long.numel(), device=block_ids_long.device, dtype=torch.long)
        return (
            int(block_ids_long.numel()),
            int(block_ids_long[0].item()),
            int(block_ids_long[-1].item()),
            int(block_ids_long.max().item()),
            int(block_ids_long.sum().item()),
            int((block_ids_long * positions).sum().item()),
        )

    @staticmethod
    def _add_token_noise(tokens, sigmas, noise=None):
        if noise is None:
            noise = torch.randn_like(tokens)
        sigmas_expanded = sigmas.unsqueeze(-1)
        noisy = (1.0 - sigmas_expanded) * tokens + sigmas_expanded * noise
        target = noise - tokens
        return noisy, target, noise

    @staticmethod
    def _causal_attention_mask(time_ids, batch_size, dtype):
        allowed = time_ids.view(-1, 1) >= time_ids.view(1, -1)
        return allowed.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    @staticmethod
    def _weighted_token_mse(pred, target, weights):
        pred = pred[:, -target.shape[1] :, :]
        loss = (pred.float() - target.float()).pow(2).mean(dim=-1)
        weights = weights.float()
        return (loss * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-8)
