import torch
import torch.distributed as dist

from lightx2v_train.runtime.distributed import get_sequence_parallel_group, get_sequence_parallel_world_size, is_sequence_parallel_enabled
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value, sequence_parallel_frame_slice, sync_sequence_parallel_gradients
from lightx2v_train.schedulers.flow_matching import CausalForcingFlowMatchScheduler
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .flow import FlowMatchingTrainer


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
        latent = sample.get("clean_latent", sample.get("latent"))
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
