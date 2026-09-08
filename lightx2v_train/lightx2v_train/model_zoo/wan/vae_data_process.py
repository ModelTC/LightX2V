"""RGB-only inputs for Wan2.1 encoder and decoder distillation."""

import torch

from lightx2v_train.utils.registry import SAMPLE_PROCESSOR_REGISTER


class Wan21VAEDistillationProcessor:
    unconditional_prompt = " "
    requires_audio = False
    load_cached_latents = False

    def __init__(self, min_source_frames=1):
        self.min_source_frames = int(min_source_frames)
        if self.min_source_frames < 1:
            raise ValueError("Wan VAE min_source_frames must be positive.")

    def __call__(self, sample):
        video = sample["inputs"]["video"]
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(f"Wan2.1 VAE expects video [3,T,H,W], got {tuple(video.shape)}.")
        source_frames = video.shape[1]
        height, width = video.shape[-2:]
        if source_frames < 1 or height % 8 or width % 8:
            raise ValueError(f"Wan2.1 VAE requires nonempty video with H/W divisible by 8, got {tuple(video.shape)}.")
        if source_frames < self.min_source_frames:
            raise ValueError(
                f"Wan2.1 VAE requires at least {self.min_source_frames} real source frames, "
                f"got {source_frames}; padding cannot satisfy the training crop length."
            )

        padding = (1 - source_frames) % 4
        if padding:
            video = torch.cat((video, video[:, -1:].expand(-1, padding, -1, -1)), dim=1)
        # VideoDataset supplies [-1,1]; shared reconstruction losses use [0,1].
        sample["inputs"]["video"] = ((video.float() + 1.0) * 0.5).clamp_(0.0, 1.0)
        sample["inputs"].pop("latents", None)
        sample["meta"].pop("latent_path", None)
        sample["meta"].update(
            source_num_frames=int(source_frames),
            num_frames=int(video.shape[1]),
            target_height=int(height),
            target_width=int(width),
        )
        return sample


@SAMPLE_PROCESSOR_REGISTER("wan21_pruned_encoder")
@SAMPLE_PROCESSOR_REGISTER("wan21_pruned_decoder")
def build_wan21_vae_processor(config):
    processor_config = config.get("data", {}).get("processor", {})
    minimum = int(processor_config.get("min_source_frames", 1))
    stages = config.get("training", {}).get("vae_distillation", {}).get("stages", ())
    maximum_crop = max((int(stage.get("crop_num_frames", 33)) for stage in stages), default=1)
    if minimum < maximum_crop:
        raise ValueError(f"Wan VAE min_source_frames={minimum} must cover the largest training crop ({maximum_crop}).")
    return Wan21VAEDistillationProcessor(minimum)
