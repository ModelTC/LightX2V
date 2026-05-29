import torch


def apply_image_condition_noise(
    image_latents,
    fake_image_latents,
    generator,
    device,
    image_noise_sigma_min,
    image_noise_sigma_max,
    video_noise_sigma_min,
    video_noise_sigma_max,
):
    image_noise_sigma = (
        torch.rand(1, device=device, generator=generator) * (image_noise_sigma_max - image_noise_sigma_min) + image_noise_sigma_min
    )
    image_latents = (
        image_noise_sigma * torch.randn(image_latents.shape, generator=generator, device=device) + (1 - image_noise_sigma) * image_latents
    )
    fake_image_noise_sigma = (
        torch.rand(1, device=device, generator=generator) * (video_noise_sigma_max - video_noise_sigma_min) + video_noise_sigma_min
    )
    fake_image_latents = (
        fake_image_noise_sigma * torch.randn(fake_image_latents.shape, generator=generator, device=device)
        + (1 - fake_image_noise_sigma) * fake_image_latents
    )
    return image_latents, fake_image_latents


def trim_generated_frames(frame_count, temporal_scale_factor):
    return ((frame_count - 1) // temporal_scale_factor) * temporal_scale_factor + 1


def finalize_video_output(history_video, video_processor, temporal_scale_factor, output_type="pt"):
    generated_frames = trim_generated_frames(history_video.size(2), temporal_scale_factor)
    history_video = history_video[:, :, :generated_frames]
    return video_processor.postprocess_video(history_video, output_type=output_type)


def pt_video_output_to_comfy_frames(video):
    if video.dim() != 5:
        raise ValueError(f"Expected [B, T, C, H, W] tensor, got shape {tuple(video.shape)}")
    return video.permute(0, 1, 3, 4, 2).flatten(0, 1).cpu()
