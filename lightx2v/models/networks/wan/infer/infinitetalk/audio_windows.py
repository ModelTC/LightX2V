import torch


def pack_later_audio_windows(audio, vae_scale, audio_window):
    """Pack audio after the first frame into latent-frame groups, keeping context at each end."""
    later = audio[:, 1:]
    if later.shape[1] % vae_scale:
        raise ValueError(f"audio frames after the first frame must be divisible by {vae_scale}, got {later.shape[1]}")
    windows = later.unflatten(1, (-1, vae_scale))
    middle = audio_window // 2
    first = windows[:, :, 0, : middle + 1]
    interior = windows[:, :, 1:-1, middle]
    last = windows[:, :, -1, middle:]
    return torch.cat((first, interior, last), dim=2)
