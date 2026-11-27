"""Official YuE2 FP32 audio VAE, exposed as a LightX2V decoder component."""

import torch


class YuE2AudioVAE:
    sample_rate = 48000

    def __init__(self, path, device, core_frames=1024, halo_frames=16):
        # Reuse the compatible decoder only; never call the upstream MoT/pipeline.
        from yue2.modeling_vae import YuE2VAE

        self.device = device
        self.core_frames = core_frames
        self.halo_frames = halo_frames
        self.model = YuE2VAE.from_pretrained(path, decoder_only=True, device="cpu", local_files_only=True).to(device)

    def decode(self, latents, check_stop):
        check_stop()
        z = latents.float().T.unsqueeze(0)

        # Cancellation must be local here: only rank zero decodes the waveform.
        def progress(completed, total):
            check_stop()

        audio = self.model.decode_tiled(z, core_frames=self.core_frames, halo_frames=self.halo_frames, output_device="cpu", on_progress=progress)
        if not torch.isfinite(audio).all():
            raise FloatingPointError("YuE2 VAE produced non-finite audio")
        return audio[0].float().clamp(-1, 1).T.contiguous()

    def close(self):
        self.model = None
