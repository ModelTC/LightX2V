from typing import Iterator

import torch

from lightx2v.models.video_encoders.hf.ltx2.audio_vae.audio_vae import AudioDecoder, AudioEncoder, decode_audio
from lightx2v.models.video_encoders.hf.ltx2.audio_vae.model_configurator import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from lightx2v.models.video_encoders.hf.ltx2.audio_vae.vocoder import Vocoder
from lightx2v.models.video_encoders.hf.ltx2.video_vae.model_configurator import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
)
from lightx2v.models.video_encoders.hf.ltx2.video_vae.tiling import TilingConfig
from lightx2v.models.video_encoders.hf.ltx2.video_vae.video_vae import VideoDecoder, VideoEncoder, decode_video
from lightx2v.utils.ltx2_media_io import *
from lightx2v.utils.ltx2_utils import *
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class LTX2VideoVAE:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        load_encoder: bool = True,
        cpu_offload: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype
        self.load_encoder_flag = load_encoder
        self.loader = SafetensorsModelStateDictLoader()
        self.encoder = None
        self.decoder = None
        self.cpu_offload = cpu_offload
        self.load()

    def load(self) -> tuple[VideoEncoder | None, VideoDecoder | None]:
        config = self.loader.metadata(self.checkpoint_path)

        if self.load_encoder_flag:
            encoder = VideoEncoderConfigurator.from_config(config)
            state_dict_obj = self.loader.load(
                self.checkpoint_path,
                sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
                device=self.device,
            )
            state_dict = state_dict_obj.sd
            if self.dtype is not None:
                state_dict = {key: value.to(dtype=self.dtype) for key, value in state_dict.items()}
            encoder.load_state_dict(state_dict, strict=False, assign=True)
            self.encoder = encoder.to(self.device).eval()

        decoder = VideoDecoderConfigurator.from_config(config)
        state_dict_obj = self.loader.load(
            self.checkpoint_path,
            sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
            device=self.device,
        )
        state_dict = state_dict_obj.sd
        if self.dtype is not None:
            state_dict = {key: value.to(dtype=self.dtype) for key, value in state_dict.items()}
        decoder.load_state_dict(state_dict, strict=False, assign=True)
        self.decoder = decoder.to(self.device).eval()

    def encode(self, video_frames: torch.Tensor) -> torch.Tensor:
        if self.cpu_offload:
            self.encoder = self.encoder.to(AI_DEVICE)
        out = self.encoder(video_frames)
        if self.cpu_offload:
            self.encoder = self.encoder.to("cpu")
        return out

    def decode(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        if self.cpu_offload:
            self.decoder = self.decoder.to(AI_DEVICE)
        out = decode_video(latent, self.decoder, tiling_config, generator)
        if self.cpu_offload:
            self.decoder = self.decoder.to("cpu")
        return out


class LTX2AudioVAE:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        cpu_offload: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype
        self.cpu_offload = cpu_offload
        self.loader = SafetensorsModelStateDictLoader()
        self.load()

    def load(self) -> tuple[AudioEncoder | None, AudioDecoder | None, Vocoder | None]:
        config = self.loader.metadata(self.checkpoint_path)

        encoder = AudioEncoderConfigurator.from_config(config)
        state_dict_obj = self.loader.load(
            self.checkpoint_path,
            sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            device=self.device,
        )
        state_dict = state_dict_obj.sd
        if self.dtype is not None:
            state_dict = {key: value.to(dtype=self.dtype) for key, value in state_dict.items()}
        encoder.load_state_dict(state_dict, strict=False, assign=True)
        self.encoder = encoder.to(self.device).eval()

        decoder = AudioDecoderConfigurator.from_config(config)
        state_dict_obj = self.loader.load(
            self.checkpoint_path,
            sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            device=self.device,
        )
        state_dict = state_dict_obj.sd
        if self.dtype is not None:
            state_dict = {key: value.to(dtype=self.dtype) for key, value in state_dict.items()}
        decoder.load_state_dict(state_dict, strict=False, assign=True)
        self.decoder = decoder.to(self.device).eval()

        vocoder = VocoderConfigurator.from_config(config)
        state_dict_obj = self.loader.load(
            self.checkpoint_path,
            sd_ops=VOCODER_COMFY_KEYS_FILTER,
            device=self.device,
        )
        state_dict = state_dict_obj.sd
        if self.dtype is not None:
            state_dict = {key: value.to(dtype=self.dtype) for key, value in state_dict.items()}
        vocoder.load_state_dict(state_dict, strict=False, assign=True)
        self.vocoder = vocoder.to(self.device).eval()

        return encoder, decoder, vocoder

    def encode(self, audio_spectrogram: torch.Tensor) -> torch.Tensor:
        if self.cpu_offload:
            self.encoder = self.encoder.to(AI_DEVICE)
        out = self.encoder(audio_spectrogram)
        if self.cpu_offload:
            self.encoder = self.encoder.to("cpu")
        return out

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.cpu_offload:
            self.decoder = self.decoder.to(AI_DEVICE)
            self.vocoder = self.vocoder.to(AI_DEVICE)
        out = decode_audio(latent, self.decoder, self.vocoder)
        if self.cpu_offload:
            self.decoder = self.decoder.to("cpu")
            self.vocoder = self.vocoder.to("cpu")
        return out


if __name__ == "__main__":
    dev = "cuda"
    dtype = torch.bfloat16

    video_vae = LTX2VideoVAE(
        checkpoint_path="/data/nvme0/gushiqiao/models/official_models/LTX-2/ltx-2-19b-distilled-fp8.safetensors",
        device=dev,
        dtype=dtype,
    )

    audio_vae = LTX2AudioVAE(
        checkpoint_path="/data/nvme0/gushiqiao/models/official_models/LTX-2/ltx-2-19b-distilled-fp8.safetensors",
        device=dev,
        dtype=dtype,
    )

    vid_enc = torch.load("/data/nvme0/gushiqiao/models/code/LightX2V/scripts/v.pth").unsqueeze(0)
    vid_dec = video_vae.decode(vid_enc)

    audio_enc = torch.load("/data/nvme0/gushiqiao/models/code/LightX2V/scripts/a.pth").unsqueeze(0)
    audio_dec = audio_vae.decode(audio_enc)

    encode_video(
        video=vid_dec,
        fps=24,
        audio=audio_dec,
        audio_sample_rate=24000,
        output_path=f"reconstructed_1.mp4",
        video_chunks_number=1,
    )
