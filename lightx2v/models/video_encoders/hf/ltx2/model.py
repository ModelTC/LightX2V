from collections.abc import Iterator

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
from lightx2v.models.video_encoders.hf.ltx2.audio_vae.vocoder import Vocoder, VocoderWithBWE
from lightx2v.models.video_encoders.hf.ltx2.upsampler.model import LatentUpsamplerConfigurator
from lightx2v.models.video_encoders.hf.ltx2.video_vae.diffusion_video_decoder import DiffusionVideoDecoder
from lightx2v.models.video_encoders.hf.ltx2.video_vae.model_configurator import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
    video_decoder_sd_ops_for_checkpoint,
)
from lightx2v.models.video_encoders.hf.ltx2.video_vae.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
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
        use_tiling: bool = False,
        cpu_offload: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype
        self.load_encoder_flag = load_encoder
        self.use_tiling = use_tiling
        self.loader = SafetensorsModelStateDictLoader()
        self.encoder = None
        self.decoder = None
        self.cpu_offload = cpu_offload
        self.grid_table = {}  # Cache for 2D grid calculations
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
        """
        Encode video frames to latent space.
        Args:
            video_frames: Input video tensor [1, C, T, H, W] or [C, T, H, W]
        Returns:
            Encoded latent tensor [C, F, H_latent, W_latent]
        """
        # Ensure video has batch dimension
        if video_frames.dim() == 4:
            video_frames = video_frames.unsqueeze(0)

        try:
            if self.cpu_offload:
                self.encoder = self.encoder.to(AI_DEVICE)

            out = self.encoder(video_frames)
            if out.dim() == 5:
                out = out.squeeze(0)
            return out
        finally:
            if self.cpu_offload:
                self.encoder = self.encoder.to("cpu")

    def decode(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        # 如果启用了tiling但没有提供配置，使用默认配置
        if self.use_tiling and tiling_config is None:
            tiling_config = TilingConfig.default()

        try:
            if self.cpu_offload:
                self.decoder = self.decoder.to(AI_DEVICE)
            yield from decode_video(latent, self.decoder, tiling_config, generator)
        finally:
            if self.cpu_offload:
                self.decoder = self.decoder.to("cpu")


class LTX25VideoVAE(LTX2VideoVAE):
    """LTX-2.5 split video VAE with a native chunked-eager DiffVAE decoder.

    The public constructor and encode/decode interface intentionally match
    :class:`LTX2VideoVAE`; only model construction and checkpoint key mapping
    differ.  Standalone ``vae/*.safetensors`` files use bare ``encoder.*`` and
    ``decoder.*`` keys, including fused decoder QKV projections.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        load_encoder: bool = True,
        use_tiling: bool = False,
        cpu_offload: bool = False,
        optimization: str = "chunked_eager",
    ):
        if optimization != "chunked_eager":
            raise ValueError(f"Unsupported LTX-2.5 DiffVAE optimization {optimization!r}; only 'chunked_eager' is available")
        self.optimization = optimization
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=dtype,
            load_encoder=load_encoder,
            use_tiling=use_tiling,
            cpu_offload=cpu_offload,
        )

    def load(self) -> tuple[VideoEncoder | None, DiffusionVideoDecoder | None]:
        config = self.loader.metadata(self.checkpoint_path)

        if self.load_encoder_flag:
            with torch.device("meta"):
                encoder = VideoEncoderConfigurator.from_config(config)
            encoder_state = self.loader.load(
                self.checkpoint_path,
                sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
                device=self.device,
            ).sd
            if self.dtype is not None:
                encoder_state = {key: value.to(dtype=self.dtype) for key, value in encoder_state.items()}
            incompatible = encoder.load_state_dict(encoder_state, strict=False, assign=True)
            if incompatible.missing_keys:
                raise ValueError(f"LTX-2.5 video encoder checkpoint is missing keys: {incompatible.missing_keys}")
            self.encoder = encoder.to(self.device).eval()

        with torch.device("meta"):
            decoder = VideoDecoderConfigurator.from_config(config)
        if not isinstance(decoder, DiffusionVideoDecoder):
            raise TypeError("LTX25VideoVAE requires a CausalDiffusionVAE checkpoint")
        decoder_state = self.loader.load(
            self.checkpoint_path,
            sd_ops=video_decoder_sd_ops_for_checkpoint(self.checkpoint_path, config),
            device=self.device,
        ).sd
        if self.dtype is not None:
            decoder_state = {key: value.to(dtype=self.dtype) for key, value in decoder_state.items()}
        incompatible = decoder.load_state_dict(decoder_state, strict=False, assign=True)
        if incompatible.missing_keys:
            raise ValueError(f"LTX-2.5 diffusion video decoder checkpoint is missing keys: {incompatible.missing_keys}")
        unexpected = [key for key in incompatible.unexpected_keys if key != "type_emb"]
        if unexpected:
            raise ValueError(f"LTX-2.5 diffusion video decoder has unexpected keys: {unexpected}")
        self.decoder = decoder.to(self.device).eval()
        return self.encoder, self.decoder

    def decode(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Yield source-compatible float video chunks in ``[0, 1]``.

        LTX-2.5's upstream ``DiffusionVideoDecoder.decode_video`` keeps the
        DiffVAE output in BF16 ``FHWC`` form until the BT.709 encoder sink.
        The older LightX2V LTX wrapper converts to RGB uint8 inside the VAE,
        which changes the later YUV conversion.  Keep this override scoped to
        LTX-2.5 so the established LTX-2.3 return contract is unchanged.
        """
        try:
            if self.cpu_offload:
                self.decoder = self.decoder.to(AI_DEVICE)
            if self.use_tiling and tiling_config is None:
                min_t, min_h, min_w = self.decoder._stage_min_sizes
                frames = (max(latent.shape[2], min_t) - 1) * 8 + 1
                height = max(latent.shape[3], min_h) * 32
                width = max(latent.shape[4], min_w) * 32
                long_side = max(height, width)
                spatial_tile = min(long_side, 1536)
                tiling_config = TilingConfig(
                    spatial_config=SpatialTilingConfig(
                        tile_size_in_pixels=spatial_tile,
                        tile_overlap_in_pixels=96 if long_side > spatial_tile else 0,
                    ),
                    temporal_config=TemporalTilingConfig(
                        tile_size_in_frames=128,
                        tile_overlap_in_frames=8 if frames > 128 else 0,
                    ),
                )

            decoded_chunks = self.decoder.tiled_decode(latent, tiling_config, generator=generator) if tiling_config is not None else iter((self.decoder(latent, generator=generator),))
            for decoded in decoded_chunks:
                video = decoded[0].permute(1, 2, 3, 0).contiguous()
                yield video.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
        finally:
            if self.cpu_offload:
                self.decoder = self.decoder.to("cpu")


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
        try:
            if self.cpu_offload:
                self.encoder = self.encoder.to(AI_DEVICE)
            return self.encoder(audio_spectrogram)
        finally:
            if self.cpu_offload:
                self.encoder = self.encoder.to("cpu")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        try:
            if self.cpu_offload:
                self.decoder = self.decoder.to(AI_DEVICE)
                self.vocoder = self.vocoder.to(AI_DEVICE)
            return decode_audio(latent, self.decoder, self.vocoder)
        finally:
            if self.cpu_offload:
                self.decoder = self.decoder.to("cpu")
                self.vocoder = self.vocoder.to("cpu")


class LTX25AudioVAE(LTX2AudioVAE):
    """Enable the released LTX-2.5 vocoder's FP32 BWE path."""

    def load(self):
        components = super().load()
        if isinstance(self.vocoder, VocoderWithBWE):
            self.vocoder.force_fp32 = True
        return components


class LTX2Upsampler:
    """
    Wrapper class for loading and using LatentUpsampler model, similar to LTX2VideoVAE.
    """

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
        self.loader = None
        self.upsampler = None
        self.load()

    def load(self):
        """
        Load upsampler model from checkpoint.

        Aligned exactly with Builder.build() in ltx_core.loader.single_gpu_model_builder:
        1. Create model on meta device (aligned with Builder.meta_model)
        2. Load state_dict and convert dtype if needed (aligned with Builder.build line 82-83)
        3. Load state_dict with assign=True (aligned with Builder.build line 84)
        4. Move to device only (aligned with Builder._return_model line 69)

        Key point: _return_model only does .to(device), NOT .to(dtype).
        This means we rely on assign=True to set correct dtype from state_dict.
        """
        self.loader = SafetensorsModelStateDictLoader()
        config = self.loader.metadata(self.checkpoint_path)

        # Handle config format: may have rational_spatial_scale instead of spatial_scale
        if "rational_spatial_scale" in config and "spatial_scale" not in config:
            config["spatial_scale"] = config["rational_spatial_scale"]

        # Create model on meta device (aligned with Builder.meta_model line 47-48)
        with torch.device("meta"):
            upsampler = LatentUpsamplerConfigurator.from_config(config)

        # Load state_dict (aligned with Builder.load_sd)
        state_dict_obj = self.loader.load(
            self.checkpoint_path,
            sd_ops=None,  # No key filtering, aligned with source code
            device=self.device,  # Directly to target device (aligned with DummyRegistry case)
        )
        state_dict = state_dict_obj.sd

        # Convert state_dict dtype if needed (aligned with Builder.build line 82-83)
        if self.dtype is not None:
            state_dict = {key: value.to(dtype=self.dtype) for key, value in state_dict.items()}

        # Load state_dict with assign=True (aligned with Builder.build line 84)
        # assign=True directly replaces parameters, so dtype should match state_dict
        upsampler.load_state_dict(state_dict, strict=False, assign=True)

        # Move to device only (aligned with Builder._return_model line 69)
        # CRITICAL: _return_model only does .to(device), NOT .to(dtype)
        # This means we rely on assign=True to have set correct dtype from state_dict
        # If state_dict contains all parameters, they should already have correct dtype
        self.upsampler = upsampler.to(self.device).eval()
        return self.upsampler

    @torch.no_grad()
    def upsample(
        self,
        latent: torch.Tensor,
        video_encoder: VideoEncoder,
    ) -> torch.Tensor:
        """
        Upsample video latent using the upsampler with proper normalization.
        Aligned with ltx_core.model.upsampler.model.upsample_video.

        This method directly calls the static upsample_video method to ensure
        exact alignment with source code implementation.

        Args:
            latent: Input latent tensor of shape [B, C, F, H, W] or [C, F, H, W].
            video_encoder: VideoEncoder with per_channel_statistics for normalization.

        Returns:
            Upsampled latent tensor of shape [B, C, F, H*2, W*2] or [C, F, H*2, W*2].
        """

        try:
            if self.cpu_offload:
                self.upsampler = self.upsampler.to(AI_DEVICE)
            return self.upsample_video(latent, video_encoder, self.upsampler)
        finally:
            if self.cpu_offload:
                self.upsampler = self.upsampler.to("cpu")

    @staticmethod
    def upsample_video(latent: torch.Tensor, video_encoder: VideoEncoder, upsampler) -> torch.Tensor:
        """
        Apply upsampling to the latent representation using the provided upsampler,
        with normalization and un-normalization based on the video encoder's per-channel statistics.

        This is a static method that can be used with any upsampler instance and video encoder.
        Aligned with ltx_core.model.upsampler.model.upsample_video.

        Args:
            latent: Input latent tensor of shape [B, C, F, H, W].
            video_encoder: VideoEncoder with per_channel_statistics for normalization.
            upsampler: LatentUpsampler module to perform upsampling.
                Note: upsampler should already be in eval mode, on correct device, and with correct dtype.

        Returns:
            torch.Tensor: Upsampled and re-normalized latent tensor.
        """
        # Aligned with source code: un_normalize -> upsampler -> normalize
        # Source code does not modify upsampler state, so we call it directly
        latent = video_encoder.per_channel_statistics.un_normalize(latent)
        latent = upsampler(latent)
        latent = video_encoder.per_channel_statistics.normalize(latent)
        return latent


if __name__ == "__main__":
    dev = "cuda"
    dtype = torch.bfloat16

    video_vae = LTX2VideoVAE(
        checkpoint_path="/data/nvme0/gushiqiao/models/official_models/LTX-2/ltx-2-19b-dev.safetensors",
        device=dev,
        dtype=dtype,
    )

    audio_vae = LTX2AudioVAE(
        checkpoint_path="/data/nvme0/gushiqiao/models/official_models/LTX-2/ltx-2-19b-dev.safetensors",
        device=dev,
        dtype=dtype,
    )

    vid_enc = torch.load("/data/nvme0/gushiqiao/models/v.pth")  # .unsqueeze(0)
    vid_dec = video_vae.decode(vid_enc)

    audio_enc = torch.load("/data/nvme0/gushiqiao/models/a.pth")  # .unsqueeze(0)
    audio_dec = audio_vae.decode(audio_enc)

    encode_video(
        video=vid_dec,
        fps=24,
        audio=audio_dec,
        output_path=f"reconstructed_1.mp4",
        video_chunks_number=1,
    )
