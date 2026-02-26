"""
TensorRT VAE implementation for Qwen Image model.

Provides accelerated VAE encoder/decoder using pre-built TensorRT engines.
Supports both single static engine and multi-aspect-ratio engine selection.
"""

import os

import torch
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

try:
    import tensorrt as trt

    HAS_TRT = True
except ImportError:
    HAS_TRT = False

try:
    from diffusers import AutoencoderKLQwenImage
    from diffusers.image_processor import VaeImageProcessor
except ImportError:
    AutoencoderKLQwenImage = None
    VaeImageProcessor = None


# Profiles used in the multi-profile engine (must match build script)
PROFILE_CONFIGS = [
    # (name, height, width, profile_idx)
    # I2I Profiles
    ("1_1_512", 512, 512, 0),
    ("1_1_1024", 1024, 1024, 1),
    ("16_9_480p", 480, 848, 2),
    ("16_9_720p", 720, 1280, 3),
    ("16_9_1080p", 1088, 1920, 4),  # Fixed to 1088 for 112-alignment
    ("9_16_720p", 1280, 720, 5),
    ("9_16_1080p", 1920, 1088, 6),  # Fixed to 1088
    ("4_3_768p", 768, 1024, 7),
    ("3_2_1080p", 1088, 1620, 8),
]
# Static resolution mapping for T2I directory mode
# (height, width) -> folder_name
STATIC_RESOLUTIONS = {
    (928, 1664): "16_9",
    (1664, 928): "9_16",
    (1328, 1328): "1_1",
    (1140, 1472): "4_3",
    (1024, 768): "3_4",
}


class TensorRTVAE:
    """TensorRT-accelerated VAE for Qwen Image model."""

    def __init__(self, config):
        if not HAS_TRT:
            raise RuntimeError("TensorRT is not available. Please install tensorrt package.")

        self.config = config
        self.dtype = GET_DTYPE()
        self.device = torch.device(AI_DEVICE)
        self.latent_channels = 16
        self.vae_latents_mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        self.vae_latents_std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]

        self.is_layered = config.get("layered", False)
        if self.is_layered:
            self.layers = config.get("layers", 4)

        # TRT config
        trt_config = config.get("trt_vae_config", {})
        # Unified engine path. Must be a directory.
        self.trt_engine_path = trt_config.get("trt_engine_path", "")
        self.multi_profile_mode = trt_config.get("multi_profile", False)

        # TRT runtime
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.stream = torch.cuda.Stream()

        # Engine cache
        self._encoder_context = None
        self._encoder_io_names = {}
        self._decoder_context = None
        self._decoder_io_names = {}

        # Static mode cache
        self._static_encoder_cache = {}
        self._static_decoder_cache = {}

        # Image processor for output
        self.image_processor = VaeImageProcessor(vae_scale_factor=config.get("vae_scale_factor", 8) * 2)

        # PyTorch VAE for decoder fallback
        self._pytorch_vae = None
        self._vae_path = config.get("vae_path", os.path.join(config.get("model_path", ""), "vae"))

        self._load_engines()

    def _load_engines(self):
        """Load TensorRT engines (Eager Loading)."""
        if not self.trt_engine_path:
            logger.warning("TensorRT VAE requires trt_engine_path configuration. Fallback to PyTorch.")
            return

        if not os.path.exists(self.trt_engine_path) or not os.path.isdir(self.trt_engine_path):
            logger.warning(f"trt_engine_path is not a valid directory: {self.trt_engine_path}. Fallback to PyTorch.")
            return

        # Directory Mode (Auto-Discovery)
        if self.multi_profile_mode:
            # I2I Multi-Profile Mode: Look for standard names in directory
            enc_path = os.path.join(self.trt_engine_path, "vae_encoder_multi_profile.trt")
            dec_path = os.path.join(self.trt_engine_path, "vae_decoder_multi_profile.trt")

            logger.info(f"Loading Multi-Profile TRT VAE from dir: {self.trt_engine_path}")
            if os.path.exists(enc_path):
                engine = self._load_engine_file(enc_path)
                self._encoder_context = engine.create_execution_context()
                self._encoder_io_names = self._get_io_names(engine)
                self._encoder_engine = engine
            else:
                logger.warning(f"Could not find encoder engine at {enc_path}, will fallback to PyTorch when needed.")

            if os.path.exists(dec_path):
                engine = self._load_engine_file(dec_path)
                self._decoder_context = engine.create_execution_context()
                self._decoder_io_names = self._get_io_names(engine)
                self._decoder_engine = engine
            else:
                logger.warning(f"Could not find decoder engine at {dec_path}, will fallback to PyTorch when needed.")
        else:
            # T2I Static Directory Mode
            logger.info(f"Loading Directory-Based Static TRT VAE (Eager Load). Root: {self.trt_engine_path}")

            # Eagerly load all supported resolutions
            loaded_count = 0
            for _, folder_name in STATIC_RESOLUTIONS.items():
                if os.path.exists(os.path.join(self.trt_engine_path, folder_name)):
                    self._load_static_engine_from_folder(self.trt_engine_path, folder_name)
                    loaded_count += 1

            if loaded_count == 0:
                logger.warning(f"No valid static engines found in {self.trt_engine_path} matching known resolutions, will fallback to PyTorch when needed.")

    def _load_static_engine_from_folder(self, root_dir, folder):
        """Lazy load static engine for a specific folder."""
        # Encoder
        enc_path = os.path.join(root_dir, folder, "vae_encoder.trt")
        if os.path.exists(enc_path):
            logger.info(f"Loading static encoder: {folder}")
            engine = self._load_engine_file(enc_path)
            self._static_encoder_cache[folder] = (engine.create_execution_context(), self._get_io_names(engine))
        else:
            logger.warning(f"Static encoder not found for {folder}: {enc_path}")

        # Decoder
        dec_path = os.path.join(root_dir, folder, "vae_decoder.trt")
        if os.path.exists(dec_path):
            logger.info(f"Loading static decoder: {folder}")
            engine = self._load_engine_file(dec_path)
            self._static_decoder_cache[folder] = (engine.create_execution_context(), self._get_io_names(engine))

    def _load_engine_file(self, path):
        """Load a TensorRT engine from file."""
        with open(path, "rb") as f:
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def _get_io_names(self, engine):
        """Get input/output tensor names from engine."""
        io_names = {"inputs": [], "outputs": []}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                io_names["inputs"].append(name)
            else:
                io_names["outputs"].append(name)
        return io_names

    def _get_static_engine(self, height, width, is_decoder=False):
        """Get pre-loaded static engine for specific resolution."""
        res_key = (height, width)
        if res_key not in STATIC_RESOLUTIONS:
            return None, None

        folder_name = STATIC_RESOLUTIONS[res_key]
        cache = self._static_decoder_cache if is_decoder else self._static_encoder_cache

        if folder_name not in cache:
            # Lazy load
            self._load_static_engine_from_folder(self.trt_engine_path, folder_name)

        return cache.get(folder_name, (None, None))

    def _run_trt_inference(self, context, io_names, input_tensor):
        """Run TensorRT inference."""
        input_name = io_names["inputs"][0]
        output_name = io_names["outputs"][0]

        output_shape = context.get_tensor_shape(output_name)
        output_buffer = torch.empty(tuple(output_shape), dtype=torch.float16, device="cuda")

        context.set_tensor_address(input_name, input_tensor.data_ptr())
        context.set_tensor_address(output_name, output_buffer.data_ptr())
        context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return output_buffer

    def _find_best_profile(self, h, w):
        """Find the profile with closest opt_shape to input."""
        best_idx = 0
        best_diff = float("inf")

        for i, (name, opt_h, opt_w, global_idx) in enumerate(PROFILE_CONFIGS):
            diff = abs(h - opt_h) + abs(w - opt_w)
            if diff < best_diff:
                best_diff = diff
                best_idx = i  # Relative index within the engine
        return best_idx

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor, layers=None):
        """Unpack latents from sequence to spatial format."""
        batchsize, num_patches, channels = latents.shape

        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        if layers:
            latents = latents.view(batchsize, layers + 1, height // 2, width // 2, channels // 4, 2, 2)
            latents = latents.permute(0, 1, 4, 2, 5, 3, 6)
            latents = latents.reshape(batchsize, layers + 1, channels // (2 * 2), height, width)
            latents = latents.permute(0, 2, 1, 3, 4)
        else:
            latents = latents.view(batchsize, height // 2, width // 2, channels // 4, 2, 2)
            latents = latents.permute(0, 3, 1, 4, 2, 5)
            latents = latents.reshape(batchsize, channels // (2 * 2), 1, height, width)

        return latents

    @staticmethod
    def _pack_latents(latents, batchsize, num_channels_latents, height, width, layers=None):
        """Pack latents from spatial to sequence format."""
        if not layers:
            latents = latents.view(batchsize, num_channels_latents, height // 2, 2, width // 2, 2)
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(batchsize, (height // 2) * (width // 2), num_channels_latents * 4)
        else:
            latents = latents.permute(0, 2, 1, 3, 4)
            latents = latents.view(batchsize, layers, num_channels_latents, height // 2, 2, width // 2, 2)
            latents = latents.permute(0, 1, 3, 5, 2, 4, 6)
            latents = latents.reshape(batchsize, layers * (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    def _encode_multi_profile(self, image):
        """Encode using multi-profile engine."""
        if self._encoder_context is None:
            return None

        b, c, t, h, w = image.shape
        # Select profile
        profile_idx = self._find_best_profile(h, w)
        self._encoder_context.set_optimization_profile_async(profile_idx, self.stream.cuda_stream)

        # Set input shape
        input_name = self._encoder_io_names["inputs"][0]
        self._encoder_context.set_input_shape(input_name, tuple(image.shape))

        # Run
        input_fp16 = image.to(torch.float16).contiguous()
        latent_dist = self._run_trt_inference(self._encoder_context, self._encoder_io_names, input_fp16)

        # Extract mean (first 16 channels)
        latent = latent_dist[:, : self.latent_channels, :, :, :]
        return latent.to(self.dtype)

    def _encode_static(self, image):
        """Encode using static engine."""
        b, c, t, h, w = image.shape
        s_ctx, s_io = self._get_static_engine(h, w, is_decoder=False)

        if s_ctx is None:
            return None

        input_fp16 = image.to(torch.float16).contiguous()
        latent_dist = self._run_trt_inference(s_ctx, s_io, input_fp16)
        latent = latent_dist[:, : self.latent_channels, :, :, :]
        return latent.to(self.dtype)

    def _decode_multi_profile(self, latents, target_h, target_w):
        """Decode using multi-profile engine."""
        if self._decoder_context is None:
            return None

        # latents: [B, C, F, H, W]
        profile_idx = self._find_best_profile(target_h, target_w)
        self._decoder_context.set_optimization_profile_async(profile_idx, self.stream.cuda_stream)

        input_name = self._decoder_io_names["inputs"][0]
        self._decoder_context.set_input_shape(input_name, tuple(latents.shape))

        input_fp16 = latents.to(torch.float16).contiguous()
        images = self._run_trt_inference(self._decoder_context, self._decoder_io_names, input_fp16)
        return images

    @torch.no_grad()
    def encode_vae_image(self, image):
        """Encode image to latent space using TensorRT."""
        num_channels_latents = self.config["in_channels"] // 4
        image = image.to(self.device)

        if image.shape[1] != self.latent_channels:
            if self.multi_profile_mode:
                image_latents = self._encode_multi_profile(image)
            else:
                image_latents = self._encode_static(image)

            if image_latents is None:
                # Fallback to PyTorch
                if self._pytorch_vae is None:
                    logger.info(f"Loading PyTorch VAE encoder from {self._vae_path}")
                    self._pytorch_vae = AutoencoderKLQwenImage.from_pretrained(self._vae_path).to(self.device).to(self.dtype)
                    self._pytorch_vae.eval()
                image_input = image.to(self.dtype)
                latent_dist = self._pytorch_vae.quant_conv(self._pytorch_vae.encoder(image_input))
                image_latents = latent_dist[:, : self.latent_channels, :, :, :]

            latents_mean = torch.tensor(self.vae_latents_mean).view(1, self.latent_channels, 1, 1, 1).to(image_latents.device, image_latents.dtype)
            latents_std = torch.tensor(self.vae_latents_std).view(1, self.latent_channels, 1, 1, 1).to(image_latents.device, image_latents.dtype)
            image_latents = (image_latents - latents_mean) / latents_std
        else:
            image_latents = image

        image_latents = torch.cat([image_latents], dim=0)
        image_latent_height, image_latent_width = image_latents.shape[3:]
        if not self.is_layered:
            image_latents = self._pack_latents(image_latents, 1, num_channels_latents, image_latent_height, image_latent_width)
        else:
            image_latents = self._pack_latents(image_latents, 1, num_channels_latents, image_latent_height, image_latent_width, 1)

        return image_latents

    @torch.no_grad()
    def decode(self, latents, input_info):
        """Decode latents to image."""
        width, height = input_info.auto_width, input_info.auto_height
        if self.is_layered:
            latents = self._unpack_latents(latents, height, width, self.config["vae_scale_factor"], self.layers)
        else:
            latents = self._unpack_latents(latents, height, width, self.config["vae_scale_factor"])

        latents = latents.to(self.dtype)
        latents_mean = torch.tensor(self.vae_latents_mean).view(1, self.latent_channels, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.vae_latents_std).view(1, self.latent_channels, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean

        images = None
        if self.multi_profile_mode:
            images = self._decode_multi_profile(latents, height, width)
        else:
            s_ctx, s_io = self._get_static_engine(height, width, is_decoder=True)
            if s_ctx is not None:
                input_fp16 = latents.to(torch.float16).contiguous()
                images = self._run_trt_inference(s_ctx, s_io, input_fp16)

        if images is not None:
            images = images[:, :, 0]

        if images is None:
            # Fallback to PyTorch
            if self._pytorch_vae is None:
                logger.info(f"Loading PyTorch VAE decoder from {self._vae_path}")
                self._pytorch_vae = AutoencoderKLQwenImage.from_pretrained(self._vae_path).to(self.device).to(self.dtype)
                self._pytorch_vae.eval()

            images = self._pytorch_vae.decode(latents).sample
            images = images[:, :, 0]

        images = self.image_processor.postprocess(images, output_type="pt" if input_info.return_result_tensor else "pil")

        return images
