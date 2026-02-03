"""
TensorRT VAE implementation for Qwen Image model.

Provides accelerated VAE encoder/decoder using pre-built TensorRT engines.
Supports both single static engine and multi-aspect-ratio engine selection.
"""

import os
import math
import torch
import torch.nn.functional as F
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

# Common aspect ratios for multi-ratio mode
# Format: (name, height, width)
# All dimensions must be divisible by 16 to ensure latent dimensions (dim/8) are divisible by 2
ASPECT_RATIO_CONFIGS = [
    ("1_1_1024", 1024, 1024),   # latent 128x128
    ("1_1_512", 512, 512),      # latent 64x64
    ("4_3_1024", 768, 1024),    # latent 96x128
    ("3_4_1024", 1024, 768),    # latent 128x96
    ("16_9_1152", 640, 1152),   # latent 80x144, ~16:9
    ("9_16_1152", 1152, 640),   # latent 144x80, ~9:16
    ("3_2_1024", 672, 1024),    # latent 84x128, ~3:2
    ("2_3_1024", 1024, 672),    # latent 128x84, ~2:3
]


def select_best_ratio(input_h, input_w, engines):
    """Select the best matching aspect ratio engine."""
    input_ratio = input_w / input_h
    input_pixels = input_h * input_w

    best_match = None
    best_score = float("inf")

    for name, (engine_h, engine_w, engine_path) in engines.items():
        engine_ratio = engine_w / engine_h
        engine_pixels = engine_h * engine_w

        ratio_diff = abs(math.log(engine_ratio / input_ratio))
        scale_penalty = 0
        if engine_pixels < input_pixels:
            scale_penalty = (input_pixels - engine_pixels) / input_pixels * 0.5

        score = ratio_diff + scale_penalty

        if score < best_score:
            best_score = score
            best_match = (name, engine_h, engine_w, engine_path)

    if best_score < 0.5: # Allow some tolerance but reject gross mismatches (e.g. 16:9 vs 1:1)
        return best_match
    return None


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
        self.engine_dir = trt_config.get("engine_dir", "")
        self.encoder_engine_path = trt_config.get("encoder_engine", "")
        self.decoder_engine_path = trt_config.get("decoder_engine", "")
        self.multi_ratio_mode = trt_config.get("multi_ratio", False)

        # TRT runtime
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.stream = torch.cuda.Stream()

        # Engine cache
        self.engines = {}
        self._current_encoder_name = None
        self._encoder_context = None
        self._encoder_io_names = {}
        self._decoder_context = None
        self._decoder_context = None
        self._decoder_io_names = {}
        self.decoder_engines = {}
        self._current_decoder_name = None

        # Image processor for output
        self.image_processor = VaeImageProcessor(vae_scale_factor=config.get("vae_scale_factor", 8) * 2)

        # PyTorch VAE for decoder fallback
        self._pytorch_vae = None
        self._vae_path = config.get("vae_path", os.path.join(config.get("model_path", ""), "vae"))

        self._load_engines()

    def _load_engines(self):
        """Load TensorRT engines."""
        if self.multi_ratio_mode and self.engine_dir:
            # Multi-ratio mode: load all available engines from directory
            logger.info(f"Loading TensorRT VAE engines from directory: {self.engine_dir}")
            for name, height, width in ASPECT_RATIO_CONFIGS:
                encoder_path = os.path.join(self.engine_dir, f"vae_encoder_{name}.trt")
                if os.path.exists(encoder_path):
                    self.engines[name] = (height, width, encoder_path)
                    logger.info(f"  Found encoder engine: {name} ({width}x{height})")
                
                decoder_path = os.path.join(self.engine_dir, f"vae_decoder_{name}.trt")
                if os.path.exists(decoder_path):
                    self.decoder_engines[name] = (height, width, decoder_path)
                    logger.info(f"  Found decoder engine: {name} ({width}x{height})")

            if not self.engines:
                raise RuntimeError(f"No TensorRT engines found in {self.engine_dir}")

            logger.info(f"Loaded {len(self.engines)} aspect ratio engines")

        elif self.encoder_engine_path:
            # Single static engine mode
            if not os.path.exists(self.encoder_engine_path):
                raise FileNotFoundError(f"Encoder engine not found: {self.encoder_engine_path}")

            logger.info(f"Loading static TensorRT VAE encoder: {self.encoder_engine_path}")
            self._load_static_encoder(self.encoder_engine_path)

            if self.decoder_engine_path and os.path.exists(self.decoder_engine_path):
                logger.info(f"Loading static TensorRT VAE decoder: {self.decoder_engine_path}")
                self._load_static_decoder(self.decoder_engine_path)
        else:
            raise ValueError("TensorRT VAE requires either engine_dir (multi-ratio) or encoder_engine path")

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

    def _load_static_encoder(self, path):
        """Load static encoder engine."""
        engine = self._load_engine_file(path)
        self._encoder_context = engine.create_execution_context()
        self._encoder_io_names = self._get_io_names(engine)
        self._encoder_engine = engine

    def _load_static_decoder(self, path):
        """Load static decoder engine."""
        engine = self._load_engine_file(path)
        self._decoder_context = engine.create_execution_context()
        self._decoder_io_names = self._get_io_names(engine)
        self._decoder_engine = engine

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

    def _encode_multi_ratio(self, image):
        """Encode using multi-ratio engine selection."""
        b, c, t, h, w = image.shape

        # Select best matching engine
        match = select_best_ratio(h, w, self.engines)
        if match is None:
            raise RuntimeError("No suitable TRT engine found")

        name, target_h, target_w, engine_path = match
        target_ratio = target_w / target_h
        input_ratio = w / h

        # Center crop to match target ratio
        if input_ratio > target_ratio:
            new_w = int(h * target_ratio)
            offset = (w - new_w) // 2
            cropped = image[:, :, :, :, offset : offset + new_w]
        else:
            new_h = int(w / target_ratio)
            offset = (h - new_h) // 2
            cropped = image[:, :, :, offset : offset + new_h, :]

        # Resize to engine resolution
        cropped_h, cropped_w = cropped.shape[3], cropped.shape[4]
        resized = F.interpolate(
            cropped.view(b * t, c, cropped_h, cropped_w), size=(target_h, target_w), mode="bilinear", align_corners=False
        ).view(b, c, t, target_h, target_w)

        # Load engine if different
        if not hasattr(self, "_current_encoder_name") or name != self._current_encoder_name:
            logger.debug(f"Switching to TRT engine: {name}")
            engine = self._load_engine_file(engine_path)
            self._encoder_context = engine.create_execution_context()
            self._encoder_io_names = self._get_io_names(engine)
            self._current_encoder_name = name
            self._encoder_engine = engine

        # Run inference
        input_fp16 = resized.to(torch.float16).contiguous()
        latent_dist = self._run_trt_inference(self._encoder_context, self._encoder_io_names, input_fp16)

        # Extract mean from latent distribution (first 16 of 32 channels)
        # The encoder outputs [mean, logvar] concatenated, we take mode = mean
        latent = latent_dist[:, :self.latent_channels, :, :, :]

        return latent.to(self.dtype)

    def _encode_static(self, image):
        """Encode using static engine."""
        input_fp16 = image.to(torch.float16).contiguous()
        latent_dist = self._run_trt_inference(self._encoder_context, self._encoder_io_names, input_fp16)

        # Extract mean from latent distribution (first 16 of 32 channels)
        latent = latent_dist[:, :self.latent_channels, :, :, :]
        return latent.to(self.dtype)

    def _decode_multi_ratio(self, latents, target_h, target_w):
        """Decode using multi-ratio engine selection."""
        if not self.decoder_engines:
            return None

        match = select_best_ratio(target_h, target_w, self.decoder_engines)
        if match is None:
            return None

        name, eng_h, eng_w, engine_path = match
        
        # Switch engine if needed
        if not hasattr(self, "_current_decoder_name") or name != self._current_decoder_name:
            logger.debug(f"Switching to TRT decoder engine: {name}")
            engine = self._load_engine_file(engine_path)
            self._decoder_context = engine.create_execution_context()
            self._decoder_io_names = self._get_io_names(engine)
            self._current_decoder_name = name
            self._decoder_engine = engine

        # Check latent shape compatibility
        lat_h_eng = eng_h // 8
        lat_w_eng = eng_w // 8
        
        if latents.shape[-2] != lat_h_eng or latents.shape[-1] != lat_w_eng:
            # Interpolate latents to match engine input
            # latents shape: [B, C, F, H, W] -> treat F as batch dim for spatial resize
            b, c, f, h, w = latents.shape
            latents_reshaped = latents.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
            latents_resized = F.interpolate(
                latents_reshaped, size=(lat_h_eng, lat_w_eng), mode="bilinear", align_corners=False
            )
            latents = latents_resized.view(b, f, c, lat_h_eng, lat_w_eng).permute(0, 2, 1, 3, 4)
            
        # Run inference
        input_fp16 = latents.to(torch.float16).contiguous()
        images = self._run_trt_inference(self._decoder_context, self._decoder_io_names, input_fp16)
        
        return images

    @torch.no_grad()
    def encode_vae_image(self, image):
        """Encode image to latent space using TensorRT."""
        num_channels_latents = self.config["in_channels"] // 4
        image = image.to(self.device)

        if image.shape[1] != self.latent_channels:
            # Actual encoding needed
            if self.multi_ratio_mode:
                image_latents = self._encode_multi_ratio(image)
            else:
                image_latents = self._encode_static(image)

            # Apply normalization
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
        """Decode latents to image.

        Uses TRT decoder if available, otherwise falls back to PyTorch decoder.
        """
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
        if self.multi_ratio_mode and self.decoder_engines:
            images = self._decode_multi_ratio(latents, height, width)
            if images is not None:
                images = images[:, :, 0] # Remove frame dim
            else:
                # Fallback if specific engine load failed unexpectedly
                self._decoder_context = None

        if images is None and self._decoder_context is not None:
            # Use static TRT decoder
            input_fp16 = latents.to(torch.float16).contiguous()
            images = self._run_trt_inference(self._decoder_context, self._decoder_io_names, input_fp16)
            images = images[:, :, 0]
        
        if images is None:
            # Fallback to PyTorch decoder
            if self._pytorch_vae is None:
                logger.info(f"Loading PyTorch VAE decoder from {self._vae_path}")
                self._pytorch_vae = AutoencoderKLQwenImage.from_pretrained(self._vae_path).to(self.device).to(self.dtype)
                self._pytorch_vae.eval()

            images = self._pytorch_vae.decode(latents).sample
            images = images[:, :, 0]

        images = self.image_processor.postprocess(images, output_type="pt" if input_info.return_result_tensor else "pil")

        return images
