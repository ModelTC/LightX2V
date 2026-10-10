import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.audio_encoders.hf.minimax_h3 import MiniMaxH3AudioVAE
from lightx2v.models.networks.minimax_h3.packing import prepare_keyframe_image
from lightx2v.models.networks.minimax_h3_world.action import action_segments_to_texts
from lightx2v.models.networks.minimax_h3_world.lora import MiniMaxH3WorldLoraAdapter
from lightx2v.models.networks.minimax_h3_world.model import MiniMaxH3WorldModel
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS, VIDEO_OUTPUT_FIELDS
from lightx2v.models.schedulers.minimax_h3_world.scheduler import MiniMaxH3WorldScheduler
from lightx2v.models.video_encoders.hf.minimax_h3_world import MiniMaxH3WorldVideoVAE
from lightx2v.utils.envs import DTYPE_MAP
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("minimax_h3_world")
class MiniMaxH3WorldRunner(MiniMaxH3Runner):
    """Specialize native MiniMax-H3 for the explicit ia2av task."""

    supported_request_fields_by_task = {
        "ia2av": COMMON_REQUEST_FIELDS | VIDEO_OUTPUT_FIELDS | {"image_path", "prompt"},
    }

    def __init__(self, config):
        if config.get("model_variant") != "fl2av":
            raise ValueError("H3-World ia2av requires model_variant='fl2av'")
        if not config.get("lora_configs"):
            raise ValueError("H3-World ia2av requires lora_configs for its trained action LoRA")
        if config.get("seq_parallel", False):
            raise NotImplementedError("H3-World's directed action mask is not mapped to sequence-parallel token shards yet")
        if config.get("warmup", False):
            raise NotImplementedError("H3-World ia2av warmup is not implemented; use warmup=false")
        super().__init__(config)

    def get_supported_tasks(self):
        return ("ia2av",)

    def prepare_request(self, request_data):
        input_info = super().prepare_request(request_data)
        if not input_info.image_path:
            raise ValueError("H3-World ia2av requires image_path")
        if not isinstance(self.config.get("world_segments"), list):
            raise ValueError("H3-World ia2av requires world_segments to be a list in the config")
        return input_info

    def init_scheduler(self):
        self.scheduler = MiniMaxH3WorldScheduler(self.config)

    def load_transformer(self):
        lora_configs = self.config["lora_configs"]
        kwargs = {"model_path": self.config["model_path"], "config": self.config, "device": self.init_device}
        if self.config.get("lora_dynamic_apply", False):
            if len(lora_configs) != 1 or not lora_configs[0].get("path") or lora_configs[0].get("alpha") is None:
                raise ValueError("H3-World dynamic LoRA requires one lora_configs entry with path and alpha")
            lora = lora_configs[0]
            return MiniMaxH3WorldModel(**kwargs, lora_path=lora["path"], lora_strength=lora.get("strength", 1.0), lora_alpha=lora["alpha"])
        if self.config.get("dit_quantized", False):
            raise ValueError("H3-World merged LoRA requires non-quantized DiT weights")
        model = MiniMaxH3WorldModel(**kwargs)
        MiniMaxH3WorldLoraAdapter(model).apply_lora(lora_configs)
        return model

    def _prepare_keyframes(self):
        value = self.input_info.image_path
        if not value or (isinstance(value, str) and "," in value):
            raise ValueError("H3-World ia2av requires exactly one first image")
        image = self._load_rgb_image(value)
        self._resolve_request_geometry(image)
        return [prepare_keyframe_image(image, self.request_height, self.request_width, stretch=False)], ("first",)

    @ProfilingContext4DebugL1("Run Text Encoder")
    def run_text_encoder(self, input_info, keyframes=None, references=None):
        action_texts = action_segments_to_texts(self.config["world_segments"], self.request_num_frames)
        output = super().run_text_encoder(input_info, keyframes=keyframes, references=references)
        encoder = self.text_encoders[0]
        # Each sentence is encoded independently, exactly as in training.
        encoded = {text: encoder.infer(text) for text in dict.fromkeys(action_texts)}
        actions = [encoded[text] for text in action_texts]
        output["action_token_lengths"] = [item["prompt_embeds"].shape[0] for item in actions]
        for key in ("prompt_embeds", "text_token_tags"):
            output[key] = torch.cat([output[key]] + [item[key] for item in actions], dim=0)
        logger.info(f"H3-World ia2av: {len(action_texts)} latent actions, {len(encoded)} distinct sentences")
        return output

    @ProfilingContext4DebugL2("Prepare DiT")
    def init_run(self):
        text = self.inputs["text_encoder_output"]
        self.scheduler.prepare(
            seed=self.input_info.seed,
            num_frames=self.request_num_frames,
            height=self.request_height,
            width=self.request_width,
            text_token_tags=text["text_token_tags"],
            keyframe_anchors=self.keyframe_anchors,
            condition_video_latents=self.condition_video_latents,
            action_token_lengths=text["action_token_lengths"],
        )
        logger.info(
            f"H3-World packed layout: text={text['prompt_embeds'].shape[0]}, audio={self.scheduler.audio_latents.shape[0]}, "
            f"video={self.scheduler.video_latents.shape[0]}, total={self.scheduler.layout.sequence_length}"
        )
        if self.config.get("cpu_offload", False) and self.config.get("offload_granularity", "model") == "model":
            self.model.to_cuda()
        getattr(torch, AI_DEVICE).synchronize()

    def load_vae(self):
        cpu_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False))
        video_vae_quantized = self.config.get("video_vae_quantized", False)
        video_vae_quant_scheme = self.config["video_vae_quant_scheme"] if video_vae_quantized else None
        video_vae_quantized_ckpt = self.config["video_vae_quantized_ckpt"] if video_vae_quantized else None
        vae_sensitive_layer_dtype = DTYPE_MAP[self.config.get("vae_sensitive_layer_dtype", "fp32")]
        video_vae = MiniMaxH3WorldVideoVAE.from_pretrained(
            self.config["model_path"],
            device=AI_DEVICE,
            cpu_offload=cpu_offload,
            checkpoint_path=video_vae_quantized_ckpt,
            quant_scheme=video_vae_quant_scheme,
            encoder_conv_mode=self.config.get("vae_encoder_conv_mode", "torch"),
            sensitive_layer_dtype=vae_sensitive_layer_dtype,
            use_compile=self.config.get("vae_use_compile", False),
            attn_type=self.config.get("vae_attn_type", "torch_sdpa"),
            offload_granularity=self.config.get("video_vae_offload_granularity", "model"),
            shared_cpu_config=self.config if self.config.get("video_vae_shared_cpu_weights", False) else None,
        )
        self._vae_decode_tile_shapes = self.config.get("vae_decode_tile_shape", {})
        self._validate_vae_decode_tile_shapes(self._vae_decode_tile_shapes, video_vae)
        if self.config.get("vae_encode_parallel", False):
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            if world_size > 1:
                video_vae.enable_encode_parallel()
                logger.info(f"MiniMax-H3 spatiotemporal-tile VAE encode parallel enabled over {world_size} ranks")
            else:
                logger.info("MiniMax-H3 VAE encode parallel disabled for single-rank inference")
        if self.config.get("vae_decode_parallel", False):
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            if world_size > 1:
                video_vae.enable_decode_parallel()
                logger.info(f"MiniMax-H3 VAE spatiotemporal tile parallelism enabled over {world_size} ranks")
            else:
                logger.info("MiniMax-H3 VAE spatiotemporal tile parallelism disabled for single-rank inference")
        audio_vae = MiniMaxH3AudioVAE.from_pretrained(self.config["model_path"], device=AI_DEVICE, cpu_offload=cpu_offload)
        configured_sample_rate = int(self.config.get("audio_sampling_rate", audio_vae.sampling_rate))
        if configured_sample_rate != audio_vae.sampling_rate:
            raise ValueError(f"MiniMax-H3 audio_sampling_rate must match the Audio VAE checkpoint: config={configured_sample_rate}, checkpoint={audio_vae.sampling_rate}")
        return video_vae, audio_vae
