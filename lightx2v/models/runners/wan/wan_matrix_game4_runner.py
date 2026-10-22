"""Matrix-Game-4 video conditioning on the shared Wan AR lifecycle."""

import torch

from lightx2v.models.input_encoders.matrix_game4 import create_streaming_encoder, encode_control_video, encode_reference_image
from lightx2v.models.networks.wan.matrix_game4_model import WanMatrixGame4Model
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.request_fields import VIDEO_REQUEST_FIELDS
from lightx2v.models.runners.wan.autoregressive_runner import WanAutoregressiveRunnerMixin
from lightx2v.models.runners.wan.streaming_vae import WanStreamingVAERunnerMixin
from lightx2v.models.runners.wan.wan_runner import Wan22DenseRunner
from lightx2v.models.schedulers.wan.matrix_game4 import WanMatrixGame4Scheduler
from lightx2v.models.video_encoders.hf.wan.matrix_game4_vae import MatrixGame4TAEVAE, set_matrix_game4_vae_norm
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.input_info import MatrixGame4InputInfo
from lightx2v.utils.profiler import ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import find_torch_model_path
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("matrix_game4")
class WanMatrixGame4Runner(WanAutoregressiveRunnerMixin, Wan22DenseRunner):
    _SUPPORTS_AUTOREGRESSIVE_WARMUP = True
    _WARMUP_TASKS = ("v2v",)
    _WARMUP_RESOLUTIONS = ((256, 448), (480, 864))
    # Short MG4 chunks dispatch faster when VAE kernel launches share the caller thread.
    vae_decode_in_worker = False
    input_info_cls_by_task = {"v2v": MatrixGame4InputInfo}
    supported_request_fields_by_task = {"v2v": VIDEO_REQUEST_FIELDS | {"video_path", "image_path"}}

    def __init__(self, config):
        if not config.get("ar_config"):
            raise ValueError("Matrix-Game-4 requires configs/matrix_game4/quality.json or configs/matrix_game4/realtime.json")
        if config.get("enable_cfg", False):
            raise ValueError("Matrix-Game-4 requires enable_cfg=false")
        if config.get("vae_cpu_offload", False):
            raise ValueError("Matrix-Game-4 streaming input encoding requires a resident VAE")
        super().__init__(config)
        self._source_fps = None

    def load_transformer(self):
        return WanMatrixGame4Model(model_path=self.config["model_path"], config=self.config, device=self.init_device)

    def init_scheduler(self):
        self.scheduler = WanMatrixGame4Scheduler(self.config)

    def load_image_encoder(self):
        return None

    def load_vae_encoder(self):
        if self.config.get("use_tae", False):
            path = find_torch_model_path(self.config, "tae_path", self.tiny_vae_name)
            return MatrixGame4TAEVAE(vae_path=path, dtype=GET_DTYPE(), device=AI_DEVICE).to(AI_DEVICE).eval().requires_grad_(False)
        vae_config = self._build_wan22_vae_config(vae_offload=False)
        vae_config["dtype"] = torch.float32
        vae_config["device"] = torch.device(AI_DEVICE)
        vae = self.vae_cls(**vae_config)
        dtype = GET_DTYPE()
        vae.model.to(dtype=dtype)
        set_matrix_game4_vae_norm(vae.model)
        if vae.encoder_model is not None:
            vae.encoder_model.to(dtype=dtype)
            set_matrix_game4_vae_norm(vae.encoder_model)
        vae.dtype = dtype
        # The released wrapper takes the reciprocal before rounding to BF16.
        vae.mean, vae.inv_std = vae.mean.to(dtype), vae.inv_std.to(dtype)
        vae.scale = [vae.mean, vae.inv_std]
        return vae

    def load_vae_decoder(self):
        if self.config.get("lightvae_path"):
            return WanStreamingVAERunnerMixin.load_vae_decoder(self)
        return self.load_vae_encoder()

    def load_vae(self):
        self.vae_encoder = self.load_vae_encoder()
        decoder = self.load_vae_decoder() if self.config.get("lightvae_path") else self.vae_encoder
        return self.vae_encoder, decoder

    def prepare_request(self, request_data):
        input_info = BaseRunner.prepare_request(self, request_data)
        if not input_info.video_path:
            raise ValueError("Matrix-Game-4 v2v requires video_path")
        if not input_info.image_path:
            raise ValueError("Matrix-Game-4 requires an independent sink image_path")
        if input_info.num_frames is not None and input_info.num_frames < 1:
            raise ValueError("Matrix-Game-4 num_frames must be positive")
        size = input_info.size or self.config["size"]
        if len(size) != 2 or any(value <= 0 or value % 32 for value in size):
            raise ValueError("Matrix-Game-4 size must contain height and width divisible by 32")
        return input_info

    @ProfilingContext4DebugL2("Run Encoders")
    def run_input_encoder(self):
        size = self.get_target_size()
        condition = encode_control_video(self.vae_encoder, self.input_info.video_path, size, self.input_info.num_frames)
        reference = encode_reference_image(self.vae_encoder, self.input_info.image_path, size, self.scheduler.sink_mode)
        self.input_info.num_frames = condition.frame_count
        self.input_info.latent_shape = list(condition.control_latents.shape)
        self._source_fps = condition.fps
        max_frames = self.config.get("rope_max_len", 8192)
        reference_frames = int(self.scheduler.sink_mode == "kv_only")
        if condition.control_latents.shape[1] + reference_frames > max_frames:
            raise ValueError(f"Matrix-Game-4 input exceeds the {max_frames}-position RoPE budget")
        text_output = self.run_text_encoder(self.input_info)
        self.maybe_empty_cache()
        return {"text_encoder_output": text_output, "image_encoder_output": {"control_latents": condition.control_latents, "reference_latents": reference}}

    def get_warmup_latent_shape(self, target_height, target_width):
        _, stride_h, stride_w = self.config["vae_stride"]
        return [48, 1, target_height // stride_h, target_width // stride_w]

    def prepare_warmup_inputs(self, height, width, latent_shape, text_output):
        self.input_info = MatrixGame4InputInfo(task="v2v", prompt="warmup", size=[height, width])
        self.input_info.latent_shape = latent_shape
        encoder = create_streaming_encoder(self.vae_encoder)
        try:
            first = encoder.encode(torch.zeros(1, 3, height, width))
            encoder.encode(torch.zeros(4, 3, height, width))
        finally:
            encoder.clear()
        reference_pixels = first.new_zeros(1, 3, 1, height, width)
        reference = self.vae_encoder.encode(reference_pixels).to(first)
        controls = first.expand(-1, latent_shape[1], -1, -1).contiguous()
        self.inputs = {"text_encoder_output": text_output, "image_encoder_output": {"control_latents": controls, "reference_latents": reference}}

    def init_run(self):
        super().init_run()
        if self.scheduler.sink_mode == "kv_only":
            self.scheduler.prepare_reference()
            self.model.infer(self.inputs)

    def run_segment(self, segment_idx=0):
        self.scheduler.prepare_segment(segment_idx)
        return super().run_segment(segment_idx)

    def get_output_fps(self):
        return self._source_fps or super().get_output_fps()

    def process_images_after_vae_decoder(self):
        self.gen_video_final = self.gen_video_final[:, :, : self.input_info.num_frames]
        return super().process_images_after_vae_decoder()

    def clear_vae_cache(self):
        self.vae_decoder.reset_decode_state()

    def clear_warmup_state(self):
        super().clear_warmup_state()
        self.clear_vae_cache()

    def end_run(self):
        self.reset_model_request_state()
        self.clear_vae_cache()
        self._source_fps = None
        super().end_run()
