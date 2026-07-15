import os

import torch

from lightx2v_train.runtime.distributed import get_sequence_parallel_world_size
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .native.wan.modules.causal_model import CausalWanModel
from .native.wan.modules.model2_2 import WanModel as Wan2_2Model
from .native.wan.modules.t5 import T5EncoderModel
from .native.wan.modules.vae2_2 import Wan2_2_VAE
from .wan_t2v import WanT2VModel


@MODEL_REGISTER("wan_ti2v_5b_ar")
@MODEL_REGISTER("wan_ti2v_5b")
class WanTI2V5BModel(WanT2VModel):
    def load_components(self, transformer_only=False, reference_model=None):
        model_config = self.config["model"]
        model_path = model_config["pretrained_model_name_or_path"]

        self.load_vae = model_config.get("load_vae", True)
        self.load_text_encoder = model_config.get("load_text_encoder", True)
        self.load_transformer = model_config.get("load_transformer", True)
        training_config = self.config.get("training", {})
        teacher_forcing_config = training_config.get("teacher_forcing", {})
        model_name = model_config.get("name")
        teacher_forcing_enabled = teacher_forcing_config.get("enabled", False)
        self.use_causal_transformer = model_name == "wan_ti2v_5b_ar" or teacher_forcing_enabled
        self.sample_posterior = model_config.get("sample_posterior", True)
        scheduler_config = self.config.get("scheduler", {})
        self.num_train_timesteps = scheduler_config.get("num_train_timesteps", 1000)
        self.max_sequence_length = model_config.get("max_sequence_length", 512)
        self.transformer_param_dtype = get_running_dtype(model_config.get("transformer_param_dtype", "fp32"))
        self.vae_dtype = get_running_dtype(model_config.get("vae_dtype", "fp32"))
        self.t5_dtype = get_running_dtype(model_config.get("t5_dtype", "bf16"))
        self.t5_cpu = model_config.get("t5_cpu", False)
        self.vae_stride = tuple(model_config.get("vae_stride", (4, 16, 16)))
        self.patch_size = tuple(model_config.get("patch_size", (1, 2, 2)))
        self.sp_size = get_sequence_parallel_world_size()
        if "num_frame_per_chunk" in teacher_forcing_config:
            self.num_frame_per_chunk = int(teacher_forcing_config["num_frame_per_chunk"])
        else:
            self.num_frame_per_chunk = int(model_config.get("num_frame_per_chunk", 1))
        if "local_attn_size" in model_config:
            self.local_attn_size = int(model_config["local_attn_size"])
        else:
            self.local_attn_size = int(teacher_forcing_config.get("local_attn_size", -1))
        if "sink_size" in model_config:
            self.sink_size = int(model_config["sink_size"])
        else:
            self.sink_size = int(teacher_forcing_config.get("sink_size", 0))
        if "defer_kv_cache_updates" in model_config:
            self.defer_kv_cache_updates = bool(model_config["defer_kv_cache_updates"])
        else:
            self.defer_kv_cache_updates = bool(model_config.get("defer_cache_updates", False))
        self.detach_kv_cache_updates = bool(model_config.get("detach_kv_cache_updates", False))
        if "independent_first_frame" in model_config:
            self.independent_first_frame = bool(model_config["independent_first_frame"])
        else:
            self.independent_first_frame = bool(teacher_forcing_config.get("independent_first_frame", False))
        self.text_encoder = None
        self.text_pipeline = None

        if transformer_only:
            if reference_model is not None:
                self.vae = reference_model.vae
                self.text_encoder = reference_model.text_encoder
                self.text_pipeline = reference_model.text_pipeline
                self.vae_stride = reference_model.vae_stride
                self.patch_size = reference_model.patch_size
                self.max_sequence_length = reference_model.max_sequence_length
                self.vae_scale_factor_temporal = reference_model.vae_scale_factor_temporal
                self.vae_scale_factor_spatial = reference_model.vae_scale_factor_spatial
            self.transformer = self._load_transformer(model_path)
            self._configure_transformer()
            self.vae_scale_factor_temporal = self.vae_stride[0]
            self.vae_scale_factor_spatial = self.vae_stride[1]
            return

        if self.load_transformer:
            self.transformer = self._load_transformer(model_path)
            self._configure_transformer()
        else:
            self.transformer = None

        if self.load_vae:
            vae_checkpoint = model_config.get("vae_checkpoint", os.path.join(model_path, "Wan2.2_VAE.pth"))
            self.vae = Wan2_2_VAE(vae_pth=vae_checkpoint, dtype=self.vae_dtype, device=self.device)
            self.vae.model.requires_grad_(False)

        if self.load_text_encoder:
            t5_checkpoint = model_config.get("t5_checkpoint", os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth"))
            t5_tokenizer = model_config.get("t5_tokenizer", os.path.join(model_path, "google/umt5-xxl"))
            self.text_encoder = T5EncoderModel(
                text_len=self.max_sequence_length,
                dtype=self.t5_dtype,
                device=torch.device("cpu"),
                checkpoint_path=t5_checkpoint,
                tokenizer_path=t5_tokenizer,
            )
            self.text_encoder.model.requires_grad_(False)
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)

        self.vae_scale_factor_temporal = self.vae_stride[0]
        self.vae_scale_factor_spatial = self.vae_stride[1]

    def _load_transformer(self, model_path):
        if self.use_causal_transformer:
            transformer = CausalWanModel.from_pretrained(
                model_path,
                torch_dtype=self.transformer_param_dtype,
                local_attn_size=self.local_attn_size,
                sink_size=self.sink_size,
                defer_kv_cache_updates=self.defer_kv_cache_updates,
                detach_kv_cache_updates=self.detach_kv_cache_updates,
            )
            return transformer.to(self.device, dtype=self.transformer_param_dtype)
        transformer = Wan2_2Model.from_pretrained(model_path, torch_dtype=self.transformer_param_dtype)
        return transformer.to(self.device, dtype=self.transformer_param_dtype)

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        timestep = timestep_or_sigma.float() * self.num_train_timesteps
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(device=self.device)

        hidden_states = denoiser_input.hidden_states.to(device=self.device, dtype=self.running_dtype)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.unsqueeze(0)
        if timestep.ndim == 1 and timestep.shape[0] == 1 and hidden_states.shape[0] > 1:
            timestep = timestep.expand(hidden_states.shape[0])

        if isinstance(self.transformer, CausalWanModel):
            timestep = self._causal_timestep(timestep, hidden_states)
            context = self._condition_to_context_tensor(condition, batch_size=hidden_states.shape[0])
            self._prepare_causal_block_mask(hidden_states, teacher_forcing=False)
            with self.transformer_forward_context():
                return self.transformer(
                    hidden_states,
                    t=timestep,
                    context=context,
                    seq_len=self._sequence_length(hidden_states),
                )

        latent_list = self._batch_to_list(hidden_states)
        context = self._condition_to_context_list(condition, batch_size=len(latent_list))
        with self.transformer_forward_context():
            prediction = self.transformer(
                latent_list,
                t=timestep,
                context=context,
                seq_len=self._sequence_length(hidden_states),
            )
        if isinstance(prediction, list):
            prediction = torch.stack(prediction, dim=0)
        return prediction

    def _latent_channels(self):
        if self.transformer is not None:
            return int(getattr(self.transformer, "in_dim", getattr(self.transformer.config, "in_dim", 48)))
        return int(self.config["model"].get("latent_channels", 48))
