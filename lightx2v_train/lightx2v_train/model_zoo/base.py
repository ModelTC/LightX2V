import json
import os

import torch
from diffusers.models.modeling_utils import SAFETENSORS_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.peft_utils import get_adapter_name
from huggingface_hub import split_torch_state_dict_into_shards
from loguru import logger
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

from lightx2v_train.model_capabilities import (
    CapabilityProvider,
    CheckpointCapability,
    ParallelCapability,
    TrainableModelCapability,
)
from lightx2v_train.model_zoo.capability_adapters.common import (
    CommonCheckpointCapability,
    CommonParallelCapability,
    CommonTrainableCapability,
)
from lightx2v_train.runtime.distributed import is_main_process
from lightx2v_train.runtime.fsdp import is_fsdp2_module
from lightx2v_train.utils.utils import get_running_dtype


class BaseModel(CapabilityProvider):
    default_unconditional_prompt = " "
    shared_condition_keys = ()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running_dtype = get_running_dtype(config["model"]["running_dtype"])
        self.device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.vae = None
        self.vae_config = None
        self.text_pipeline = None
        self.image_processor = None

    def register_capabilities(self):
        self.capabilities.register(
            TrainableModelCapability,
            CommonTrainableCapability(self),
        )
        self.capabilities.register(
            ParallelCapability,
            CommonParallelCapability(self),
        )
        self.capabilities.register(
            CheckpointCapability,
            CommonCheckpointCapability(self),
        )

    def load_components(
        self,
        *,
        load_transformer,
        load_vae,
        load_condition_encoder,
    ):
        """Load the requested model weights; lightweight config metadata may still be read."""
        raise NotImplementedError

    def reuse_frozen_components_from(self, source):
        """Reuse the frozen VAE and condition components owned by another model."""
        self.vae = source.vae
        self.vae_config = source.vae_config
        self.text_pipeline = source.text_pipeline
        self.image_processor = source.image_processor

    def denoiser_module(self):
        raise NotImplementedError(f"{self.__class__.__name__} must define denoiser_module().")

    def add_lora(self, rank, alpha, target_modules):
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.denoiser_module().add_adapter(lora_config)

    def set_lora_trainable(self):
        denoiser = self.denoiser_module()
        denoiser.requires_grad_(False)
        denoiser.train()
        for name, param in denoiser.named_parameters():
            param.requires_grad = "lora" in name

    def set_full_trainable(self):
        denoiser = self.denoiser_module()
        denoiser.requires_grad_(True)
        denoiser.train()

    def trainable_parameters(self):
        return (p for p in self.denoiser_module().parameters() if p.requires_grad)

    def enable_gradient_checkpointing(self):
        denoiser = self.denoiser_module()
        if hasattr(denoiser, "enable_gradient_checkpointing"):
            denoiser.enable_gradient_checkpointing()

    def set_denoiser_eval(self):
        self.denoiser_module().eval()

    def is_fsdp2_wrapped(self):
        return is_fsdp2_module(self.denoiser_module())

    def fsdp2_state_module(self):
        return self.denoiser_module()

    def set_fsdp2_gradient_sync(self, enabled):
        denoiser = self.denoiser_module()
        if hasattr(denoiser, "set_requires_gradient_sync"):
            denoiser.set_requires_gradient_sync(enabled)
        if hasattr(denoiser, "set_is_last_backward"):
            denoiser.set_is_last_backward(enabled)

    def fsdp2_shard_plan(self, fsdp_config):
        raise NotImplementedError(f"{self.__class__.__name__} must define fsdp2_shard_plan().")

    def log_model_structure(self):
        logger.info("[model] class={}", self.__class__.__name__)
        text_encoder = getattr(getattr(self, "text_pipeline", None), "text_encoder", None)
        if text_encoder is not None:
            logger.info("[model] text_encoder structure:\n{}", text_encoder)
        if self.vae is not None:
            logger.info("[model] vae structure:\n{}", self.vae)
        logger.info("[model] denoiser structure:\n{}", self.denoiser_module())

    def encode_to_latent(self, sample):
        raise NotImplementedError

    def encode_condition(self, sample):
        raise NotImplementedError

    @property
    def unconditional_prompt(self):
        return self.config["model"].get("unconditional_prompt", self.default_unconditional_prompt)

    def encode_condition_roles(self, sample, prompts, *, contextual_roles=()):
        """Encode named prompts, retaining sample context only for requested roles."""
        contextual_roles = set(contextual_roles)
        contextual_names = [name for name in prompts if name in contextual_roles]
        conditions = {name: self.encode_prompt_condition(prompt) for name, prompt in prompts.items() if name not in contextual_roles}
        if contextual_names:
            contextual_prompts = [prompts[name] for name in contextual_names]
            contextual_conditions = self.encode_conditions_with_context(sample, contextual_prompts)
            conditions.update(zip(contextual_names, contextual_conditions, strict=True))
        return {name: conditions[name] for name in prompts}

    def encode_conditions_with_context(self, sample, prompts):
        conditions = []
        for prompt in prompts:
            contextual_sample = dict(sample)
            contextual_sample["conditioning"] = {
                **sample["conditioning"],
                "prompt": prompt,
            }
            conditions.append(self.encode_condition(contextual_sample))
        return conditions

    def encode_to_cache_latent(self, sample):
        """Encode a deterministic target latent for reuse across training epochs."""
        return self.encode_to_latent(sample)

    def encode_inference_condition(self, sample, *, is_negative=False):
        del is_negative
        return self.encode_condition(sample)

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        raise NotImplementedError

    def denoise(self, denoiser_input, timesteps, condition):
        raise NotImplementedError

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        raise NotImplementedError

    def apply_cfg(self, positive, negative, guidance_scale):
        return negative + guidance_scale * (positive - negative)

    def denoiser_prediction_type(self):
        """Return the quantity predicted by the denoiser.

        LightX2V's current diffusion backbones are trained as rectified-flow
        velocity predictors.  Keeping this declaration on the model avoids
        baking that assumption into training objectives and leaves room
        for models that predict x0, noise, or another parameterization.
        """
        return "velocity"

    def predict_denoiser_output(self, noisy_latent, timestep_or_sigma, condition, **denoiser_kwargs):
        """Run the model-specific denoiser path and return latent-shaped output.

        Consistency objectives operate on latent tensors, while individual
        models may pack those tensors before the transformer forward.  This
        method is the common boundary between the two layers.  Extra keyword
        arguments are intentionally forwarded for algorithms such as
        MeanFlow, whose denoisers can require an additional endpoint time.
        """
        denoiser_input = self.prepare_denoiser_input(noisy_latent, condition=condition)
        prediction = self.denoise(
            denoiser_input,
            timestep_or_sigma,
            condition,
            **denoiser_kwargs,
        )
        return self.postprocess_denoiser_output(prediction, denoiser_input)

    def prepare_infer_latents(self, height, width, generator=None):
        raise NotImplementedError

    def decode_latent(self, latent):
        raise NotImplementedError

    def load_lora_for_infer(self, lora_path, adapter_name=None):
        denoiser = self.denoiser_module()
        if adapter_name is None:
            adapter_name = get_adapter_name(denoiser)
        denoiser.load_lora_adapter(lora_path, adapter_name=adapter_name)
        self._infer_lora_adapter_name = adapter_name

    def unload_lora_for_infer(self):
        adapter_name = getattr(self, "_infer_lora_adapter_name", None)
        if adapter_name is not None:
            self.denoiser_module().delete_adapters(adapter_name)
            self._infer_lora_adapter_name = None

    def save_lora_weights(
        self,
        save_dir,
        adapter_name=None,
        weights_subdir=None,
        *,
        auxiliary_parameter_names=(),
        auxiliary_weights_name=None,
    ):
        peft_state_dict, auxiliary_state_dict = self._get_lora_and_auxiliary_state_dict_for_save(
            adapter_name=adapter_name,
            auxiliary_parameter_names=auxiliary_parameter_names,
        )
        if not is_main_process():
            return

        output_dir = os.path.join(save_dir, weights_subdir) if weights_subdir else save_dir
        os.makedirs(output_dir, exist_ok=True)
        lora_state_dict = convert_state_dict_to_diffusers(peft_state_dict)
        if hasattr(self.pipeline_cls, "save_lora_weights"):
            self.pipeline_cls.save_lora_weights(output_dir, lora_state_dict, safe_serialization=True)
        else:
            save_file(lora_state_dict, os.path.join(output_dir, "pytorch_lora_weights.safetensors"))
        if auxiliary_state_dict:
            if not auxiliary_weights_name:
                raise ValueError("auxiliary_weights_name is required when auxiliary parameters are saved.")
            save_file(
                auxiliary_state_dict,
                os.path.join(output_dir, auxiliary_weights_name),
            )

    def _get_lora_state_dict_for_save(self, adapter_name=None):
        return self._get_lora_and_auxiliary_state_dict_for_save(adapter_name=adapter_name)[0]

    def _get_lora_and_auxiliary_state_dict_for_save(
        self,
        adapter_name=None,
        auxiliary_parameter_names=(),
    ):
        denoiser = self.denoiser_module()
        peft_kwargs = {} if adapter_name is None else {"adapter_name": adapter_name}
        if not is_fsdp2_module(denoiser):
            state_dict = denoiser.state_dict()
        else:
            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=False,
                strict=False,
            )
            state_dict, _ = get_state_dict(denoiser, (), options=options)
            if not is_main_process():
                return {}, {}

        peft_state_dict = get_peft_model_state_dict(denoiser, state_dict=state_dict, **peft_kwargs)
        auxiliary_names = set(auxiliary_parameter_names)
        missing = auxiliary_names - state_dict.keys()
        if missing:
            raise RuntimeError(f"Auxiliary parameters are missing from the model state: {sorted(missing)}")
        auxiliary_state_dict = {name: state_dict[name].detach().cpu().contiguous() for name in auxiliary_names}
        return peft_state_dict, auxiliary_state_dict

    def load_lora_weights_for_resume(self, lora_path, adapter_name=None, weights_subdir=None):
        weights_dir = os.path.join(lora_path, weights_subdir) if weights_subdir else lora_path
        raw = load_file(os.path.join(weights_dir, "pytorch_lora_weights.safetensors"))
        peft_state_dict = {}
        for key, value in raw.items():
            new_key = key.removeprefix("transformer.")
            new_key = new_key.replace(".lora.down.weight", ".lora_A.weight")
            new_key = new_key.replace(".lora.up.weight", ".lora_B.weight")
            peft_state_dict[new_key] = value

        load_kwargs = {} if adapter_name is None else {"adapter_name": adapter_name}
        incompatible = set_peft_model_state_dict(self.denoiser_module(), peft_state_dict, **load_kwargs)
        if incompatible and incompatible.unexpected_keys:
            logger.warning("Unexpected keys when resuming LoRA: {}", incompatible.unexpected_keys)

    def load_auxiliary_weights(
        self,
        checkpoint_dir,
        parameter_names,
        *,
        weights_name,
    ):
        names = set(parameter_names)
        if not names:
            return
        path = os.path.join(checkpoint_dir, weights_name)
        if not os.path.exists(path):
            raise RuntimeError(f"Auxiliary weights were not found at {path}.")
        incompatible = self.denoiser_module().load_state_dict(load_file(path), strict=False)
        missing = [name for name in incompatible.missing_keys if name in names]
        unexpected = [name for name in incompatible.unexpected_keys if name in names]
        if missing:
            raise RuntimeError(f"Missing auxiliary keys in {path}: {missing}")
        if unexpected:
            logger.warning("Unexpected auxiliary keys in {}: {}", path, unexpected)

    def load_full_weights_for_resume(self, resume_ckpt_path):
        raise NotImplementedError(f"{self.__class__.__name__} must define load_full_weights_for_resume().")

    def prepare_consolidated_state_dict(self, state_dict):
        return state_dict

    def consolidated_safetensors_metadata(self):
        return {"format": "pt"}

    def save_consolidated_weights(self, output_path):
        denoiser = self.denoiser_module()
        if is_fsdp2_module(denoiser):
            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=False,
                strict=False,
            )
            logger.info("[checkpoint] gathering consolidated model state dict")
            state_dict, _ = get_state_dict(denoiser, (), options=options)
        elif is_main_process():
            state_dict = denoiser.state_dict()
        else:
            state_dict = {}

        if not is_main_process():
            return

        state_dict = self.prepare_consolidated_state_dict(state_dict)
        state_dict = {key: value.detach().cpu().contiguous() for key, value in state_dict.items()}
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        tmp_path = f"{output_path}.tmp"
        logger.info("[checkpoint] saving consolidated model weights to {}", output_path)
        save_file(state_dict, tmp_path, metadata=self.consolidated_safetensors_metadata())
        os.replace(tmp_path, output_path)
        logger.info("[checkpoint] saved consolidated model weights to {}", output_path)

    def save_full_model(self, save_dir):
        denoiser = self.denoiser_module()
        transformer_dir = os.path.join(save_dir, "transformer")
        if not is_fsdp2_module(denoiser):
            if is_main_process():
                denoiser.save_pretrained(transformer_dir, safe_serialization=True)
            return

        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=False,
            strict=False,
        )
        logger.info("[checkpoint] gathering consolidated full model state dict")
        state_dict, _ = get_state_dict(denoiser, (), options=options)
        if is_main_process():
            self._save_full_state_dict(transformer_dir, denoiser, state_dict)

    def _save_full_state_dict(self, save_dir, denoiser, state_dict):
        logger.info("[checkpoint] saving consolidated transformer weights to {}", save_dir)
        os.makedirs(save_dir, exist_ok=True)
        denoiser.save_config(save_dir)

        weights_name_pattern = SAFETENSORS_WEIGHTS_NAME.replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict,
            max_shard_size="10GB",
            filename_pattern=weights_name_pattern,
        )

        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
            save_file(shard, os.path.join(save_dir, filename), metadata={"format": "pt"})

        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            index_path = os.path.join(save_dir, SAFE_WEIGHTS_INDEX_NAME)
            with open(index_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(index, indent=2, sort_keys=True) + "\n")

        logger.info("[checkpoint] saved consolidated transformer weights to {}", save_dir)
