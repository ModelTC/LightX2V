import torch
import torch.nn.functional as F

from lightx2v.common.ops.mm.mm_weight import MMWeight
from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.utils import sinusoidal_embedding_1d
from lightx2v.models.schedulers.wan.edenlive.scheduler import EdenLiveARScheduler


class EdenLivePreInfer(WanPreInfer):
    def __init__(self, config):
        super().__init__(config)
        self.is_wan21 = config["model_cls"] in ["wan21_edenlive", "wan21_edenlive_ar"]
        self.clear_projection_cache()

    def clear_request_state(self):
        self.clear_projection_cache()

    def clear_projection_cache(self):
        """Release request-owned projections after warmup, requests, or reloads."""
        self._projection_request = None
        self._projection_weight_signature = None
        self._projection_weight_refs = ()
        self._projection_schedule = None
        self._projection_schedule_signature = None
        self._text_projection_cache = {}
        self._time_projection_cache = {}
        self._proxy_interpolation_cache = None
        self._projection_generation = getattr(self, "_projection_generation", 0) + 1

    @staticmethod
    def _tensor_signature(tensor):
        if not torch.is_tensor(tensor):
            return None
        try:
            version = tensor._version
        except RuntimeError:
            # Inference tensors do not track in-place edits: use the ordinary
            # path rather than silently assuming their contents are immutable.
            return None
        return (id(tensor), version, tensor.device, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()))

    def _prepare_projection_cache(self, weights, device):
        config = self.config
        scheduler = self.scheduler
        if (
            type(scheduler) is not EdenLiveARScheduler
            or not isinstance(getattr(scheduler, "generator", None), torch.Generator)
            or config.get("cpu_offload", False)
            or config.get("lazy_load", False)
            or config.get("seq_parallel", False)
            or config.get("tensor_parallel", False)
            or config.get("use_image_encoder", False)
            or config.get("lora_configs")
            or config.get("lora_dynamic_apply", False)
        ):
            self.clear_projection_cache()
            return False
        signatures, references = [], []
        for name in ("text_embedding_0", "text_embedding_2", "time_embedding_0", "time_embedding_2", "time_projection_1"):
            module = getattr(weights, name)
            if (
                type(module) is not MMWeight
                or module.has_lora_branch
                or module.has_diff
                or hasattr(module, "weight_diff")
                or hasattr(module, "bias_diff")
                or module.lazy_load
                or module.create_cuda_buffer
                or module.create_cpu_buffer
            ):
                self.clear_projection_cache()
                return False
            tensors = (module.weight, getattr(module, "bias", None))
            signature = tuple(self._tensor_signature(tensor) if tensor is not None else None for tensor in tensors)
            if signature[0] is None or any(tensor is not None and (tensor.device != device or entry is None) for tensor, entry in zip(tensors, signature)):
                self.clear_projection_cache()
                return False
            signatures.append((id(module), signature))
            references.extend(tensor for tensor in tensors if tensor is not None)
        signature = (id(weights), tuple(signatures), self.infer_dtype, self.sensitive_layer_dtype, self.freq_dim, self.dim)
        if self._projection_request is not scheduler.generator or self._projection_weight_signature != signature:
            self.clear_projection_cache()
            # Retain the generator and source tensors, preventing identity reuse.
            self._projection_request = scheduler.generator
            self._projection_weight_signature = signature
            self._projection_weight_refs = tuple(references)
        return True

    def _time_projection_key(self, t):
        scheduler = self.scheduler
        if t.numel() != 1 or scheduler.chunk_start == 0:
            return None
        schedule = scheduler.timesteps
        if schedule.ndim != 1 or not schedule.is_contiguous():
            return None
        signature = self._tensor_signature(schedule)
        timestep_signature = self._tensor_signature(t)
        step = scheduler.step_index
        if signature is None or timestep_signature is None or not isinstance(step, int) or not 0 <= step < schedule.numel():
            return None
        if self._projection_schedule is not schedule or self._projection_schedule_signature != signature:
            self._time_projection_cache.clear()
            self._projection_schedule = schedule
            self._projection_schedule_signature = signature
        if scheduler.cache_update:
            # EdenLiveARScheduler creates a fresh zero scalar for cache commits.
            # A modified scalar is deliberately excluded without reading CUDA.
            if timestep_signature[1] != 0 or t.dtype != torch.float32 or t.device != schedule.device:
                return None
            return (True,)
        # Normal scalar steps must be the schedule slice made by step_pre().
        if t.dtype != schedule.dtype or t.device != schedule.device or t.data_ptr() != schedule.data_ptr() + step * schedule.element_size() or timestep_signature[1] != signature[1]:
            return None
        return (False, step)

    def _project_time(self, weights, t):
        embed = sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        embed = weights.time_embedding_2.apply(F.silu(weights.time_embedding_0.apply(embed), inplace=True))
        embed0 = weights.time_projection_1.apply(F.silu(embed)).unflatten(1, (6, self.dim))
        return embed, embed0

    def _project_text(self, weights, source):
        context = source.squeeze(0).to(self.sensitive_layer_dtype)
        return weights.text_embedding_2.apply(F.gelu(weights.text_embedding_0.apply(context), approximate="tanh"))

    def _interpolate_proxy(self, source, x, cache_projections):
        zero = not self.scheduler.infer_condition and self.config["proxy_unconditional_control"] == "zero"
        signature = self._tensor_signature(source) if cache_projections else None
        key = (signature, tuple(x.shape[1:]), x.device, x.dtype, zero, self.scheduler.chunk_start) if signature is not None else None
        cached = self._proxy_interpolation_cache
        if cached is not None and cached[0] == key and cached[1] is source and cached[3] == self._tensor_signature(cached[2]):
            return cached[2]
        control = source.to(x)
        if zero:
            control = torch.zeros_like(control)
        control = F.interpolate(control, size=x.shape[1:], mode="trilinear", align_corners=False)[0]
        output_signature = self._tensor_signature(control) if key is not None else None
        self._proxy_interpolation_cache = (key, source, control, output_signature) if output_signature is not None else None
        return control

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        scheduler = self.scheduler
        x = scheduler.latents_input.to(self.infer_dtype)
        condition = inputs["image_encoder_output"]
        if self.is_wan21:
            start = scheduler.chunk_start
            image_latent = condition["vae_encoder_out"][:, start : start + x.shape[1]]
            x = torch.cat((x, image_latent), dim=0)
        cache_projections = self._prepare_projection_cache(weights, x.device)
        control = self._interpolate_proxy(inputs["proxy_features"], x, cache_projections)
        if self.config["proxy_condition_rmsnorm"]:
            control = F.rms_norm(control.permute(1, 2, 3, 0), (1024,), weights.condition_norm.tensor, 1e-6)
            control = control.permute(3, 0, 1, 2)
        x = weights.patch_embedding.apply(torch.cat((x, control), dim=0).unsqueeze(0))
        grid = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)[0].contiguous()
        t = scheduler.timestep_input
        if t.numel() > 1:
            t = t.repeat_interleave(grid[1] * grid[2])
        time_key = self._time_projection_key(t) if cache_projections else None
        cached_time = self._time_projection_cache.get(time_key) if time_key is not None else None
        if cached_time is not None and cached_time[2] == tuple(self._tensor_signature(tensor) for tensor in cached_time[:2]):
            embed, embed0 = cached_time[:2]
        else:
            embed, embed0 = self._project_time(weights, t)
            if time_key is not None:
                signatures = (self._tensor_signature(embed), self._tensor_signature(embed0))
                if all(signature is not None for signature in signatures):
                    self._time_projection_cache[time_key] = (embed, embed0, signatures)
        text_key = "context" if scheduler.infer_condition else "context_null"
        text_source = inputs["text_encoder_output"][text_key]
        text_signature = self._tensor_signature(text_source) if cache_projections else None
        cached_text = self._text_projection_cache.get(text_key) if text_signature is not None else None
        if cached_text is not None and cached_text[0] is text_source and cached_text[1] == text_signature and cached_text[3] == self._tensor_signature(cached_text[2]):
            context, context_id = cached_text[2], cached_text[4]
        else:
            context = self._project_text(weights, text_source)
            context_id = id(text_source)
            context_signature = self._tensor_signature(context) if text_signature is not None else None
            if context_signature is not None:
                self._projection_generation += 1
                context_id = (id(text_source), self._projection_generation)
                # Keep only one prompt per CFG branch, bounding scheduled-prompt
                # memory while preserving the original tensor's lifetime.
                self._text_projection_cache[text_key] = (text_source, text_signature, context, context_signature, context_id)
        if self.config["use_image_encoder"]:
            image = condition["clip_encoder_out"].squeeze(0)
            image = weights.proj_1.apply(weights.proj_0.apply(image).to(self.infer_dtype))
            image = weights.proj_4.apply(weights.proj_3.apply(F.gelu(image, approximate="none")))
            context = torch.cat((image.squeeze(0).to(self.infer_dtype), context), dim=0)
        return WanPreInferModuleOutput(
            embed=embed,
            embed0=embed0,
            x=x,
            context=context,
            grid_sizes=GridOutput(torch.tensor([grid], device=x.device, dtype=torch.int32), grid),
            adapter_args={"text_segments": inputs.get("text_segments"), "context_id": context_id},
        )
