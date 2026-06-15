from dataclasses import replace

import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.dreamzero.module_io import DreamZeroPreInferOutput
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


def _tensor(weight):
    if hasattr(weight, "tensor"):
        return weight.tensor
    return weight.pin_tensor.to(AI_DEVICE)


def _category_linear(x, weights):
    weight = _tensor(weights.W)[0].to(device=x.device, dtype=x.dtype)
    bias = _tensor(weights.b)[0].to(device=x.device, dtype=x.dtype)
    return torch.matmul(x, weight) + bias


def _swish(x):
    return x * torch.sigmoid(x)


def _action_pos_encoding(timesteps, embedding_dim):
    timesteps = timesteps.float()
    half_dim = embedding_dim // 2
    exponent = -torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / half_dim)
    freqs = timesteps.unsqueeze(-1) * exponent.exp()
    return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


def _dreamzero_sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half, dtype=position.dtype, device=position.device).div(half)),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


class DreamZeroPreInfer:
    def __init__(self, config):
        self.config = config
        self.freq_dim = config["freq_dim"]
        self.dim = config["dim"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.dim // self.num_heads
        self.patch_size = tuple(config.get("patch_size", (1, 2, 2)))
        self.num_action_per_block = int(config.get("num_action_per_block", config.get("action_horizon", 24)))
        self.num_state_per_block = int(config.get("num_state_per_block", 1))
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()
        self._freqs = None
        self._freqs_action = None
        self._freqs_state = None
        self._context_projection_cache = {}

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def clear_cache(self):
        self._context_projection_cache.clear()

    @staticmethod
    def _rope_params(max_seq_len, dim, theta=10000):
        assert dim % 2 == 0
        freqs = torch.outer(
            torch.arange(max_seq_len, dtype=torch.float64),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float64).div(dim)),
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    def _ensure_freqs(self, device):
        if self._freqs is not None and self._freqs[0].device == device:
            return
        d = self.head_dim
        self._freqs = [
            self._rope_params(1024, d - 4 * (d // 6)).to(device),
            self._rope_params(1024, 2 * (d // 6)).to(device),
            self._rope_params(1024, 2 * (d // 6)).to(device),
        ]
        self._freqs_action = self._rope_params(1024 * 10, d).to(device)
        self._freqs_state = self._rope_params(1024, d).to(device)

    def _create_freqs(self, grid_size, start_frame, device):
        self._ensure_freqs(device)
        f, h, w = grid_size
        return torch.cat(
            [
                self._freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self._freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self._freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)

    def _action_encoder(self, weights, action, timestep_action):
        action_features = _category_linear(action, weights.action_encoder.W1)
        tau_emb = _action_pos_encoding(timestep_action, action_features.shape[-1]).to(dtype=action_features.dtype)
        action_features = _swish(_category_linear(torch.cat([action_features, tau_emb], dim=-1), weights.action_encoder.W2))
        return _category_linear(action_features, weights.action_encoder.W3)

    def _state_encoder(self, weights, state):
        state_features = F.relu(_category_linear(state, weights.state_encoder.layer1))
        return _category_linear(state_features, weights.state_encoder.layer2)

    @staticmethod
    def _tensor_cache_id(tensor):
        if tensor is None:
            return None
        return (tensor.data_ptr(), tuple(tensor.shape), str(tensor.device), str(tensor.dtype))

    def _context_cache_key(self, inputs, context, clip_feature, device, dtype):
        return (
            str(inputs.get("context_cache_name", inputs.get("cache_name", "pos"))),
            self._tensor_cache_id(context),
            self._tensor_cache_id(clip_feature),
            str(device),
            str(dtype),
        )

    @torch.no_grad()
    def project_context(self, weights, inputs, device, dtype):
        context = inputs.get("context")
        clip_feature = inputs.get("clip_feature")
        if context is None:
            raise ValueError("DreamZero context projection requires context.")
        if clip_feature is None:
            raise ValueError("DreamZero context projection requires clip_feature.")

        cache_key = self._context_cache_key(inputs, context, clip_feature, device, dtype)
        cached = self._context_projection_cache.get(cache_key)
        if cached is not None:
            return cached

        context = context.to(device=device).to(GET_DTYPE()).squeeze(0)
        context = weights.text_embedding_0.apply(context.to(self.sensitive_layer_dtype if self.sensitive_layer_dtype != self.infer_dtype else context.dtype))
        context = F.gelu(context, approximate="tanh")
        context = weights.text_embedding_2.apply(context)

        clip_feature = clip_feature.to(device=device).to(GET_DTYPE())
        if clip_feature.dim() == 3:
            clip_feature = clip_feature.squeeze(0)
        context_clip = weights.proj_0.apply(clip_feature)
        context_clip = weights.proj_1.apply(context_clip)
        context_clip = F.gelu(context_clip, approximate="none")
        context_clip = weights.proj_3.apply(context_clip)
        context_clip = weights.proj_4.apply(context_clip)
        projected_context = torch.cat([context_clip, context], dim=0)
        self._context_projection_cache[cache_key] = projected_context
        return projected_context

    @torch.no_grad()
    def infer_shared(self, weights, inputs):
        x = inputs["video_latents"].to(AI_DEVICE).to(GET_DTYPE())
        if x.shape[0] != 1:
            raise ValueError(f"DreamZero native inference expects batch_size=1, got {x.shape[0]}.")
        y = inputs.get("y")
        if y is not None and self.config.get("concat_first_frame_latent", True):
            x = torch.cat([x, y.to(device=x.device, dtype=x.dtype)], dim=1)

        x = weights.patch_embedding.apply(x)
        grid_size = tuple(int(v) for v in x.shape[2:])
        seq_len = grid_size[0] * grid_size[1] * grid_size[2]
        x = x.flatten(start_dim=2).transpose(1, 2).contiguous().squeeze(0)

        action = inputs.get("action")
        timestep_action = inputs.get("timestep_action")
        state = inputs.get("state")
        action_length = 0
        action_register_length = None
        if action is not None:
            action = action.to(device=x.device, dtype=x.dtype)
            timestep_action = timestep_action.to(device=x.device)
            state = state.to(device=x.device, dtype=x.dtype)
            action_features = self._action_encoder(weights, action, timestep_action)
            state_features = self._state_encoder(weights, state)
            action_register = torch.cat([action_features, state_features], dim=1).squeeze(0)
            action_length = int(action_features.shape[1])
            action_register_length = int(action_register.shape[0])
            x = torch.cat([x, action_register], dim=0)

        timestep = inputs["timestep"].to(device=x.device)
        frame_count = int(timestep.shape[1])
        if frame_count <= seq_len:
            repeat = (seq_len + frame_count - 1) // frame_count
            timestep_tokens = timestep.repeat_interleave(repeat, dim=1)[:, :seq_len]
        else:
            indices = torch.linspace(0, frame_count - 1, seq_len, device=timestep.device, dtype=torch.long)
            timestep_tokens = timestep[:, indices]
        if action is not None:
            stride = timestep_action.shape[1] // state.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep_tokens = torch.cat([timestep_tokens, timestep_action, timestep_state], dim=1)

        embed = _dreamzero_sinusoidal_embedding_1d(self.freq_dim, timestep_tokens.flatten()).to(device=x.device, dtype=x.dtype)
        embed = weights.time_embedding_0.apply(embed.to(self.sensitive_layer_dtype if self.sensitive_layer_dtype != self.infer_dtype else embed.dtype))
        embed = F.silu(embed)
        embed = weights.time_embedding_2.apply(embed)
        embed0 = weights.time_projection_1.apply(F.silu(embed)).reshape(-1, 6, self.dim)

        freqs = self._create_freqs(grid_size, int(inputs.get("current_start_frame", 0)), x.device)
        return DreamZeroPreInferOutput(
            x=x,
            embed=embed,
            embed0=embed0,
            context=None,
            freqs=freqs,
            freqs_action=self._freqs_action,
            freqs_state=self._freqs_state,
            grid_size=grid_size,
            seq_len=seq_len,
            action_length=action_length,
            action_register_length=action_register_length,
            current_start_frame=int(inputs.get("current_start_frame", 0)),
            update_cache=bool(inputs.get("update_cache", False)),
            cache_name=str(inputs.get("cache_name", "pos")),
        )

    @torch.no_grad()
    def with_context(self, weights, pre_infer_out, inputs):
        context = self.project_context(weights, inputs, pre_infer_out.x.device, pre_infer_out.x.dtype)
        return replace(
            pre_infer_out,
            context=context,
            update_cache=bool(inputs.get("update_cache", False)),
            cache_name=str(inputs.get("cache_name", pre_infer_out.cache_name)),
        )

    @torch.no_grad()
    def infer(self, weights, inputs):
        return self.with_context(weights, self.infer_shared(weights, inputs), inputs)
