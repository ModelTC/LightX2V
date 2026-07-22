import torch
import torch.nn.functional as F
from einops import rearrange

from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

from .rope import RotaryPositionalEmbedding1D


class WanInfiniteTalkPreInfer(WanPreInfer):
    def __init__(self, config):
        super().__init__(config)
        self.audio_window = config.get("audio_window", 5)
        self.vae_scale = config.get("infinitetalk_vae_scale", 4)
        self.context_tokens = config.get("infinitetalk_context_tokens", 32)
        self.audio_output_dim = config.get("infinitetalk_audio_output_dim", 768)
        self.audio_rope = None
        self.rope_1d = RotaryPositionalEmbedding1D(self.head_size)
        self._audio_rope_cache = {}

    def set_audio_rope(self, rope):
        self.audio_rope = rope
        self._audio_rope_cache.clear()

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        original_latents = self.scheduler.latents
        image_encoder_output = inputs.get("image_encoder_output", {})
        original_vae_encoder_out = image_encoder_output.get("vae_encoder_out")

        self.scheduler.latents = original_latents.to(GET_DTYPE())
        if original_vae_encoder_out is not None:
            image_encoder_output["vae_encoder_out"] = original_vae_encoder_out.to(GET_DTYPE())
        try:
            pre_out = super().infer(weights, inputs, kv_start=kv_start, kv_end=kv_end)
        finally:
            self.scheduler.latents = original_latents
            if original_vae_encoder_out is not None:
                image_encoder_output["vae_encoder_out"] = original_vae_encoder_out
        if self.task == "s2v" and self.config.get("use_image_encoder", True):
            self._prepend_clip_context(weights, inputs, pre_out)
        self._append_audio_adapter_args(weights, inputs, pre_out)
        self._append_ref_masks(inputs, pre_out)
        self._append_audio_rope_cache(pre_out)
        return pre_out

    def _prepend_clip_context(self, weights, inputs, pre_out):
        clip_fea = inputs["image_encoder_output"]["clip_encoder_out"]
        if GET_SENSITIVE_DTYPE() != GET_DTYPE():
            context_clip = weights.proj_0.apply(clip_fea.to(GET_SENSITIVE_DTYPE()))
        else:
            context_clip = weights.proj_0.apply(clip_fea.to(GET_DTYPE()))
        context_clip = weights.proj_1.apply(context_clip)
        context_clip = torch.nn.functional.gelu(context_clip, approximate="none")
        context_clip = weights.proj_3.apply(context_clip)
        context_clip = weights.proj_4.apply(context_clip)
        pre_out.context = torch.concat([context_clip, pre_out.context], dim=0)

    def _append_audio_adapter_args(self, weights, inputs, pre_out):
        audio_cond = inputs["audio_encoder_output"].to(device=AI_DEVICE, dtype=GET_DTYPE())
        first_frame_audio = audio_cond[:, :1]
        latter_audio = audio_cond[:, 1:]
        if latter_audio.shape[1] % self.vae_scale != 0:
            raise ValueError(f"audio frames after the first frame must be divisible by {self.vae_scale}, got {latter_audio.shape[1]}")

        latter_audio = rearrange(latter_audio, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale)
        middle_index = self.audio_window // 2
        latter_first = rearrange(latter_audio[:, :, :1, : middle_index + 1], "b n_t n w s c -> b n_t (n w) s c")
        latter_middle = rearrange(latter_audio[:, :, 1:-1, middle_index : middle_index + 1], "b n_t n w s c -> b n_t (n w) s c")
        latter_last = rearrange(latter_audio[:, :, -1:, middle_index:], "b n_t n w s c -> b n_t (n w) s c")
        latter_frame_audio = torch.concat([latter_first, latter_middle, latter_last], dim=2)

        audio_embedding = self._audio_proj(weights, first_frame_audio, latter_frame_audio)
        human_num = audio_embedding.shape[0]
        audio_embedding = torch.concat(audio_embedding.split(1, dim=0), dim=2).squeeze(0).to(GET_DTYPE())
        pre_out.adapter_args["audio_embedding"] = audio_embedding
        pre_out.adapter_args["human_num"] = human_num

    def _audio_proj(self, weights, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        humans = audio_embeds.shape[0]

        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) (w b c)")
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) (w b c)")

        audio_embeds = torch.relu(weights.audio_proj_proj1.apply(audio_embeds))
        audio_embeds_vf = torch.relu(weights.audio_proj_proj1_vf.apply(audio_embeds_vf))
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=humans)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=humans)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1)
        audio_embeds_c = rearrange(audio_embeds_c, "bz f c -> (bz f) c")
        audio_embeds_c = torch.relu(weights.audio_proj_proj2.apply(audio_embeds_c))
        context_tokens = weights.audio_proj_proj3.apply(audio_embeds_c).reshape(humans * video_length, self.context_tokens, self.audio_output_dim)
        if hasattr(weights, "audio_proj_norm"):
            context_tokens = weights.audio_proj_norm.apply(context_tokens).to(GET_DTYPE())
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)
        return context_tokens

    def _append_ref_masks(self, inputs, pre_out):
        ref_target_masks = inputs.get("ref_target_masks")
        if ref_target_masks is None:
            pre_out.adapter_args["ref_target_masks"] = None
            return
        _, grid_h, grid_w = pre_out.grid_sizes.tuple
        masks = ref_target_masks.to(device=AI_DEVICE, dtype=torch.float32).unsqueeze(0)
        masks = F.interpolate(masks, size=(grid_h, grid_w), mode="nearest").squeeze(0)
        masks = (masks > 0).flatten(1).to(GET_DTYPE())
        pre_out.adapter_args["ref_target_masks"] = masks

    def _append_audio_rope_cache(self, pre_out):
        human_num = pre_out.adapter_args.get("human_num", 1)
        ref_target_masks = pre_out.adapter_args.get("ref_target_masks")
        if human_num <= 1 or ref_target_masks is None:
            return
        if self.audio_rope is None:
            raise RuntimeError("InfiniteTalk audio RoPE is not initialized.")

        human_map_count = min(int(human_num), max(0, ref_target_masks.shape[0] - 1))
        if human_map_count <= 0:
            return

        audio_embedding = pre_out.adapter_args["audio_embedding"]
        grid_t = pre_out.grid_sizes.tuple[0]
        audio_tokens = audio_embedding.shape[1]
        device = audio_embedding.device
        dtype = audio_embedding.dtype
        cache_key = (
            human_map_count,
            grid_t,
            audio_tokens,
            device.type,
            device.index,
            dtype,
        )
        cached = self._audio_rope_cache.get(cache_key)
        if cached is None:
            h1 = torch.tensor((0, self.config.get("infinitetalk_class_interval", 4)), dtype=dtype, device=device)
            class_range = self.config.get("infinitetalk_class_range", 24)
            h2 = torch.tensor(
                (class_range - self.config.get("infinitetalk_class_interval", 4), class_range),
                dtype=dtype,
                device=device,
            )
            if human_map_count == 2:
                rope_ranges = torch.stack([h1, h2], dim=0)
            else:
                starts = torch.linspace(h1[0], h2[0], human_map_count, dtype=dtype, device=device)
                ends = torch.linspace(h1[1], h2[1], human_map_count, dtype=dtype, device=device)
                rope_ranges = torch.stack([starts, ends], dim=1)

            per_frame = torch.zeros(audio_tokens, dtype=dtype, device=device)
            token_edges = torch.linspace(
                0,
                audio_tokens,
                human_map_count + 1,
                dtype=torch.int64,
                device=device,
            )
            rope_centers = rope_ranges.mean(dim=1)
            for idx in range(human_map_count):
                per_frame[token_edges[idx] : token_edges[idx + 1]] = rope_centers[idx]
            encoder_pos = per_frame.repeat(grid_t)
            encoder_freqs, encoder_positions = self.rope_1d.prepare(self.audio_rope, encoder_pos)
            cached = (rope_ranges, encoder_freqs, encoder_positions)
            self._audio_rope_cache[cache_key] = cached

        pre_out.adapter_args["audio_rope_ranges"] = cached[0]
        pre_out.adapter_args["audio_encoder_rope"] = cached[1:]
