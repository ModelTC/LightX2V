"""Qwen3-VL conditioning for causal H3 prompt travel."""

import gc

import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl import MINIMAX_H3_TEXT_HIDDEN_SIZE, MINIMAX_H3_TEXT_TAG, MiniMaxH3Qwen3VLTextEncoder, _empty_device_cache
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class MiniMaxH3CausalQwen3VLTextEncoder(MiniMaxH3Qwen3VLTextEncoder):
    def _encode_vision(self, input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, *, return_cpu=True):
        if pixel_values is None and pixel_values_videos is None:
            return None, None, None
        vision_encoder = self.load_vision_encoder().to(AI_DEVICE)
        parameter = next(vision_encoder.parameters())
        image_features = image_deepstack = video_features = video_deepstack = None
        try:
            with torch.no_grad():
                if pixel_values is not None:
                    image_features, image_deepstack = vision_encoder(pixel_values.to(AI_DEVICE, parameter.dtype), image_grid_thw.to(AI_DEVICE))
                if pixel_values_videos is not None:
                    video_features, video_deepstack = vision_encoder(pixel_values_videos.to(AI_DEVICE, parameter.dtype), video_grid_thw.to(AI_DEVICE))
            image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
            image_mask, video_mask = input_ids == image_token_id, input_ids == video_token_id
            vision_mask = image_mask | video_mask
            feature_dim = image_features.shape[-1] if image_features is not None else video_features.shape[-1]
            combined = torch.empty((int(vision_mask.sum()), feature_dim), device=AI_DEVICE, dtype=parameter.dtype)
            image_joint = image_mask[vision_mask]
            video_joint = video_mask[vision_mask]
            if image_features is not None:
                combined[image_joint] = image_features
            if video_features is not None:
                combined[video_joint] = video_features
            deepstack = []
            source = image_deepstack if image_deepstack is not None else video_deepstack
            for layer_index in range(len(source)):
                one = torch.empty_like(combined)
                if image_deepstack is not None:
                    one[image_joint] = image_deepstack[layer_index]
                if video_deepstack is not None:
                    one[video_joint] = video_deepstack[layer_index]
                deepstack.append(one.cpu() if return_cpu else one)
            return vision_mask.cpu(), combined.cpu() if return_cpu else combined, deepstack
        finally:
            if self.cpu_offload:
                vision_encoder.to("cpu")
                _empty_device_cache()
                gc.collect()

    @torch.inference_mode()
    def prepare_reference_context(self, references):
        """Prepare the reference prefix and vision features reused across actions."""
        self._ensure_loaded()
        token_ids, tags, pixels, image_grid, video_pixels, video_grid = self._prepare_reference_inputs("", references)
        vision = self._encode_vision(torch.tensor(token_ids, dtype=torch.long), pixels, image_grid, video_pixels, video_grid, return_cpu=False)
        return {"token_ids": token_ids, "token_tags": tags, "image_grid_thw": image_grid, "video_grid_thw": video_grid, "vision": vision}

    @torch.inference_mode()
    def infer(self, prompt, image_list=None, references=None, *, pad_to_len=None, pad_mode="token_id", reference_context=None):
        """Return unbatched ``[tokens, 5120]`` conditioning and text tags."""
        if pad_mode not in ("token_id", "zero_feature"):
            raise ValueError(f"Unknown H3 text padding mode: {pad_mode}")
        self._ensure_loaded()
        try:
            # Input encoding happens before DefaultRunner enters its main-model
            # try/finally.  Keep migration here so a partially failed transfer
            # still reaches the conditioner-specific offload cleanup below.
            if reference_context is not None:
                # The presentation tokenizer keeps the reference prefix and
                # prompt separate, so changing text cannot alter prefix BPE.
                prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                prepared = (
                    reference_context["token_ids"] + prompt_ids,
                    reference_context["token_tags"] + [MINIMAX_H3_TEXT_TAG] * len(prompt_ids),
                    None,
                    reference_context["image_grid_thw"],
                    None,
                    reference_context["video_grid_thw"],
                )
            elif references is not None:
                prepared = self._prepare_reference_inputs(prompt, references)
            elif image_list:
                prepared = self._prepare_keyframe_inputs(prompt, image_list)
            else:
                prepared = None
            if prepared is None:
                input_ids = self._prepare_t2av_input_ids(prompt, "cpu")
                token_tags = torch.full((input_ids.shape[0],), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
                position_ids = vision_mask = vision_embeds = deepstack = None
            else:
                token_ids, tags, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = prepared
                if pad_to_len is not None:
                    if len(token_ids) > pad_to_len:
                        raise ValueError(f"H3 presentation {len(token_ids)} exceeds pad_to_len {pad_to_len}")
                    if pad_mode == "token_id":
                        extra = pad_to_len - len(token_ids)
                        token_ids = list(token_ids) + [0] * extra
                        tags = list(tags) + [MINIMAX_H3_TEXT_TAG] * extra
                input_ids = torch.tensor(token_ids, dtype=torch.long)
                token_tags = torch.tensor(tags, dtype=torch.long)
                processor = self.load_processor()
                mm_types = torch.tensor(processor.create_mm_token_type_ids([token_ids])[0], dtype=torch.long)
                position_ids = self._get_rope_index(
                    input_ids,
                    mm_types,
                    processor.image_processor.merge_size,
                    image_grid_thw,
                    video_grid_thw,
                )
                if reference_context is None:
                    vision_mask, vision_embeds, deepstack = self._encode_vision(input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw)
                else:
                    vision_mask, vision_embeds, deepstack = reference_context["vision"]
                    if vision_mask is not None:
                        vision_mask = F.pad(vision_mask, (0, input_ids.shape[0] - vision_mask.shape[0]), value=False)
            if self.cpu_offload and not self.block_offload:
                self.text_encoder.to_cuda()
            elif self.block_offload:
                # Recreate transient device slots if the previous request was
                # configured to release them after text encoding.
                self.text_encoder.init_block_offload()
            device = self.text_encoder.device
            input_ids = input_ids.to(device)
            prompt_embeds = self.text_encoder.forward(
                input_ids,
                None if position_ids is None else position_ids.to(device),
                None if vision_mask is None else vision_mask.to(device),
                None if vision_embeds is None else vision_embeds.to(device),
                None if deepstack is None else [value.to(device) for value in deepstack],
            )
            expected_shape = (input_ids.shape[0], MINIMAX_H3_TEXT_HIDDEN_SIZE)
            if tuple(prompt_embeds.shape) != expected_shape:
                raise RuntimeError(f"MiniMax-H3 expected conditioner hidden shape {expected_shape}, but native Qwen3-VL returned {tuple(prompt_embeds.shape)}")

            prompt_embeds = prompt_embeds.to(device=AI_DEVICE, dtype=GET_DTYPE()).contiguous()
            if pad_to_len is not None and pad_mode == "zero_feature":
                extra = pad_to_len - prompt_embeds.shape[0]
                if extra < 0:
                    raise ValueError(f"H3 presentation {prompt_embeds.shape[0]} exceeds pad_to_len {pad_to_len}")
                prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, extra))
                token_tags = F.pad(token_tags, (0, extra), value=MINIMAX_H3_TEXT_TAG)
            return {
                "prompt_embeds": prompt_embeds,
                "text_token_tags": token_tags.to(prompt_embeds.device),
            }
        finally:
            if self.block_offload:
                if self.release_block_offload_buffers and self.text_encoder is not None:
                    self.text_encoder.release_block_offload_buffers()
            elif self.cpu_offload and self.text_encoder is not None:
                try:
                    self.text_encoder.to_cpu()
                except Exception as error:
                    logger.warning(f"Best-effort MiniMax-H3 text-encoder offload failed: {error}")
                _empty_device_cache()
                gc.collect()
