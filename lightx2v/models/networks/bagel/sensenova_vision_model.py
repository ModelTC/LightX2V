# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image

from lightx2v.models.networks.bagel.data_utils import pil_img2rgb
from lightx2v.models.networks.bagel.model import (
    GEN_THINK_SYSTEM_PROMPT,
    VLM_THINK_SYSTEM_PROMPT,
    BagelModel,
)
from lightx2v.models.networks.bagel.model_io import BagelInputs, cache_init
from lightx2v.models.networks.bagel.sensenova_tasks import get_mode_profile
from lightx2v_platform.base.global_var import AI_DEVICE


@dataclass
class SenseNovaPreparedInputs:
    inputs: Optional[BagelInputs] = None
    text_outputs: list[str] = field(default_factory=list)
    image_shape: tuple[int, int] = (1024, 1024)
    output_packed_seqlens: Optional[torch.Tensor] = None
    preprocessed_images: list[Image.Image] = field(default_factory=list)
    output_raw_tensor: bool = False


class SenseNovaVisionModel(BagelModel):
    """SenseNova-Vision orchestration on top of LightX2V's Bagel kernels."""

    @staticmethod
    def _collapse_generation_batch(generation_input):
        generation_input["packed_seqlens"] = generation_input["packed_seqlens"].sum(
            dim=0,
            keepdim=True,
            dtype=generation_input["packed_seqlens"].dtype,
        )
        generation_input["key_values_lens"] = generation_input["key_values_lens"].sum(
            dim=0,
            keepdim=True,
            dtype=generation_input["key_values_lens"].dtype,
        )

    @staticmethod
    def _collapse_cfg_batch(generation_input):
        generation_input["cfg_key_values_lens"] = generation_input["cfg_key_values_lens"].sum(
            dim=0,
            keepdim=True,
            dtype=generation_input["cfg_key_values_lens"].dtype,
        )

    def prepare_sensenova_inputs(
        self,
        input_info,
        scheduler,
        vae_model,
        input_lists,
        mode,
        vae_transform,
        vit_transform,
        output_multiple_vae=False,
        output_raw_tensor=False,
    ):
        self.set_scheduler(scheduler)
        profile = get_mode_profile(mode)
        think = bool(profile.get("think", False))
        caption = bool(profile.get("caption", False))
        understanding_output = bool(profile.get("understanding_output", False))
        max_think_token_n = int(profile.get("max_think_token_n", 1000))
        do_sample = bool(profile.get("do_sample", False))
        text_temperature = float(profile.get("text_temperature", self.config.get("text_temperature", 0.3)))

        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        self.transformer_infer.gen_context = gen_context
        self.transformer_infer.cfg_text_context = cfg_text_context
        self.transformer_infer.cfg_img_context = cfg_img_context

        target_shape = tuple(getattr(input_info, "target_shape", None) or ())
        image_shape = target_shape if len(target_shape) == 2 else (1024, 1024)
        text_outputs = []
        preprocessed_images = []
        device_type = torch.device(AI_DEVICE).type

        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            if think:
                system_prompt = VLM_THINK_SYSTEM_PROMPT if understanding_output else GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)
                elif isinstance(input_term, Image.Image):
                    input_term = vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(
                        input_term,
                        gen_context,
                        vae_model=vae_model,
                        vae=not understanding_output,
                        vit=True,
                        vae_transform=vae_transform,
                        vit_transform=vit_transform,
                    )
                    image_shape = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                    preprocessed_images.append(input_term)
                else:
                    raise TypeError(f"Unsupported SenseNova-Vision input type: {type(input_term)}")

            if understanding_output:
                text_outputs.append(
                    self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n,
                    )
                )
                return SenseNovaPreparedInputs(
                    text_outputs=text_outputs,
                    image_shape=image_shape,
                    preprocessed_images=preprocessed_images,
                )

            if think or caption:
                generated_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                gen_context = self.update_context_text(generated_text, gen_context)
                text_outputs.append(generated_text)

        input_image_count = sum(isinstance(item, Image.Image) for item in input_lists)
        num_output_vae = max(input_image_count, 1) if output_multiple_vae else 1
        output_image_shapes = [image_shape] * num_output_vae

        curr_kvlens = gen_context["kv_lens"] + [0] * (num_output_vae - 1)
        curr_ropes = [gen_context["ropes"][0] + index for index in range(num_output_vae)]
        scheduler.generator = torch.Generator(device="cpu").manual_seed(int(getattr(input_info, "seed", 42)))
        generation_input = scheduler.prepare_vae_latent(
            curr_kvlens=curr_kvlens,
            curr_rope=curr_ropes,
            image_sizes=output_image_shapes,
            new_token_ids=self.new_token_ids,
            seed=getattr(input_info, "seed", 42),
        )
        output_packed_seqlens = generation_input["packed_seqlens"].clone()
        self._collapse_generation_batch(generation_input)

        cfg_text_past_key_values = cfg_text_context["past_key_values"]
        cfg_text_kvlens = cfg_text_context["kv_lens"] + [0] * (num_output_vae - 1)
        cfg_text_ropes = [cfg_text_context["ropes"][0] + index for index in range(num_output_vae)]
        generation_input_cfg_text = scheduler.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_kvlens,
            curr_rope=cfg_text_ropes,
            image_sizes=output_image_shapes,
        )
        self._collapse_cfg_batch(generation_input_cfg_text)

        cfg_img_past_key_values = cfg_img_context["past_key_values"]
        cfg_img_kvlens = cfg_img_context["kv_lens"] + [0] * (num_output_vae - 1)
        cfg_img_ropes = [cfg_img_context["ropes"][0] + index for index in range(num_output_vae)]
        generation_input_cfg_img = scheduler.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_kvlens,
            curr_rope=cfg_img_ropes,
            image_sizes=output_image_shapes,
        )
        self._collapse_cfg_batch(generation_input_cfg_img)

        scheduler.generation_input = generation_input
        scheduler.generation_input_cfg_text = generation_input_cfg_text
        scheduler.generation_input_cfg_image = generation_input_cfg_img
        scheduler.latents = generation_input["packed_init_noises"]

        if self.enable_taylorseer:
            model_pred_cache_dic, model_pred_current = cache_init(self, scheduler.infer_steps)
            model_pred_text_cache_dic, model_pred_text_current = cache_init(self, scheduler.infer_steps)
            model_pred_img_cache_dic, model_pred_img_current = cache_init(self, scheduler.infer_steps)
        else:
            model_pred_cache_dic = model_pred_current = None
            model_pred_text_cache_dic = model_pred_text_current = None
            model_pred_img_cache_dic = model_pred_img_current = None

        inputs = BagelInputs(
            image_shapes=output_image_shapes,
            gen_context=gen_context,
            cfg_text_precontext=cfg_text_context,
            cfg_img_precontext=cfg_img_context,
            model_pred_cache_dic=model_pred_cache_dic,
            model_pred_current=model_pred_current,
            model_pred_text_cache_dic=model_pred_text_cache_dic,
            model_pred_text_current=model_pred_text_current,
            model_pred_img_cache_dic=model_pred_img_cache_dic,
            model_pred_img_current=model_pred_img_current,
            generation_input=generation_input,
            generation_input_cfg_text=generation_input_cfg_text,
            generation_input_cfg_img=generation_input_cfg_img,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
        )
        return SenseNovaPreparedInputs(
            inputs=inputs,
            text_outputs=text_outputs,
            image_shape=image_shape,
            output_packed_seqlens=output_packed_seqlens,
            preprocessed_images=preprocessed_images,
            output_raw_tensor=output_raw_tensor,
        )
