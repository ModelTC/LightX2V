import copy
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from .wan_wrapper import WanDiffusionWrapper


def get_current_action(mode="universal"):
    CAM_VALUE = 0.1
    if mode == "universal":
        print()
        print("-" * 30)
        print("PRESS [I, K, J, L, U] FOR CAMERA TRANSFORM\n (I: up, K: down, J: left, L: right, U: no move)")
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        print("-" * 30)
        CAMERA_VALUE_MAP = {"i": [CAM_VALUE, 0], "k": [-CAM_VALUE, 0], "j": [0, -CAM_VALUE], "l": [0, CAM_VALUE], "u": [0, 0]}
        KEYBOARD_IDX = {"w": [1, 0, 0, 0], "s": [0, 1, 0, 0], "a": [0, 0, 1, 0], "d": [0, 0, 0, 1], "q": [0, 0, 0, 0]}
        flag = 0
        while flag != 1:
            try:
                idx_mouse = input("Please input the mouse action (e.g. `U`):\n").strip().lower()
                idx_keyboard = input("Please input the keyboard action (e.g. `W`):\n").strip().lower()
                if idx_mouse in CAMERA_VALUE_MAP.keys() and idx_keyboard in KEYBOARD_IDX.keys():
                    flag = 1
            except:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    elif mode == "gta_drive":
        print()
        print("-" * 30)
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        print("-" * 30)
        CAMERA_VALUE_MAP = {"a": [0, -CAM_VALUE], "d": [0, CAM_VALUE], "q": [0, 0]}
        KEYBOARD_IDX = {"w": [1, 0], "s": [0, 1], "q": [0, 0]}
        flag = 0
        while flag != 1:
            try:
                indexes = input("Please input the actions (split with ` `):\n(e.g. `W` for forward, `W A` for forward and left)\n").strip().lower().split(" ")
                idx_mouse = []
                idx_keyboard = []
                for i in indexes:
                    if i in CAMERA_VALUE_MAP.keys():
                        idx_mouse += [i]
                    elif i in KEYBOARD_IDX.keys():
                        idx_keyboard += [i]
                if len(idx_mouse) == 0:
                    idx_mouse += ["q"]
                if len(idx_keyboard) == 0:
                    idx_keyboard += ["q"]
                assert idx_mouse in [["a"], ["d"], ["q"]] and idx_keyboard in [["q"], ["w"], ["s"]]
                flag = 1
            except:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse[0]]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard[0]]).cuda()
    elif mode == "templerun":
        print()
        print("-" * 30)
        print("PRESS [W, S, A, D, Z, C, Q] FOR ACTIONS\n (W: jump, S: slide, A: left side, D: right side, Z: turn left, C: turn right, Q: no move)")
        print("-" * 30)
        KEYBOARD_IDX = {
            "w": [0, 1, 0, 0, 0, 0, 0],
            "s": [0, 0, 1, 0, 0, 0, 0],
            "a": [0, 0, 0, 0, 0, 1, 0],
            "d": [0, 0, 0, 0, 0, 0, 1],
            "z": [0, 0, 0, 1, 0, 0, 0],
            "c": [0, 0, 0, 0, 1, 0, 0],
            "q": [1, 0, 0, 0, 0, 0, 0],
        }
        flag = 0
        while flag != 1:
            try:
                idx_keyboard = input("Please input the action: \n(e.g. `W` for forward, `Z` for turning left)\n").strip().lower()
                if idx_keyboard in KEYBOARD_IDX.keys():
                    flag = 1
            except:
                pass
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()

    if mode != "templerun":
        return {"mouse": mouse_cond, "keyboard": keyboard_cond}
    return {"keyboard": keyboard_cond}


def cond_current(conditional_dict, current_start_frame, num_frame_per_block, replace=None, mode="universal"):
    new_cond = {}

    new_cond["cond_concat"] = conditional_dict["cond_concat"][:, :, current_start_frame : current_start_frame + num_frame_per_block]
    new_cond["visual_context"] = conditional_dict["visual_context"]
    if replace != None:
        if current_start_frame == 0:
            last_frame_num = 1 + 4 * (num_frame_per_block - 1)
        else:
            last_frame_num = 4 * num_frame_per_block
        final_frame = 1 + 4 * (current_start_frame + num_frame_per_block - 1)
        if mode != "templerun":
            conditional_dict["mouse_cond"][:, -last_frame_num + final_frame : final_frame] = replace["mouse"][None, None, :].repeat(1, last_frame_num, 1)
        conditional_dict["keyboard_cond"][:, -last_frame_num + final_frame : final_frame] = replace["keyboard"][None, None, :].repeat(1, last_frame_num, 1)
    if mode != "templerun":
        new_cond["mouse_cond"] = conditional_dict["mouse_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]
    new_cond["keyboard_cond"] = conditional_dict["keyboard_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]

    if replace != None:
        return new_cond, conditional_dict
    else:
        return new_cond


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
        self,
        args,
        device="cuda",
        generator=None,
        vae_decoder=None,
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator

        self.vae_decoder = vae_decoder
        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict,
        initial_latent=None,
        return_latents=False,
        mode="universal",
        profile=False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """

        assert noise.shape[1] == 16
        batch_size, num_channels, num_frames, height, width = noise.shape

        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        output = torch.zeros([batch_size, num_channels, num_output_frames, height, width], device=noise.device, dtype=noise.dtype)
        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None

        self.kv_cache1 = self.kv_cache_keyboard = self.kv_cache_mouse = self.crossattn_cache = None
        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_kv_cache_mouse_and_keyboard(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[:, :, current_start_frame : current_start_frame + self.num_frame_per_block]
                output[:, :, current_start_frame : current_start_frame + self.num_frame_per_block] = current_ref_latents

                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if profile:
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
        for current_num_frames in tqdm(all_num_frames):
            noisy_input = noise[:, :, current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            if profile:
                torch.cuda.synchronize()
                diffusion_start.record()
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        rearrange(denoised_pred, "b c f h w -> (b f) c h w"),  # .flatten(0, 1),
                        torch.randn_like(rearrange(denoised_pred, "b c f h w -> (b f) c h w")),
                        next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    )
                    noisy_input = rearrange(noisy_input, "(b f) c h w -> b c f h w", b=denoised_pred.shape[0])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            # Step 3.2: record the model's output
            output[:, :, current_start_frame : current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise

            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_cache_mouse=self.kv_cache_mouse,
                kv_cache_keyboard=self.kv_cache_keyboard,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

            denoised_pred = denoised_pred.transpose(1, 2)
            video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
            videos += [video]

            if profile:
                torch.cuda.synchronize()
                diffusion_end.record()
                diffusion_time = diffusion_start.elapsed_time(diffusion_end)
                print(f"diffusion_time: {diffusion_time}", flush=True)
                fps = video.shape[1] * 1000 / diffusion_time
                print(f"  - FPS: {fps:.2f}")

        if return_latents:
            return output
        else:
            return videos

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 15 * 1 * self.frame_seq_length  # 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append(
                {
                    "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
            kv_cache_mouse.append(
                {
                    "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
        self.kv_cache_keyboard = kv_cache_keyboard  # always store the clean cache
        self.kv_cache_mouse = kv_cache_mouse  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {"k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device), "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device), "is_init": False}
            )
        self.crossattn_cache = crossattn_cache


class CausalInferenceStreamingPipeline(torch.nn.Module):
    def __init__(
        self,
        args,
        device="cuda",
        vae_decoder=None,
        generator=None,
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.vae_decoder = vae_decoder

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880  # 1590 # HW/4

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self, noise: torch.Tensor, conditional_dict, initial_latent: Optional[torch.Tensor] = None, return_latents: bool = False, output_folder=None, name=None, mode="universal"
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """

        assert noise.shape[1] == 16
        batch_size, num_channels, num_frames, height, width = noise.shape

        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        output = torch.zeros([batch_size, num_channels, num_output_frames, height, width], device=noise.device, dtype=noise.dtype)
        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None
        # Set up profiling if requested
        self.kv_cache1 = self.kv_cache_keyboard = self.kv_cache_mouse = self.crossattn_cache = None
        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_kv_cache_mouse_and_keyboard(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0

            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[:, :, current_start_frame : current_start_frame + self.num_frame_per_block]
                output[:, :, current_start_frame : current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, replace=True),
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks

        for current_num_frames in all_num_frames:
            noisy_input = noise[:, :, current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames]

            current_actions = get_current_action(mode=mode)
            new_act, conditional_dict = cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, replace=current_actions, mode=mode)
            # Step 3.1: Spatial denoising loop

            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        rearrange(denoised_pred, "b c f h w -> (b f) c h w"),  # .flatten(0, 1),
                        torch.randn_like(rearrange(denoised_pred, "b c f h w -> (b f) c h w")),
                        next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    )
                    noisy_input = rearrange(noisy_input, "(b f) c h w -> b c f h w", b=denoised_pred.shape[0])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            # Step 3.2: record the model's output
            output[:, :, current_start_frame : current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise

            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=new_act,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_cache_mouse=self.kv_cache_mouse,
                kv_cache_keyboard=self.kv_cache_keyboard,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # Step 3.4: update the start and end frame indices
            denoised_pred = denoised_pred.transpose(1, 2)
            video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
            videos += [video]
            video = rearrange(video, "B T C H W -> B T H W C")
            video = ((video.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
            video = np.ascontiguousarray(video)
            mouse_icon = "assets/images/mouse.png"
            if mode != "templerun":
                config = (
                    conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block - 1)].float().cpu().numpy(),
                    conditional_dict["mouse_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block - 1)].float().cpu().numpy(),
                )
            else:
                config = conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block - 1)].float().cpu().numpy()
            process_video(video.astype(np.uint8), output_folder + f"/{name}_current.mp4", config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)
            current_start_frame += current_num_frames

            if input("Continue? (Press `n` to break)").strip() == "n":
                break

        videos_tensor = torch.cat(videos, dim=1)
        videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos)
        mouse_icon = "assets/images/mouse.png"
        if mode != "templerun":
            config = (
                conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block - 1)].float().cpu().numpy(),
                conditional_dict["mouse_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block - 1)].float().cpu().numpy(),
            )
        else:
            config = conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block - 1)].float().cpu().numpy()
        process_video(video.astype(np.uint8), output_folder + f"/{name}_icon.mp4", config, mouse_icon, mouse_scale=0.1, mode=mode)
        process_video(video.astype(np.uint8), output_folder + f"/{name}.mp4", config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)

        if return_latents:
            return output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 15 * 1 * self.frame_seq_length  # 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append(
                {
                    "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
            kv_cache_mouse.append(
                {
                    "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
        self.kv_cache_keyboard = kv_cache_keyboard  # always store the clean cache
        self.kv_cache_mouse = kv_cache_mouse  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {"k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device), "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device), "is_init": False}
            )
        self.crossattn_cache = crossattn_cache
