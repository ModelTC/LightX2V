import gc

import torch
from PIL import Image
from einops import rearrange
from loguru import logger
from safetensors.torch import load_file
from torchvision.transforms import v2

from lightx2v.models.input_encoders.hf.wanx_vae_src import CLIPModel, VAEDecoderWrapper, WanVAE
from lightx2v.models.networks.wan.infer.matrixgame.conditions import Bench_actions_gta_drive, Bench_actions_templerun, Bench_actions_universal
from lightx2v.models.networks.wan.infer.matrixgame.wan_wrapper import WanDiffusionWrapper
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import ProfilingContext4Debug
from lightx2v.utils.registry_factory import RUNNER_REGISTER

model_ckpt_path_map = {
    "universal": "base_distilled_model/base_distill.safetensors",
    "gta_drive": "gta_distilled_model/gta_keyboard2dim.safetensors",
    "templerun": "templerun_distilled_model/templerun_7dim_onlykey.safetensors",
}


@RUNNER_REGISTER("wan2.1_matrixgame")
class WanMatrixGameRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)

        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None

        self.num_frame_per_block = config.get("num_frame_per_block", 1)
        self.frame_process = v2.Compose(
            [
                v2.Resize(size=(352, 640), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def load_transformer(self):
        model = WanDiffusionWrapper(model_config=os.environ["matrix_game_config_path"], timestep_shift=self.config.timestep_shift)
        state_dict = load_file(os.path.join(self.config.model_path, model_ckpt_path_map[self.config.get("mode", "universal")]))
        model.load_state_dict(state_dict)

        model = model.to(device=self.device, dtype=self.weight_dtype)

        self.local_attn_size = model.model.local_attn_size
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        self.scheduler = model.scheduler
        self.denoising_step_list = torch.tensor(self.config.denoising_step_list, dtype=torch.long)
        if self.config.warp_denoising_step:
            timesteps = torch.cat((model.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        if self.num_frame_per_block > 1:
            model.model.num_frame_per_block = self.num_frame_per_block
        return model

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image

    def read_image_input(self, img_path):
        image = Image.open(img_path).convert("RGB")
        image = self._resizecrop(image, 352, 640)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)
        return image

    def load_image_encoder(self):
        image_encoder = CLIPModel(checkpoint_path=os.path.join(self.config.model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))
        image_encoder = image_encoder.to(dtype=self.weight_dtype, device=self.device)
        return image_encoder

    def run_image_encoder(self, image):
        clip_encoder_out = self.image_encoder.encode_video(image)
        return clip_encoder_out

    def load_vae(self):
        vae_encoder = self.load_vae_encoder()
        vae_decoder = self.load_vae_decoder()
        return vae_encoder, vae_decoder

    def load_vae_encoder(self):
        vae_encoder = WanVAE(pretrained_path=os.path.join(self.config.model_path, "Wan2.1_VAE.pth")).to(torch.float16)
        # vae_encoder = WanVAE(pretrained_path=os.path.join(self.config.model_path, "Wan2.1_VAE.pth"))
        vae_encoder = vae_encoder.to(dtype=self.weight_dtype, device=self.device)
        return vae_encoder

    def load_vae_decoder(self):
        vae_decoder = VAEDecoderWrapper(self.config.model_path)
        vae_state_dict = torch.load(os.path.join(self.config.model_path, "Wan2.1_VAE.pth"), map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if "decoder." in key or "conv2" in key:
                decoder_state_dict[key] = value
        vae_decoder.load_state_dict(decoder_state_dict)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()
        vae_decoder.compile(mode="max-autotune-no-cudagraphs")
        vae_decoder.to(dtype=self.weight_dtype, device=self.device)
        vae_decoder = vae_decoder.to(torch.float16)
        return vae_decoder

    def run_vae_encoder(self, image):
        # Encode the input image as the first latent
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.config.num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae_encoder.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
        return cond_concat

    def load_text_encoder(self):
        pass

    def run_text_encoder(self, prompt, img):
        pass

    def run_pipeline(self, save_video=True):
        self.inputs = self.run_input_encoder()

        self.run()

        gen_video = self.process_images_after_vae_decoder(save_video=save_video)

        self.end_run()

        return gen_video

    def run(self):
        inputs = self.inputs
        sampled_noise = torch.randn([1, 16, self.config.num_output_frames, 44, 80], device=self.device, dtype=self.weight_dtype)
        num_frames = (self.config.num_output_frames - 1) * 4 + 1

        conditional_dict = {
            "cond_concat": inputs["image_encoder_output"]["vae_encoder_out"],
            "visual_context": inputs["image_encoder_output"]["clip_encoder_out"],
        }

        mode = self.config.get("mode", "universal")
        if mode == "universal":
            cond_data = Bench_actions_universal(num_frames)
            mouse_condition = cond_data["mouse_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict["mouse_cond"] = mouse_condition
        elif mode == "gta_drive":
            cond_data = Bench_actions_gta_drive(num_frames)
            mouse_condition = cond_data["mouse_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict["mouse_cond"] = mouse_condition
        else:
            cond_data = Bench_actions_templerun(num_frames)
        keyboard_condition = cond_data["keyboard_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        conditional_dict["keyboard_cond"] = keyboard_condition

        with torch.no_grad():
            videos = self.inference(noise=sampled_noise, conditional_dict=conditional_dict, return_latents=False, mode=mode, profile=False)

        videos_tensor = torch.cat(videos, dim=1)
        gen_video = rearrange(videos_tensor, "B T C H W -> B C T H W")

        self.gen_video = gen_video

        return gen_video

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
        vae_cache = [None for _ in range(32)]

        self.kv_cache1 = self.kv_cache_keyboard = self.kv_cache_mouse = self.crossattn_cache = None
        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self._initialize_kv_cache_mouse_and_keyboard(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        # Step 2: Cache context feature
        current_start_frame = 0

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if profile:
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
        for current_index, current_num_frames in enumerate(all_num_frames):
            logger.info(f"========> block_idx: {current_index + 1} / {num_blocks}")
            noisy_input = noise[:, :, current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            if profile:
                torch.cuda.synchronize()
                diffusion_start.record()
            with ProfilingContext4Debug("🚀 infer_main"):
                for index, current_timestep in enumerate(self.denoising_step_list):
                    logger.info(f"=====> step_idx: {index + 1} / {len(self.denoising_step_list)}")
                    # set current timestep
                    timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep

                    if index < len(self.denoising_step_list) - 1:
                        _, denoised_pred = self.model(
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
                        _, denoised_pred = self.model(
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
            context_timestep = torch.ones_like(timestep) * self.config.context_noise

            self.model(
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

    def end_run(self):
        gc.collect()
        torch.cuda.empty_cache()


def cond_current(conditional_dict, current_start_frame, num_frame_per_block, replace=None, mode="universal"):
    new_cond = {}

    new_cond["cond_concat"] = conditional_dict["cond_concat"][:, :, current_start_frame : current_start_frame + num_frame_per_block]
    new_cond["visual_context"] = conditional_dict["visual_context"]
    if replace is not None:
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

    if replace is not None:
        return new_cond, conditional_dict
    else:
        return new_cond
