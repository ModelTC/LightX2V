import gc

import torch
from loguru import logger

from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import ProfilingContext4Debug
from lightx2v.utils.registry_factory import RUNNER_REGISTER

from lightx2v.models.networks.wan.infer.matrixgame.wan_wrapper import WanDiffusionWrapper

from safetensors.torch import load_file
from torchvision.transforms import v2


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
        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        

    def load_transformer(self):
        model = WanDiffusionWrapper(model_config=self.config.model_config_path, timestep_shift=self.config.timestep_shift)
        state_dict = load_file(self.config.model_path)
        model.load_state_dict(state_dict)

        self.local_attn_size = model.model.local_attn_size
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

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


    def run_image_encoder(self, image):
        image = self._resizecrop(image, 352, 640)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)
        clip_encoder_out = self.image_encoder.visual([image[None, :, :, :]]).squeeze(0).to(GET_DTYPE())
        print('clip_encode_out', clip_encoder_out)
        return clip_encoder_out

    def init_scheduler(self):
        self.scheduler = self.model.scheduler
        # self.model.scheduler.prepare(self.inputs["image_encoder_output"])
        # self.model.scheduler.clear()

        self.denoising_step_list = torch.tensor(
            self.config.denoising_step_list, dtype=torch.long)
        if self.config.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # self.model.scheduler.step_pre(step_index=step_index)
        # self.model.scheduler.step_post()

    def prepare_inputs(self):
        
        pass

    def run(self):
        inputs = self.inputs
        print(inputs)

        return [], None

    def end_run(self):
        gc.collect()
        torch.cuda.empty_cache()
