import os

import torch
from diffusers.utils.loading_utils import load_image
from torchvision.transforms import v2

from lightx2v.models.input_encoders.hf.wan.matrix_game2.clip import CLIPModel
from lightx2v.models.input_encoders.hf.wan.matrix_game2.conditions import Bench_actions_gta_drive, Bench_actions_templerun, Bench_actions_universal
from lightx2v.models.networks.wan.matrix_game2_model import WanSFMtxg2Model
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner
from lightx2v.models.video_encoders.hf.wan.vae_sf import WanMtxg2VAE
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER


class VAEWrapper:
    def __init__(self, vae):
        self.vae = vae

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self.vae, name)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, latents):
        return NotImplementedError


class WanxVAEWrapper(VAEWrapper):
    def __init__(self, vae, clip):
        self.vae = vae
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.clip = clip
        if clip is not None:
            self.clip.requires_grad_(False)
            self.clip.eval()

    def encode(self, x, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        x = self.vae.encode(x, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)  # already scaled
        return x  # torch.stack(x, dim=0)

    def clip_img(self, x):
        x = self.clip(x)
        return x

    def decode(self, latents, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        videos = self.vae.decode(latents, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return videos  # self.vae.decode(videos, dim=0) # already scaled

    def to(self, device, dtype):
        # 移动 vae 到指定设备
        self.vae = self.vae.to(device, dtype)

        # 如果 clip 存在，也移动到指定设备
        if self.clip is not None:
            self.clip = self.clip.to(device, dtype)

        return self


def get_wanx_vae_wrapper(model_path, weight_dtype):
    vae = WanMtxg2VAE(pretrained_path=os.path.join(model_path, "Wan2.1_VAE.pth")).to(weight_dtype)
    clip = CLIPModel(checkpoint_path=os.path.join(model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"), tokenizer_path=os.path.join(model_path, "xlm-roberta-large"))
    return WanxVAEWrapper(vae, clip)


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
            except Exception as e:
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
            except Exception as e:
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
            except Exception as e:
                pass
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()

    if mode != "templerun":
        return {"mouse": mouse_cond, "keyboard": keyboard_cond}
    return {"keyboard": keyboard_cond}


@RUNNER_REGISTER("wan2.1_sf_mtxg2")
class WanSFMtxg2Runner(WanSFRunner):
    def __init__(self, config):
        super().__init__(config)
        self.frame_process = v2.Compose(
            [
                v2.Resize(size=(352, 640), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

    def load_text_encoder(self):
        from lightx2v.models.input_encoders.hf.wan.matrix_game2.conditions import MatrixGame2_Bench

        return MatrixGame2_Bench()

    def load_image_encoder(self):
        wrapper = get_wanx_vae_wrapper(self.config["model_path"], torch.float16)
        wrapper.requires_grad_(False)
        wrapper.eval()
        return wrapper.to(self.device, self.weight_dtype)

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

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        # image
        image = load_image(self.input_info.image_path)
        image = self._resizecrop(image, 352, 640)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.config["num_output_frames"] - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.image_encoder.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
        visual_context = self.image_encoder.clip.encode_video(image)
        image_encoder_output = {"cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype), "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)}

        # text
        text_encoder_output = {}
        num_frames = (self.config["num_output_frames"] - 1) * 4 + 1
        if self.config["mode"] == "universal":
            cond_data = Bench_actions_universal(num_frames)
            mouse_condition = cond_data["mouse_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            text_encoder_output["mouse_cond"] = mouse_condition
        elif self.config["mode"] == "gta_drive":
            cond_data = Bench_actions_gta_drive(num_frames)
            mouse_condition = cond_data["mouse_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            text_encoder_output["mouse_cond"] = mouse_condition
        else:
            cond_data = Bench_actions_templerun(num_frames)
        keyboard_condition = cond_data["keyboard_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        text_encoder_output["keyboard_cond"] = keyboard_condition

        # set shape
        self.input_info.latent_shape = [16, self.config["num_output_frames"], 44, 80]

        return {"text_encoder_output": text_encoder_output, "image_encoder_output": image_encoder_output}

    def load_transformer(self):
        model = WanSFMtxg2Model(
            self.config["model_path"],
            self.config,
            self.init_device,
        )
        return model

    def init_run_segment(self, segment_idx):
        self.segment_idx = segment_idx

        if self.config["streaming"]:
            self.inputs["current_actions"] = get_current_action(mode=self.config["mode"])

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        self.init_run()
        if self.config.get("compile", False):
            self.model.select_graph_for_compile(self.input_info)

        stop = ""
        while stop != "n":
            for segment_idx in range(self.video_segment_num):
                logger.info(f"🔄 start segment {segment_idx + 1}/{self.video_segment_num}")
                with ProfilingContext4DebugL1(
                    f"segment end2end {segment_idx + 1}/{self.video_segment_num}",
                    recorder_mode=GET_RECORDER_MODE(),
                    metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                    metrics_labels=["DefaultRunner"],
                ):
                    self.check_stop()
                    # 1. default do nothing
                    self.init_run_segment(segment_idx)
                    # 2. main inference loop
                    latents = self.run_segment(segment_idx=segment_idx)
                    # 3. vae decoder
                    self.gen_video = self.run_vae_decoder(latents)
                    # 4. default do nothing
                    self.end_run_segment(segment_idx)

                # 5. stop or not
                if self.config["streaming"]:
                    stop = input("Press `n` to stop generation: ").strip().lower()
                    if stop == "n":
                        break
            stop = "n"

        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final

    @ProfilingContext4DebugL2("Run DiT")
    def run_main_live(self, total_steps=None):
        try:
            self.init_video_recorder()
            logger.info(f"init video_recorder: {self.video_recorder}")
            rank, world_size = self.get_rank_and_world_size()
            if rank == world_size - 1:
                assert self.video_recorder is not None, "video_recorder is required for stream audio input for rank 2"
                self.video_recorder.start(self.width, self.height)
            if world_size > 1:
                dist.barrier()
            self.init_run()
            if self.config.get("compile", False):
                self.model.select_graph_for_compile(self.input_info)

            stop = ""
            while stop != "n":
                for segment_idx in range(self.video_segment_num):
                    logger.info(f"🔄 start segment {segment_idx + 1}/{self.video_segment_num}")
                    with ProfilingContext4DebugL1(
                        f"segment end2end {segment_idx + 1}/{self.video_segment_num}",
                        recorder_mode=GET_RECORDER_MODE(),
                        metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                        metrics_labels=["DefaultRunner"],
                    ):
                        self.check_stop()
                        # 1. default do nothing
                        self.init_run_segment(segment_idx)
                        # 2. main inference loop
                        latents = self.run_segment(segment_idx=segment_idx)
                        # 3. vae decoder
                        self.gen_video = self.run_vae_decoder(latents)
                        # 4. default do nothing
                        self.end_run_segment(segment_idx)

                    # 5. stop or not
                    if self.config["streaming"]:
                        stop = input("Press `n` to stop generation: ").strip().lower()
                        if stop == "n":
                            break
                stop = "n"
        finally:
            if hasattr(self.model, "inputs"):
                self.end_run()
            if self.video_recorder:
                self.video_recorder.stop()
                self.video_recorder = None
