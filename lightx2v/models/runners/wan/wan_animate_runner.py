import gc
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

try:
    from decord import VideoReader
except ImportError:
    VideoReader = None
    logger.info("If you need run animate model, please install decord.")


from lightx2v.models.input_encoders.hf.animate.face_encoder import FaceEncoder
from lightx2v.models.input_encoders.hf.animate.motion_encoder import Generator
from lightx2v.models.networks.wan.animate_model import WanAnimateModel
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import load_weights, remove_substrings_from_keys


@RUNNER_REGISTER("wan2.2_animate")
class WanAnimateRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)
        assert self.config.task == "animate"

    def inputs_padding(self, array, target_len):
        idx = 0
        flip = False
        target_array = []
        while len(target_array) < target_len:
            target_array.append(deepcopy(array[idx]))
            if flip:
                idx -= 1
            else:
                idx += 1
            if idx == 0 or idx == len(array) - 1:
                flip = not flip
        return target_array[:target_len]

    def get_valid_len(self, real_len, clip_len=81, overlap=1):
        real_clip_len = clip_len - overlap
        last_clip_num = (real_len - overlap) % real_clip_len
        if last_clip_num == 0:
            extra = 0
        else:
            extra = real_clip_len - last_clip_num
        target_len = real_len + extra
        return target_len

    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, dtype=GET_DTYPE(), device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def padding_resize(
        self,
        img_ori,
        height=512,
        width=512,
        padding_color=(0, 0, 0),
        interpolation=cv2.INTER_LINEAR,
    ):
        ori_height = img_ori.shape[0]
        ori_width = img_ori.shape[1]
        channel = img_ori.shape[2]

        img_pad = np.zeros((height, width, channel))
        if channel == 1:
            img_pad[:, :, 0] = padding_color[0]
        else:
            img_pad[:, :, 0] = padding_color[0]
            img_pad[:, :, 1] = padding_color[1]
            img_pad[:, :, 2] = padding_color[2]

        if (ori_height / ori_width) > (height / width):
            new_width = int(height / ori_height * ori_width)
            img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
            padding = int((width - new_width) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            img_pad[:, padding : padding + new_width, :] = img
        else:
            new_height = int(width / ori_width * ori_height)
            img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
            padding = int((height - new_height) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            img_pad[padding : padding + new_height, :, :] = img

        img_pad = np.uint8(img_pad)

        return img_pad

    def prepare_source(self, src_pose_path, src_face_path, src_ref_path):
        pose_video_reader = VideoReader(src_pose_path)
        pose_len = len(pose_video_reader)
        pose_idxs = list(range(pose_len))
        cond_images = pose_video_reader.get_batch(pose_idxs).asnumpy()

        face_video_reader = VideoReader(src_face_path)
        face_len = len(face_video_reader)
        face_idxs = list(range(face_len))
        face_images = face_video_reader.get_batch(face_idxs).asnumpy()
        height, width = cond_images[0].shape[:2]
        refer_images = cv2.imread(src_ref_path)[..., ::-1]
        refer_images = self.padding_resize(refer_images, height=height, width=width)
        return cond_images, face_images, refer_images

    def prepare_source_for_replace(self, src_bg_path, src_mask_path):
        bg_video_reader = VideoReader(src_bg_path)
        bg_len = len(bg_video_reader)
        bg_idxs = list(range(bg_len))
        bg_images = bg_video_reader.get_batch(bg_idxs).asnumpy()

        mask_video_reader = VideoReader(src_mask_path)
        mask_len = len(mask_video_reader)
        mask_idxs = list(range(mask_len))
        mask_images = mask_video_reader.get_batch(mask_idxs).asnumpy()
        mask_images = mask_images[:, :, :, 0] / 255
        return bg_images, mask_images

    @ProfilingContext4DebugL2("Run Image Encoders")
    def run_image_encoders(
        self,
        conditioning_pixel_values,
        refer_t_pixel_values,
        bg_pixel_values,
        mask_pixel_values,
        face_pixel_values,
    ):
        clip_encoder_out = self.run_image_encoder(self.refer_pixel_values)
        vae_encoder_out, pose_latents = self.run_vae_encoder(
            conditioning_pixel_values,
            refer_t_pixel_values,
            bg_pixel_values,
            mask_pixel_values,
        )
        return {"image_encoder_output": {"clip_encoder_out": clip_encoder_out, "vae_encoder_out": vae_encoder_out, "pose_latents": pose_latents, "face_pixel_values": face_pixel_values}}

    def run_vae_encoder(
        self,
        conditioning_pixel_values,
        refer_t_pixel_values,
        bg_pixel_values,
        mask_pixel_values,
    ):
        H, W = self.refer_pixel_values.shape[-2], self.refer_pixel_values.shape[-1]
        pose_latents = self.vae_encoder.encode(conditioning_pixel_values.unsqueeze(0))  #  c t h w
        ref_latents = self.vae_encoder.encode(self.refer_pixel_values.unsqueeze(1).unsqueeze(0))  #  c t h w

        mask_ref = self.get_i2v_mask(1, self.config.lat_h, self.config.lat_w, 1)
        y_ref = torch.concat([mask_ref, ref_latents])

        if self.mask_reft_len > 0:
            if self.config.replace_flag:
                y_reft = self.vae_encoder.encode(
                    torch.concat(
                        [
                            refer_t_pixel_values.unsqueeze(2)[0, :, : self.mask_reft_len],
                            bg_pixel_values[:, self.mask_reft_len :],
                        ],
                        dim=1,
                    )
                    .cuda()
                    .unsqueeze(0)
                )
                mask_pixel_values = 1 - mask_pixel_values
                mask_pixel_values = mask_pixel_values.permute(1, 0, 2, 3)
                mask_pixel_values = F.interpolate(mask_pixel_values, size=(H // 8, W // 8), mode="nearest")
                mask_pixel_values = mask_pixel_values[:, 0, :, :]

                msk_reft = self.get_i2v_mask(
                    self.config.lat_t,
                    self.config.lat_h,
                    self.config.lat_w,
                    self.mask_reft_len,
                    mask_pixel_values=mask_pixel_values.unsqueeze(0),
                )
            else:
                y_reft = self.vae_encoder.encode(
                    torch.concat(
                        [
                            torch.nn.functional.interpolate(
                                refer_t_pixel_values.unsqueeze(2)[0, :, : self.mask_reft_len].cpu(),
                                size=(H, W),
                                mode="bicubic",
                            ),
                            torch.zeros(3, self.config.target_video_length - self.mask_reft_len, H, W, dtype=GET_DTYPE()),
                        ],
                        dim=1,
                    )
                    .cuda()
                    .unsqueeze(0)
                )
                msk_reft = self.get_i2v_mask(self.config.lat_t, self.config.lat_h, self.config.lat_w, self.mask_reft_len)
        else:
            if self.config.replace_flag:
                mask_pixel_values = 1 - mask_pixel_values
                mask_pixel_values = mask_pixel_values.permute(1, 0, 2, 3)
                mask_pixel_values = F.interpolate(mask_pixel_values, size=(H // 8, W // 8), mode="nearest")
                mask_pixel_values = mask_pixel_values[:, 0, :, :]
                y_reft = self.vae_encoder.encode(bg_pixel_values.unsqueeze(0))
                msk_reft = self.get_i2v_mask(
                    self.config.lat_t,
                    self.config.lat_h,
                    self.config.lat_w,
                    self.mask_reft_len,
                    mask_pixel_values=mask_pixel_values.unsqueeze(0),
                )
            else:
                y_reft = self.vae_encoder.encode(torch.zeros(1, 3, self.config.target_video_length - self.mask_reft_len, H, W, dtype=GET_DTYPE(), device="cuda"))
                msk_reft = self.get_i2v_mask(self.config.lat_t, self.config.lat_h, self.config.lat_w, self.mask_reft_len)

        y_reft = torch.concat([msk_reft, y_reft])
        y = torch.concat([y_ref, y_reft], dim=1)

        return y, pose_latents

    def prepare_input(self):
        src_pose_path = self.config.get("src_pose_path", None)
        src_face_path = self.config.get("src_face_path", None)
        src_ref_path = self.config.get("src_ref_images", None)
        self.cond_images, self.face_images, self.refer_images = self.prepare_source(src_pose_path, src_face_path, src_ref_path)
        self.refer_pixel_values = torch.tensor(self.refer_images / 127.5 - 1, dtype=GET_DTYPE(), device="cuda").permute(2, 0, 1)  # chw
        self.real_frame_len = len(self.cond_images)
        target_len = self.get_valid_len(
            self.real_frame_len,
            self.config.target_video_length,
            overlap=self.config.get("refert_num", 1),
        )
        logger.info("real frames: {} target frames: {}".format(self.real_frame_len, target_len))
        self.cond_images = self.inputs_padding(self.cond_images, target_len)
        self.face_images = self.inputs_padding(self.face_images, target_len)

        if self.config.get("replace_flag", False):
            src_bg_path = self.config.get("src_bg_path")
            src_mask_path = self.config.get("src_mask_path")
            self.bg_images, self.mask_images = self.prepare_source_for_replace(src_bg_path, src_mask_path)
            self.bg_images = self.inputs_padding(self.bg_images, target_len)
            self.mask_images = self.inputs_padding(self.mask_images, target_len)

    def get_video_segment_num(self):
        total_frames = len(self.cond_images)
        self.move_frames = self.config.target_video_length - self.config.refert_num
        if total_frames <= self.config.target_video_length:
            self.video_segment_num = 1
        else:
            self.video_segment_num = 1 + (total_frames - self.config.target_video_length + self.move_frames - 1) // self.move_frames

    def init_run(self):
        self.all_out_frames = []
        self.prepare_input()
        super().init_run()

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_decoder(self, latents):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_decoder = self.load_vae_decoder()
        images = self.vae_decoder.decode(latents[:, 1:].to(GET_DTYPE()))
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_decoder
            torch.cuda.empty_cache()
            gc.collect()
        return images

    def init_run_segment(self, segment_idx):
        start = segment_idx * self.move_frames
        end = start + self.config.target_video_length
        if start == 0:
            self.mask_reft_len = 0
        else:
            self.mask_reft_len = self.config.refert_num

        conditioning_pixel_values = torch.tensor(
            np.stack(self.cond_images[start:end]) / 127.5 - 1,
            device="cuda",
            dtype=GET_DTYPE(),
        ).permute(3, 0, 1, 2)  # c t h w

        face_pixel_values = torch.tensor(
            np.stack(self.face_images[start:end]) / 127.5 - 1,
            device="cuda",
            dtype=GET_DTYPE(),
        ).permute(0, 3, 1, 2)  # thwc->tchw

        if start == 0:
            height, width = self.refer_images.shape[:2]
            refer_t_pixel_values = torch.zeros(
                3,
                self.config.refert_num,
                height,
                width,
                device="cuda",
                dtype=GET_DTYPE(),
            )  # c t h w
        else:
            refer_t_pixel_values = self.gen_video[0, :, -self.config.refert_num :].transpose(0, 1).clone().detach()  # c t h w

        bg_pixel_values, mask_pixel_values = None, None
        if self.config.replace_flag:
            bg_pixel_values = torch.tensor(
                np.stack(self.bg_images[start:end]) / 127.5 - 1,
                device="cuda",
                dtype=GET_DTYPE(),
            ).permute(3, 0, 1, 2)  # c t h w,

            mask_pixel_values = torch.tensor(
                np.stack(self.mask_images[start:end])[:, :, :, None],
                device="cuda",
                dtype=GET_DTYPE(),
            ).permute(3, 0, 1, 2)  # c t h w,

        self.inputs.update(
            self.run_image_encoders(
                conditioning_pixel_values,
                refer_t_pixel_values,
                bg_pixel_values,
                mask_pixel_values,
                face_pixel_values,
            )
        )

        if start != 0:
            self.model.scheduler.reset(self.input_info.seed, self.input_info.latent_shape)

    def end_run_segment(self, segment_idx):
        if segment_idx != 0:
            self.gen_video = self.gen_video[:, :, self.config["refert_num"] :]
        self.all_out_frames.append(self.gen_video.cpu())

    def process_images_after_vae_decoder(self, save_video=True):
        self.gen_video = torch.cat(self.all_out_frames, dim=2)[:, :, : self.real_frame_len]
        del self.all_out_frames
        gc.collect()
        super().process_images_after_vae_decoder(save_video)

    def set_target_shape(self):
        self.config.target_video_length = self.config.target_video_length
        self.config.lat_h = self.refer_pixel_values.shape[-2] // 8
        self.config.lat_w = self.refer_pixel_values.shape[-1] // 8
        self.config.lat_t = self.config.target_video_length // 4 + 1
        self.config.target_shape = [16, self.config.lat_t + 1, self.config.lat_h, self.config.lat_w]

    def run_image_encoder(self, img):  # CHW
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.image_encoder = self.load_image_encoder()
        clip_encoder_out = self.image_encoder.visual([img.unsqueeze(0)]).squeeze(0).to(GET_DTYPE())
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.image_encoder
            torch.cuda.empty_cache()
            gc.collect()
        return clip_encoder_out

    def load_transformer(self):
        model = WanAnimateModel(
            self.config.model_path,
            self.config,
            self.init_device,
        )
        motion_encoder, face_encoder = self.load_encoders()
        model.set_animate_encoders(motion_encoder, face_encoder)
        return model

    def load_encoders(self):
        motion_encoder = Generator(size=512, style_dim=512, motion_dim=20).eval().requires_grad_(False).to(GET_DTYPE()).cuda()
        face_encoder = FaceEncoder(in_dim=512, hidden_dim=5120, num_heads=4).eval().requires_grad_(False).to(GET_DTYPE()).cuda()
        motion_weight_dict = remove_substrings_from_keys(load_weights(self.config["model_path"], include_keys=["motion_encoder"]), "motion_encoder.")
        face_weight_dict = remove_substrings_from_keys(load_weights(self.config["model_path"], include_keys=["face_encoder"]), "face_encoder.")
        motion_encoder.load_state_dict(motion_weight_dict)
        face_encoder.load_state_dict(face_weight_dict)
        return motion_encoder, face_encoder
