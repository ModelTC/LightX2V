import math

import torch

from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.utils.envs import *


class WanAnimatePreInfer(WanPreInfer):
    def __init__(self, config):
        super().__init__(config)
        self.encode_bs = 8

    def set_animate_encoders(self, motion_encoder, face_encoder):
        self.motion_encoder = motion_encoder
        self.face_encoder = face_encoder

    @torch.no_grad()
    def after_patch_embedding(self, weights, x, pose_latents, face_pixel_values):
        pose_latents = weights.pose_patch_embedding.apply(pose_latents)
        x[:, :, 1:].add_(pose_latents)

        face_pixel_values_tmp = []
        for i in range(math.ceil(face_pixel_values.shape[0] / self.encode_bs)):
            face_pixel_values_tmp.append(self.motion_encoder.get_motion(face_pixel_values[i * self.encode_bs : (i + 1) * self.encode_bs]))

        motion_vec = torch.cat(face_pixel_values_tmp)
        motion_vec = self.face_encoder(motion_vec.unsqueeze(0).to(GET_DTYPE())).squeeze(0)
        pad_face = torch.zeros(1, motion_vec.shape[1], motion_vec.shape[2], dtype=motion_vec.dtype, device="cuda")
        motion_vec = torch.cat([pad_face, motion_vec], dim=0)
        return x, motion_vec
