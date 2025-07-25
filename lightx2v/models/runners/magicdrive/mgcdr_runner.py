import os
import gc
import einops
import json
import copy
from copy import deepcopy
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from einops import repeat, rearrange
from torchvision.io import write_video
from typing import Optional, Tuple, List, Dict, Any
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.mgcdr.scheduler import MagicDriverScheduler
# from lightx2v.models.schedulers.mgcdr.datasets.carla import CARLAVariableDataset
from lightx2v.models.networks.mgcdr.model import MagicDriveModel
from magicdrivedit.models.vae.vae_cogvideox import VideoAutoencoderKLCogVideoX
from magicdrivedit.models.text_encoder.t5 import T5Encoder
from lightx2v.utils.profiler import ProfilingContext4Debug
from magicdrivedit.datasets import save_sample
from magicdrivedit.utils.inference_utils import apply_mask_strategy, edit_pos, concat_n_views_pt
from magicdrivedit.utils.misc import collate_bboxes_to_maxlen, move_to
from magicdrivedit.datasets.carla import CARLAVariableDataset
from magicdrivedit.schedulers.rf.rectified_flow import timestep_transform
from loguru import logger
from lightx2v.models.schedulers.mgcdr.feature_caching.scheduler import (
    MagicDriverSchedulerTeaCaching,
)


@RUNNER_REGISTER("mgcdr")
class MagicDriverRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_class = CARLAVariableDataset
        self.dtype = torch.bfloat16
        self.save_fps = self.config.get("save_video_fps", 8)
        self.num_sampling_steps = self.config.get("infer_steps", 1)
        self.num_timesteps = self.config.get("num_timesteps", 1000)
        self.guidance_scale = self.config.get("guidance_scale",  2.0)
    
    def get_encoder_output_i2v(self):
        pass
    
    def load_image_encoder(self):
        pass
    
    def run_image_encoder(self):
        pass
        
    def init_modules(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()
    
    def load_transformer(self):
        model = MagicDriveModel(config=self.config, device=self.init_device)
        return model
    
    def load_text_encoder(self):
        t5_path = self.config.get('t5_path')
        model_max_length = self.config.get('t5_max_length', 300)
        t5_config = {
            'from_pretrained': t5_path,
            'model_max_length': model_max_length,
            'shardformer': True
        }
        text_encoder = T5Encoder(
            **t5_config
        )
        text_encoders = [text_encoder]
        return text_encoders
    
    def load_vae(self):
        # import pdb; pdb.set_trace()
        vae_path = self.config.get('vae_path')
        vae_micro_frame_size = self.config.get('micro_frame_size', 32)
        vae_micro_batch_size = self.config.get('micro_batch_size', 1)
        vae_config = {
            'from_pretrained': vae_path, 
            'subfolder': 'vae', 
            'micro_frame_size': vae_micro_frame_size, 
            'micro_batch_size': vae_micro_batch_size
        }
        # vae_config.update(additional_config)
        vae = VideoAutoencoderKLCogVideoX(
            **vae_config
        )
        return vae.to('cuda', torch.bfloat16).eval()
    
    def run_text_encoder(self, text, neg_text):
        self.text_encoders[0].t5.model.to('cuda')
        # import pdb; pdb.set_trace()
        n = len(text)
        # text_encoder_output = {}
        # import pdb; pdb.set_trace()
        text_encoder_output = self.text_encoders[0].encode(text)
        # text_encoder_output['x_mask'] = text_encoder_output.pop('mask')
        # text_encoder_output["y"] = y
        if neg_text is None:
            text_encoder_output["y_null"] = self.model.pre_weight.y_embedder_y_embedding.tensor[None].repeat(n, 1, 1)[:, None].to(self.init_device)
        else:
            text_encoder_output["y_null"] = self.text_encoders[0].encode(neg_text)
        # text_encoder_output["mask"] = mask
        self.text_encoders[0].t5.model.to('cpu')
        return text_encoder_output
    
    def init_scheduler(self):
        if self.config.feature_caching == "NoCaching":
            scheduler = MagicDriverScheduler(self.config)
        elif self.config.feature_caching == "Tea":
            scheduler = MagicDriverSchedulerTeaCaching(self.config)
        self.model.set_scheduler(scheduler)

    def run_vae_encoder(self, img):
        pass
    
    def set_target_shape(self):
        pass
    
    def save_video_func(self, samples):
        sample = samples[0]
        video_clips = []
        vid_samples = []
        pose_vis_i = rearrange(self.pose_vis[0], "T H W C -> T C H W")
        if pose_vis_i.shape[-2:] != sample.shape[-2:]: # go
            pose_vis_i = F.interpolate(pose_vis_i, size=sample.shape[-2:])
        pose_vis_i = rearrange(pose_vis_i, "T C H W -> 1 C T H W")
        sample = torch.cat([sample, pose_vis_i])
        vid_samples.append(concat_n_views_pt(sample, oneline=False))
        samples = torch.stack(vid_samples, dim=0)
        video_clips.append(samples)
        del vid_samples
        del samples
        
        # import pdb; pdb.set_trace()
        video = [video_clips[0][0]]
        video = torch.cat(video, dim=1)
        
        save_sample(
            video,
            fps=self.save_fps, # 8
            save_path=self.config.get('save_video_path'),
            high_quality=True,
            verbose=False,
            save_per_n_frame=self.config.get("save_per_n_frame", -1), # -1
            force_image=self.config.get("force_image", False), # False
        )
        
    def replace_with_null_condition(self, _model_args, uncond_cam, uncond_rel_pos,
                                uncond_y, keys, append=False):
        unchanged_keys = ["mv_order_map", "t_order_map", "height", "width", "num_frames", "fps"]
        handled_keys = []
        model_args = {}
        if "y" in keys and "y" in _model_args:
            handled_keys.append("y")
            if append:
                model_args["y"] = torch.cat([_model_args["y"], uncond_y], 0)
            else:
                model_args['y'] = uncond_y
            keys.remove("y")

        if "bbox" in keys and "bbox" in _model_args:
            handled_keys.append("bbox")
            _bbox = _model_args['bbox']
            bbox = {}
            for k in _bbox.keys():
                null_item = torch.zeros_like(_bbox[k])
                if append:
                    bbox[k] = torch.cat([_bbox[k], null_item], dim=0)
                else:
                    bbox[k] = null_item
            model_args['bbox'] = bbox
            keys.remove("bbox")

        if "cams" in keys and "cams" in _model_args:
            handled_keys.append("cams")
            cams = _model_args['cams']  # BxNC, T, 1, 3, 7
            null_cams = torch.zeros_like(cams)
            BNC, T, L = null_cams.shape[:3]
            null_cams = null_cams.reshape(-1, 3, 7)
            null_cams[:] = uncond_cam[None]
            null_cams = null_cams.reshape(BNC, T, L, 3, 7)
            if append:
                model_args['cams'] = torch.cat([cams, null_cams], dim=0)
            else:
                model_args['cams'] = null_cams
            keys.remove("cams")

        if "rel_pos" in keys and "rel_pos" in _model_args:
            handled_keys.append("rel_pos")
            rel_pos = _model_args['rel_pos'][..., :-1, :]  # BxNC, T, 1, 4, 4
            null_rel_pos = torch.zeros_like(rel_pos)
            BNC, T, L = null_rel_pos.shape[:3]
            null_rel_pos = null_rel_pos.reshape(-1, 3, 4)
            null_rel_pos[:] = uncond_rel_pos[None]
            null_rel_pos = null_rel_pos.reshape(BNC, T, L, 3, 4)
            if append:
                model_args['rel_pos'] = torch.cat([rel_pos, null_rel_pos], dim=0)
            else:
                model_args['rel_pos'] = null_rel_pos
            keys.remove("rel_pos")

        if "maps" in keys and "maps" in _model_args:
            handled_keys.append("maps")
            maps = _model_args["maps"]
            null_maps = torch.zeros_like(maps)
            if append:
                model_args['maps'] = torch.cat([maps, null_maps], dim=0)
            else:
                model_args['maps'] = null_maps
            keys.remove("maps")

        if len(keys) > 0:
            raise RuntimeError(f"{keys} left unhandled with {_model_args.keys()}")
        for k in _model_args.keys():
            if k in handled_keys:
                continue
            elif k in unchanged_keys:
                model_args[k] = _model_args[k]
            elif k == "bbox":
                _bbox = _model_args['bbox']
                bbox = {}
                for kb in _bbox.keys():
                    bbox[kb] = repeat(_bbox[kb], "b ... -> (2 b) ...")
                model_args['bbox'] = bbox
            else:
                if append:
                    model_args[k] = repeat(_model_args[k], "b ... -> (2 b) ...")
                else:
                    model_args[k] = deepcopy(_model_args[k])
        return model_args

    def get_additional_inputs(self):
        # import pdb; pdb.set_trace()
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]     
        dataset_params_json = self.config.get("dataset_params_json")
        with open(dataset_params_json, 'r') as file:
            dataset_args_dict = json.load(file)
        camera_params_json = self.config.get("camera_params_json")
        raw_meta_files = [self.config.get("raw_meta_files")]
        scene_description_file = self.config.get("scene_description_file")
        dataset_args_dict.update(
            {
                "pap_cam_init_path": camera_params_json,
                "raw_meta_files": raw_meta_files,
                "scene_description_file": scene_description_file
            }
        )
        self.dataset = self.dataset_class(**dataset_args_dict)
        batch = self.dataset['0-65-10']
        batch["pixel_values"] = batch["pixel_values"].unsqueeze(0)
        B, T, NC = batch["pixel_values"].shape[:3]
        self.B, self.T, self.NC = B, T, NC
        latent_size = self.vae.get_latent_size((T, *batch["pixel_values"].shape[-2:]))
        x = batch.pop("pixel_values").to(self.init_device, self.dtype)
        x = rearrange(x, "B T NC C ... -> (B NC) C T ...")
        y = [batch.pop("captions")[0]]
        maps = batch.pop("bev_map_with_aux").to(self.init_device, self.dtype).unsqueeze(0)
        bbox = [batch.pop("bboxes_3d_data")]
        bbox = [bbox_i.data for bbox_i in bbox]
        bbox = collate_bboxes_to_maxlen(bbox, self.init_device, self.dtype, NC, T)
        cams = torch.tensor(batch.pop("camera_param")).to(self.init_device, self.dtype).unsqueeze(0)
        cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # BxNC, T, 1, 3, 7
        rel_pos = torch.tensor(batch.pop("frame_emb")).to(self.init_device, self.dtype).unsqueeze(0)
        trans_scale = self.config.get("trans_scale", 1)
        rel_pos, self.pose_vis = edit_pos(rel_pos, self.config.get("traj", None), trans_scale,
            edit_param1=self.config.get("traj_param1", None),
            edit_param2=self.config.get("traj_param2", None),
            edit_param3=self.config.get("traj_param3", None),
        ) # time-consuming pre op
        rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)
        condition_frame_length = self.config.get("condition_frame_length", 0)
        prompts = y
        neg_prompts = None
        refs = [""] * len(y)
        ms = [""] * len(y)
        
        model_args = {}
        model_args["maps"] = maps
        model_args["bbox"] = bbox
        model_args["cams"] = cams
        model_args["rel_pos"] = rel_pos
        model_args["fps"] = torch.tensor([batch.pop('fps')])
        model_args['drop_cond_mask'] = torch.ones((B))  # camera
        model_args['drop_frame_mask'] = torch.ones((B, T))  # box & rel_pos
        model_args["height"] = torch.tensor([batch.pop("height")])
        model_args["width"] = torch.tensor([batch.pop("width")])
        model_args["num_frames"] = torch.tensor([batch.pop("num_frames")])
        model_args = move_to(model_args, device=self.init_device, dtype=self.dtype)
        # no need to move these
        model_args["mv_order_map"] = self.config.get(
            "mv_order_map", 
            {
                0: [5, 1, 6],
                1: [0, 2, 6],
                2: [1, 3],
                3: [2, 4],
                4: [3, 5],
                5: [4, 0, 6],
                6: [5, 1, 0]
            }
        )
        model_args["t_order_map"] = self.config.get("t_order_map", None)
        # import pdb; pdb.set_trace()
        bbox = self.add_box_latent(bbox, B, NC, T)
        new_bbox = {}
        for k, v in bbox.items():
            new_bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")  # BxNC, T, len, 3, 7
        model_args["bbox"] = move_to(new_bbox, device=self.init_device, dtype=self.dtype)
        z = torch.randn(len(prompts), self.config.get("in_channels", 16) * NC, *latent_size, device=self.init_device, dtype=self.dtype)
        mask = apply_mask_strategy(z, refs, ms, 0, align=None)
        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)
        # mask_t = mask * self.num_timesteps
        # x0 = z.clone()
        # x_noise = self.add_noise(x0, torch.randn_like(x0), timesteps)
        # mask_t_upper = mask_t >= timesteps.unsqueeze(1)
        # mask_add_noise = mask_t_upper & ~noise_added
        # z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
        # noise_added = mask_t_upper
        model_args["x_mask"] = mask
        model_args["x"] = z
        # import pdb; pdb.set_trace()
        self.timesteps = [torch.tensor([t] * z.shape[0], device=self.init_device) for t in timesteps] 
        self.timesteps = [timestep_transform(t, model_args, num_timesteps=self.num_timesteps, cog_style=True) for t in self.timesteps]
        # self.timesteps = torch.tensor([(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)], device=self.init_device)
        
        return prompts, neg_prompts, model_args
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise
    
    def add_box_latent(self, bbox, B, NC, T, len_dim=3):
        # == add latent, NC and T share the same set of latents ==
        max_len = bbox['bboxes'].shape[len_dim]
        
        # _bbox_latent = sample_func(B * max_len)
        _bbox_latent = torch.randn(
            (B*max_len, self.config.get("hidden_size", 1152))
        ) 
        
        if _bbox_latent is not None:
            _bbox_latent = _bbox_latent.view(B, max_len, -1)
            # finally, add to bbox
            bbox['box_latent'] = einops.repeat(
                _bbox_latent, "B ... -> B T NC ...", NC=NC, T=T)
        return bbox
    
    def run_input_encoder(self):
        cond_inputs = {}
        prompts, neg_prompts, model_args = self.get_additional_inputs()
        cond_inputs.update(model_args)
        text_encoder_outputs = self.run_text_encoder(prompts, neg_prompts)
        # cond_y = text_encoder_outputs.pop('y')
        uncond_y = text_encoder_outputs.pop('y_null')
        cond_inputs.update(text_encoder_outputs)
        uncond_inputs = copy.deepcopy(cond_inputs)
        uncond_cam = self.model.pre_weight.camera_embedder_uncond_cam.tensor
        uncond_rel_pos = self.model.pre_weight.frame_embedder_uncond_cam.tensor
        # import pdb; pdb.set_trace()
        uncond_inputs = self.replace_with_null_condition(uncond_inputs, uncond_cam, uncond_rel_pos, uncond_y, keys=["y", "bbox", "cams", "rel_pos", "maps"], append=False)

        self.cond_inputs = cond_inputs
        self.uncond_inputs = uncond_inputs
        self.model_args = model_args
        
    def run_vae_decoer(self, samples):
        samples = rearrange(samples, "B (C NC) T ... -> (B NC) C T ...", NC=self.NC)
        num_frames = self.model_args["num_frames"]
        # del self.cond_inputs
        # del self.uncond_inputs
        # del self.model_args
        # del self.dataset
        torch.cuda.empty_cache()
        # self.vae.to('cuda')
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            samples = self.vae.decode(samples.to(torch.bfloat16), num_frames=num_frames)
        # self.vae.to('cpu')
        samples = rearrange(samples, "(B NC) C T ... -> B NC C T ...", NC=self.NC)
        # import pdb; pdb.set_trace()
        return samples
    
    def run(self):
        if self.cond_inputs['x_mask'] is not None:
            noise_added = torch.zeros_like(self.cond_inputs['x_mask'], dtype=torch.bool)
            noise_added = noise_added | (self.cond_inputs['x_mask'] == 1)
            
        for step_index in range(self.model.scheduler.infer_steps):
            logger.info(f"==> step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")

            with ProfilingContext4Debug("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)
            # import pdb; pdb.set_trace()
            if self.cond_inputs['x_mask'] is not None:
                mask_t = self.cond_inputs['x_mask'] * self.num_timesteps
                x0 = self.cond_inputs['x'].clone()
                x_noise = self.add_noise(x0, torch.randn_like(x0), self.timesteps[step_index])
                mask_t_upper = mask_t >= self.timesteps[step_index].unsqueeze(1)
                mask_add_noise = mask_t_upper & ~noise_added
                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                self.cond_inputs['x'] = z
                self.cond_inputs['timestep'] = self.timesteps[step_index]
                self.uncond_inputs['x'] = z
                self.uncond_inputs['timestep'] = self.timesteps[step_index]
                noise_added = mask_t_upper
            # import pdb; pdb.set_trace()
            with ProfilingContext4Debug("cond infer"):
                if self.cond_inputs['x_mask'] is not None:
                    self.cond_inputs['x_mask'] = mask_t_upper
                self.model.infer(self.cond_inputs)
                if self.config["feature_caching"] == "Tea":
                    self.model.scheduler.cnt += 1
                    if self.model.scheduler.cnt >= self.model.scheduler.num_steps:
                        self.model.scheduler.cnt = 0
            # import pdb; pdb.set_trace()
            with ProfilingContext4Debug("uncond infer"):
                if self.uncond_inputs['x_mask'] is not None:
                    self.uncond_inputs['x_mask'] = mask_t_upper
                self.model.infer(self.uncond_inputs)
                if self.config["feature_caching"] == "Tea":
                    self.model.scheduler.cnt += 1
                    if self.model.scheduler.cnt >= self.model.scheduler.num_steps:
                        self.model.scheduler.cnt = 0
            # import pdb; pdb.set_trace()
            v_pred = self.uncond_inputs['x'] + self.guidance_scale * (self.cond_inputs['x'] - self.uncond_inputs['x'])
            dt = self.timesteps[step_index] - self.timesteps[step_index + 1] if step_index < len(self.timesteps) - 1 else self.timesteps[step_index]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]
            # import pdb; pdb.set_trace()
            if self.cond_inputs['x_mask'] is not None and self.uncond_inputs['x_mask'] is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)
                # import pdb; pdb.set_trace()
                self.cond_inputs['x'] = z
                self.uncond_inputs['x'] = z
            with ProfilingContext4Debug("step_post"):
                self.model.scheduler.step_post()
        
        samples = self.cond_inputs['x'].clone().detach()
        return samples
    
    def run_dit(self):
        self.init_scheduler()
        # self.model.scheduler.prepare()
        samples = self.run()
        return samples
    
    async def run_pipeline(self):
        self.run_input_encoder()
        samples = self.run_dit()
        samples = self.run_vae_decoer(samples)
        self.save_video(samples)
        del samples
        torch.cuda.empty_cache()
        gc.collect()
