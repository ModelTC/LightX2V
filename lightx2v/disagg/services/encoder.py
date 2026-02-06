import math
import torch
import numpy as np
from typing import Dict, Any, Optional, List

from lightx2v.disagg.services.base import BaseService
from lightx2v.disagg.conn import DataArgs, DataManager, DataSender, DisaggregationMode, DataPoll
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import seed_all
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v.disagg.utils import (
    load_wan_text_encoder,
    load_wan_image_encoder,
    load_wan_vae_encoder,
    read_image_input,
)

class EncoderService(BaseService):
    def __init__(self, config):
        super().__init__(config)
        self.text_encoder = None
        self.image_encoder = None
        self.vae_encoder = None
        self.engine_rank = 0
        self.data_mgr = None
        self.data_sender = None
        self._rdma_buffers: List[torch.Tensor] = []
        
        # Load models based on config
        self.load_models()
        
        # Seed everything if seed is in config
        if "seed" in self.config:
            seed_all(self.config["seed"])

        data_bootstrap_addr = self.config.get("data_bootstrap_addr", "127.0.0.1")
        data_bootstrap_room = self.config.get("data_bootstrap_room", 0)
        
        if data_bootstrap_addr is not None and data_bootstrap_room is not None:
            data_ptrs, data_lens, data_item_lens = self.alloc_bufs()
            data_args = DataArgs(
                engine_rank=self.engine_rank,
                data_ptrs=data_ptrs,
                data_lens=data_lens,
                data_item_lens=data_item_lens,
                ib_device=self.config.get("ib_device", ""),
            )
            self.data_mgr = DataManager(data_args, DisaggregationMode.ENCODE)
            self.data_sender = DataSender(
                self.data_mgr, data_bootstrap_addr, int(data_bootstrap_room)
            )

    def load_models(self):
        self.logger.info("Loading Encoder Models...")
        
        # T5 Text Encoder
        text_encoders = load_wan_text_encoder(self.config)
        self.text_encoder = text_encoders[0] if text_encoders else None
        
        # CLIP Image Encoder (Optional per usage in wan_i2v.py)
        if self.config.get("use_image_encoder", False):
           self.image_encoder = load_wan_image_encoder(self.config)
        
        # VAE Encoder (Required for I2V)
        # Note: wan_i2v.py logic: if vae_encoder is None: raise RuntimeError
        # But we only load if needed or always? Let's check the config flags.
        # It seems always loaded for I2V task, but might be offloaded.
        # For simplicity of this service, we load it if the task implies it or just try to load.
        # But `load_wan_vae_encoder` will look at the config.
        self.vae_encoder = load_wan_vae_encoder(self.config)
        
        self.logger.info("Encoder Models loaded successfully.")

    def _get_latent_shape_with_lat_hw(self, latent_h, latent_w):
        return [
            self.config.get("num_channels_latents", 16),
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            latent_h,
            latent_w,
        ]

    def _compute_latent_shape_from_image(self, image_tensor: torch.Tensor):
        h, w = image_tensor.shape[2:]
        aspect_ratio = h / w
        max_area = self.config["target_height"] * self.config["target_width"]

        latent_h = round(
            np.sqrt(max_area * aspect_ratio)
            // self.config["vae_stride"][1]
            // self.config["patch_size"][1]
            * self.config["patch_size"][1]
        )
        latent_w = round(
            np.sqrt(max_area / aspect_ratio)
            // self.config["vae_stride"][2]
            // self.config["patch_size"][2]
            * self.config["patch_size"][2]
        )
        latent_shape = self._get_latent_shape_with_lat_hw(latent_h, latent_w)
        return latent_shape, latent_h, latent_w

    def _get_vae_encoder_output(self, first_frame: torch.Tensor, latent_h: int, latent_w: int):
        h = latent_h * self.config["vae_stride"][1]
        w = latent_w * self.config["vae_stride"][2]

        msk = torch.ones(
            1,
            self.config["target_video_length"],
            latent_h,
            latent_w,
            device=torch.device(AI_DEVICE),
        )
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, latent_h, latent_w)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat(
            [
                torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                torch.zeros(3, self.config["target_video_length"] - 1, h, w),
            ],
            dim=1,
        ).to(AI_DEVICE)

        vae_encoder_out = self.vae_encoder.encode(vae_input.unsqueeze(0).to(GET_DTYPE()))
        vae_encoder_out = torch.concat([msk, vae_encoder_out]).to(GET_DTYPE())
        return vae_encoder_out

    def alloc_bufs(self):
        text_len = int(self.config.get("text_len", 512))
        enable_cfg = bool(self.config.get("enable_cfg", False))
        use_image_encoder = bool(self.config.get("use_image_encoder", True))
        task = self.config.get("task", "i2v")

        text_dim = int(self.config.get("text_encoder_dim", 4096))
        clip_dim = int(self.config.get("clip_embed_dim", 1024))
        z_dim = int(self.config.get("vae_z_dim", 16))

        vae_stride = self.config.get("vae_stride", (4, 8, 8))
        stride_t = int(vae_stride[0])
        stride_h = int(vae_stride[1])
        stride_w = int(vae_stride[2])

        target_video_length = int(self.config.get("target_video_length", 81))
        target_height = int(self.config.get("target_height", 480))
        target_width = int(self.config.get("target_width", 832))

        t_prime = 1 + (target_video_length - 1) // stride_t
        h_prime = int(math.ceil(target_height / stride_h))
        w_prime = int(math.ceil(target_width / stride_w))

        self._rdma_buffers = []
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        data_item_lens: List[int] = []

        def _alloc_buffer(shape, dtype):
            buf = torch.empty(shape, dtype=dtype, device=torch.device(f"cuda:{self.engine_rank}"))
            self._rdma_buffers.append(buf)
            nbytes = buf.numel() * buf.element_size()
            data_ptrs.append(buf.data_ptr())
            data_lens.append(nbytes)
            data_item_lens.append(nbytes)

        _alloc_buffer((1, text_len, text_dim), GET_DTYPE())
        if enable_cfg:
            _alloc_buffer((1, text_len, text_dim), GET_DTYPE())

        if task == "i2v":
            if use_image_encoder:
                _alloc_buffer((clip_dim,), GET_DTYPE())
            _alloc_buffer((z_dim + 4, t_prime, h_prime, w_prime), GET_DTYPE())

        _alloc_buffer((4,), torch.int64)
        return data_ptrs, data_lens, data_item_lens

    def process(self) -> Dict[str, Any]:
        """
        Generates encoder outputs from prompt and image input.
        """
        self.logger.info("Starting processing in EncoderService...")
        
        prompt = self.config.get("prompt")
        negative_prompt = self.config.get("negative_prompt")
        if prompt is None:
            raise ValueError("prompt is required in config.")

        # 1. Text Encoding
        text_len = self.config.get("text_len", 512)
        
        context = self.text_encoder.infer([prompt])
        context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])

        if self.config.get("enable_cfg", False):
            if negative_prompt is None:
                raise ValueError("negative_prompt is required in config when enable_cfg is True.")
            context_null = self.text_encoder.infer([negative_prompt])
            context_null = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context_null])
        else:
            context_null = None

        text_encoder_output = {
            "context": context,
            "context_null": context_null,
        }
        
        task = self.config.get("task")

        if task == "t2v":
            latent_h = self.config["target_height"] // self.config["vae_stride"][1]
            latent_w = self.config["target_width"] // self.config["vae_stride"][2]
            latent_shape = [
                self.config.get("num_channels_latents", 16),
                (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
                latent_h,
                latent_w,
            ]
            image_encoder_output = None
        elif task == "i2v":
            image_path = self.config.get("image_path")
            if image_path is None:
                raise ValueError("image_path is required for i2v task.")

            # 2. Image Encoding + VAE Encoding
            img, _ = read_image_input(image_path)

            clip_encoder_out = None
            if self.image_encoder is not None:
                # Assuming image_encoder.visual handles list of images
                clip_encoder_out = self.image_encoder.visual([img]).squeeze(0).to(GET_DTYPE())

            if self.vae_encoder is None:
                raise RuntimeError("VAE encoder is required but was not loaded.")

            latent_shape, latent_h, latent_w = self._compute_latent_shape_from_image(img)
            vae_encoder_out = self._get_vae_encoder_output(img, latent_h, latent_w)

            image_encoder_output = {
                "clip_encoder_out": clip_encoder_out,
                "vae_encoder_out": vae_encoder_out,
            }
        else:
            raise ValueError(f"Unsupported task: {task}")

        if self.data_mgr is not None and self.data_sender is not None:
            buffer_index = 0
            self._rdma_buffers[buffer_index].copy_(context)
            buffer_index += 1
            if self.config.get("enable_cfg", False):
                self._rdma_buffers[buffer_index].copy_(context_null)
                buffer_index += 1

            if task == "i2v":
                if self.config.get("use_image_encoder", True):
                    if image_encoder_output.get("clip_encoder_out") is not None:
                        self._rdma_buffers[buffer_index].copy_(image_encoder_output["clip_encoder_out"])
                    else:
                        self._rdma_buffers[buffer_index].zero_()
                    buffer_index += 1

                vae_buf = self._rdma_buffers[buffer_index]
                vae_buf.zero_()
                vae_flat = vae_buf.view(-1)
                src_flat = image_encoder_output["vae_encoder_out"].reshape(-1)
                vae_flat[: src_flat.numel()].copy_(src_flat)
                buffer_index += 1

            latent_tensor = torch.tensor(latent_shape, device=AI_DEVICE, dtype=torch.int64)
            self._rdma_buffers[buffer_index].copy_(latent_tensor)

            buffer_ptrs = [buf.data_ptr() for buf in self._rdma_buffers]
            self.data_sender.send(buffer_ptrs)

            import time
            while True:
                status = self.data_sender.poll()
                if status == DataPoll.Success:
                    break
                time.sleep(0.01)

    def release_memory(self):
        """
        Releases the RDMA buffers and clears GPU cache.
        """
        self._rdma_buffers = []
        torch.cuda.empty_cache()
