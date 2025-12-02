from abc import ABC

import torch
import torch.distributed as dist

from lightx2v_platform.base.global_var import AI_DEVICE


class BaseRunner(ABC):
    """Abstract base class for all Runners

    Defines interface methods that all subclasses must implement
    """

    def __init__(self, config):
        self.config = config
        self.vae_encoder_need_img_original = False
        self.input_info = None

    def load_transformer(self):
        """Load transformer model

        Returns:
            Loaded transformer model instance
        """
        pass

    def load_text_encoder(self):
        """Load text encoder

        Returns:
            Text encoder instance or list of text encoder instances
        """
        pass

    def load_image_encoder(self):
        """Load image encoder

        Returns:
            Image encoder instance or None if not needed
        """
        pass

    def load_vae(self):
        """Load VAE encoder and decoder

        Returns:
            Tuple[vae_encoder, vae_decoder]: VAE encoder and decoder instances
        """
        pass

    def run_image_encoder(self, img):
        """Run image encoder

        Args:
            img: Input image

        Returns:
            Image encoding result
        """
        pass

    def run_vae_encoder(self, img):
        """Run VAE encoder

        Args:
            img: Input image

        Returns:
            Tuple of VAE encoding result and additional parameters
        """
        pass

    def run_text_encoder(self, prompt, img):
        """Run text encoder

        Args:
            prompt: Input text prompt
            img: Optional input image (for some models)

        Returns:
            Text encoding result
        """
        pass

    def get_encoder_output_i2v(self, clip_encoder_out, vae_encoder_out, text_encoder_output, img):
        """Combine encoder outputs for i2v task

        Args:
            clip_encoder_out: CLIP encoder output
            vae_encoder_out: VAE encoder output
            text_encoder_output: Text encoder output
            img: Original image

        Returns:
            Combined encoder output dictionary
        """
        pass

    def init_scheduler(self):
        """Initialize scheduler"""
        pass

    def load_vae_decoder(self):
        """Load VAE decoder

        Default implementation: get decoder from load_vae method
        Subclasses can override this method to provide different loading logic

        Returns:
            VAE decoder instance
        """
        if not hasattr(self, "vae_decoder") or self.vae_decoder is None:
            _, self.vae_decoder = self.load_vae()
        return self.vae_decoder

    def get_video_segment_num(self):
        self.video_segment_num = 1

    def init_run(self):
        pass

    def init_run_segment(self, segment_idx):
        self.segment_idx = segment_idx

    def run_segment(self, segment_idx=0):
        pass

    def end_run_segment(self, segment_idx=None):
        self.gen_video_final = self.gen_video

    def end_run(self):
        pass

    def check_stop(self):
        """Check if the stop signal is received"""

        rank, world_size = 0, 1
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        signal_rank = world_size - 1

        stopped = 0
        if rank == signal_rank and hasattr(self, "stop_signal") and self.stop_signal:
            stopped = 1

        if world_size > 1:
            if rank == signal_rank:
                t = torch.tensor([stopped], dtype=torch.int32).to(device=AI_DEVICE)
            else:
                t = torch.zeros(1, dtype=torch.int32, device=AI_DEVICE)
            dist.broadcast(t, src=signal_rank)
            stopped = t.item()

        if stopped == 1:
            raise Exception(f"find rank: {rank} stop_signal, stop running, it's an expected behavior")
