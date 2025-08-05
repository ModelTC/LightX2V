import imageio
import numpy as np

from lightx2v.models.input_encoders.hf.t5_v1_1_xxl.model import T5EncoderModel_v1_1_xxl
from lightx2v.models.networks.cogvideox.model import CogvideoxModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.cogvideox.scheduler import CogvideoxXDPMScheduler
from lightx2v.models.video_encoders.hf.cogvideox.model import CogvideoxVAE
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("cogvideox")
class CogvideoxRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_transformer(self):
        model = CogvideoxModel(self.config)
        return model

    def load_image_encoder(self):
        return None

    def load_text_encoder(self):
        text_encoder = T5EncoderModel_v1_1_xxl(self.config)
        text_encoders = [text_encoder]
        return text_encoders

    def load_vae(self):
        vae_model = CogvideoxVAE(self.config)
        return vae_model, vae_model

    def init_scheduler(self):
        scheduler = CogvideoxXDPMScheduler(self.config)
        self.model.set_scheduler(scheduler)

    def run_text_encoder(self, text, img):
        text_encoder_output = {}
        n_prompt = self.config.get("negative_prompt", "")
        context = self.text_encoders[0].infer([text], self.config)
        context_null = self.text_encoders[0].infer([n_prompt if n_prompt else ""], self.config)
        text_encoder_output["context"] = context
        text_encoder_output["context_null"] = context_null
        return text_encoder_output

    def run_vae_encoder(self, img):
        # TODO: implement vae encoder for Cogvideox
        raise NotImplementedError("I2V inference is not implemented for Cogvideox.")

    def get_encoder_output_i2v(self, clip_encoder_out, vae_encoder_out, text_encoder_output, img):
        # TODO: Implement image encoder for Cogvideox-I2V
        raise ValueError(f"Unsupported model class: {self.config['model_cls']}")

    def set_target_shape(self):
        ret = {}
        if self.config.task == "i2v":
            # TODO: implement set_target_shape for Cogvideox-I2V
            raise NotImplementedError("I2V inference is not implemented for Cogvideox.")
        else:
            num_frames = self.config.target_video_length
            latent_frames = (num_frames - 1) // self.config.vae_scale_factor_temporal + 1
            additional_frames = 0
            patch_size_t = self.config.patch_size_t
            if patch_size_t is not None and latent_frames % patch_size_t != 0:
                additional_frames = patch_size_t - latent_frames % patch_size_t
                num_frames += additional_frames * self.config.vae_scale_factor_temporal
            self.config.target_shape = (
                self.config.batch_size,
                (num_frames - 1) // self.config.vae_scale_factor_temporal + 1,
                self.config.latent_channels,
                self.config.height // self.config.vae_scale_factor_spatial,
                self.config.width // self.config.vae_scale_factor_spatial,
            )
            ret["target_shape"] = self.config.target_shape
        return ret

    def save_video_func(self, images):
        with imageio.get_writer(self.config.save_video_path, fps=16) as writer:
            for pil_image in images:
                frame_np = np.array(pil_image, dtype=np.uint8)
                writer.append_data(frame_np)
