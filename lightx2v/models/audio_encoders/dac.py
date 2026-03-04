import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from lightx2v.models.networks.mova.dac_vae import DAC

class DACVAE(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, checkpoint_path, device, dtype):
        super().__init__()
        self.model = DAC.from_pretrained(checkpoint_path).to(device).to(dtype)
        self.model.eval()
        self._device = device
        self._dtype = dtype

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def encode(self, x):
        # x shape: [B, 1, T]
        x = self.model.preprocess(x, self.model.sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.model.encode(x)
        return z

    def decode(self, z):
        return self.model.decode(z)