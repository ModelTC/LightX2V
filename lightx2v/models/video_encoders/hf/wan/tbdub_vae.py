"""TBDub's released-pipeline conventions on the native Wan2.2 codec."""

import torch

from lightx2v.models.video_encoders.hf.wan.matrix_game4_vae import set_matrix_game4_vae_norm
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import CausalConv3d, Wan2_2_VAE, get_wan22_vae_scale


class _ChunkLocalEncoderHead(torch.nn.Module):
    """Keep the final encoder convolution local to each four-frame chunk.

    The released TBDub pipeline wraps its encoder head's CausalConv3d in a
    generic module. Its encoder tests the outer type, so this final convolution
    receives no temporal cache. Preserve that observable checkpoint convention
    without the upstream VRAM manager or changes to the shared Wan codec.
    """

    def __init__(self, convolution):
        super().__init__()
        self.convolution = convolution
        self.train(convolution.training)

    def forward(self, hidden_states):
        return self.convolution(hidden_states)


class TBDubVAE(Wan2_2_VAE):
    def __init__(self, *args, **kwargs):
        if kwargs.get("vae_type", "wan2.2") != "wan2.2":
            raise ValueError("TBDub requires the released full Wan2.2 VAE")
        super().__init__(*args, **kwargs)
        # Load the original checkpoint keys before adapting the modules.
        set_matrix_game4_vae_norm(self.model)
        head = self.model.encoder.head[-1]
        if not isinstance(head, CausalConv3d):
            raise TypeError("TBDub expects a causal convolution at the encoder head")
        self.model.encoder.head[-1] = _ChunkLocalEncoderHead(head)
        # Source computes reciprocal in FP32, then casts both scale tensors.
        mean, std, inv_std = get_wan22_vae_scale(torch.float32, self.device)
        self.mean, self.std, self.inv_std = (value.to(self.dtype) for value in (mean, std, inv_std))
        self.scale = [self.mean, self.inv_std]
