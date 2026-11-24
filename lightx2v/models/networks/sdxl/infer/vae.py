import torch.nn.functional as F

from lightx2v.models.networks.native_components.ops import Attention

from .unet import group_norm, resnet


class SDXLVAEInfer:
    def __init__(self):
        self.attention = Attention()

    def infer(self, w, latents):
        # Official SDXL VAE needs FP32 to avoid overflow at 1024px.
        x = w.conv("post_quant_conv")(latents.float())
        x = w.conv("decoder.conv_in", padding=1)(x)
        x = resnet(w, "decoder.mid_block.resnets.0", x, eps=1e-6)
        p = "decoder.mid_block.attentions.0"
        b, c, h, width = x.shape
        y = group_norm(w, p + ".group_norm", x, 1e-6).flatten(2).transpose(1, 2)
        q, k, v = [w.linear(p + ".to_" + t)(y).unsqueeze(1) for t in ("q", "k", "v")]
        y = self.attention(q, k, v).squeeze(1)
        x = x + w.linear(p + ".to_out.0")(y).transpose(1, 2).reshape(b, c, h, width)
        x = resnet(w, "decoder.mid_block.resnets.1", x, eps=1e-6)
        for i in range(4):
            for j in range(3):
                x = resnet(w, f"decoder.up_blocks.{i}.resnets.{j}", x, eps=1e-6)
            if i < 3:
                x = w.conv(f"decoder.up_blocks.{i}.upsamplers.0.conv", padding=1)(F.interpolate(x, scale_factor=2, mode="nearest"))
        return w.conv("decoder.conv_out", padding=1)(F.silu(group_norm(w, "decoder.conv_norm_out", x, 1e-6)))
