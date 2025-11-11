import torch

from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.models.networks.wan.infer.self_forcing.pre_infer import WanSFPreInfer, sinusoidal_embedding_1d
from lightx2v.utils.envs import *


def cond_current(conditional_dict, current_start_frame, num_frame_per_block, replace=None, mode="universal"):
    new_cond = {}

    new_cond["cond_concat"] = conditional_dict["image_encoder_output"]["cond_concat"][:, :, current_start_frame : current_start_frame + num_frame_per_block]
    new_cond["visual_context"] = conditional_dict["image_encoder_output"]["visual_context"]
    if replace != None:
        if current_start_frame == 0:
            last_frame_num = 1 + 4 * (num_frame_per_block - 1)
        else:
            last_frame_num = 4 * num_frame_per_block
        final_frame = 1 + 4 * (current_start_frame + num_frame_per_block - 1)
        if mode != "templerun":
            conditional_dict["text_encoder_output"]["mouse_cond"][:, -last_frame_num + final_frame : final_frame] = replace["mouse"][None, None, :].repeat(1, last_frame_num, 1)
        conditional_dict["text_encoder_output"]["keyboard_cond"][:, -last_frame_num + final_frame : final_frame] = replace["keyboard"][None, None, :].repeat(1, last_frame_num, 1)
    if mode != "templerun":
        new_cond["mouse_cond"] = conditional_dict["text_encoder_output"]["mouse_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]
    new_cond["keyboard_cond"] = conditional_dict["text_encoder_output"]["keyboard_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]

    if replace != None:
        return new_cond, conditional_dict
    else:
        return new_cond


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


class WanMtxg2PreInfer(WanSFPreInfer):
    def __init__(self, config):
        super().__init__(config)
        d = config["dim"] // config["num_heads"]
        self.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1).to(torch.device("cuda"))
        self.dim = config["dim"]

    def img_emb(self, weights, x):
        x = weights.img_emb_0.apply(x)
        x = weights.img_emb_1.apply(x.squeeze(0))
        x = torch.nn.functional.gelu(x, approximate="none")
        x = weights.img_emb_3.apply(x)
        x = weights.img_emb_4.apply(x)
        x = x.unsqueeze(0)
        return x

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        x = self.scheduler.latents_input
        t = self.scheduler.timestep_input
        current_start_frame = self.scheduler.seg_index * self.scheduler.num_frame_per_block
        current_conditional_dict = cond_current(inputs, current_start_frame, self.scheduler.num_frame_per_block, mode=self.config["mode"])
        cond_concat = current_conditional_dict["cond_concat"]
        visual_context = current_conditional_dict["visual_context"]

        x = torch.cat([x.unsqueeze(0), cond_concat], dim=1)

        # embeddings
        x = weights.patch_embedding.apply(x)
        grid_sizes_t, grid_sizes_h, grid_sizes_w = torch.tensor(x.shape[2:], dtype=torch.long)
        grid_sizes = GridOutput(tensor=torch.tensor([[grid_sizes_t, grid_sizes_h, grid_sizes_w]], dtype=torch.int32, device=x.device), tuple=(grid_sizes_t, grid_sizes_h, grid_sizes_w))

        x = x.flatten(2).transpose(1, 2)  # B FHW C'
        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long, device=torch.device("cuda"))
        assert seq_lens[0] <= 15 * 1 * 880

        embed_tmp = sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)  # torch.Size([3, 256])
        embed = self.time_embedding(weights, embed_tmp)  # torch.Size([3, 1536])
        embed0 = self.time_projection(weights, embed).unflatten(dim=0, sizes=t.shape)

        # context
        context_lens = None
        context = self.img_emb(weights, visual_context)

        # data = torch.load("/data/nvme2/wushuo/Matrix-Game/Matrix-Game-2/pre_infer.pt")
        # breakpoint()
        # print((data['x']-x).sum())
        return WanPreInferModuleOutput(
            embed=embed,
            grid_sizes=grid_sizes,
            x=x.squeeze(0),
            embed0=embed0.squeeze(0),
            seq_lens=seq_lens,
            freqs=self.freqs,
            context=context[0],
            conditional_dict=current_conditional_dict,
        )
