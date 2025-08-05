import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.models.networks.wan.infer.utils import pad_freqs


class WanTransformerDistInfer(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.seq_p_group = self.config["device_mesh"].get_group(mesh_dim="seq_p")

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, audio_dit_blocks=None):
        x = self.dist_pre_process(x)
        x = super().infer(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, audio_dit_blocks)
        x = self.dist_post_process(x)
        return x

    def compute_freqs(self, q, grid_sizes, freqs):
        if "audio" in self.config.get("model_cls", ""):
            freqs_i = self.compute_freqs_audio_dist(q.size(0), q.size(2) // 2, grid_sizes, freqs)
        else:
            freqs_i = self.compute_freqs_dist(q.size(0), q.size(2) // 2, grid_sizes, freqs)
        return freqs_i

    def dist_pre_process(self, x):
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)

        padding_size = (world_size - (x.shape[0] % world_size)) % world_size

        if padding_size > 0:
            # 使用 F.pad 填充第一维
            x = F.pad(x, (0, 0, 0, padding_size))  # (后维度填充, 前维度填充)

        x = torch.chunk(x, world_size, dim=0)[cur_rank]
        return x

    def dist_post_process(self, x):
        world_size = dist.get_world_size(self.seq_p_group)

        # 创建一个列表，用于存储所有进程的输出
        gathered_x = [torch.empty_like(x) for _ in range(world_size)]

        # 收集所有进程的输出
        dist.all_gather(gathered_x, x, group=self.seq_p_group)

        # 在指定的维度上合并所有进程的输出
        combined_output = torch.cat(gathered_x, dim=0)

        return combined_output  # 返回合并后的输出

    def compute_freqs_dist(self, s, c, grid_sizes, freqs):
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        f, h, w = grid_sizes[0].tolist()
        seq_len = f * h * w
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        freqs_i = pad_freqs(freqs_i, s * world_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(cur_rank * s_per_rank) : ((cur_rank + 1) * s_per_rank), :, :]
        return freqs_i_rank

    def compute_freqs_audio_dist(self, s, c, grid_sizes, freqs):
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        f, h, w = grid_sizes[0].tolist()
        f = f + 1
        seq_len = f * h * w
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        freqs_i = pad_freqs(freqs_i, s * world_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(cur_rank * s_per_rank) : ((cur_rank + 1) * s_per_rank), :, :]
        return freqs_i_rank
