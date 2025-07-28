import torch
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
import torch.distributed as dist
import torch.nn.functional as F
from lightx2v.models.networks.wan.infer.utils import compute_freqs_dist, compute_freqs_audio_dist


class WanTransformerDistInfer(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, audio_dit_blocks=None):
        x = self.dist_pre_process(x)
        x = super().infer(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, audio_dit_blocks)
        x = self.dist_post_process(x)
        return x

    def compute_freqs(self, q, grid_sizes, freqs):
        if "audio" in self.config.get("model_cls", ""):
            freqs_i = compute_freqs_audio_dist(q.size(0), q.size(2) // 2, grid_sizes, freqs)
        else:
            freqs_i = compute_freqs_dist(q.size(0), q.size(2) // 2, grid_sizes, freqs)
        return freqs_i

    def dist_pre_process(self, x):
        world_size = dist.get_world_size()
        cur_rank = dist.get_rank()

        padding_size = (world_size - (x.shape[0] % world_size)) % world_size

        if padding_size > 0:
            # 使用 F.pad 填充第一维
            x = F.pad(x, (0, 0, 0, padding_size))  # (后维度填充, 前维度填充)

        x = torch.chunk(x, world_size, dim=0)[cur_rank]
        return x

    def dist_post_process(self, x):
        # 获取当前进程的世界大小
        world_size = dist.get_world_size()

        # 创建一个列表，用于存储所有进程的输出
        gathered_x = [torch.empty_like(x) for _ in range(world_size)]

        # 收集所有进程的输出
        dist.all_gather(gathered_x, x)

        # 在指定的维度上合并所有进程的输出
        combined_output = torch.cat(gathered_x, dim=0)

        return combined_output  # 返回合并后的输出
