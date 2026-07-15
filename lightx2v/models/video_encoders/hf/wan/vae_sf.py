import torch
import torch.nn as nn
from einops import rearrange, repeat

from lightx2v.models.video_encoders.hf.wan.vae import WanVAE_, _video_vae
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WanSFVAE:
    def __init__(
        self,
        z_dim=16,
        vae_path="cache/vae_step_411000.pth",
        dtype=torch.float,
        device="cuda",
        parallel=False,
        use_tiling=False,
        cpu_offload=False,
        use_2d_split=True,
        load_from_rank0=False,
        **kwargs,
    ):
        self.dtype = dtype
        self.device = device
        self.parallel = parallel
        self.use_tiling = use_tiling
        self.cpu_offload = cpu_offload
        self.use_2d_split = use_2d_split

        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(pretrained_path=vae_path, z_dim=z_dim, cpu_offload=cpu_offload, dtype=dtype, load_from_rank0=load_from_rank0).eval().requires_grad_(False).to(device).to(dtype)
        self.model.clear_cache()
        self.upsampling_factor = 8

    def to_cpu(self):
        self.model.encoder = self.model.encoder.to("cpu")
        self.model.decoder = self.model.decoder.to("cpu")
        self.model = self.model.to("cpu")
        self.mean = self.mean.cpu()
        self.inv_std = self.inv_std.cpu()
        self.scale = [self.mean, self.inv_std]

    def to_cuda(self):
        self.model.encoder = self.model.encoder.to(AI_DEVICE)
        self.model.decoder = self.model.decoder.to(AI_DEVICE)
        self.model = self.model.to(AI_DEVICE)
        self.mean = self.mean.to(AI_DEVICE)
        self.inv_std = self.inv_std.to(AI_DEVICE)
        self.scale = [self.mean, self.inv_std]

    def decode(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        latent = latent.transpose(0, 1).unsqueeze(0)
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4).squeeze(0)
        return output

    def tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = (T + 3) // 4
        weight = torch.zeros((1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)
        values = torch.zeros((1, 16, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)
        for h, h_, w, w_ in tasks:  # tqdm(tasks, desc="VAE encoding"):
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch, is_bound=(h == 0, h_ >= H, w == 0, w_ >= W), border_width=((size_h - stride_h) // self.upsampling_factor, (size_w - stride_w) // self.upsampling_factor)
            ).to(dtype=video.dtype, device=data_device)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values

    def single_encode(self, video, device):
        video = video.to(device)
        x = self.model.encode(video, self.scale)
        return x

    def encode(self, videos, device, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        videos = [video.to("cpu") for video in videos]
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size = (tile_size[0] * 8, tile_size[1] * 8)
                tile_stride = (tile_stride[0] * 8, tile_stride[1] * 8)
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states


class WanMtxg2VAE(nn.Module):
    def __init__(self, pretrained_path=None, z_dim=16):
        super().__init__()
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]
        # init model
        self.model = (
            WanVAE_(
                dim=96,
                z_dim=z_dim,
                num_res_blocks=2,
                dim_mult=[1, 2, 4, 4],
                temperal_downsample=[False, True, True],
                dropout=0.0,
                pruning_rate=0.0,
            )
            .eval()
            .requires_grad_(False)
        )
        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path, map_location="cpu"), assign=True)
        self.upsampling_factor = 8

    def to(self, *args, **kwargs):
        self.mean = self.mean.to(*args, **kwargs)
        self.std = self.std.to(*args, **kwargs)
        self.scale = [self.mean, 1.0 / self.std]
        self.model = self.model.to(*args, **kwargs)
        return self

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    def tiled_decode(self, hidden_states, device, tile_size, tile_stride):
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"  # TODO
        computation_device = device

        out_T = T * 4 - 3
        weight = torch.zeros((1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)
        values = torch.zeros((1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)

        for h, h_, w, w_ in tasks:  # tqdm(tasks, desc="VAE decoding"):
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.decode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch, is_bound=(h == 0, h_ >= H, w == 0, w_ >= W), border_width=((size_h - stride_h) * self.upsampling_factor, (size_w - stride_w) * self.upsampling_factor)
            ).to(dtype=hidden_states.dtype, device=data_device)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        values = values.clamp_(-1, 1)
        return values

    def tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = (T + 3) // 4
        weight = torch.zeros((1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)
        values = torch.zeros((1, 16, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)

        for h, h_, w, w_ in tasks:  # tqdm(tasks, desc="VAE encoding"):
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch, is_bound=(h == 0, h_ >= H, w == 0, w_ >= W), border_width=((size_h - stride_h) // self.upsampling_factor, (size_w - stride_w) // self.upsampling_factor)
            ).to(dtype=video.dtype, device=data_device)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values

    def single_encode(self, video, device):
        video = video.to(device)
        x = self.model.encode(video, self.scale)
        return x

    def single_decode(self, hidden_state, device):
        hidden_state = hidden_state.to(device)
        video = self.model.decode(hidden_state, self.scale)
        return video.clamp_(-1, 1)

    def encode(self, videos, device, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        # videos: torch.Size([1, 3, 597, 352, 640]), device='cuda:0', dtype=torch.bfloat16
        videos = [video.to("cpu") for video in videos]
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)  # torch.Size([1, 3, 597, 352, 640])  torch.bfloat16  device(type='cpu')
            if tiled:  # True
                tile_size = (tile_size[0] * 8, tile_size[1] * 8)
                tile_stride = (tile_stride[0] * 8, tile_stride[1] * 8)
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states

    def decode(self, hidden_states, device, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        hidden_states = [hidden_state.to("cpu") for hidden_state in hidden_states]
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if tiled:
                video = self.tiled_decode(hidden_state, device, tile_size, tile_stride)
            else:
                video = self.single_decode(hidden_state, device)
            video = video.squeeze(0)
            videos.append(video)
        videos = torch.stack(videos)
        return videos
