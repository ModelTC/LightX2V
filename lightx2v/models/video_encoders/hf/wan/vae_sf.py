import torch

from lightx2v.models.video_encoders.hf.wan.vae import _video_vae
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
