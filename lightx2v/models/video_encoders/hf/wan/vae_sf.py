import torch

from lightx2v.models.video_encoders.hf.wan.vae import _video_vae


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

        # init model
        self.model = _video_vae(pretrained_path=vae_path, z_dim=z_dim, cpu_offload=cpu_offload, dtype=dtype, load_from_rank0=load_from_rank0).eval().requires_grad_(False).to(device).to(dtype)
        self.model.clear_cache()

    def to_cpu(self):
        self.model.encoder = self.model.encoder.to("cpu")
        self.model.decoder = self.model.decoder.to("cpu")
        self.model = self.model.to("cpu")
        self.mean = self.mean.cpu()
        self.inv_std = self.inv_std.cpu()
        self.scale = [self.mean, self.inv_std]

    def to_cuda(self):
        self.model.encoder = self.model.encoder.to("cuda")
        self.model.decoder = self.model.decoder.to("cuda")
        self.model = self.model.to("cuda")
        self.mean = self.mean.cuda()
        self.inv_std = self.inv_std.cuda()
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
