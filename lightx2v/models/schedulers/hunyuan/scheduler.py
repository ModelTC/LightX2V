import torch
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
from typing import Union, Tuple, List
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from lightx2v.models.schedulers.scheduler import BaseScheduler


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Precompute the frequency tensor for complex exponential (cis) with given dimensions.
    (Note: `cis` means `cos + i * sin`, where i is the imaginary unit.)

    This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (int or torch.FloatTensor): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.
        theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.

    Returns:
        freqs_cis: Precomputed frequency tensor with complex exponential. [S, D/2]
        freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
    """
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [D/2]
    # assert interpolation_factor == 1.0, f"interpolation_factor: {interpolation_factor}"
    freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


def get_meshgrid_nd(start, *args, dim=2):
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (np.ndarray): [dim, ...]
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

    return grid


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
):
    """
    This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
            part and an imaginary part separately.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.

    Returns:
        pos_embed (torch.Tensor): [HW, D/2]
    """

    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))  # [3, W, H, D] / [2, W, H]

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(rope_dim_list), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(rope_dim_list), "len(interpolation_factor) should equal to len(rope_dim_list)"

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )  # 2 x [WHD, rope_dim_list[i]]
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
        sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)  # (WHD, D/2)
        return emb


def set_timesteps_sigmas(num_inference_steps, shift, device, num_train_timesteps=1000):
    sigmas = torch.linspace(1, 0, num_inference_steps + 1)
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32, device=device)
    return timesteps, sigmas


def get_1d_rotary_pos_embed_riflex(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # [S]

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device)[: (dim // 2)].float() / dim))  # [D/2]

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k - 1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===

    freqs = torch.outer(pos, freqs)  # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


class HunyuanScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.shift = 7.0
        self.timesteps, self.sigmas = set_timesteps_sigmas(self.infer_steps, self.shift, device=torch.device("cuda"))
        assert len(self.timesteps) == self.infer_steps
        self.embedded_guidance_scale = 6.0
        self.generator = [torch.Generator("cuda").manual_seed(seed) for seed in [self.config.seed]]
        self.noise_pred = None

    def prepare(self, image_encoder_output):
        self.image_encoder_output = image_encoder_output
        self.prepare_latents(shape=self.config.target_shape, dtype=torch.float16, image_encoder_output=image_encoder_output)
        self.prepare_guidance()
        self.prepare_rotary_pos_embedding(video_length=self.config.target_video_length, height=self.config.target_height, width=self.config.target_width)

    def prepare_guidance(self):
        self.guidance = torch.tensor([self.embedded_guidance_scale], dtype=torch.bfloat16, device=torch.device("cuda")) * 1000.0

    def step_post(self):
        if self.config.task == "t2v":
            sample = self.latents.to(torch.float32)
            dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
            self.latents = sample + self.noise_pred.to(torch.float32) * dt
        else:
            sample = self.latents[:, :, 1:, :, :].to(torch.float32)
            dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
            latents = sample + self.noise_pred[:, :, 1:, :, :].to(torch.float32) * dt
            self.latents = torch.concat([self.image_encoder_output["img_latents"], latents], dim=2)

    def prepare_latents(self, shape, dtype, image_encoder_output):
        if self.config.task == "t2v":
            self.latents = randn_tensor(shape, generator=self.generator, device=torch.device("cuda"), dtype=dtype)
        else:
            x1 = image_encoder_output["img_latents"].repeat(1, 1, (self.config.target_video_length - 1) // 4 + 1, 1, 1)
            x0 = randn_tensor(shape, generator=self.generator, device=torch.device("cuda"), dtype=dtype)
            t = torch.tensor([0.999]).to(device=torch.device("cuda"))
            self.latents = x0 * t + x1 * (1 - t)
            self.latents = self.latents.to(dtype=dtype)
            self.latents = torch.concat([image_encoder_output["img_latents"], self.latents[:, :, 1:, :, :]], dim=2)

    def prepare_rotary_pos_embedding(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2
        # 884
        vae = "884-16c-hy"
        patch_size = [1, 2, 2]
        hidden_size = 3072
        heads_num = 24
        rope_theta = 256
        rope_dim_list = [16, 56, 56]
        if "884" in vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]

        if isinstance(patch_size, int):
            assert all(s % patch_size == 0 for s in latents_size), f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), but got {latents_size}."
            rope_sizes = [s // patch_size for s in latents_size]
        elif isinstance(patch_size, list):
            assert all(s % patch_size[idx] == 0 for idx, s in enumerate(latents_size)), f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), but got {latents_size}."
            rope_sizes = [s // patch_size[idx] for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis

        if self.config.task == "t2v":
            head_dim = hidden_size // heads_num
            rope_dim_list = rope_dim_list
            if rope_dim_list is None:
                rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
            assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
            self.freqs_cos, self.freqs_sin = get_nd_rotary_pos_embed(
                rope_dim_list,
                rope_sizes,
                theta=rope_theta,
                use_real=True,
                theta_rescale_factor=1,
            )
            self.freqs_cos = self.freqs_cos.to(dtype=torch.bfloat16, device=torch.device("cuda"))
            self.freqs_sin = self.freqs_sin.to(dtype=torch.bfloat16, device=torch.device("cuda"))

        else:
            L_test = rope_sizes[0]  # Latent frames
            L_train = 25  # Training length from HunyuanVideo
            actual_num_frames = video_length  # Use input video_length directly

            head_dim = hidden_size // heads_num
            rope_dim_list = rope_dim_list or [head_dim // target_ndim for _ in range(target_ndim)]
            assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) must equal head_dim"

            if actual_num_frames > 192:
                k = 2 + ((actual_num_frames + 3) // (4 * L_train))
                k = max(4, min(8, k))

                # Compute positional grids for RIFLEx
                axes_grids = [torch.arange(size, device=torch.device("cuda"), dtype=torch.float32) for size in rope_sizes]
                grid = torch.meshgrid(*axes_grids, indexing="ij")
                grid = torch.stack(grid, dim=0)  # [3, t, h, w]
                pos = grid.reshape(3, -1).t()  # [t * h * w, 3]

                # Apply RIFLEx to temporal dimension
                freqs = []
                for i in range(3):
                    if i == 0:  # Temporal with RIFLEx
                        freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(rope_dim_list[i], pos[:, i], theta=rope_theta, use_real=True, k=k, L_test=L_test)
                    else:  # Spatial with default RoPE
                        freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(rope_dim_list[i], pos[:, i], theta=rope_theta, use_real=True, k=None, L_test=None)
                    freqs.append((freqs_cos, freqs_sin))

                freqs_cos = torch.cat([f[0] for f in freqs], dim=1)
                freqs_sin = torch.cat([f[1] for f in freqs], dim=1)
            else:
                # 20250316 pftq: Original code for <= 192 frames
                freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                    rope_dim_list,
                    rope_sizes,
                    theta=rope_theta,
                    use_real=True,
                    theta_rescale_factor=1,
                )

            self.freqs_cos = freqs_cos.to(dtype=torch.bfloat16, device=torch.device("cuda"))
            self.freqs_sin = freqs_sin.to(dtype=torch.bfloat16, device=torch.device("cuda"))
