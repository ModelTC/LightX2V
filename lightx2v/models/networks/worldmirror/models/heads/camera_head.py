# inspired by https://github.com/facebookresearch/vggt/blob/main/src/models/heads/camera_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import MlpFP32
from ..layers.block import Block


class CameraHead(nn.Module):
    """
    Camera head module: predicts camera parameters from token representations using iterative refinement

    Processes dedicated camera tokens through a series of transformer blocks
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
        block_fn: nn.Module = Block,
    ):
        super().__init__()

        self.out_dim = 9
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.depth = trunk_depth

        # Build refinement network using transformer block sequence
        self.refine_net = nn.Sequential(*[block_fn(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values) for _ in range(trunk_depth)])

        # Normalization for camera tokens and network output
        self.token_norm = nn.LayerNorm(dim_in)
        self.out_norm = nn.LayerNorm(dim_in)

        # Learnable initial camera parameter token
        self.init_token = nn.Parameter(torch.zeros(1, 1, self.out_dim))
        self.param_embed = nn.Linear(self.out_dim, dim_in)

        # Generate adaptive normalization parameters: shift, scale, and gate
        self.adapt_norm_gen = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization (no learnable parameters)
        self.adapt_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        # self.param_predictor = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.out_dim, drop=0)
        self.param_predictor = MlpFP32(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.out_dim, drop=0)

    def to(self, *args, **kwargs):
        self.refine_net = self.refine_net.to(*args, **kwargs)
        self.token_norm = self.token_norm.to(*args, **kwargs)
        self.out_norm = self.out_norm.to(*args, **kwargs)
        self.adapt_norm_gen = self.adapt_norm_gen.to(*args, **kwargs)
        self.adapt_norm = self.adapt_norm.to(*args, **kwargs)
        self.param_predictor = self.param_predictor.to(*args, **kwargs)

        # keep these parameters in FP32
        args, kwargs = MlpFP32.map_to_args_to_float(args, kwargs)
        self.init_token = nn.Parameter(self.init_token.to(*args, **kwargs))
        self.param_embed = self.param_embed.to(*args, **kwargs)

        return self

    def forward(self, feat_seq: list, steps: int = 4) -> list:
        """
        Forward pass to predict camera parameters

        Args:
            feat_seq: List of token tensors from network, last one used for prediction
            steps: Number of iterative refinement steps, default 4

        Returns:
            List of predicted camera encodings (post-activation) from each iteration
        """
        # Use tokens from last block for camera prediction
        latest_feat = feat_seq[-1]

        # Extract camera tokens
        cam_tokens = latest_feat[:, :, 0]
        cam_tokens = self.token_norm(cam_tokens)

        # Iteratively refine camera pose predictions
        b, seq_len, feat_dim = cam_tokens.shape  # seq_len expected to be 1
        curr_pred = None
        pred_seq = []

        for step in range(steps):
            # Use learned initial token for first iteration
            if curr_pred is None:
                net_input = self.param_embed(self.init_token.expand(b, seq_len, -1))
            else:
                curr_pred = curr_pred.detach()
                net_input = self.param_embed(curr_pred)
            net_input = net_input.to(cam_tokens.dtype)
            norm_shift, norm_scale, norm_gate = self.adapt_norm_gen(net_input).chunk(3, dim=-1)
            mod_cam_feat = norm_gate * self.apply_adaptive_modulation(self.adapt_norm(cam_tokens), norm_shift, norm_scale)
            mod_cam_feat = mod_cam_feat + cam_tokens

            proc_feat = self.refine_net(mod_cam_feat)
            param_delta = self.param_predictor(self.out_norm(proc_feat))

            if curr_pred is None:
                curr_pred = param_delta
            else:
                curr_pred = curr_pred + param_delta

            # Apply final activation functions for translation, quaternion, and field-of-view
            activated_params = self.apply_camera_parameter_activation(curr_pred)
            pred_seq.append(activated_params)

        return pred_seq

    def apply_camera_parameter_activation(self, params: torch.Tensor) -> torch.Tensor:
        """
        Apply activation functions to camera parameter components

        Args:
            params: Tensor containing camera parameters [translation, quaternion, focal_length]

        Returns:
            Activated camera parameters tensor
        """
        trans_vec = params[..., :3]
        quat_vec = params[..., 3:7]
        fl_vec = params[..., 7:]  # or field of view

        trans_vec = self.apply_parameter_activation(trans_vec, self.trans_act)
        quat_vec = self.apply_parameter_activation(quat_vec, self.quat_act)
        fl_vec = self.apply_parameter_activation(fl_vec, self.fl_act)

        activated_params = torch.cat([trans_vec, quat_vec, fl_vec], dim=-1)
        return activated_params

    def apply_parameter_activation(self, tensor: torch.Tensor, act_type: str) -> torch.Tensor:
        """
        Apply specified activation function to parameter tensor

        Args:
            tensor: Tensor containing parameter values
            act_type: Activation type ("linear", "inv_log", "exp", "relu")

        Returns:
            Activated parameter tensor
        """
        if act_type == "linear":
            return tensor
        elif act_type == "inv_log":
            return self.apply_inverse_logarithm_transform(tensor)
        elif act_type == "exp":
            return torch.exp(tensor)
        elif act_type == "relu":
            return F.relu(tensor)
        else:
            raise ValueError(f"Unknown activation_type: {act_type}")

    def apply_inverse_logarithm_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse logarithm transform: sign(y) * (exp(|y|) - 1)

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        return torch.sign(x) * (torch.expm1(torch.abs(x)))

    def apply_adaptive_modulation(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive modulation to input tensor using scaling and shifting parameters
        """
        # Modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
        return x * (1 + scale) + shift
