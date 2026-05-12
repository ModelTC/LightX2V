"""
Transformer-block weight holders for Lyra2WanDiT.

Each Lyra2AttentionBlock (lyra_2/_src/networks/wan2pt1_lyra2.py L50-212) owns:

  # Original submodules:
  # norm1        = WanLayerNorm(dim, eps)           – elementwise_affine=False → NO params
  # self_attn    = WanSelfAttention(dim, num_heads, ...)
  #     .q / .k / .v / .o  = nn.Linear(dim, dim)
  #     .norm_q / .norm_k  = WanRMSNorm(dim, eps)  (if qk_norm=True)
  #     .attn_op            = DotProductAttention    – no learnable params
  # norm3        = WanLayerNorm(dim, eps, elementwise_affine=True)  ← only if cross_attn_norm
  #              = nn.Identity()                                     ← default (cross_attn_norm=False)
  # cross_attn   = WanT2VCrossAttention or WanI2VCrossAttention
  #     t2v: same q/k/v/o/norm_q/norm_k as self_attn
  #     i2v: adds k_img, v_img (Linear), norm_k_img (RMSNorm), attn_op_image
  # norm2        = WanLayerNorm(dim, eps)           – elementwise_affine=False → NO params
  # ffn          = nn.Sequential(Linear(dim, ffn_dim), GELU, Linear(ffn_dim, dim))
  # modulation   = nn.Parameter(torch.randn(1, 6, dim))
  # cam_encoder  = nn.Linear(cam_dim, dim, bias=False)   ← if use_plucker_condition
  # buffer_encoder = nn.Sequential(Linear(buf_embed, 256, bias=False),
  #                                Linear(256, dim, bias=False))  ← if use_correspondence + mlp_squeeze
"""

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, TENSOR_REGISTER


class Lyra2SelfAttnWeights(WeightModule):
    """
    Weights for WanSelfAttention.

    Original WanSelfAttention.__init__ (wan2pt1.py L281-322):
      self.q = nn.Linear(dim, dim)
      self.k = nn.Linear(dim, dim)
      self.v = nn.Linear(dim, dim)
      self.o = nn.Linear(dim, dim)
      self.norm_q = WanRMSNorm(dim, eps)  # if qk_norm
      self.norm_k = WanRMSNorm(dim, eps)  # if qk_norm
    """

    def __init__(self, block_prefix: str, rms_norm_type: str = "torch"):
        super().__init__()
        p = block_prefix

        # q / k / v / o projections
        # Original: self.q = nn.Linear(dim, dim)
        self.add_module("q", MM_WEIGHT_REGISTER["Default"](f"{p}.self_attn.q.weight", f"{p}.self_attn.q.bias"))
        self.add_module("k", MM_WEIGHT_REGISTER["Default"](f"{p}.self_attn.k.weight", f"{p}.self_attn.k.bias"))
        self.add_module("v", MM_WEIGHT_REGISTER["Default"](f"{p}.self_attn.v.weight", f"{p}.self_attn.v.bias"))
        self.add_module("o", MM_WEIGHT_REGISTER["Default"](f"{p}.self_attn.o.weight", f"{p}.self_attn.o.bias"))

        # QK-norm (RMSNorm)
        # Original: self.norm_q = WanRMSNorm(dim, eps)  → state key: norm_q.weight
        self.add_module("norm_q", RMS_WEIGHT_REGISTER[rms_norm_type](f"{p}.self_attn.norm_q.weight"))
        self.add_module("norm_k", RMS_WEIGHT_REGISTER[rms_norm_type](f"{p}.self_attn.norm_k.weight"))

    def to_cpu(self, non_blocking=True):
        for m in self._modules.values():
            if hasattr(m, "to_cpu"):
                m.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for m in self._modules.values():
            if hasattr(m, "to_cuda"):
                m.to_cuda(non_blocking=non_blocking)


class Lyra2CrossAttnWeights(WeightModule):
    """
    Weights for WanT2VCrossAttention or WanI2VCrossAttention.

    T2V shares q/k/v/o/norm_q/norm_k with WanSelfAttention.
    I2V additionally has:
      self.k_img     = nn.Linear(dim, dim)
      self.v_img     = nn.Linear(dim, dim)
      self.norm_k_img = WanRMSNorm(dim, eps)

    Original cross-attn forward (wan2pt1.py L359-380 / L410-445):
      q = norm_q(q_proj(x))
      k = norm_k(k_proj(context))
      v = v_proj(context)
      (i2v also: k_img = norm_k_img(k_img_proj(context_img)), v_img = v_img_proj(context_img))
    """

    def __init__(self, block_prefix: str, cross_attn_type: str = "i2v_cross_attn", rms_norm_type: str = "torch"):
        super().__init__()
        p = block_prefix
        self.cross_attn_type = cross_attn_type

        # Shared q/k/v/o (inherited from WanSelfAttention)
        self.add_module("q", MM_WEIGHT_REGISTER["Default"](f"{p}.cross_attn.q.weight", f"{p}.cross_attn.q.bias"))
        self.add_module("k", MM_WEIGHT_REGISTER["Default"](f"{p}.cross_attn.k.weight", f"{p}.cross_attn.k.bias"))
        self.add_module("v", MM_WEIGHT_REGISTER["Default"](f"{p}.cross_attn.v.weight", f"{p}.cross_attn.v.bias"))
        self.add_module("o", MM_WEIGHT_REGISTER["Default"](f"{p}.cross_attn.o.weight", f"{p}.cross_attn.o.bias"))
        self.add_module("norm_q", RMS_WEIGHT_REGISTER[rms_norm_type](f"{p}.cross_attn.norm_q.weight"))
        self.add_module("norm_k", RMS_WEIGHT_REGISTER[rms_norm_type](f"{p}.cross_attn.norm_k.weight"))

        if cross_attn_type == "i2v_cross_attn":
            # Original WanI2VCrossAttention (wan2pt1.py L382-396):
            # self.k_img     = nn.Linear(dim, dim)
            # self.v_img     = nn.Linear(dim, dim)
            # self.norm_k_img = WanRMSNorm(dim, eps)
            self.add_module("k_img", MM_WEIGHT_REGISTER["Default"](f"{p}.cross_attn.k_img.weight", f"{p}.cross_attn.k_img.bias"))
            self.add_module("v_img", MM_WEIGHT_REGISTER["Default"](f"{p}.cross_attn.v_img.weight", f"{p}.cross_attn.v_img.bias"))
            self.add_module("norm_k_img", RMS_WEIGHT_REGISTER[rms_norm_type](f"{p}.cross_attn.norm_k_img.weight"))

    def to_cpu(self, non_blocking=True):
        for m in self._modules.values():
            if hasattr(m, "to_cpu"):
                m.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for m in self._modules.values():
            if hasattr(m, "to_cuda"):
                m.to_cuda(non_blocking=non_blocking)


class Lyra2BlockWeights(WeightModule):
    """
    All learnable weights for one Lyra2AttentionBlock.

    State-dict prefix: "blocks.{i}.*"

    Notes:
    - norm1, norm2 have elementwise_affine=False → NO weight/bias in state dict.
    - norm3 is nn.Identity when cross_attn_norm=False (ZoomGS default) → no params.
    - modulation is a raw nn.Parameter → loaded via TENSOR_REGISTER.
    - cam_encoder has bias=False → bias_name=None.
    - buffer_encoder.{0,1} have bias=False → bias_name=None.
    """

    def __init__(
        self,
        block_index: int,
        cross_attn_type: str = "i2v_cross_attn",
        use_plucker: bool = True,
        use_correspondence: bool = True,
        buffer_mlp_squeeze_dim: int = 256,
        cross_attn_norm: bool = False,
        rms_norm_type: str = "torch",
    ):
        super().__init__()
        p = f"blocks.{block_index}"

        # --------------------------------------------------------------
        # modulation  – nn.Parameter(1, 6, dim)
        # Original: self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        # State dict key: "blocks.{i}.modulation"  (no .weight suffix)
        # --------------------------------------------------------------
        self.add_module("modulation", TENSOR_REGISTER["Default"](f"{p}.modulation"))

        # norm1, norm2 – WanLayerNorm(dim, eps, elementwise_affine=False)
        # → No learnable parameters; pure computation (handled in infer).

        # --------------------------------------------------------------
        # Self-attention
        # Original: self.self_attn = WanSelfAttention(dim, num_heads, ...)
        # --------------------------------------------------------------
        self.add_module("self_attn", Lyra2SelfAttnWeights(p, rms_norm_type))

        # --------------------------------------------------------------
        # norm3  – only has params when cross_attn_norm=True
        # Default in ZoomGS: cross_attn_norm=False → Identity → no params.
        # Original: self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        #           if cross_attn_norm else nn.Identity()
        # --------------------------------------------------------------
        # (no weight module needed for the ZoomGS case)

        # --------------------------------------------------------------
        # Cross-attention
        # Original: self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](...)
        # --------------------------------------------------------------
        self.add_module("cross_attn", Lyra2CrossAttnWeights(p, cross_attn_type, rms_norm_type))

        # norm2 – same as norm1, no params (elementwise_affine=False).

        # --------------------------------------------------------------
        # FFN  – Sequential(Linear(dim, ffn_dim), GELU, Linear(ffn_dim, dim))
        # indices: 0=Linear, 1=GELU(no params), 2=Linear
        # Original: self.ffn = nn.Sequential(
        #     nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        # )
        # --------------------------------------------------------------
        self.add_module("ffn_0", MM_WEIGHT_REGISTER["Default"](f"{p}.ffn.0.weight", f"{p}.ffn.0.bias"))
        self.add_module("ffn_2", MM_WEIGHT_REGISTER["Default"](f"{p}.ffn.2.weight", f"{p}.ffn.2.bias"))

        # --------------------------------------------------------------
        # cam_encoder  – nn.Linear(1536, dim, bias=False)
        # Original: self.cam_encoder = nn.Linear(self.cam_dim, self.dim, bias=False)
        #           if self.cam_dim > 0 else None
        # --------------------------------------------------------------
        if use_plucker:
            # bias_name=None  → no bias loaded (bias=False in Linear)
            self.add_module(
                "cam_encoder",
                MM_WEIGHT_REGISTER["Default"](f"{p}.cam_encoder.weight", None),
            )

        # --------------------------------------------------------------
        # buffer_encoder
        # When buffer_mlp_squeeze_dim > 0:
        #   Sequential(Linear(buf_embed_dim, squeeze, bias=False),
        #              Linear(squeeze, dim, bias=False))
        # Original: self.buffer_encoder = nn.Sequential(
        #     nn.Linear(buffer_embed_dim, self.buffer_mlp_squeeze_dim, bias=False),
        #     nn.Linear(self.buffer_mlp_squeeze_dim, self.dim, bias=False),
        # )
        # --------------------------------------------------------------
        if use_correspondence:
            self.add_module(
                "buffer_enc_0",
                MM_WEIGHT_REGISTER["Default"](f"{p}.buffer_encoder.0.weight", None),
            )
            if buffer_mlp_squeeze_dim > 0:
                self.add_module(
                    "buffer_enc_1",
                    MM_WEIGHT_REGISTER["Default"](f"{p}.buffer_encoder.1.weight", None),
                )

    def to_cpu(self, non_blocking=True):
        for m in self._modules.values():
            if hasattr(m, "to_cpu"):
                m.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for m in self._modules.values():
            if hasattr(m, "to_cuda"):
                m.to_cuda(non_blocking=non_blocking)


class Lyra2TransformerWeights(WeightModule):
    """
    Container for all Lyra2AttentionBlock weights (blocks.0 … blocks.N-1).

    Original: self.blocks = nn.ModuleList([Lyra2AttentionBlock(...) for _ in range(num_layers)])
    """

    def __init__(self, config: dict):
        super().__init__()
        num_layers = config["num_layers"]
        cross_attn_type = "t2v_cross_attn" if config.get("model_type", "i2v") == "t2v" else "i2v_cross_attn"
        use_plucker = config.get("use_plucker_condition", True)
        use_correspondence = config.get("use_correspondence", True)
        buffer_mlp_squeeze_dim = config.get("buffer_mlp_squeeze_dim", 256)
        cross_attn_norm = config.get("cross_attn_norm", False)
        rms_norm_type = config.get("rms_norm_type", "torch")

        blocks = []
        for i in range(num_layers):
            blk = Lyra2BlockWeights(
                block_index=i,
                cross_attn_type=cross_attn_type,
                use_plucker=use_plucker,
                use_correspondence=use_correspondence,
                buffer_mlp_squeeze_dim=buffer_mlp_squeeze_dim,
                cross_attn_norm=cross_attn_norm,
                rms_norm_type=rms_norm_type,
            )
            blocks.append(blk)

        self.blocks = blocks
        for i, blk in enumerate(blocks):
            self.add_module(f"block_{i}", blk)

    def to_cpu(self, non_blocking=True):
        for m in self._modules.values():
            if hasattr(m, "to_cpu"):
                m.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for m in self._modules.values():
            if hasattr(m, "to_cuda"):
                m.to_cuda(non_blocking=non_blocking)
