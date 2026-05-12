"""
Pre-transformer weight holders for Lyra2WanDiT.

Covers the following Lyra2WanModel submodules (lyra_2/_src/networks/wan2pt1_lyra2.py):

  # Original Lyra2WanModel.__init__ (L292-356):
  # patch_embedding  = nn.Linear(in_dim*pt*ph*pw, dim)
  # text_embedding   = nn.Sequential(Linear(text_dim, dim), GELU, Linear(dim, dim))
  # time_embedding   = nn.Sequential(Linear(freq_dim, dim), SiLU, Linear(dim, dim))
  # time_projection  = nn.Sequential(SiLU, Linear(dim, dim*6))
  # img_emb          = MLPProj(1280, dim)   # only for i2v / flf2v
  # clean_patch_embeddings = nn.ModuleList([Linear(in_dim*new_pt*..., dim), ...])
  # rope_position_embedding = VideoRopePosition3DEmb(...)  # no learnable params
"""

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class Lyra2PreWeights(WeightModule):
    """
    Holds all pre-transformer weights: patch embed, text/time embeddings,
    optional image proj (MLPProj), and optional clean patch embeddings.

    Weight key convention: keys are relative to model.net.*
    (strip the 'net.' prefix before passing the state-dict to .load()).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_type = config.get("model_type", "i2v")

        # ------------------------------------------------------------------
        # patch_embedding  (Linear, no Conv3d – conv_patchify defaults False)
        # Original: self.patch_embedding = nn.Linear(in_dim*pt*ph*pw, dim)
        # State dict key: "patch_embedding.weight", "patch_embedding.bias"
        # ------------------------------------------------------------------
        self.add_module(
            "patch_embedding",
            MM_WEIGHT_REGISTER["Default"]("patch_embedding.weight", "patch_embedding.bias"),
        )

        # ------------------------------------------------------------------
        # text_embedding: Sequential(Linear(text_dim, dim), GELU, Linear(dim, dim))
        # indices: 0=Linear, 1=GELU(no params), 2=Linear
        # Original: self.text_embedding = nn.Sequential(
        #     nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        # )
        # ------------------------------------------------------------------
        self.add_module(
            "text_emb_0",
            MM_WEIGHT_REGISTER["Default"]("text_embedding.0.weight", "text_embedding.0.bias"),
        )
        self.add_module(
            "text_emb_2",
            MM_WEIGHT_REGISTER["Default"]("text_embedding.2.weight", "text_embedding.2.bias"),
        )

        # ------------------------------------------------------------------
        # time_embedding: Sequential(Linear(freq_dim, dim), SiLU, Linear(dim, dim))
        # indices: 0=Linear, 1=SiLU(no params), 2=Linear
        # Original: self.time_embedding = nn.Sequential(
        #     nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        # )
        # ------------------------------------------------------------------
        self.add_module(
            "time_emb_0",
            MM_WEIGHT_REGISTER["Default"]("time_embedding.0.weight", "time_embedding.0.bias"),
        )
        self.add_module(
            "time_emb_2",
            MM_WEIGHT_REGISTER["Default"]("time_embedding.2.weight", "time_embedding.2.bias"),
        )

        # ------------------------------------------------------------------
        # time_projection: Sequential(SiLU, Linear(dim, dim*6))
        # index 0=SiLU(no params), index 1=Linear
        # Original: self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        # ------------------------------------------------------------------
        self.add_module(
            "time_proj_1",
            MM_WEIGHT_REGISTER["Default"]("time_projection.1.weight", "time_projection.1.bias"),
        )

        # ------------------------------------------------------------------
        # img_emb (MLPProj) – only for i2v / flf2v model types
        # Original MLPProj.proj = Sequential(
        #   0: LayerNorm(in_dim),  1: Linear(in_dim, in_dim),
        #   2: GELU,               3: Linear(in_dim, out_dim),
        #   4: LayerNorm(out_dim),
        # )
        # State dict keys under "img_emb.proj.*"
        # ------------------------------------------------------------------
        self.has_img_emb = model_type in ("i2v", "flf2v")
        if self.has_img_emb:
            self.add_module(
                "img_emb_ln0",
                LN_WEIGHT_REGISTER["torch"](
                    "img_emb.proj.0.weight", "img_emb.proj.0.bias", eps=1e-5
                ),
            )
            self.add_module(
                "img_emb_fc1",
                MM_WEIGHT_REGISTER["Default"]("img_emb.proj.1.weight", "img_emb.proj.1.bias"),
            )
            self.add_module(
                "img_emb_fc3",
                MM_WEIGHT_REGISTER["Default"]("img_emb.proj.3.weight", "img_emb.proj.3.bias"),
            )
            self.add_module(
                "img_emb_ln4",
                LN_WEIGHT_REGISTER["torch"](
                    "img_emb.proj.4.weight", "img_emb.proj.4.bias", eps=1e-5
                ),
            )

        # ------------------------------------------------------------------
        # clean_patch_embeddings – ModuleList of Linear, one per kernel size.
        # Created by Lyra2WanModel.init_clean_patch_embeddings().
        # State dict keys: "clean_patch_embeddings.{i}.weight/bias"
        # Number of entries is determined at runtime; we register them lazily.
        # ------------------------------------------------------------------
        self._clean_emb_modules = []  # filled by init_clean_patch_embeddings()

    def init_clean_patch_embeddings(self, num_kernels: int):
        """
        Register weight modules for each clean patch embedding.
        Must be called after the model config (kernel count) is known.

        Original: net.init_clean_patch_embeddings(kernel_sizes, kernel_types)
          → appends nn.Linear per kernel to net.clean_patch_embeddings (ModuleList)
        """
        for i in range(num_kernels):
            mod = MM_WEIGHT_REGISTER["Default"](
                f"clean_patch_embeddings.{i}.weight",
                f"clean_patch_embeddings.{i}.bias",
            )
            name = f"clean_emb_{i}"
            self.add_module(name, mod)
            self._clean_emb_modules.append(mod)

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)
