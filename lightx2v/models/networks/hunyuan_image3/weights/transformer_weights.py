from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.hunyuan_image3.weights.common import (
    HunyuanImage3AttentionWeights,
    HunyuanImage3MLPPhaseWeights,
)


class HunyuanImage3TransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.config = config
        self.blocks_num = int(config.get("num_layers") or config["num_hidden_layers"])
        self.mm_type = config.get("dit_quant_scheme", "Default")
        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True
        self.lazy_load = config.get("lazy_load", False)
        self.blocks = WeightModuleList(
            [
                HunyuanImage3TransformerBlock(
                    block_index=i,
                    config=config,
                    mm_type=self.mm_type,
                    create_cuda_buffer=False,
                    create_cpu_buffer=False,
                    block_prefix="model.layers",
                    lazy_load=self.lazy_load,
                    lazy_load_path=lazy_load_path,
                    lora_path=lora_path,
                )
                for i in range(self.blocks_num)
            ]
        )
        self.register_offload_buffers(config, lazy_load_path, lora_path)
        self.add_module("blocks", self.blocks)

    def register_offload_buffers(self, config, lazy_load_path, lora_path):
        self.offload_block_cuda_buffers = None
        self.offload_phase_cuda_buffers = None
        self.offload_block_cpu_buffers = None
        self.offload_phase_cpu_buffers = None
        if not config.get("cpu_offload", False):
            return

        if config.get("offload_granularity", "block") == "block":
            self.offload_blocks_num = 2
            self.offload_block_cuda_buffers = WeightModuleList(
                [
                    HunyuanImage3TransformerBlock(
                        i,
                        config,
                        self.mm_type,
                        True,
                        False,
                        "model.layers",
                        self.lazy_load,
                        lazy_load_path,
                        lora_path,
                    )
                    for i in range(self.offload_blocks_num)
                ]
            )
            self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
            if self.lazy_load:
                self.offload_block_cpu_buffers = WeightModuleList(
                    [
                        HunyuanImage3TransformerBlock(
                            i,
                            config,
                            self.mm_type,
                            False,
                            True,
                            "model.layers",
                            self.lazy_load,
                            lazy_load_path,
                            lora_path,
                        )
                        for i in range(self.offload_blocks_num)
                    ]
                )
                self.add_module("offload_block_cpu_buffers", self.offload_block_cpu_buffers)
        elif config.get("offload_granularity") == "phase":
            self.offload_phase_cuda_buffers = HunyuanImage3TransformerBlock(
                0,
                config,
                self.mm_type,
                True,
                False,
                "model.layers",
                self.lazy_load,
                lazy_load_path,
                lora_path,
            ).compute_phases
            self.add_module("offload_phase_cuda_buffers", self.offload_phase_cuda_buffers)
            if self.lazy_load:
                self.offload_phase_cpu_buffers = WeightModuleList(
                    [
                        HunyuanImage3TransformerBlock(
                            i,
                            config,
                            self.mm_type,
                            False,
                            True,
                            "model.layers",
                            self.lazy_load,
                            lazy_load_path,
                            lora_path,
                        ).compute_phases
                        for i in range(2)
                    ]
                )
                self.add_module("offload_phase_cpu_buffers", self.offload_phase_cpu_buffers)

    def non_block_weights_to_cuda(self):
        pass

    def non_block_weights_to_cpu(self):
        pass


class HunyuanImage3TransformerBlock(WeightModule):
    def __init__(
        self,
        block_index,
        config,
        mm_type,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        block_prefix="model.layers",
        lazy_load=False,
        lazy_load_path=None,
        lora_path=None,
    ):
        super().__init__()
        self.block_index = block_index
        lazy_load_file = lazy_load_path if lazy_load else None
        self.compute_phases = WeightModuleList(
            [
                HunyuanImage3AttentionWeights(
                    block_prefix,
                    block_index,
                    config,
                    mm_type,
                    create_cuda_buffer,
                    create_cpu_buffer,
                    lazy_load,
                    lazy_load_file,
                    lora_path,
                ),
                HunyuanImage3MLPPhaseWeights(
                    block_prefix,
                    block_index,
                    config,
                    mm_type,
                    create_cuda_buffer,
                    create_cpu_buffer,
                    lazy_load,
                    lazy_load_file,
                    lora_path,
                ),
            ]
        )
        self.add_module("compute_phases", self.compute_phases)
