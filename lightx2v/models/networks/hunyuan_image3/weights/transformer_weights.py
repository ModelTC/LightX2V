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
        # Execution slots need the loaded CPU shapes and are created by the model.
        self.add_module("blocks", self.blocks)

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
