from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, TENSOR_REGISTER, LN_WEIGHT_REGISTER
from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList


class MagicDrivePostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.register_parameter(
            'final_layer_scale_shift_table', TENSOR_REGISTER['Default']('final_layer.scale_shift_table')
        )
        
        self.add_module(
            'final_layer_norm',
            LN_WEIGHT_REGISTER['Default']()
        )
        
        self.add_module(
            'final_layer_linear',
            MM_WEIGHT_REGISTER['Flinear']('final_layer.linear.weight', 'final_layer.linear.bias')
        )