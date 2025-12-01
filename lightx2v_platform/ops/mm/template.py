from abc import ABCMeta, abstractmethod


class MMWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.create_cuda_buffer = create_cuda_buffer
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.is_post_adapter = is_post_adapter
        self.config = {}

    @abstractmethod
    def load(self, weight_dict):
        pass

    @abstractmethod
    def apply(self):
        pass

    def set_config(self, config={}):
        self.config = config

    def to_cuda(self, non_blocking=False):
        self.weight = self.pin_weight.cuda(non_blocking=non_blocking)
        if hasattr(self, "pin_weight_scale"):
            self.weight_scale = self.pin_weight_scale.cuda(non_blocking=non_blocking)
        if hasattr(self, "pin_bias") and self.pin_bias is not None:
            self.bias = self.pin_bias.cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pin_weight"):
            self.weight = self.pin_weight.copy_(self.weight, non_blocking=non_blocking).cpu()
            if hasattr(self, "weight_scale_name"):
                self.weight_scale = self.pin_weight_scale.copy_(self.weight_scale, non_blocking=non_blocking).cpu()
            if self.bias is not None:
                self.bias = self.pin_bias.copy_(self.bias, non_blocking=non_blocking).cpu()
        else:
            self.weight = self.weight.to("cpu", non_blocking=non_blocking)
            if hasattr(self, "weight_scale"):
                self.weight_scale = self.weight_scale.to("cpu", non_blocking=non_blocking)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias = self.bias.to("cpu", non_blocking=non_blocking)


class MMWeightQuantTemplate(MMWeightTemplate):
    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.weight_scale_name = self.weight_name.removesuffix(".weight") + ".weight_scale"
        self.load_func = None
        self.weight_need_transpose = True
        self.act_quant_func = None
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.infer_dtype = GET_DTYPE()
