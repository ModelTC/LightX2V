import torch
from loguru import logger

try:
    from qtorch.quant import float_quantize
except Exception:
    logger.warning("qtorch not found, please install qtorch.Please install qtorch (pip install qtorch).")
    float_quantize = None


class BaseQuantizer(object):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        self.bit = bit
        self.sym = symmetric
        self.granularity = granularity
        self.kwargs = kwargs
        if self.granularity == "per_group":
            self.group_size = self.kwargs["group_size"]
        self.calib_algo = self.kwargs.get("calib_algo", "minmax")

    def get_tensor_range(self, tensor):
        if self.calib_algo == "minmax":
            return self.get_minmax_range(tensor)
        elif self.calib_algo == "mse":
            return self.get_mse_range(tensor)
        else:
            raise ValueError(f"Unsupported calibration algorithm: {self.calib_algo}")

    def get_minmax_range(self, tensor):
        if self.granularity == "per_tensor":
            max_val = torch.max(tensor)
            min_val = torch.min(tensor)
        else:
            max_val = tensor.amax(dim=-1, keepdim=True)
            min_val = tensor.amin(dim=-1, keepdim=True)
        return (min_val, max_val)

    def get_mse_range(self, tensor):
        raise NotImplementedError

    def get_qparams(self, tensor_range, device):
        min_val, max_val = tensor_range[0], tensor_range[1]
        qmin = self.qmin.to(device)
        qmax = self.qmax.to(device)
        if self.sym:
            abs_max = torch.max(max_val.abs(), min_val.abs())
            abs_max = abs_max.clamp(min=1e-5)
            scales = abs_max / qmax
            zeros = torch.tensor(0.0)
        else:
            scales = (max_val - min_val).clamp(min=1e-5) / (qmax - qmin)
            zeros = (qmin - torch.round(min_val / scales)).clamp(qmin, qmax)
        return scales, zeros, qmax, qmin

    def reshape_tensor(self, tensor, allow_padding=False):
        if self.granularity == "per_group":
            t = tensor.reshape(-1, self.group_size)
        else:
            t = tensor
        return t

    def restore_tensor(self, tensor, shape):
        if tensor.shape == shape:
            t = tensor
        else:
            t = tensor.reshape(shape)
        return t

    def get_tensor_qparams(self, tensor):
        tensor = self.reshape_tensor(tensor)
        tensor_range = self.get_tensor_range(tensor)
        scales, zeros, qmax, qmin = self.get_qparams(tensor_range, tensor.device)
        return tensor, scales, zeros, qmax, qmin

    def fake_quant_tensor(self, tensor):
        org_shape = tensor.shape
        org_dtype = tensor.dtype
        tensor, scales, zeros, qmax, qmin = self.get_tensor_qparams(tensor)
        tensor = self.quant_dequant(tensor, scales, zeros, qmax, qmin)
        tensor = self.restore_tensor(tensor, org_shape).to(org_dtype)
        return tensor

    def real_quant_tensor(self, tensor):
        org_shape = tensor.shape
        tensor, scales, zeros, qmax, qmin = self.get_tensor_qparams(tensor)
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.restore_tensor(tensor, org_shape)
        if self.sym:
            zeros = None
        return tensor, scales, zeros


class IntegerQuantizer(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)
        if "int_range" in self.kwargs:
            self.qmin = self.kwargs["int_range"][0]
            self.qmax = self.kwargs["int_range"][1]
        else:
            if self.sym:
                self.qmin = -(2 ** (self.bit - 1))
                self.qmax = 2 ** (self.bit - 1) - 1
            else:
                self.qmin = 0.0
                self.qmax = 2**self.bit - 1

        self.qmin = torch.tensor(self.qmin)
        self.qmax = torch.tensor(self.qmax)
        self.dst_nbins = 2**bit

    def quant(self, tensor, scales, zeros, qmax, qmin):
        tensor = torch.clamp(torch.round(tensor / scales) + zeros, qmin, qmax)
        return tensor

    def dequant(self, tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor

    def quant_dequant(
        self,
        tensor,
        scales,
        zeros,
        qmax,
        qmin,
    ):
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.dequant(tensor, scales, zeros)
        return tensor


class FloatQuantizer(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)
        assert self.bit in ["e4m3", "e5m2"], f"Unsupported bit configuration: {self.bit}"
        assert self.sym

        if self.bit == "e4m3":
            self.e_bits = 4
            self.m_bits = 3
            self.fp_dtype = torch.float8_e4m3fn
        elif self.bit == "e5m2":
            self.e_bits = 5
            self.m_bits = 2
            self.fp_dtype = torch.float8_e5m2
        else:
            raise ValueError(f"Unsupported bit configuration: {self.bit}")

        finfo = torch.finfo(self.fp_dtype)
        self.qmin, self.qmax = finfo.min, finfo.max

        self.qmax = torch.tensor(self.qmax)
        self.qmin = torch.tensor(self.qmin)

    def quant(self, tensor, scales, zeros, qmax, qmin):
        scaled_tensor = tensor / scales + zeros
        scaled_tensor = torch.clip(scaled_tensor, self.qmin.cuda(), self.qmax.cuda())
        org_dtype = scaled_tensor.dtype
        q_tensor = float_quantize(scaled_tensor.float(), self.e_bits, self.m_bits, rounding="nearest")
        q_tensor.to(org_dtype)
        return q_tensor

    def dequant(self, tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor

    def quant_dequant(self, tensor, scales, zeros, qmax, qmin):
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.dequant(tensor, scales, zeros)
        return tensor


if __name__ == "__main__":
    weight = torch.randn(4096, 4096, dtype=torch.bfloat16).cuda()
    quantizer = IntegerQuantizer(4, False, "per_group", group_size=128)
    q_weight = quantizer.fake_quant_tensor(weight)
    logger.info(weight)
    logger.info(q_weight)
    logger.info(f"cosine = {torch.cosine_similarity(weight.view(1, -1).to(torch.float64), q_weight.view(1, -1).to(torch.float64))}")

    realq_weight, scales, zeros = quantizer.real_quant_tensor(weight)
    logger.info(f"realq_weight = {realq_weight}, {realq_weight.shape}")
    logger.info(f"scales = {scales}, {scales.shape}")
    logger.info(f"zeros = {zeros}, {zeros.shape}")

    weight = torch.randn(8192, 4096, dtype=torch.bfloat16).cuda()
    quantizer = FloatQuantizer("e4m3", True, "per_channel")
    q_weight = quantizer.fake_quant_tensor(weight)
    logger.info(weight)
    logger.info(q_weight)
    logger.info(f"cosine = {torch.cosine_similarity(weight.view(1, -1).to(torch.float64), q_weight.view(1, -1).to(torch.float64))}")

    realq_weight, scales, zeros = quantizer.real_quant_tensor(weight)
    logger.info(f"realq_weight = {realq_weight}, {realq_weight.shape}")
    logger.info(f"scales = {scales}, {scales.shape}")
    logger.info(f"zeros = {zeros}")
