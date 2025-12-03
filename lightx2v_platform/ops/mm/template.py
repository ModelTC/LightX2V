from abc import ABCMeta, abstractmethod

import torch


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
        self.infer_dtype = torch.bfloat16  # bias dtype
        self.bias_force_fp32 = False

    # =========================
    # weight load functions
    # =========================
    def load(self, weight_dict):
        self.load_quantized(weight_dict)
        if self.weight_need_transpose:
            if hasattr(self, "weight") and self.weight is not None:
                self.weight = self.weight.t()
            if hasattr(self, "pin_weight") and self.pin_weight is not None:
                self.pin_weight = self.pin_weight.t()
            if hasattr(self, "weight_cuda_buffer") and self.weight_cuda_buffer is not None:
                self.weight_cuda_buffer = self.weight_cuda_buffer.t()

    def load_quantized(self, weight_dict):
        if self.create_cuda_buffer:
            self._load_cuda_buffers(weight_dict)
        elif self.create_cpu_buffer:
            self._load_cpu_pin_buffers()
        else:
            self._load_default_tensors(weight_dict)

    def _load_cuda_buffers(self, weight_dict):
        source = self.lazy_load_file if self.lazy_load else weight_dict
        self.weight_cuda_buffer, self.weight_scale_cuda_buffer = self._get_cuda_tensor_pair(source, self.lazy_load)
        self.bias_cuda_buffer = self._get_cuda_bias_tensor(source, self.lazy_load)

    def _get_cuda_tensor_pair(self, source, is_lazy):
        if is_lazy:
            weight = source.get_tensor(self.weight_name).to(AI_DEVICE)
            scale = source.get_tensor(self.weight_scale_name).float().to(AI_DEVICE)
        else:
            weight = source[self.weight_name].to(AI_DEVICE)
            scale = source[self.weight_scale_name].float().to(AI_DEVICE)
        return weight, scale

    def _get_cuda_bias_tensor(self, source, is_lazy):
        if self.bias_name is None:
            return None
        if is_lazy:
            bias = source.get_tensor(self.bias_name)
            dtype = self.infer_dtype
        else:
            bias = source[self.bias_name]
            dtype = bias.dtype
        if self.bias_force_fp32:
            bias = bias.to(torch.float32)
        else:
            bias = bias.to(dtype)
        return bias.to(AI_DEVICE)

    def _load_cpu_pin_buffers(self):
        self.pin_weight, self.pin_weight_scale = self._get_cpu_pin_tensor_pair(self.lazy_load_file, is_lazy=True)
        self.pin_bias = self._get_cpu_pin_bias_tensor(self.lazy_load_file, is_lazy=True)
        self.bias = None

    def _get_cpu_pin_tensor_pair(self, source, is_lazy):
        if is_lazy:
            weight_tensor = source.get_tensor(self.weight_name)
            scale_tensor = source.get_tensor(self.weight_scale_name)
            scale_dtype = torch.float
        else:
            weight_tensor = source[self.weight_name]
            scale_tensor = source[self.weight_scale_name]
            scale_dtype = torch.float

        pin_weight = self._create_pin_tensor(weight_tensor)
        pin_scale = self._create_pin_tensor(scale_tensor, scale_dtype)
        return pin_weight, pin_scale

    def _get_cpu_pin_bias_tensor(self, source, is_lazy):
        if self.bias_name is None:
            return None
        if is_lazy:
            bias_tensor = source.get_tensor(self.bias_name)
            if not self.bias_force_fp32:
                bias_tensor = bias_tensor.to(self.infer_dtype)
        else:
            bias_tensor = source[self.bias_name]
        if self.bias_force_fp32:
            bias_tensor = bias_tensor.to(torch.float32)
        return self._create_pin_tensor(bias_tensor)

    def _create_pin_tensor(self, tensor, dtype=None):
        dtype = dtype or tensor.dtype
        pin_tensor = torch.empty(tensor.shape, pin_memory=True, dtype=dtype)
        pin_tensor.copy_(tensor)
        del tensor
        return pin_tensor

    def _load_default_tensors(self, weight_dict):
        if not self.lazy_load:
            self.weight, self.weight_scale, self.pin_weight, self.pin_weight_scale = self._get_device_tensor_pair(weight_dict)
            self._load_default_bias(weight_dict)
        else:
            self.bias = None
            self.pin_bias = None

    def _get_device_tensor_pair(self, source):
        device = source[self.weight_name].device
        if device.type == "cpu":
            pin_weight, pin_scale = self._get_cpu_pin_tensor_pair(source, is_lazy=False)
            return None, None, pin_weight, pin_scale
        else:
            return source[self.weight_name], source[self.weight_scale_name].float(), None, None

    def _load_default_bias(self, source):
        if self.bias_name is None:
            self.bias = None
            self.pin_bias = None
            self.bias_cuda_buffer = None
            return

        if self.create_cuda_buffer:
            self.bias_cuda_buffer = self._get_cuda_bias_tensor(source, is_lazy=False)
            self.bias = None
            self.pin_bias = None
        else:
            bias_tensor = source[self.bias_name].float() if self.bias_force_fp32 else source[self.bias_name]
            device = bias_tensor.device
            if device.type == "cpu":
                self.pin_bias = self._get_cpu_pin_bias_tensor(source, is_lazy=False)
                self.bias = None
            else:
                self.bias = bias_tensor
                self.pin_bias = None

    def load_fp8_perchannel_sym(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            self.weight = weight_dict[self.weight_name].to(torch.float32)
            w_quantizer = FloatQuantizer("e4m3", True, "per_channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.float8_e4m3fn)
            self.weight_scale = self.weight_scale.to(torch.float32)
        else:
            self.load_quantized(weight_dict)

    def load_int8_perchannel_sym(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            self.weight = weight_dict[self.weight_name].to(torch.float32)
            w_quantizer = IntegerQuantizer(8, True, "per_channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.int8)
            self.weight_scale = self.weight_scale.to(torch.float32)
        else:
            self.load_quantized(weight_dict)

    def load_mxfp4(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            device = weight_dict[self.weight_name].device
            self.weight = weight_dict[self.weight_name].to(AI_DEVICE).to(torch.bfloat16)
            self.weight, self.weight_scale = scaled_mxfp4_quant(self.weight)
            self.weight, self.weight_scale = self.weight.to(device), self.weight_scale.to(device)
        else:
            device = weight_dict[self.weight_name].device
            if device.type == "cpu":
                weight_shape = weight_dict[self.weight_name].shape
                weight_dtype = weight_dict[self.weight_name].dtype
                self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
                self.pin_weight.copy_(weight_dict[self.weight_name])

                weight_scale_shape = weight_dict[self.weight_scale_name].shape
                weight_scale_dtype = weight_dict[self.weight_scale_name].dtype
                self.pin_weight_scale = torch.empty(weight_scale_shape, pin_memory=True, dtype=weight_scale_dtype)
                self.pin_weight_scale.copy_(weight_dict[self.weight_scale_name])
                del weight_dict[self.weight_name]
            else:
                self.weight = weight_dict[self.weight_name]
                self.weight_scale = weight_dict[self.weight_scale_name]

    def load_mxfp6(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            device = weight_dict[self.weight_name].device
            self.weight = weight_dict[self.weight_name].to(AI_DEVICE).to(torch.bfloat16)
            self.weight, self.weight_scale = scaled_mxfp6_quant(self.weight)
            self.weight, self.weight_scale = self.weight.to(device), self.weight_scale.to(device)
        else:
            device = weight_dict[self.weight_name].device
            if device.type == "cpu":
                weight_shape = weight_dict[self.weight_name].shape
                weight_dtype = weight_dict[self.weight_name].dtype
                self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
                self.pin_weight.copy_(weight_dict[self.weight_name])

                weight_scale_shape = weight_dict[self.weight_scale_name].shape
                weight_scale_dtype = weight_dict[self.weight_scale_name].dtype
                self.pin_weight_scale = torch.empty(weight_scale_shape, pin_memory=True, dtype=weight_scale_dtype)
                self.pin_weight_scale.copy_(weight_dict[self.weight_scale_name])
                del weight_dict[self.weight_name]
            else:
                self.weight = weight_dict[self.weight_name]
                self.weight_scale = weight_dict[self.weight_scale_name]

    def load_mxfp8(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            device = weight_dict[self.weight_name].device
            self.weight = weight_dict[self.weight_name].to(AI_DEVICE).to(torch.bfloat16)
            self.weight, self.weight_scale = scaled_mxfp8_quant(self.weight)
            self.weight, self.weight_scale = self.weight.to(device), self.weight_scale.to(device)
        else:
            device = weight_dict[self.weight_name].device
            if device.type == "cpu":
                weight_shape = weight_dict[self.weight_name].shape
                weight_dtype = weight_dict[self.weight_name].dtype
                self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
                self.pin_weight.copy_(weight_dict[self.weight_name])

                weight_scale_shape = weight_dict[self.weight_scale_name].shape
                weight_scale_dtype = weight_dict[self.weight_scale_name].dtype
                self.pin_weight_scale = torch.empty(weight_scale_shape, pin_memory=True, dtype=weight_scale_dtype)
                self.pin_weight_scale.copy_(weight_dict[self.weight_scale_name])
                del weight_dict[self.weight_name]
            else:
                self.weight = weight_dict[self.weight_name]
                self.weight_scale = weight_dict[self.weight_scale_name]

    def load_nvfp4(self, weight_dict):
        device = weight_dict[self.weight_name].device

        input_absmax = weight_dict[self.weight_name.replace(".weight", ".input_absmax")]
        input_global_scale = (2688.0 / input_absmax).to(torch.float32)
        weight_global_scale = weight_dict[f"{self.weight_name}_global_scale"]
        alpha = 1.0 / (input_global_scale * weight_global_scale)

        if device.type == "cpu":
            weight_shape = weight_dict[self.weight_name].shape
            weight_dtype = weight_dict[self.weight_name].dtype
            self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
            self.pin_weight.copy_(weight_dict[self.weight_name])

            weight_scale_shape = weight_dict[self.weight_scale_name].shape
            weight_scale_dtype = weight_dict[self.weight_scale_name].dtype
            self.pin_weight_scale = torch.empty(weight_scale_shape, pin_memory=True, dtype=weight_scale_dtype)
            self.pin_weight_scale.copy_(weight_dict[self.weight_scale_name])

            input_global_scale_shape = input_global_scale.shape
            input_global_scale_dtype = input_global_scale.dtype
            self.pin_input_global_scale = torch.empty(input_global_scale_shape, pin_memory=True, dtype=input_global_scale_dtype)
            self.pin_input_global_scale.copy_(input_global_scale)

            alpha_shape = alpha.shape
            alpha_dtype = alpha.dtype
            self.pin_alpha = torch.empty(alpha_shape, pin_memory=True, dtype=alpha_dtype)
            self.pin_alpha.copy_(alpha)

            del weight_dict[self.weight_name]
        else:
            self.weight = weight_dict[self.weight_name]
            self.weight_scale = weight_dict[self.weight_scale_name]
            self.input_global_scale = input_global_scale
            self.alpha = alpha

        if self.bias_name is not None:
            if self.create_cuda_buffer:
                self.bias_cuda_buffer = weight_dict[self.bias_name].to(AI_DEVICE)
            else:
                device = weight_dict[self.bias_name].device
                if device.type == "cpu":
                    bias_shape = weight_dict[self.bias_name].shape
                    bias_dtype = weight_dict[self.bias_name].dtype
                    self.pin_bias = torch.empty(bias_shape, pin_memory=True, dtype=bias_dtype)
                    self.pin_bias.copy_(weight_dict[self.bias_name])
                else:
                    self.bias = weight_dict[self.bias_name]
        else:
            self.bias = None
            self.pin_bias = None

    def load_fp8_perblock128_sym(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            self.weight = weight_dict[self.weight_name]
            self.weight, self.weight_scale = self.per_block_cast_to_fp8(self.weight)
        else:
            self.load_quantized(weight_dict)

    def per_block_cast_to_fp8(self, x):
        assert x.dim() == 2
        m, n = x.shape
        x_padded = torch.zeros(
            (deep_gemm.ceil_div(m, 128) * 128, deep_gemm.ceil_div(n, 128) * 128),
            dtype=x.dtype,
            device=x.device,
        )
        x_padded[:m, :n] = x
        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

    # =========================
    # act quant kernels
    # =========================
    def act_quant_int8_perchannel_sym_torchao(self, x):
        input_tensor_quant, input_tensor_scale = quantize_activation_per_token_absmax(x)
        return input_tensor_quant, input_tensor_scale

    def act_quant_fp8_perchannel_sym_vllm(self, x):
        input_tensor_quant, input_tensor_scale = ops.scaled_fp8_quant(x, None, scale_ub=None, use_per_token_if_dynamic=True)
        return input_tensor_quant, input_tensor_scale

    def act_quant_fp8_perchannel_sym_sgl(self, x):
        m, k = x.shape
        input_tensor_quant = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda", requires_grad=False)
        input_tensor_scale = torch.empty((m, 1), dtype=torch.float32, device="cuda", requires_grad=False)
        sgl_kernel.sgl_per_token_quant_fp8(x, input_tensor_quant, input_tensor_scale)
        return input_tensor_quant, input_tensor_scale

    def act_quant_int8_perchannel_sym_vllm(self, x):
        input_tensor_quant, input_tensor_scale, _ = ops.scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
        return input_tensor_quant, input_tensor_scale

    def act_quant_nvfp4(self, x):
        input_tensor_quant, input_tensor_scale = scaled_nvfp4_quant(x, self.input_global_scale)
        return input_tensor_quant, input_tensor_scale

    def act_quant_mxfp4(self, x):
        input_tensor_quant, input_tensor_scale = scaled_mxfp4_quant(x)
        return input_tensor_quant, input_tensor_scale

    def act_quant_mxfp8(self, x):
        input_tensor_quant, input_tensor_scale = scaled_mxfp8_quant(x)
        return input_tensor_quant, input_tensor_scale

    def act_quant_fp8_perchannelgroup128_sym_deepgemm(self, x):
        assert x.dim() == 2 and x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

    def act_quant_fp8_perchannelgroup128_sym_sgl(self, x):
        m, k = x.shape
        input_tensor_quant = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda", requires_grad=False)
        input_tensor_scale = torch.empty((m, k // 128), dtype=torch.float32, device="cuda", requires_grad=False)
        sgl_kernel.sgl_per_token_group_quant_fp8(
            x,
            input_tensor_quant,
            input_tensor_scale,
            group_size=128,
            eps=1e-10,
            fp8_min=-448.0,
            fp8_max=448.0,
        )
        return input_tensor_quant, input_tensor_scale

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.weight_name] = self.pin_weight if hasattr(self, "pin_weight") else self.weight
        if self.bias_name is not None:
            destination[self.bias_name] = self.pin_bias if hasattr(self, "pin_bias") else self.bias
        destination[self.weight_scale_name] = self.pin_weight_scale if hasattr(self, "pin_weight_scale") else self.weight_scale
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        if self.is_post_adapter:
            weight_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_name, count=1)
            weight_scale_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_scale_name, count=1)
        else:
            weight_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_name, count=1)
            weight_scale_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_scale_name, count=1)

        if weight_name not in destination:
            self.weight = None
            return

        self.weight = self.weight_cuda_buffer.copy_(destination[weight_name], non_blocking=True)
        self.weight_scale = self.weight_scale_cuda_buffer.copy_(destination[weight_scale_name], non_blocking=True)

        if self.bias_name is not None:
            bias_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.bias_name, count=1)
            self.bias = self.bias_cuda_buffer.copy_(destination[bias_name], non_blocking=True)
        else:
            self.bias = None

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        if self.is_post_adapter:
            self.weight_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_name, count=1)
            self.weight_scale_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_scale_name, count=1)
        else:
            self.weight_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_name, count=1)
            self.weight_scale_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_scale_name, count=1)

        if self.weight_need_transpose:
            weight_tensor = self.lazy_load_file.get_tensor(self.weight_name).t()
        else:
            weight_tensor = self.lazy_load_file.get_tensor(self.weight_name)
        self.pin_weight = self.pin_weight.copy_(weight_tensor)

        weight_scale_tensor = self.lazy_load_file.get_tensor(self.weight_scale_name)
        self.pin_weight_scale = self.pin_weight_scale.copy_(weight_scale_tensor)

        del weight_tensor

        if self.bias_name is not None:
            if self.is_post_adapter:
                assert adapter_block_index is not None
                self.bias_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.bias_name, count=1)
            else:
                self.bias_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.bias_name, count=1)

            bias_tensor = self.lazy_load_file.get_tensor(self.bias_name)
            self.pin_bias.copy_(bias_tensor)
            del bias_tensor
