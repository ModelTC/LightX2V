from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER, PLATFORM_LAYERNORM_WEIGHT_REGISTER, PLATFORM_MM_WEIGHT_REGISTER, PLATFORM_RMS_WEIGHT_REGISTER, PLATFORM_ROPE_REGISTER


class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def __call__(self, target_or_name):
        if callable(target_or_name):
            return self.register(target_or_name)
        else:
            return lambda x: self.register(x, key=target_or_name)

    def register(self, target, key=None):
        if not callable(target):
            raise Exception(f"Error: {target} must be callable!")

        if key is None:
            key = target.__name__

        if key in self._dict:
            raise Exception(f"{key} already exists.")

        self[key] = target
        return target

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def merge(self, other_register):
        for key, value in other_register.items():
            if key in self._dict:
                raise Exception(f"{key} already exists in target register.")
            self[key] = value


MM_WEIGHT_REGISTER = Register()
ATTN_WEIGHT_REGISTER = Register()
RMS_WEIGHT_REGISTER = Register()
LN_WEIGHT_REGISTER = Register()
CONV3D_WEIGHT_REGISTER = Register()
CONV2D_WEIGHT_REGISTER = Register()
TENSOR_REGISTER = Register()
CONVERT_WEIGHT_REGISTER = Register()
EMBEDDING_WEIGHT_REGISTER = Register()
RUNNER_REGISTER = Register()
ROPE_REGISTER = Register()
SPARSE_MASK_GENERATOR_REGISTER = Register()
SPARSE_OPERATOR_REGISTER = Register()

ATTN_WEIGHT_REGISTER.merge(PLATFORM_ATTN_WEIGHT_REGISTER)
MM_WEIGHT_REGISTER.merge(PLATFORM_MM_WEIGHT_REGISTER)
RMS_WEIGHT_REGISTER.merge(PLATFORM_RMS_WEIGHT_REGISTER)
LN_WEIGHT_REGISTER.merge(PLATFORM_LAYERNORM_WEIGHT_REGISTER)
ROPE_REGISTER.merge(PLATFORM_ROPE_REGISTER)
