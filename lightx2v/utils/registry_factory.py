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
