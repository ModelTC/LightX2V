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


MODEL_REGISTER = Register()
TRAINER_REGISTER = Register()
DATA_REGISTER = Register()


def build_model(config):
    name = config["model"]["name"]
    if name not in MODEL_REGISTER:
        available = ", ".join(sorted(MODEL_REGISTER.keys()))
        raise ValueError(f"Unknown model {name!r}. Available models: {available}")
    return MODEL_REGISTER[name](config)


def build_trainer(config):
    name = config["training"]["method"]
    if name not in TRAINER_REGISTER:
        available = ", ".join(sorted(TRAINER_REGISTER.keys()))
        raise ValueError(f"Unknown trainer {name!r}. Available trainers: {available}")
    return TRAINER_REGISTER[name](config)


def build_data(config, train_or_val):
    data_config = config.get("data", {})
    if train_or_val not in data_config:
        available_splits = ", ".join(repr(k) for k in sorted(data_config.keys()))
        raise ValueError(f"config['data'] has no key {train_or_val!r}. Available keys: {available_splits}")
    data_config_split = data_config[train_or_val]
    data_name = data_config_split.get("name", "image_dataset")
    if data_name not in DATA_REGISTER:
        available_names = ", ".join(sorted(DATA_REGISTER.keys()))
        raise ValueError(f"Unknown data {data_name!r}. Available data: {available_names}")
    return DATA_REGISTER[data_name](data_config_split, train_or_val=train_or_val)
