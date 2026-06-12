import math


class StaticTimeShiftMu:
    def __init__(self, time_shift_settings):
        self.mu = time_shift_settings.get("time_shift_mu", 5.0)

    def __call__(self, latent_hw=None, num_steps=None):
        return self.mu


class LinearDynamicTimeShiftMu:
    def __init__(self, time_shift_settings):
        self.shift_x1 = time_shift_settings["shift_x1"]
        self.shift_x2 = time_shift_settings["shift_x2"]
        self.shift_y1 = time_shift_settings["shift_y1"]
        self.shift_y2 = time_shift_settings["shift_y2"]
        self.patch_size = time_shift_settings.get("patch_size", [2, 2])
        self._mu_slope = (self.shift_y2 - self.shift_y1) / (self.shift_x2 - self.shift_x1)
        self._mu_bias = self.shift_y1 - self._mu_slope * self.shift_x1

    def __call__(self, latent_hw=None, num_steps=None):
        image_seq_len = self._image_seq_len(latent_hw)
        return self._mu_slope * image_seq_len + self._mu_bias

    def _image_seq_len(self, latent_hw):
        if latent_hw is None:
            raise ValueError("latent_hw=(H, W) must be provided when dynamic_shift=True")
        h, w = latent_hw
        return (h // self.patch_size[0]) * (w // self.patch_size[1])


class Flux2EmpiricalTimeShiftMu:
    def __init__(self, time_shift_settings):
        self.shift_mu_num_steps = time_shift_settings.get("shift_mu_num_steps", 50)
        self.patch_size = time_shift_settings.get("patch_size", [1, 1])

    def __call__(self, latent_hw=None, num_steps=None):
        image_seq_len = self._image_seq_len(latent_hw)
        return self._compute_mu(
            image_seq_len=image_seq_len,
            num_steps=num_steps or self.shift_mu_num_steps,
        )

    def _image_seq_len(self, latent_hw):
        if latent_hw is None:
            raise ValueError("latent_hw=(H, W) must be provided when dynamic_shift=True")
        h, w = latent_hw
        return (h // self.patch_size[0]) * (w // self.patch_size[1])

    @staticmethod
    def _compute_mu(image_seq_len, num_steps):
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            return float(a2 * image_seq_len + b2)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        return float(a * num_steps + b)


class WanVideoDynamicTimeShiftMu:
    def __init__(self, time_shift_settings):
        self.base_shift = time_shift_settings.get("base_shift", 3.0)
        self.max_shift = time_shift_settings.get("max_shift", 5.0)
        self.vae_scale_factor_spatial = time_shift_settings.get("vae_scale_factor_spatial", 8)
        self.base_resolution = time_shift_settings.get("base_resolution", [480, 832])
        self.max_resolution = time_shift_settings.get("max_resolution", [720, 1280])
        self.shift_x1 = time_shift_settings.get("shift_x1", self._resolution_length(self.base_resolution))
        self.shift_x2 = time_shift_settings.get("shift_x2", self._resolution_length(self.max_resolution))
        self.shift_y1 = time_shift_settings.get("shift_y1", math.log(self.base_shift))
        self.shift_y2 = time_shift_settings.get("shift_y2", math.log(self.max_shift))
        if self.shift_x2 == self.shift_x1:
            self._mu_slope = 0.0
            self._mu_bias = self.shift_y1
        else:
            self._mu_slope = (self.shift_y2 - self.shift_y1) / (self.shift_x2 - self.shift_x1)
            self._mu_bias = self.shift_y1 - self._mu_slope * self.shift_x1

    def _resolution_length(self, resolution):
        height, width = resolution
        return math.sqrt(height * width) / self.vae_scale_factor_spatial

    def __call__(self, latent_hw=None, num_steps=None):
        if latent_hw is None:
            raise ValueError("latent_hw=(H, W) must be provided when shift_mu_strategy='wan_video'")
        h, w = latent_hw
        length = math.sqrt(h * w)
        return self._mu_slope * length + self._mu_bias


def build_time_shift_mu(time_shift_settings):
    if not time_shift_settings.get("dynamic_shift", False):
        return StaticTimeShiftMu(time_shift_settings)

    shift_mu_strategy = time_shift_settings.get("shift_mu_strategy", "linear")
    if shift_mu_strategy == "linear":
        return LinearDynamicTimeShiftMu(time_shift_settings)
    if shift_mu_strategy == "flux2_empirical":
        return Flux2EmpiricalTimeShiftMu(time_shift_settings)
    if shift_mu_strategy == "wan_video":
        return WanVideoDynamicTimeShiftMu(time_shift_settings)
    raise ValueError(f"Unsupported shift_mu_strategy: {shift_mu_strategy}")
