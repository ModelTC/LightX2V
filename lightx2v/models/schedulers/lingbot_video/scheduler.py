import torch
from loguru import logger

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class LingBotVideoScheduler(WanScheduler):
    def __init__(self, config):
        BaseScheduler.__init__(self, config)
        self.infer_steps = int(config["infer_steps"])
        self.sample_shift = float(config["sample_shift"])
        self.shift = 1
        self.num_train_timesteps = 1000
        self.disable_corrector = []
        self.solver_order = 2
        self.noise_pred = None
        self.sample_guide_scale = float(config["sample_guide_scale"])
        self.caching_records_2 = [True] * self.infer_steps
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = None
        self._begin_index = None
        self.current_timestep = None
        self.keep_latents_dtype_in_scheduler = True

    def refresh_from_config(self, config):
        self.config = config
        self.infer_steps = int(config["infer_steps"])
        self.sample_shift = float(config["sample_shift"])
        self.sample_guide_scale = float(config["sample_guide_scale"])
        self.caching_records = [True] * self.infer_steps
        self.caching_records_2 = [True] * self.infer_steps
        self.step_index = 0

    def prepare(self, input_info):
        super().prepare(int(input_info.seed), input_info.target_shape)

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
        if self.generator is None:
            self.generator = torch.Generator(device=AI_DEVICE).manual_seed(int(seed))
        else:
            logger.info("Generator is not None, using existing generator for latents")
        self.latents = torch.randn(
            tuple(latent_shape),
            dtype=dtype,
            device=AI_DEVICE,
            generator=self.generator,
        )

    def _transformer_timestep(self, timestep):
        sigma = timestep.float() / self.num_train_timesteps
        sigma = sigma.to(GET_DTYPE())
        return (sigma * self.num_train_timesteps).float().reshape(1).to(AI_DEVICE)

    def step_pre(self, step_index):
        super().step_pre(step_index)
        self.current_timestep = self._transformer_timestep(self.timesteps[self.step_index])

    def step_post(self):
        super().step_post()
        self.noise_pred = None

    def clear(self):
        self.noise_pred = None
        self.current_timestep = None
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None
        self.this_order = None
        self.lower_order_nums = 0
