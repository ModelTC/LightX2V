from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from .flow_match_pair import FlowMatchPairScheduler
from .flow_match import FlowMatchScheduler

# from mova.registry import DIFFUSION_SCHEDULERS

# DIFFUSION_SCHEDULERS.register_module(
#     name="FlowMatchEulerDiscreteScheduler", module=FlowMatchEulerDiscreteScheduler
# )