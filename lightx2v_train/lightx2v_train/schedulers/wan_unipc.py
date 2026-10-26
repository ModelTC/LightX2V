from lightx2v_train.model_zoo.native.wan.utils.fm_solvers_unipc import (
    FlowUniPCMultistepScheduler,
)


def build_wan_unipc_scheduler(
    num_train_timesteps,
    num_inference_steps,
    device,
    shift=5.0,
):
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=int(num_train_timesteps),
        shift=1.0,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(
        int(num_inference_steps),
        device=device,
        shift=float(shift),
    )
    return scheduler


def wan_unipc_timestep_to_sigma(timestep, num_train_timesteps):
    return timestep.float() / int(num_train_timesteps)
