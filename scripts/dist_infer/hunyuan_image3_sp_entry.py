"""Start one HunyuanImage3 pipeline lane per sequence-parallel rank.

CUDA_VISIBLE_DEVICES must be narrowed before importing torch because each
HunyuanImage3 SP rank owns a complete, pipeline-sharded model replica.
"""

import os
import sys


def _pipeline_layout_from_environment_or_argv():
    layout = os.getenv("HUNYUAN_IMAGE3_PIPELINE_LAYOUT", "interleaved")
    option_names = ("--hunyuan_pipeline_layout", "--hunyuan-pipeline-layout")
    arguments = sys.argv[1:]
    for index, argument in enumerate(arguments):
        if argument in option_names and index + 1 < len(arguments):
            return arguments[index + 1]
        for option_name in option_names:
            prefix = f"{option_name}="
            if argument.startswith(prefix):
                return argument[len(prefix) :]
    return layout


def _configure_pipeline_lane():
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    if local_world_size == 1:
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    visible_devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if not visible_devices:
        raise RuntimeError("HunyuanImage3 SP launcher requires CUDA_VISIBLE_DEVICES to be set before torchrun.")
    if len(visible_devices) % local_world_size:
        raise RuntimeError(
            "Visible GPU count must be divisible by LOCAL_WORLD_SIZE: "
            f"devices={visible_devices}, local_world_size={local_world_size}."
        )

    layout = _pipeline_layout_from_environment_or_argv().strip().lower()
    if layout == "interleaved":
        owned_devices = visible_devices[local_rank::local_world_size]
    elif layout == "contiguous":
        devices_per_lane = len(visible_devices) // local_world_size
        start = local_rank * devices_per_lane
        owned_devices = visible_devices[start : start + devices_per_lane]
    else:
        raise RuntimeError(f"HUNYUAN_IMAGE3_PIPELINE_LAYOUT must be interleaved or contiguous, got {layout!r}.")

    if not owned_devices:
        raise RuntimeError(f"HunyuanImage3 SP local rank {local_rank} owns no pipeline devices.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(owned_devices)
    os.environ["LIGHTX2V_CUDA_DEVICE_INDEX"] = "0"
    os.environ["HUNYUAN_IMAGE3_PIPELINE_LANE_DEVICES"] = ",".join(owned_devices)
    print(
        f"[HunyuanImage3 SP] local_rank={local_rank}/{local_world_size}, "
        f"layout={layout}, physical_lane={owned_devices}",
        flush=True,
    )


def main():
    _configure_pipeline_lane()

    # Importing this decorator imports torch, so it must happen after lane isolation.
    from torch.distributed.elastic.multiprocessing.errors import record

    from lightx2v.infer import main as infer_main

    record(infer_main)()


if __name__ == "__main__":
    main()
