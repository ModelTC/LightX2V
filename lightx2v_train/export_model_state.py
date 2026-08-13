import argparse
import os

import torch
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import cleanup_distributed, init_distributed, load_config, setup_logger
from lightx2v_train.runtime.distributed import barrier, is_main_process
from lightx2v_train.runtime.parallel import apply_parallel


def parse_args():
    parser = argparse.ArgumentParser(description="Export model_state.pt from an FSDP2 dist_state checkpoint.")
    parser.add_argument("--config", required=True, help="Training config used when the checkpoint was saved.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory containing dist_state/.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    init_distributed(config)
    setup_logger(config)

    try:
        checkpoint_dir = args.checkpoint
        dist_state_path = os.path.join(checkpoint_dir, "dist_state")
        output_path = os.path.join(checkpoint_dir, "model_state.pt")

        if not os.path.isdir(dist_state_path):
            raise FileNotFoundError(f"dist_state/ not found in {checkpoint_dir}")

        model = build_model(config)
        model.load_components()
        model.set_full_trainable()
        apply_parallel(model, config)

        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-6)

        shard_options = StateDictOptions(ignore_frozen_params=True, strict=False)
        model_state, optim_state = get_state_dict(model.fsdp2_state_module(), optimizer, options=shard_options)
        state = {"student_model": model_state, "student_optimizer": optim_state}
        dcp.load(state, checkpoint_id=dist_state_path)
        set_state_dict(
            model.fsdp2_state_module(),
            optimizer,
            model_state_dict=state["student_model"],
            optim_state_dict=state["student_optimizer"],
            options=shard_options,
        )

        full_options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=True,
            strict=False,
        )
        full_state_dict, _ = get_state_dict(model.fsdp2_state_module(), (), options=full_options)

        barrier()
        if is_main_process():
            torch.save(full_state_dict, output_path)
            logger.info("Exported consolidated model_state.pt to {}", output_path)
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
