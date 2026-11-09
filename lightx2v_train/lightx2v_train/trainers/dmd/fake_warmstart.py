"""Weights-only initialization of the fake role from an existing DMD run."""

from pathlib import Path

import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

from lightx2v_train.model_capabilities import ParallelCapability


def load_fake_checkpoint(trainer, checkpoint_path, *, strict=True):
    """Load fake model weights after wrapping; never load optimizer/cursor state."""
    checkpoint = Path(checkpoint_path)
    distributed = checkpoint / "dist_state"
    if checkpoint.name == "dist_state":
        distributed = checkpoint
    if distributed.is_dir():
        module = trainer.fake_model.ensure_capabilities().require(ParallelCapability).state_module()
        options = StateDictOptions(ignore_frozen_params=True, strict=strict)
        state = {"fake_model": get_model_state_dict(module, options=options)}
        # Full DMD checkpoints store fake_model, whereas generic/CM checkpoints
        # contain model. Reject the latter instead of silently loading a student.
        dcp.load(state, checkpoint_id=str(distributed), process_group=trainer.checkpoint_manager._get_checkpoint_process_group())
        set_model_state_dict(module, model_state_dict=state["fake_model"], options=options)
        logger.info("[train] fake weights-only warm-start from {} (fresh optimizer, schedule and data cursor)", distributed)
        return
    directory = trainer.role_registry.weight_directory_name("fake")
    weights = checkpoint / directory
    if not weights.is_dir():
        raise FileNotFoundError(f"Fake warm-start requires DMD dist_state or {directory}/ weights: {checkpoint}")
    trainer._load_model_weights(trainer.fake_model, str(weights), role="fake")
    logger.info("[train] fake weights-only warm-start from {} (fresh optimizer, schedule and data cursor)", weights)
