import os
import shutil


def prune_checkpoints(output_dir, total_limit):
    if total_limit is None:
        return
    if not os.path.exists(output_dir):
        return

    checkpoints = [name for name in os.listdir(output_dir) if name.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda name: int(name.split("-")[-1]))
    if len(checkpoints) < total_limit:
        return

    for name in checkpoints[: len(checkpoints) - total_limit + 1]:
        shutil.rmtree(os.path.join(output_dir, name))
