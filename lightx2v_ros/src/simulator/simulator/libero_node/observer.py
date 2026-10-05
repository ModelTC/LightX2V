"""Interactive defaults over the shared, ROS-free LIBERO runtime."""

from pathlib import Path

import numpy as np

from .runtime import (
    LIBERO_BENCHMARKS,
    LiberoRuntime,
)
from .runtime import (
    add_python_path as add_python_path,
)
from .runtime import (
    default_libero_root as default_libero_root,
)
from .runtime import (
    load_libero as load_libero,
)
from .runtime import (
    setup_libero_config as setup_libero_config,
)


def load_init_states(get_libero_path, task, init_state_id):
    state, _ = load_init_state(get_libero_path, task, init_state_id)
    return state


def load_init_state(get_libero_path, task, init_state_id):
    import torch

    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    init_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
    index = int(init_state_id)
    if index < 0 or index >= len(init_states):
        raise ValueError(f"init_state_id {index} is out of range for {task.name!r}; expected 0..{len(init_states) - 1}")
    return init_states[index], len(init_states)


def build_task_catalog(benchmark_module):
    """Return stable UI task ids mapped to their LIBERO suite/task metadata."""
    factories = benchmark_module.get_benchmark_dict()
    catalog = {}
    for benchmark_name in LIBERO_BENCHMARKS:
        factory = factories.get(benchmark_name)
        if factory is None:
            continue
        task_suite = factory()
        for task_id in range(task_suite.get_num_tasks()):
            task = task_suite.get_task(task_id)
            key = f"{benchmark_name}/{task_id}"
            catalog[key] = {
                "benchmark": benchmark_name,
                "task_id": task_id,
                "task_name": task.name,
                "language": task.language,
            }
    return catalog


class LiberoActionObserver(LiberoRuntime):
    def __init__(self, benchmark_name="libero_spatial", task_id=0, init_state_id=0, image_size=224, seed=0, libero_root=None):
        super().__init__(
            benchmark_name,
            task_id,
            init_state_id,
            image_size,
            seed,
            libero_root,
            camera_names=["robot0_eye_in_hand", "agentview", "frontview", "galleryview"],
            eager_reset=True,
        )

    def step(self, action):
        return super().step(np.asarray(action, dtype=np.float32))
