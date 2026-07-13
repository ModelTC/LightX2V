"""Single source of truth for the simulator/inference/visualization contract.

This module is intentionally dependency-free (pure Python, no ROS/torch imports)
so it can be imported by every ROS node as well as plain scripts. It defines, per
simulation environment, the ROS topic namespace, the set of cameras, the
action/state dimensions and the inference profile. All three nodes derive their
topic names and tensor dimensions from the same `EnvContract`, which removes the
LIBERO-specific hard-coding that used to live in each node.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class EnvContract:
    # Logical environment id, e.g. "libero" or "robotwin".
    name: str
    # ROS topic prefix, e.g. "/libero".
    namespace: str
    # All cameras the simulator publishes (logical names, also used by the viewer).
    cameras: Tuple[str, ...]
    # Subset of `cameras` fed to the policy, in the exact order the policy expects.
    policy_input_cameras: Tuple[str, ...]
    # Robot action / proprio-state vector dimensions exchanged over ROS.
    action_dim: int
    state_dim: int
    # Square render/publish size hint (used by simulators that render on demand).
    image_size: int
    # Inference assembly profile understood by FastWAMPolicy ("libero"|"robotwin").
    policy_profile: str
    # Action/state normalization mode ("minmax"|"zscore").
    normalize_mode: str
    # Whether to apply the LIBERO single-gripper sign/binarize post-processing.
    gripper_postprocess: bool
    # Optional human-readable description.
    description: str = field(default="")

    # ----- derived topic helpers (kept in one place so nodes never disagree) -----
    @property
    def action_topic(self) -> str:
        return f"{self.namespace}/action"

    @property
    def state_topic(self) -> str:
        return f"{self.namespace}/state"

    @property
    def success_topic(self) -> str:
        return f"{self.namespace}/success"

    @property
    def observation_ready_topic(self) -> str:
        return f"{self.namespace}/observation_ready"

    @property
    def task_topic(self) -> str:
        return f"{self.namespace}/task_description"

    @property
    def episode_topic(self) -> str:
        # Monotonic episode counter; lets the policy reset its per-episode state when
        # the simulator loops into a fresh episode.
        return f"{self.namespace}/episode"

    @property
    def control_topic(self) -> str:
        # JSON control commands (start/pause/resume/restart/set_task) from UIs.
        return f"{self.namespace}/control"

    @property
    def status_topic(self) -> str:
        # JSON status snapshots (state machine, episode history, available tasks).
        return f"{self.namespace}/status"

    def camera_topic(self, camera: str) -> str:
        if camera not in self.cameras:
            raise KeyError(f"camera '{camera}' is not part of contract '{self.name}': {self.cameras}")
        return f"{self.namespace}/{camera}/image_raw"

    def camera_topics(self) -> Dict[str, str]:
        return {camera: self.camera_topic(camera) for camera in self.cameras}


LIBERO_CONTRACT = EnvContract(
    name="libero",
    namespace="/libero",
    cameras=("agentview", "wrist", "frontview", "galleryview"),
    policy_input_cameras=("agentview", "wrist"),
    action_dim=7,
    state_dim=8,
    image_size=224,
    policy_profile="libero",
    normalize_mode="minmax",
    gripper_postprocess=True,
    description="LIBERO (robosuite/mujoco) single-arm tabletop manipulation.",
)


ROBOTWIN_CONTRACT = EnvContract(
    name="robotwin",
    namespace="/robotwin",
    # front_camera is a static scene camera already rendered every frame by
    # RoboTwin; observer_camera is an extra on-demand third-person view. Both are
    # viewer-only: the policy consumes exactly `policy_input_cameras`.
    cameras=("head_camera", "left_camera", "right_camera", "front_camera", "observer_camera"),
    policy_input_cameras=("head_camera", "left_camera", "right_camera"),
    action_dim=14,
    state_dim=14,
    image_size=384,
    policy_profile="robotwin",
    normalize_mode="zscore",
    gripper_postprocess=False,
    description="RoboTwin 2.0 (SAPIEN) dual-arm manipulation.",
)


CONTRACTS: Dict[str, EnvContract] = {
    LIBERO_CONTRACT.name: LIBERO_CONTRACT,
    ROBOTWIN_CONTRACT.name: ROBOTWIN_CONTRACT,
}


def get_contract(name: str) -> EnvContract:
    key = str(name).strip().lower()
    if key not in CONTRACTS:
        raise KeyError(f"unknown environment '{name}'. Available: {sorted(CONTRACTS)}")
    return CONTRACTS[key]
