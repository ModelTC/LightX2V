"""Generic, contract-driven simulator ROS node.

`SimulatorNode` is environment-agnostic: it derives every topic name and the
action/state dimensions from the `EnvContract` and drives any `BaseSimEnv`
implementation. An `env_factory(node) -> BaseSimEnv` callback lets each concrete
environment declare its own ROS parameters on the node before construction.

Evaluation control plane
------------------------
The node is a small state machine driven by JSON commands on
``{namespace}/control`` and reporting on ``{namespace}/status``:

    ready ──start──▶ running ──success/step-cap──▶ success | failure
      ▲                │  ▲                             │
      │              pause resume                    start/restart
      └────set_task────┴──┴─────────────────────────────┘

- ``pause`` freezes the evaluation stream: incoming actions are dropped and the
  observation counter stops advancing, so the policy (which only reacts to newer
  ``observation_ready`` indices) goes quiet. The scene itself stays put.
- ``resume``/``start`` bumps the observation counter so the policy sees a fresh
  observation and continues from the current physical state.
- ``restart`` tears the episode down and rebuilds it with a new seed.
- ``set_task`` rebuilds the env with a different task/scenario (may take tens of
  seconds; the node publishes a "switching" status first).
"""

import json
import time
from typing import Callable

import numpy as np
import rclpy
from common.contract import EnvContract
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Int32, String

from .base_env import BaseSimEnv

# Node states
READY = "ready"
RUNNING = "running"
PAUSED = "paused"
SUCCESS = "success"
FAILURE = "failure"
SWITCHING = "switching"

FINISHED_STATES = (SUCCESS, FAILURE)


def rgb_to_image_msg(image, stamp, frame_id):
    image = np.ascontiguousarray(image)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(image.strides[0])
    msg.data = image.tobytes()
    return msg


class SimulatorNode(Node):
    def __init__(
        self,
        contract: EnvContract,
        env_factory: Callable[["SimulatorNode"], BaseSimEnv],
        *,
        node_name: str = "simulator_node",
    ):
        super().__init__(node_name)
        self.contract = contract

        self.declare_parameter("republish_period", 1.0)
        # Auto-start the first episode on launch (old behaviour). Default is to
        # wait for a `start` control command (e.g. from the web viewer).
        self.declare_parameter("autostart", False)
        # Continuous-eval loop: when true, the node automatically starts a fresh
        # episode after each episode ends instead of stopping.
        self.declare_parameter("loop", False)
        # Per-episode step cap; <=0 means "use the env hint (env.max_steps)".
        # Hitting the cap ends the episode as FAILURE.
        self.declare_parameter("max_episode_steps", 0)
        self.republish_period = float(self.get_parameter("republish_period").value)
        self.loop = bool(self.get_parameter("loop").value)

        # env_factory may declare/read its own parameters via `self`.
        self.env = env_factory(self)
        if self.env.contract is not contract:
            raise ValueError("env_factory returned an env bound to a different contract")
        self.env.set_frame_callback(self.publish_intermediate_frames)

        param_max_steps = int(self.get_parameter("max_episode_steps").value)
        env_hint = getattr(self.env, "max_steps", None)
        if param_max_steps > 0:
            self.max_episode_steps = param_max_steps
        elif env_hint:
            self.max_episode_steps = int(env_hint)
        else:
            self.max_episode_steps = 0

        self.state_pub = self.create_publisher(Float32MultiArray, contract.state_topic, 10)
        self.image_pubs = {cam: self.create_publisher(Image, contract.camera_topic(cam), 10) for cam in contract.cameras}
        self.success_pub = self.create_publisher(Bool, contract.success_topic, 10)
        self.observation_ready_pub = self.create_publisher(Int32, contract.observation_ready_topic, 10)
        self.task_pub = self.create_publisher(String, contract.task_topic, 10)
        self.episode_pub = self.create_publisher(Int32, contract.episode_topic, 10)
        self.status_pub = self.create_publisher(String, contract.status_topic, 10)
        self.action_sub = self.create_subscription(Float32MultiArray, contract.action_topic, self.on_action, 10)
        self.control_sub = self.create_subscription(String, contract.control_topic, self.on_control, 10)

        # `step_index` is a monotonic global observation counter (never reset), so the
        # policy's "process only newer observations" logic keeps working across episodes.
        self.step_index = 0
        # `episode_step` counts steps within the current episode (drives the step cap).
        self.episode_step = 0
        self.episode_index = 0
        self.success = False
        self.state = READY
        self.history = []  # [{episode, task, config, seed, outcome, steps}]
        self._in_env_step = False

        self.obs = self.env.reset()
        self.env.validate(self.obs)

        self.get_logger().info(
            f"[{contract.name}] cameras={list(contract.cameras)} "
            f"action_dim={contract.action_dim} state_dim={contract.state_dim}; "
            f"loop={self.loop} max_episode_steps={self.max_episode_steps or 'unlimited'}; "
            f"control on {contract.control_topic}, status on {contract.status_topic}"
        )
        self.timer = self.create_timer(self.republish_period, self.republish)
        if bool(self.get_parameter("autostart").value):
            self.state = RUNNING
        self.publish_observation()
        self.publish_status()

    # ------------------------------------------------------------- publishing
    def republish(self):
        self.publish_observation()
        self.publish_status()

    def publish_observation(self):
        stamp = self.get_clock().now().to_msg()
        for cam, pub in self.image_pubs.items():
            image = self.obs.images.get(cam)
            if image is not None:
                pub.publish(rgb_to_image_msg(image, stamp, cam))

        state_msg = Float32MultiArray()
        state_msg.data = np.asarray(self.obs.state, dtype=np.float32).reshape(-1).tolist()
        self.state_pub.publish(state_msg)

        task_msg = String()
        task_msg.data = self.env.task_description or ""
        self.task_pub.publish(task_msg)

        success_msg = Bool()
        success_msg.data = self.success
        self.success_pub.publish(success_msg)

        episode_msg = Int32()
        episode_msg.data = self.episode_index
        self.episode_pub.publish(episode_msg)

        # Only advertise the observation to the policy while running: the policy
        # ignores indices it has already processed, so a paused/finished node that
        # never advances `step_index` keeps the policy quiet.
        if self.state == RUNNING:
            ready_msg = Int32()
            ready_msg.data = self.step_index
            self.observation_ready_pub.publish(ready_msg)

    def publish_intermediate_frames(self, images):
        """Publish viewer-only frames rendered mid-action (no observation_ready)."""
        stamp = self.get_clock().now().to_msg()
        for cam, image in images.items():
            pub = self.image_pubs.get(cam)
            if pub is not None and image is not None:
                pub.publish(rgb_to_image_msg(image, stamp, cam))

    def build_status(self) -> dict:
        succ = sum(1 for h in self.history if h["outcome"] == "success")
        return {
            "env": self.contract.name,
            "state": self.state,
            "task_name": getattr(self.env, "task_name", self.contract.name),
            "task_config": getattr(self.env, "task_config", ""),
            "seed": getattr(self.env, "seed", None),
            "instruction": self.env.task_description or "",
            "episode": self.episode_index,
            "episode_step": self.episode_step,
            "max_episode_steps": self.max_episode_steps,
            "loop": self.loop,
            "history": self.history[-50:],
            "stats": {"episodes": len(self.history), "successes": succ},
            "available_tasks": self.env.list_tasks(),
            "available_task_configs": self.env.list_task_configs(),
            "supports_task_switch": self.env.supports_task_switch,
            "cameras": list(self.contract.cameras),
            "policy_cameras": list(self.contract.policy_input_cameras),
            "timestamp": time.time(),
        }

    def publish_status(self):
        msg = String()
        msg.data = json.dumps(self.build_status())
        self.status_pub.publish(msg)

    # ---------------------------------------------------------------- actions
    def on_action(self, msg):
        if self.state != RUNNING:
            return

        action = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if action.size != self.contract.action_dim:
            self.get_logger().error(f"expected action length {self.contract.action_dim}, got {action.size}")
            return

        self._in_env_step = True
        try:
            self.obs, success = self.env.step(action)
        finally:
            self._in_env_step = False
        self.step_index += 1
        self.episode_step += 1
        self.success = bool(success)

        capped = self.max_episode_steps > 0 and self.episode_step >= self.max_episode_steps
        if self.success or capped:
            self._finish_episode("success" if self.success else "failure")
            return

        self.publish_observation()

    def _finish_episode(self, outcome: str):
        self.state = SUCCESS if outcome == "success" else FAILURE
        self.history.append(
            {
                "episode": self.episode_index,
                "task": getattr(self.env, "task_name", self.contract.name),
                "config": getattr(self.env, "task_config", ""),
                "seed": getattr(self.env, "seed", None),
                "outcome": outcome,
                "steps": self.episode_step,
            }
        )
        self.get_logger().info(f"episode {self.episode_index} ended [{outcome.upper()}] after {self.episode_step} steps (global step {self.step_index})")
        self.publish_observation()
        self.publish_status()
        if self.loop:
            self._start_next_episode()

    def _start_next_episode(self):
        self.state = SWITCHING
        self.publish_status()
        try:
            self.obs = self.env.new_episode()
            self.env.validate(self.obs)
        except Exception as exc:
            self.get_logger().error(f"failed to start next episode: {exc}")
            self.state = FAILURE
            self.publish_status()
            raise
        self._begin_episode()

    def _begin_episode(self):
        self.episode_index += 1
        self.episode_step = 0
        # Keep the global observation counter strictly increasing so the policy always
        # sees the new episode's first frame as "newer" than anything it processed.
        self.step_index += 1
        self.success = False
        self.state = RUNNING
        self._refresh_max_steps()
        self.get_logger().info(f"episode {self.episode_index} started (global step {self.step_index}): {self.env.task_description!r}")
        self.publish_observation()
        self.publish_status()

    def _refresh_max_steps(self):
        param_max_steps = int(self.get_parameter("max_episode_steps").value)
        if param_max_steps > 0:
            return
        env_hint = getattr(self.env, "max_steps", None)
        if env_hint:
            self.max_episode_steps = int(env_hint)

    # ---------------------------------------------------------------- control
    def on_control(self, msg):
        try:
            command = json.loads(msg.data)
            cmd = str(command.get("cmd", "")).strip().lower()
        except Exception as exc:
            self.get_logger().error(f"bad control message {msg.data!r}: {exc}")
            return
        self.get_logger().info(f"control command: {command}")

        try:
            if cmd == "start":
                self._cmd_start()
            elif cmd == "pause":
                self._cmd_pause()
            elif cmd == "resume":
                self._cmd_resume()
            elif cmd == "restart":
                self._cmd_restart()
            elif cmd == "set_task":
                self._cmd_set_task(command)
            else:
                self.get_logger().error(f"unknown control command: {cmd!r}")
        except Exception as exc:
            self.get_logger().error(f"control command {cmd!r} failed: {exc}")
            self.publish_status()

    def _cmd_start(self):
        if self.state == READY:
            # First run on the freshly built episode.
            self.state = RUNNING
            self.step_index += 1
            self.publish_observation()
            self.publish_status()
        elif self.state == PAUSED:
            self._cmd_resume()
        elif self.state in FINISHED_STATES:
            # The current scene is spent; start a fresh episode.
            self._start_next_episode()
        # RUNNING: no-op

    def _cmd_pause(self):
        if self.state != RUNNING:
            return
        self.state = PAUSED
        self.publish_status()

    def _cmd_resume(self):
        if self.state != PAUSED:
            return
        self.state = RUNNING
        # Bump the counter: an action published for the pre-pause observation may
        # have been dropped, so re-advertise the current state as a new observation.
        self.step_index += 1
        self.publish_observation()
        self.publish_status()

    def _cmd_restart(self):
        self._start_next_episode()

    def _cmd_set_task(self, command):
        if not self.env.supports_task_switch:
            self.get_logger().error(f"env '{self.contract.name}' does not support task switching")
            return
        task_name = str(command.get("task_name", "")).strip()
        task_config = str(command.get("task_config", "")).strip()
        seed = command.get("seed", None)
        if not task_name:
            self.get_logger().error("set_task requires task_name")
            return
        self.state = SWITCHING
        self.publish_status()
        try:
            self.obs = self.env.set_task(task_name, task_config=task_config, seed=seed)
            self.env.validate(self.obs)
        except Exception as exc:
            self.get_logger().error(f"set_task({task_name!r}, {task_config!r}) failed: {exc}")
            self.state = FAILURE
            self.publish_status()
            return
        # New scene, fresh episode counters; wait in READY for an explicit start.
        self.episode_index += 1
        self.episode_step = 0
        self.step_index += 1
        self.success = False
        self.state = READY
        self._refresh_max_steps()
        self.get_logger().info(f"switched to task {task_name!r} config {task_config!r}: {self.env.task_description!r}")
        self.publish_observation()
        self.publish_status()

    def destroy_node(self):
        try:
            if hasattr(self, "env"):
                self.env.close()
        finally:
            super().destroy_node()


def run_simulator_node(
    contract: EnvContract,
    env_factory: Callable[["SimulatorNode"], BaseSimEnv],
    *,
    node_name: str,
    args=None,
):
    rclpy.init(args=args)
    node = SimulatorNode(contract, env_factory, node_name=node_name)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
