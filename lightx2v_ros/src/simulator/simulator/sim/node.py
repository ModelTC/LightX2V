"""Generic, contract-driven simulator ROS node.

`SimulatorNode` is environment-agnostic: it derives every topic name and the
action/state dimensions from the `EnvContract` and drives any `BaseSimEnv`
implementation. An `env_factory(node) -> BaseSimEnv` callback lets each concrete
environment declare its own ROS parameters on the node before construction.
"""

from typing import Callable

import numpy as np
import rclpy
from common.contract import EnvContract
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Int32, String

from .base_env import BaseSimEnv


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
        self.declare_parameter("idle_republish_period", 2.0)
        # Continuous-eval loop: when true, the node automatically starts a fresh
        # episode after each success (or step cap) instead of stopping.
        self.declare_parameter("loop", False)
        # Per-episode step cap; <=0 means "use the env hint (env.max_steps) or run
        # until success". Failed episodes rely on this cap to eventually loop.
        self.declare_parameter("max_episode_steps", 0)
        self.republish_period = float(self.get_parameter("republish_period").value)
        self.idle_republish_period = float(self.get_parameter("idle_republish_period").value)
        self.loop = bool(self.get_parameter("loop").value)

        # env_factory may declare/read its own parameters via `self`.
        self.env = env_factory(self)
        if self.env.contract is not contract:
            raise ValueError("env_factory returned an env bound to a different contract")

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
        self.action_sub = self.create_subscription(Float32MultiArray, contract.action_topic, self.on_action, 10)

        # `step_index` is a monotonic global observation counter (never reset), so the
        # policy's "process only newer observations" logic keeps working across episodes.
        self.step_index = 0
        # `episode_step` counts steps within the current episode (drives the step cap).
        self.episode_step = 0
        self.episode_index = 0
        self.success = False
        self._slowed = False

        self.obs = self.env.reset()
        self.env.validate(self.obs)

        self.get_logger().info(
            f"[{contract.name}] cameras={list(contract.cameras)} "
            f"action_dim={contract.action_dim} state_dim={contract.state_dim}; "
            f"loop={self.loop} max_episode_steps={self.max_episode_steps or 'unlimited'}; "
            f"listening for actions on {contract.action_topic}"
        )
        self.timer = self.create_timer(self.republish_period, self.republish)
        self.publish_observation()

    def republish(self):
        self.publish_observation()

    def on_action(self, msg):
        if self.success:
            # In loop mode `success` is reset to False synchronously in
            # `_start_next_episode`, so this only drops late actions that raced the
            # episode boundary; in single-episode mode it stops the rollout.
            if not self.loop:
                self.get_logger().warning("episode already succeeded; ignoring action")
            return

        action = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if action.size != self.contract.action_dim:
            self.get_logger().error(f"expected action length {self.contract.action_dim}, got {action.size}")
            return

        self.obs, success = self.env.step(action)
        self.step_index += 1
        self.episode_step += 1
        self.success = bool(success)

        capped = self.max_episode_steps > 0 and self.episode_step >= self.max_episode_steps
        if self.loop and (self.success or capped):
            outcome = "SUCCESS" if self.success else f"step cap ({self.max_episode_steps})"
            self.get_logger().info(f"episode {self.episode_index} ended [{outcome}] after {self.episode_step} steps (global step {self.step_index}); starting next episode...")
            # Emit the final frame (success flag reflects the outcome) before rebuilding.
            self.publish_observation()
            self._start_next_episode()
            return

        self.publish_observation()

        if self.success and not self.loop:
            self.get_logger().info(f"episode succeeded at step {self.step_index}")
            self._slow_down_timer()

    def _start_next_episode(self):
        try:
            self.obs = self.env.new_episode()
            self.env.validate(self.obs)
        except Exception as exc:
            self.get_logger().error(f"failed to start next episode: {exc}")
            raise
        self.episode_index += 1
        self.episode_step = 0
        # Keep the global observation counter strictly increasing so the policy always
        # sees the new episode's first frame as "newer" than anything it processed.
        self.step_index += 1
        self.success = False
        self.get_logger().info(f"episode {self.episode_index} started (global step {self.step_index}): {self.env.task_description!r}")
        self.publish_observation()

    def _slow_down_timer(self):
        # After success the env stops stepping. Keep republishing the final frame at a
        # low rate so the web viewer keeps showing the last image instead of going blank.
        if self._slowed:
            return
        self._slowed = True
        try:
            self.timer.cancel()
        except Exception:
            pass
        self.timer = self.create_timer(self.idle_republish_period, self.republish)

    def publish_observation(self):
        stamp = self.get_clock().now().to_msg()
        for cam, pub in self.image_pubs.items():
            pub.publish(rgb_to_image_msg(self.obs.images[cam], stamp, cam))

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

        ready_msg = Int32()
        ready_msg.data = self.step_index
        self.observation_ready_pub.publish(ready_msg)

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
