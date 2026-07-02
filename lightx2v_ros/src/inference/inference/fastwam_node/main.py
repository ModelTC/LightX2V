import numpy as np
import rclpy
from common.contract import get_contract
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Int32, String

from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
from lightx2v.utils.set_config import auto_calc_config, get_default_config


class FastWAMNode(Node):
    def __init__(self):
        super().__init__("fastwam_node")

        self.declare_parameter("env", "libero")
        self.declare_parameter("config_json", "")
        self.declare_parameter("model_path", "")
        self.declare_parameter("num_steps_wait", -1)

        env = str(self.get_parameter("env").value).strip().lower()
        self.contract = get_contract(env)

        self.get_logger().info(f"[{self.contract.name}] loading FastWAM policy")
        self.policy_config = self.build_policy_config()
        self.policy = FastWAMPolicy.from_config(self.policy_config)
        self.get_logger().info(f"[{self.contract.name}] FastWAM policy loaded")

        self.images = {cam: None for cam in self.contract.policy_input_cameras}
        self.state = None
        self.task_description = None
        self.success = False
        self.episode_index = 0
        self.last_processed_observation = -1

        ns_wait = int(self.get_parameter("num_steps_wait").value)
        self.num_steps_wait = ns_wait if ns_wait >= 0 else int(self.policy_config.get("num_steps_wait", 0))
        self.dummy_action = self._build_dummy_action()

        self.action_pub = self.create_publisher(Float32MultiArray, self.contract.action_topic, 10)
        self._camera_subs = []
        for cam in self.contract.policy_input_cameras:
            self._camera_subs.append(self.create_subscription(Image, self.contract.camera_topic(cam), self._make_image_cb(cam), 10))
        self.create_subscription(Float32MultiArray, self.contract.state_topic, self.on_state, 10)
        self.create_subscription(String, self.contract.task_topic, self.on_task, 10)
        self.create_subscription(Bool, self.contract.success_topic, self.on_success, 10)
        self.create_subscription(Int32, self.contract.episode_topic, self.on_episode, 10)
        self.create_subscription(Int32, self.contract.observation_ready_topic, self.on_observation_ready, 10)

        self.get_logger().info(
            f"[{self.contract.name}] fastwam_node ready: input_cameras={list(self.contract.policy_input_cameras)} "
            f"action_dim={self.contract.action_dim} state_dim={self.contract.state_dim} num_steps_wait={self.num_steps_wait}"
        )

    def build_policy_config(self):
        config_json = str(self.get_parameter("config_json").value).strip()
        if not config_json:
            raise ValueError("FastWAM ROS node requires `config_json`.")
        model_path = str(self.get_parameter("model_path").value).strip()
        if not model_path:
            raise ValueError("FastWAM ROS node requires `model_path`.")
        config = get_default_config()
        config.update(
            {
                "model_cls": "fastwam",
                "task": "i2va",
                "model_path": model_path,
                "config_json": config_json,
            }
        )
        config = auto_calc_config(config)

        # The config_json is authoritative for policy params; warn loudly on any
        # mismatch with the environment contract so dimension bugs surface early.
        for key, expected in (("action_dim", self.contract.action_dim), ("robot_state_dim", self.contract.state_dim)):
            actual = int(config.get(key, expected))
            if actual != expected:
                self.get_logger().warning(f"config `{key}`={actual} disagrees with env '{self.contract.name}' contract ({expected})")
        return config

    def _build_dummy_action(self):
        action = np.zeros(self.contract.action_dim, dtype=np.float32)
        if self.contract.gripper_postprocess:
            # LIBERO warmup keeps the gripper open while the scene settles.
            action[-1] = -1.0
        return action

    def _make_image_cb(self, camera):
        def _cb(msg):
            self.images[camera] = image_msg_to_rgb(msg)

        return _cb

    def on_state(self, msg):
        state = np.asarray(msg.data, dtype=np.float32)
        if state.size != self.contract.state_dim:
            self.get_logger().error(f"expected {self.contract.state_topic} length {self.contract.state_dim}, got {state.size}")
            return
        self.state = state

    def on_task(self, msg):
        self.task_description = msg.data

    def on_success(self, msg):
        self.success = bool(msg.data)

    def on_episode(self, msg):
        episode = int(msg.data)
        if episode == self.episode_index:
            return
        # Simulator looped into a fresh episode: drop the queued action chunk and any
        # stale success/observation bookkeeping so we start clean on the new rollout.
        self.episode_index = episode
        self.success = False
        self.last_processed_observation = -1
        self.policy.reset()
        self.get_logger().info(f"new episode {episode}; policy state reset for fresh rollout")

    def on_observation_ready(self, msg):
        observation_index = int(msg.data)
        if observation_index <= self.last_processed_observation:
            return
        if self.success:
            self.get_logger().info(f"environment succeeded at observation {observation_index}; stop publishing actions")
            self.last_processed_observation = observation_index
            return

        missing = self.missing_inputs()
        if missing:
            self.get_logger().warning(f"waiting for inputs before observation {observation_index}: {missing}")
            return

        if observation_index < self.num_steps_wait:
            action = self.dummy_action.copy()
            self.get_logger().info(f"observation {observation_index}: publishing warmup dummy action")
        else:
            self.get_logger().info(f"observation {observation_index}: running FastWAM inference/action queue")
            action = self.policy.next_action(
                images={cam: self.images[cam] for cam in self.contract.policy_input_cameras},
                state=self.state,
                task_description=self.task_description,
            )

        self.publish_action(action)
        self.last_processed_observation = observation_index

    def missing_inputs(self):
        missing = [cam for cam in self.contract.policy_input_cameras if self.images.get(cam) is None]
        if self.state is None:
            missing.append("state")
        if not self.task_description:
            missing.append("task_description")
        return missing

    def publish_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.contract.action_dim:
            raise ValueError(f"expected action length {self.contract.action_dim}, got {action.size}")
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)
        self.get_logger().info(f"published action: {msg.data}")

    def destroy_node(self):
        if hasattr(self, "policy"):
            self.policy.close()
        super().destroy_node()


def image_msg_to_rgb(msg):
    encoding = msg.encoding.lower()
    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")

    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = row[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if encoding == "bgr8":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image.copy())


def main(args=None):
    rclpy.init(args=args)
    node = FastWAMNode()
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


if __name__ == "__main__":
    main()
