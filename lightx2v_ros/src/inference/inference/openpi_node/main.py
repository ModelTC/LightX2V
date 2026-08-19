import numpy as np
import rclpy
from common.contract import get_contract
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Int32, String

from lightx2v.models.runners.openpi.openpi_runner import OpenPIPolicy
from lightx2v.utils.set_config import auto_calc_config, get_default_config


class OpenPINode(Node):
    """Thin ROS bridge between the shared LIBERO simulator and OpenPI."""

    def __init__(self):
        super().__init__("openpi_node")

        self.declare_parameter("env", "libero")
        self.declare_parameter("config_json", "")
        self.declare_parameter("model_path", "")
        self.declare_parameter("seed", 7)
        self.declare_parameter("actions_per_plan", -1)
        self.declare_parameter("num_steps_wait", -1)

        env = str(self.get_parameter("env").value).strip().lower()
        self.contract = get_contract(env)
        if self.contract.name != "libero":
            raise ValueError("OpenPI ROS integration currently supports only LIBERO.")

        self.get_logger().info("[libero] loading OpenPI policy")
        self.policy_config = self.build_policy_config()
        self.policy = OpenPIPolicy.from_config(self.policy_config)
        self.get_logger().info("[libero] OpenPI policy loaded")

        self.images = {camera: None for camera in self.contract.policy_input_cameras}
        self.state = None
        self.task_description = None
        self.success = False
        self.episode_index = 0
        self.episode_observation_count = 0
        self.last_processed_observation = -1

        configured_wait = int(self.get_parameter("num_steps_wait").value)
        self.num_steps_wait = configured_wait if configured_wait >= 0 else int(self.policy_config.get("num_steps_wait", 10))
        self.dummy_action = np.zeros(self.contract.action_dim, dtype=np.float32)
        self.dummy_action[-1] = -1.0

        self.action_pub = self.create_publisher(Float32MultiArray, self.contract.action_topic, 10)
        self._camera_subs = []
        for camera in self.contract.policy_input_cameras:
            self._camera_subs.append(self.create_subscription(Image, self.contract.camera_topic(camera), self._make_image_cb(camera), 10))
        self.create_subscription(Float32MultiArray, self.contract.state_topic, self.on_state, 10)
        self.create_subscription(String, self.contract.task_topic, self.on_task, 10)
        self.create_subscription(Bool, self.contract.success_topic, self.on_success, 10)
        self.create_subscription(Int32, self.contract.episode_topic, self.on_episode, 10)
        self.create_subscription(Int32, self.contract.observation_ready_topic, self.on_observation_ready, 10)

        self.get_logger().info(
            "[libero] openpi_node ready: "
            f"cameras={list(self.contract.policy_input_cameras)} action_dim={self.contract.action_dim} "
            f"state_dim={self.contract.state_dim} actions_per_plan={self.policy_config.get('actions_per_plan', 5)} "
            f"num_steps_wait={self.num_steps_wait}"
        )

    def build_policy_config(self):
        config_json = str(self.get_parameter("config_json").value).strip()
        if not config_json:
            raise ValueError("OpenPI ROS node requires `config_json`.")
        model_path = str(self.get_parameter("model_path").value).strip()
        if not model_path:
            raise ValueError("OpenPI ROS node requires `model_path`.")

        seed = int(self.get_parameter("seed").value)
        config = get_default_config()
        config.update(
            {
                "model_cls": "openpi",
                "task": "i2va",
                "model_path": model_path,
                "config_json": config_json,
                "seed": seed,
            }
        )
        config = auto_calc_config(config)

        # Explicit ROS parameters are runtime overrides and must win over JSON.
        config["model_cls"] = "openpi"
        config["task"] = "i2va"
        config["model_path"] = model_path
        config["seed"] = seed
        actions_per_plan = int(self.get_parameter("actions_per_plan").value)
        if actions_per_plan > 0:
            config["actions_per_plan"] = actions_per_plan

        expected_dims = {
            "state_dim": self.contract.state_dim,
            "output_action_dim": self.contract.action_dim,
        }
        for key, expected in expected_dims.items():
            if key in config and int(config[key]) != expected:
                raise ValueError(f"OpenPI config `{key}`={config[key]} does not match LIBERO contract ({expected}).")
        return config

    def _make_image_cb(self, camera):
        def _callback(msg):
            # The shared LIBERO simulator already applies the official OpenPI
            # 180-degree observation rotation. Do not flip the image again here.
            self.images[camera] = image_msg_to_rgb(msg)

        return _callback

    def on_state(self, msg):
        state = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if state.size != self.contract.state_dim:
            self.get_logger().error(f"expected state length {self.contract.state_dim}, got {state.size}")
            return
        self.state = state

    def on_task(self, msg):
        self.task_description = str(msg.data).strip()

    def on_success(self, msg):
        self.success = bool(msg.data)

    def on_episode(self, msg):
        episode = int(msg.data)
        if episode == self.episode_index:
            return
        self.episode_index = episode
        self.success = False
        self.episode_observation_count = 0
        self.last_processed_observation = -1
        self.policy.reset()
        self.get_logger().info(f"new episode {episode}; cleared OpenPI action queue")

    def missing_inputs(self):
        missing = [camera for camera in self.contract.policy_input_cameras if self.images.get(camera) is None]
        if self.state is None:
            missing.append("state")
        if not self.task_description:
            missing.append("task_description")
        return missing

    def on_observation_ready(self, msg):
        observation_index = int(msg.data)
        if observation_index <= self.last_processed_observation:
            return
        if self.success:
            self.last_processed_observation = observation_index
            return

        missing = self.missing_inputs()
        if missing:
            self.get_logger().warning(f"observation {observation_index} waiting for: {missing}")
            return

        if self.episode_observation_count < self.num_steps_wait:
            action = self.dummy_action.copy()
            self.get_logger().info(
                f"observation {observation_index}: publishing LIBERO warmup action "
                f"({self.episode_observation_count + 1}/{self.num_steps_wait})"
            )
        else:
            self.get_logger().info(f"observation {observation_index}: running/consuming OpenPI action chunk")
            action = self.policy.next_action(
                images={camera: self.images[camera] for camera in self.contract.policy_input_cameras},
                state=self.state,
                task_description=self.task_description,
            )

        self.publish_action(action)
        self.episode_observation_count += 1
        self.last_processed_observation = observation_index

    def publish_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.contract.action_dim:
            raise ValueError(f"expected action length {self.contract.action_dim}, got {action.size}")
        if not np.isfinite(action).all():
            raise ValueError("OpenPI produced a non-finite action.")
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)

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
    node = OpenPINode()
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
