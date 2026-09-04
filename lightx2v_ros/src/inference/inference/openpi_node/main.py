import json
from collections import deque

import numpy as np
import rclpy
from common.contract import get_contract
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Float64MultiArray, MultiArrayDimension, String

from lightx2v.models.runners.openpi.openpi_runner import OpenPIPolicy
from lightx2v.utils.set_config import auto_calc_config, get_default_config


class OpenPINode(Node):
    def __init__(self):
        super().__init__("openpi_node")

        self.declare_parameter("config_json", "")
        self.declare_parameter("model_path", "")
        self.declare_parameter("seed", 0)
        self.declare_parameter("actions_per_plan", 5)
        self.declare_parameter("numeric_precision", "float64")

        self.contract = get_contract("libero")

        precision = str(self.get_parameter("numeric_precision").value).strip().lower()
        numeric_types = {
            "float32": (Float32MultiArray, np.float32),
            "float64": (Float64MultiArray, np.float64),
        }
        if precision not in numeric_types:
            raise ValueError("numeric_precision must be 'float32' or 'float64'")
        self.numeric_message_type, self.numeric_dtype = numeric_types[precision]

        config = self._policy_config()
        self.actions_per_plan = int(self.get_parameter("actions_per_plan").value)
        action_horizon = int(config["action_horizon"])
        if not 1 <= self.actions_per_plan <= action_horizon:
            raise ValueError(f"actions_per_plan must be in [1, {action_horizon}]")

        self.get_logger().info("loading OpenPI policy")
        self.policy = OpenPIPolicy(config)
        self.get_logger().info("OpenPI policy loaded")

        self.images = {camera: None for camera in self.contract.policy_input_cameras}
        self.state = None
        self.task_description = ""
        self.episode_index = None
        self.plan_epoch = None
        self.pending_observation = None
        self.last_processed_observation = None
        self.pending_actions = deque()

        self.action_pub = self.create_publisher(self.numeric_message_type, self.contract.action_topic, 10)
        self._camera_subscriptions = [self.create_subscription(Image, self.contract.camera_topic(camera), self._image_callback(camera), 10) for camera in self.contract.policy_input_cameras]
        self.state_sub = self.create_subscription(self.numeric_message_type, self.contract.state_topic, self._on_state, 10)
        self.context_sub = self.create_subscription(String, self.contract.observation_context_topic, self._on_context, 10)

        self.get_logger().info(f"OpenPI ready on {self.contract.namespace}: horizon={action_horizon}, actions_per_plan={self.actions_per_plan}, precision={precision}")

    def _policy_config(self):
        config_json = str(self.get_parameter("config_json").value).strip()
        model_path = str(self.get_parameter("model_path").value).strip()
        if not config_json or not model_path:
            raise ValueError("OpenPI requires model_path and config_json")
        seed = int(self.get_parameter("seed").value)

        config = get_default_config()
        config.update(
            {
                "model_cls": "openpi",
                "task": "i2va",
                "model_path": model_path,
                "config_json": config_json,
            }
        )
        config = auto_calc_config(config)
        # ROS parameters take precedence over values loaded from the model JSON.
        config["model_path"] = model_path
        config["seed"] = seed

        expected = {"state_dim": self.contract.state_dim, "output_action_dim": self.contract.action_dim}
        for name, dimension in expected.items():
            if int(config[name]) != dimension:
                raise ValueError(f"OpenPI {name}={config[name]} does not match the LIBERO contract ({dimension})")
        return config

    def _image_callback(self, camera):
        def callback(msg):
            identity = observation_identity(msg.header.frame_id)
            if identity is None:
                return
            episode, observation = identity
            self.images[camera] = (episode, observation, image_msg_to_rgb(msg))
            self._try_process_observation()

        return callback

    def _on_state(self, msg):
        if not msg.layout.dim:
            return
        identity = state_identity(msg.layout.dim[0].label)
        if identity is None:
            return
        episode, observation = identity
        state = np.asarray(msg.data, dtype=self.numeric_dtype).reshape(-1)
        if state.size != self.contract.state_dim:
            self.get_logger().error(f"expected state length {self.contract.state_dim}, got {state.size}")
            return
        self.state = (episode, observation, state)
        self._try_process_observation()

    def _on_context(self, msg):
        try:
            context = json.loads(msg.data)
            episode = int(context["episode"])
            observation = int(context["observation"])
            plan_epoch = int(context["plan_epoch"])
            task_description = str(context["task_description"]).strip()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"invalid observation context: {exc}")
            return

        if episode != self.episode_index:
            self.episode_index = episode
            self.pending_actions.clear()
            self.last_processed_observation = None
            self.get_logger().info(f"episode {episode}: cleared ROS action queue")
        if plan_epoch != self.plan_epoch:
            if self.plan_epoch is not None:
                self.pending_actions.clear()
                self.get_logger().info(f"plan epoch {plan_epoch}: cleared ROS action queue")
            self.plan_epoch = plan_epoch
        identity = (episode, observation)
        if identity == self.last_processed_observation:
            return
        self.task_description = task_description
        self.pending_observation = identity
        self._try_process_observation()

    def _try_process_observation(self):
        if self.pending_observation is None or not self.task_description:
            return
        identity = self.pending_observation
        if self.state is None or self.state[:2] != identity:
            return
        if any(value is None or value[:2] != identity for value in self.images.values()):
            return

        episode, observation = identity
        if not self.pending_actions:
            chunk = self.policy.predict_action_chunk(
                images={camera: self.images[camera][2] for camera in self.contract.policy_input_cameras},
                state=self.state[2],
                task_description=self.task_description,
            )
            self.pending_actions.extend(action.copy() for action in chunk[: self.actions_per_plan])
            self.get_logger().info(f"episode {episode} observation {observation}: predicted {len(chunk)} actions")

        action = self.pending_actions.popleft()
        self.last_processed_observation = identity
        self.pending_observation = None
        self._publish_action(action, episode, observation)

    def _publish_action(self, action, episode, observation):
        action = np.asarray(action, dtype=self.numeric_dtype).reshape(-1)
        if action.size != self.contract.action_dim:
            raise ValueError(f"expected action length {self.contract.action_dim}, got {action.size}")
        if not np.isfinite(action).all():
            raise ValueError("OpenPI produced a non-finite action")
        msg = self.numeric_message_type()
        msg.layout.dim = [
            MultiArrayDimension(
                label=f"episode={episode};observation={observation};plan_epoch={self.plan_epoch}",
                size=action.size,
                stride=action.size,
            )
        ]
        msg.data = action.tolist()
        self.action_pub.publish(msg)


def observation_identity(frame_id):
    parts = str(frame_id).rsplit("|", 2)
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def state_identity(label):
    fields = {}
    for item in str(label).split(";"):
        name, separator, value = item.partition("=")
        if separator:
            fields[name] = value
    try:
        return int(fields["episode"]), int(fields["observation"])
    except (KeyError, ValueError):
        return None


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
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
