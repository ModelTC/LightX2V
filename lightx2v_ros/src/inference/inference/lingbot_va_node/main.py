import numpy as np
import rclpy
from common.contract import get_contract
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Int32, String

from lightx2v.models.runners.wan.wan_lingbot_va_runner import LingbotVAPolicy
from lightx2v.utils.set_config import auto_calc_config, get_default_config


class LingbotVANode(Node):
    def __init__(self):
        super().__init__("lingbot_va_node")

        self.declare_parameter("env", "libero")
        self.declare_parameter("config_json", "")
        self.declare_parameter("model_path", "")
        self.declare_parameter("seed", 0)
        self.declare_parameter("num_steps_wait", -1)
        # The shared LIBERO simulator publishes images rotated 180 degrees for
        # FastWAM.  Released LingBot-VA evaluation data only flips vertically, so
        # undo the extra horizontal flip at this policy boundary by default.
        self.declare_parameter("undo_libero_horizontal_flip", True)

        env = str(self.get_parameter("env").value).strip().lower()
        self.contract = get_contract(env)
        if self.contract.name not in {"libero", "robotwin"}:
            raise ValueError("LingBot-VA ROS integration supports only LIBERO and RoboTwin.")
        self.seed = int(self.get_parameter("seed").value)
        self.undo_libero_horizontal_flip = bool(self.get_parameter("undo_libero_horizontal_flip").value)

        self.get_logger().info(f"[{self.contract.name}] loading LingBot-VA policy")
        self.policy_config = self.build_policy_config()
        self.policy = LingbotVAPolicy.from_config(self.policy_config, policy_profile=self.contract.policy_profile)
        self.get_logger().info(f"[{self.contract.name}] LingBot-VA policy loaded")

        expected_action_dim = 16 if self.contract.name == "robotwin" else self.contract.action_dim
        if self.policy.output_action_dim != expected_action_dim:
            raise ValueError(f"LingBot-VA config produces {self.policy.output_action_dim} actions for {self.contract.name}; expected {expected_action_dim}.")

        self.images = {cam: None for cam in self.contract.policy_input_cameras}
        self.state = None
        self.task_description = None
        self.success = False
        self.episode_index = 0
        self.last_processed_observation = -1

        configured_wait = int(self.get_parameter("num_steps_wait").value)
        default_wait = 5 if self.contract.name == "libero" else 0
        self.num_steps_wait = configured_wait if configured_wait >= 0 else int(self.policy_config.get("num_steps_wait", default_wait))

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
            f"[{self.contract.name}] lingbot_va_node ready: cameras={list(self.contract.policy_input_cameras)} "
            f"model_action_dim={self.policy.output_action_dim} keyframe_interval={self.policy.keyframe_interval} "
            f"num_steps_wait={self.num_steps_wait}"
        )

    def build_policy_config(self):
        config_json = str(self.get_parameter("config_json").value).strip()
        if not config_json:
            raise ValueError("LingBot-VA ROS node requires `config_json`.")
        model_path = str(self.get_parameter("model_path").value).strip()
        if not model_path:
            raise ValueError("LingBot-VA ROS node requires `model_path`.")
        config = get_default_config()
        config.update(
            {
                "model_cls": "lingbot_va",
                "task": "i2va",
                "model_path": model_path,
                "config_json": config_json,
                "seed": self.seed,
            }
        )
        return auto_calc_config(config)

    def _make_image_cb(self, camera):
        def _callback(msg):
            image = image_msg_to_rgb(msg)
            if self.contract.name == "libero" and self.undo_libero_horizontal_flip:
                image = np.ascontiguousarray(image[:, ::-1])
            self.images[camera] = image

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
        self.last_processed_observation = -1
        self.policy.reset()
        self.get_logger().info(f"new episode {episode}; LingBot-VA video/action/cache state reset")

    def missing_inputs(self):
        missing = [camera for camera in self.contract.policy_input_cameras if self.images.get(camera) is None]
        if not self.task_description:
            missing.append("task_description")
        if self.contract.name == "robotwin" and self.num_steps_wait > 0 and self.state is None:
            missing.append("state")
        return missing

    def _warmup_action(self):
        if self.contract.name == "robotwin":
            if self.state is None:
                raise RuntimeError("RoboTwin warmup requires the current 14-D qpos state.")
            return self.state.copy()
        # Matches the released LingBot-VA LIBERO evaluation's five zero steps.
        return np.zeros(self.contract.action_dim, dtype=np.float32)

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

        if observation_index < self.num_steps_wait:
            action = self._warmup_action()
            self.get_logger().info(f"observation {observation_index}: publishing warmup action")
        else:
            self.get_logger().info(f"observation {observation_index}: running/consuming LingBot-VA action chunk")
            action = self.policy.next_action(
                images={camera: self.images[camera] for camera in self.contract.policy_input_cameras},
                task_description=self.task_description,
                seed=self.seed,
            )

        self.publish_action(action)
        self.last_processed_observation = observation_index

    def publish_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        accepted = {self.contract.action_dim}
        if self.contract.name == "robotwin":
            accepted.add(16)
        if action.size not in accepted:
            raise ValueError(f"expected action length in {sorted(accepted)}, got {action.size}")
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
    node = LingbotVANode()
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
