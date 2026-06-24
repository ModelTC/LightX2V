import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Int32, String

from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
from lightx2v.utils.set_config import auto_calc_config, get_default_config

DEFAULT_DUMMY_ACTION = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
ACTION_TOPIC = "/libero/action"
AGENTVIEW_TOPIC = "/libero/agentview/image_raw"
WRIST_TOPIC = "/libero/wrist/image_raw"
STATE_TOPIC = "/libero/state"
OBSERVATION_READY_TOPIC = "/libero/observation_ready"
SUCCESS_TOPIC = "/libero/success"
TASK_TOPIC = "/libero/task_description"


class FastWAMNode(Node):
    def __init__(self):
        super().__init__("fastwam_node")

        self.declare_parameter("config_json", "")
        self.declare_parameter("model_path", "")

        self.get_logger().info("loading FastWAM policy")
        self.policy_config = self.build_policy_config()
        self.policy = FastWAMPolicy.from_config(self.policy_config)
        self.get_logger().info("FastWAM policy loaded")

        self.agentview = None
        self.wrist = None
        self.state = None
        self.task_description = None
        self.success = False
        self.last_processed_observation = -1
        self.num_steps_wait = int(self.policy_config.get("num_steps_wait", 30))

        self.action_pub = self.create_publisher(Float32MultiArray, ACTION_TOPIC, 10)
        self.create_subscription(
            Image,
            AGENTVIEW_TOPIC,
            self.on_agentview,
            10,
        )
        self.create_subscription(
            Image,
            WRIST_TOPIC,
            self.on_wrist,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            STATE_TOPIC,
            self.on_state,
            10,
        )
        self.create_subscription(
            String,
            TASK_TOPIC,
            self.on_task,
            10,
        )
        self.create_subscription(
            Bool,
            SUCCESS_TOPIC,
            self.on_success,
            10,
        )
        self.create_subscription(
            Int32,
            OBSERVATION_READY_TOPIC,
            self.on_observation_ready,
            10,
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
        return auto_calc_config(config)

    def on_agentview(self, msg):
        self.agentview = image_msg_to_rgb(msg)

    def on_wrist(self, msg):
        self.wrist = image_msg_to_rgb(msg)

    def on_state(self, msg):
        state = np.asarray(msg.data, dtype=np.float32)
        if state.shape != (8,):
            self.get_logger().error(f"expected /libero/state length 8, got {state.size}")
            return
        self.state = state

    def on_task(self, msg):
        self.task_description = msg.data

    def on_success(self, msg):
        self.success = bool(msg.data)

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
            action = DEFAULT_DUMMY_ACTION.copy()
            self.get_logger().info(f"observation {observation_index}: publishing warmup dummy action")
        else:
            self.get_logger().info(f"observation {observation_index}: running FastWAM inference/action queue")
            action = self.policy.next_action(
                agentview_rgb=self.agentview,
                wrist_rgb=self.wrist,
                state=self.state,
                task_description=self.task_description,
            )

        self.publish_action(action)
        self.last_processed_observation = observation_index

    def missing_inputs(self):
        missing = []
        if self.agentview is None:
            missing.append("agentview")
        if self.wrist is None:
            missing.append("wrist")
        if self.state is None:
            missing.append("state")
        if not self.task_description:
            missing.append("task_description")
        return missing

    def publish_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (7,):
            raise ValueError(f"expected action length 7, got {action.size}")
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
