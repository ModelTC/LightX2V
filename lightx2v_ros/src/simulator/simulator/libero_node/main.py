import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Int32, String

from .observer import LiberoActionObserver, default_libero_root


class LiberoNode(Node):
    def __init__(self):
        super().__init__("libero_node")

        self.declare_parameter("libero_root", str(default_libero_root()))
        self.declare_parameter("benchmark", "libero_spatial")
        self.declare_parameter("task_id", 0)
        self.declare_parameter("init_state_id", 0)
        self.declare_parameter("image_size", 224)
        self.declare_parameter("seed", 0)
        self.declare_parameter("action_topic", "/libero/action")
        self.declare_parameter("state_topic", "/libero/state")
        self.declare_parameter("agentview_topic", "/libero/agentview/image_raw")
        self.declare_parameter("wrist_topic", "/libero/wrist/image_raw")
        self.declare_parameter("frontview_topic", "/libero/frontview/image_raw")
        self.declare_parameter("galleryview_topic", "/libero/galleryview/image_raw")
        self.declare_parameter("success_topic", "/libero/success")
        self.declare_parameter("observation_ready_topic", "/libero/observation_ready")
        self.declare_parameter("task_topic", "/libero/task_description")

        self.state_pub = self.create_publisher(Float32MultiArray, self.get_parameter("state_topic").value, 10)
        self.agentview_pub = self.create_publisher(Image, self.get_parameter("agentview_topic").value, 10)
        self.wrist_pub = self.create_publisher(Image, self.get_parameter("wrist_topic").value, 10)
        self.frontview_pub = self.create_publisher(Image, self.get_parameter("frontview_topic").value, 10)
        self.galleryview_pub = self.create_publisher(Image, self.get_parameter("galleryview_topic").value, 10)
        self.success_pub = self.create_publisher(Bool, self.get_parameter("success_topic").value, 10)
        self.observation_ready_pub = self.create_publisher(Int32, self.get_parameter("observation_ready_topic").value, 10)
        self.task_pub = self.create_publisher(String, self.get_parameter("task_topic").value, 10)
        self.action_sub = self.create_subscription(
            Float32MultiArray,
            self.get_parameter("action_topic").value,
            self.on_action,
            10,
        )

        self.observer = LiberoActionObserver(
            benchmark_name=self.get_parameter("benchmark").value,
            task_id=int(self.get_parameter("task_id").value),
            init_state_id=int(self.get_parameter("init_state_id").value),
            image_size=int(self.get_parameter("image_size").value),
            seed=int(self.get_parameter("seed").value),
            libero_root=self.get_parameter("libero_root").value,
        )
        self.step_index = 0
        self.success = False
        self.observation_timer = self.create_timer(1.0, self.republish_observation)

        self.get_logger().info(f"listening for actions on {self.get_parameter('action_topic').value}")
        self.publish_observation()

    def republish_observation(self):
        if self.success:
            self.observation_timer.cancel()
            return
        self.publish_observation()

    def on_action(self, msg):
        if self.success:
            self.get_logger().warning("episode already succeeded; ignoring action")
            return

        action = np.asarray(msg.data, dtype=np.float32)
        if action.shape != (7,):
            self.get_logger().error(f"expected action length 7, got {action.size}")
            return

        self.get_logger().info(f"received action: {action.tolist()}")
        _, _, success, _ = self.observer.step(action)
        self.step_index += 1
        self.success = bool(success)
        self.publish_observation()

        if self.success:
            self.get_logger().info(f"episode succeeded at step {self.step_index}")

    def publish_observation(self):
        obs = self.observer.obs
        stamp = self.get_clock().now().to_msg()

        self.state_pub.publish(self.make_state_msg(obs))
        self.agentview_pub.publish(self.make_image_msg(obs["agentview_image"][::-1, ::-1], stamp, "agentview"))
        self.wrist_pub.publish(self.make_image_msg(obs["robot0_eye_in_hand_image"][::-1, ::-1], stamp, "wrist"))
        self.frontview_pub.publish(self.make_image_msg(obs["frontview_image"][::-1, ::-1], stamp, "frontview"))
        self.galleryview_pub.publish(self.make_image_msg(obs["galleryview_image"][::-1, ::-1], stamp, "galleryview"))

        task_msg = String()
        task_msg.data = self.observer.task_description
        self.task_pub.publish(task_msg)

        success_msg = Bool()
        success_msg.data = self.success
        self.success_pub.publish(success_msg)

        observation_ready_msg = Int32()
        observation_ready_msg.data = self.step_index
        self.observation_ready_pub.publish(observation_ready_msg)

    def make_state_msg(self, obs):
        pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        axis_angle = quat_to_axis_angle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)

        msg = Float32MultiArray()
        msg.data = np.concatenate([pos, axis_angle, gripper]).astype(np.float32).tolist()
        return msg

    def make_image_msg(self, image, stamp, frame_id):
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

    def destroy_node(self):
        if hasattr(self, "observer"):
            self.observer.close()
        super().destroy_node()


def quat_to_axis_angle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = LiberoNode()
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
