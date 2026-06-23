import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

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
        self.declare_parameter("pose_topic", "/libero/ee_pose")
        self.declare_parameter("agentview_topic", "/libero/agentview/image_raw")
        self.declare_parameter("wrist_topic", "/libero/wrist/image_raw")
        self.declare_parameter("frame_id", "libero_world")

        self.pose_pub = self.create_publisher(PoseStamped, self.get_parameter("pose_topic").value, 10)
        self.agentview_pub = self.create_publisher(Image, self.get_parameter("agentview_topic").value, 10)
        self.wrist_pub = self.create_publisher(Image, self.get_parameter("wrist_topic").value, 10)
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

        self.get_logger().info(f"listening for actions on {self.get_parameter('action_topic').value}")

    def on_action(self, msg):
        action = np.asarray(msg.data, dtype=np.float32)
        if action.shape != (7,):
            self.get_logger().error(f"expected action length 7, got {action.size}")
            return

        self.get_logger().info(f"received action: {action.tolist()}")
        self.observer.step(action)
        obs = self.observer.obs
        stamp = self.get_clock().now().to_msg()

        self.pose_pub.publish(self.make_pose_msg(obs, stamp))
        self.agentview_pub.publish(self.make_image_msg(obs["agentview_image"][::-1, ::-1], stamp, "agentview"))
        self.wrist_pub.publish(self.make_image_msg(obs["robot0_eye_in_hand_image"][::-1, ::-1], stamp, "wrist"))

    def make_pose_msg(self, obs, stamp):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.get_parameter("frame_id").value

        pos = obs["robot0_eef_pos"]
        quat = obs["robot0_eef_quat"]
        pose.pose.position.x = float(pos[0])
        pose.pose.position.y = float(pos[1])
        pose.pose.position.z = float(pos[2])
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])
        return pose

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
