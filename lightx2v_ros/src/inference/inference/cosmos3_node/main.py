import os

import numpy as np
import rclpy
import torch
import torch.distributed as dist
from common.contract import get_contract
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Int32, String

from lightx2v.models.runners.cosmos3.cosmos3_runner import Cosmos3Policy
from lightx2v.utils.set_config import auto_calc_config, get_default_config, set_parallel_config
from lightx2v.utils.utils import seed_all
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def image_msg_to_rgb(msg):
    encoding = msg.encoding.lower()
    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = row[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if encoding == "bgr8":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image.copy())


class Cosmos3Node(Node):
    """ROS bridge for a model-parallel Cosmos3 Policy-DROID runner.

    Rank 0 owns ROS I/O and broadcasts complete observations to the remaining
    torchrun ranks.  Every rank advances the same action queue, ensuring that
    collective model inference is entered in lockstep while only rank 0 sends
    robot commands to the simulator.
    """

    def __init__(self):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        super().__init__(f"cosmos3_node_rank_{local_rank}")

        self.declare_parameter("env", "robolab")
        self.declare_parameter("config_json", "")
        self.declare_parameter("model_path", "")
        self.declare_parameter("actions_per_plan", 32)
        self.declare_parameter("binarize_gripper", True)
        self.declare_parameter("prompt_format", "official_text")
        self.declare_parameter("seed", 0)

        env = str(self.get_parameter("env").value).strip().lower()
        self.contract = get_contract(env)
        if self.contract.name != "robolab":
            raise ValueError("Cosmos3-Nano-Policy-DROID ROS currently requires env='robolab'.")

        self.policy_config = self._build_policy_config()
        self._initialize_parallel()
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._broadcast_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', self.rank))}") if dist.is_initialized() else None

        seed_all(int(self.get_parameter("seed").value))
        self.get_logger().info(f"rank {self.rank}/{self.world_size}: loading Cosmos3 Policy-DROID")
        self.policy = Cosmos3Policy.from_config(
            self.policy_config,
            actions_per_plan=int(self.get_parameter("actions_per_plan").value),
            binarize_gripper=bool(self.get_parameter("binarize_gripper").value),
        )

        self.images = {camera: None for camera in self.contract.policy_input_cameras}
        self.state = None
        self.task_description = None
        self.success = False
        self.episode_index = 0
        self.last_processed_observation = -1
        self._workers_stopped = False

        # Only rank 0 participates in DDS. Other ranks wait for rank 0's
        # distributed commands in worker_loop().
        self.action_pub = None
        self._ros_subscriptions = []
        if self.rank == 0:
            self.action_pub = self.create_publisher(Float32MultiArray, self.contract.action_topic, 10)
            for camera in self.contract.policy_input_cameras:
                self._ros_subscriptions.append(self.create_subscription(Image, self.contract.camera_topic(camera), self._make_image_cb(camera), 10))
            self._ros_subscriptions.extend(
                [
                    self.create_subscription(Float32MultiArray, self.contract.state_topic, self.on_state, 10),
                    self.create_subscription(String, self.contract.task_topic, self.on_task, 10),
                    self.create_subscription(Bool, self.contract.success_topic, self.on_success, 10),
                    self.create_subscription(Int32, self.contract.episode_topic, self.on_episode, 10),
                    self.create_subscription(Int32, self.contract.observation_ready_topic, self.on_observation_ready, 10),
                ]
            )
            self.get_logger().info(
                f"Cosmos3 Policy-DROID ready on {self.contract.namespace}: "
                f"cameras={list(self.contract.policy_input_cameras)}, actions_per_plan={self.policy.actions_per_plan}, "
                f"prompt_format={self.policy.prompt_format}"
            )

    def _build_policy_config(self):
        config_json = str(self.get_parameter("config_json").value).strip()
        model_path = str(self.get_parameter("model_path").value).strip()
        prompt_format = str(self.get_parameter("prompt_format").value).strip().lower()
        if not config_json:
            raise ValueError("Cosmos3 ROS node requires `config_json`.")
        if not model_path:
            raise ValueError("Cosmos3 ROS node requires `model_path`.")

        config = get_default_config()
        config.update(
            {
                "model_cls": "cosmos3",
                "task": "i2va",
                "model_path": model_path,
                "config_json": config_json,
                "seed": int(self.get_parameter("seed").value),
            }
        )
        config = auto_calc_config(config)
        # ROS parameters are explicit runtime overrides and must win over the
        # values loaded from config_json by auto_calc_config().
        config["policy_prompt_format"] = prompt_format
        if int(config.get("raw_action_dim", 8)) != self.contract.action_dim:
            raise ValueError(f"Cosmos3 raw_action_dim={config.get('raw_action_dim')} != RoboLab action_dim={self.contract.action_dim}")
        return config

    def _initialize_parallel(self):
        parallel = self.policy_config.get("parallel", False)
        if not parallel:
            return
        expected = int(parallel.get("cfg_p_size", 1)) * int(parallel.get("seq_p_size", 1))
        actual = int(os.environ.get("WORLD_SIZE", 1))
        if actual != expected:
            raise RuntimeError(f"Cosmos3 config requires {expected} torchrun ranks, got WORLD_SIZE={actual}. Launch with torchrun --nproc_per_node={expected}.")
        if not dist.is_initialized():
            platform = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
            if platform is None:
                raise RuntimeError(f"unsupported LightX2V platform: {os.getenv('PLATFORM', 'cuda')}")
            platform.init_parallel_env()
        set_parallel_config(self.policy_config)

    def _broadcast(self, payload):
        if not dist.is_initialized():
            return
        objects = [payload]
        dist.broadcast_object_list(objects, src=0, device=self._broadcast_device)

    def worker_loop(self):
        if self.rank == 0:
            raise RuntimeError("rank 0 must use rclpy.spin(), not worker_loop()")
        while True:
            objects = [None]
            dist.broadcast_object_list(objects, src=0, device=self._broadcast_device)
            payload = objects[0]
            command = payload.get("command")
            if command == "stop":
                self._workers_stopped = True
                return
            if command == "reset":
                self.policy.reset()
                continue
            if command != "step":
                raise RuntimeError(f"unknown rank-0 Cosmos3 worker command: {command!r}")
            self.policy.next_action(
                images=payload["images"],
                state=payload["state"],
                task_description=payload["task_description"],
            )

    def _make_image_cb(self, camera):
        def _callback(msg):
            self.images[camera] = image_msg_to_rgb(msg)

        return _callback

    def on_state(self, msg):
        state = np.asarray(msg.data, dtype=np.float32)
        if state.size != self.contract.state_dim:
            self.get_logger().error(f"expected state length {self.contract.state_dim}, got {state.size}")
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
        self.episode_index = episode
        self.success = False
        self.last_processed_observation = -1
        self._broadcast({"command": "reset"})
        self.policy.reset()
        self.get_logger().info(f"new episode {episode}; cleared Cosmos3 action queue")

    def _log_inference_prompt(self, observation_index):
        border = "=" * 80
        self.get_logger().info(f"\n{border}\n[ROBOLAB PROMPT -> COSMOS3 INFERENCE] observation={observation_index}\n{self.task_description}\n{border}")

    def on_observation_ready(self, msg):
        observation_index = int(msg.data)
        if observation_index <= self.last_processed_observation:
            return
        if self.success:
            self.last_processed_observation = observation_index
            return
        missing = [name for name, image in self.images.items() if image is None]
        if self.state is None:
            missing.append("state")
        if not self.task_description:
            missing.append("task_description")
        if missing:
            self.get_logger().warning(f"observation {observation_index} waiting for: {missing}")
            return

        if not self.policy.pending_actions:
            self._log_inference_prompt(observation_index)

        payload = {
            "command": "step",
            "images": self.images,
            "state": self.state,
            "task_description": self.task_description,
        }
        self._broadcast(payload)
        action = self.policy.next_action(
            images=self.images,
            state=self.state,
            task_description=self.task_description,
        )
        self.publish_action(action)
        self.last_processed_observation = observation_index

    def publish_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.contract.action_dim:
            raise ValueError(f"expected action length {self.contract.action_dim}, got {action.size}")
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)

    def destroy_node(self):
        try:
            if self.rank == 0 and dist.is_initialized() and not self._workers_stopped:
                self._broadcast({"command": "stop"})
                self._workers_stopped = True
            if hasattr(self, "policy"):
                self.policy.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Cosmos3Node()
    try:
        if node.rank == 0:
            rclpy.spin(node)
        else:
            node.worker_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if dist.is_initialized():
            dist.destroy_process_group()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
