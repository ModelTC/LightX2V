"""Run one official-order LIBERO suite through the ROS control plane."""

import json
import os
import time
from pathlib import Path

import rclpy
from common.contract import get_contract
from rclpy.node import Node
from std_msgs.msg import String

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
NUM_TASKS = 10
NUM_INIT_STATES = 50
FINISHED_STATES = ("success", "failure")


class LiberoSuiteEvaluator(Node):
    def __init__(self):
        super().__init__("libero_suite_evaluator")
        self.declare_parameter("task_suite_name", "libero_spatial")
        self.declare_parameter("output_dir", "data/libero/ros_evaluation")
        self.declare_parameter("command_timeout", 180.0)
        self.declare_parameter("overwrite", False)
        self.contract = get_contract("libero")

        self.suite = str(self.get_parameter("task_suite_name").value).strip().lower()
        if self.suite not in SUITES:
            raise ValueError(f"task_suite_name must be one of {SUITES}, got {self.suite!r}")

        output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        self.output_dir = output_dir / self.suite
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        if self.command_timeout <= 0:
            raise ValueError("command_timeout must be positive")
        self._prepare_output(bool(self.get_parameter("overwrite").value))

        self.control_pub = self.create_publisher(String, self.contract.control_topic, 10)
        self.status_sub = self.create_subscription(String, self.contract.status_topic, self._on_status, 10)
        self.timer = self.create_timer(0.1, self._tick)

        self.status = None
        self.phase = "waiting_for_simulator"
        self.task_id = 0
        self.init_state_id = 0
        self.active_episode = None
        self.set_task_after_episode = None
        self.command_sent_at = None
        self.results = []
        self.failure_reason = None
        self.started_at = time.time()
        self.get_logger().info(f"waiting for {self.contract.status_topic}; suite={self.suite}, output={self.output_dir}")

    @property
    def task_key(self):
        return f"{self.suite}/{self.task_id}"

    def _prepare_output(self, overwrite):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = [path for path in (self.episodes_path, self.summary_path) if path.exists()]
        if existing and not overwrite:
            paths = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"evaluation output already exists: {paths}; set overwrite:=true to replace it")
        if overwrite:
            for path in existing:
                path.unlink()

    def _on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"ignoring invalid {self.contract.status_topic} message: {exc}")

    def _tick(self):
        if self.status is None:
            return
        if bool(self.status.get("loop")):
            self._abort("simulator parameter loop must be false for ordered suite evaluation")
            return

        if self.phase == "waiting_for_simulator":
            if self.control_pub.get_subscription_count() == 0:
                return
            if self.status.get("state") != "ready" or int(self.status.get("episode", -1)) != 0 or self.status.get("history"):
                self._abort("simulator must be a fresh READY instance; launch libero_node with autostart:=false and loop:=false so no policy RNG is consumed before task 0/init 0")
                return
            if self._matches_target():
                self.active_episode = 0
                self._publish_control({"cmd": "start"})
                self.phase = "waiting_for_start"
            else:
                self._set_task()
        elif self.phase == "waiting_for_ready":
            episode = int(self.status.get("episode", -1))
            if self._matches_target() and self.status.get("state") == "ready" and episode > self.set_task_after_episode:
                self.active_episode = episode
                self._publish_control({"cmd": "start"})
                self.phase = "waiting_for_start"
            else:
                self._check_timeout("set_task")
        elif self.phase == "waiting_for_start":
            if self._matches_active_episode() and self.status.get("state") == "running":
                self.phase = "running"
                self.command_sent_at = None
            else:
                self._check_timeout("start")
        elif self.phase == "running":
            if not self._matches_active_episode():
                return
            if self.status.get("state") in FINISHED_STATES:
                self._record_episode()
                if len(self.results) == NUM_TASKS * NUM_INIT_STATES:
                    self._finish()
                else:
                    self._advance()
                    self._set_task()

    def _set_task(self):
        self.set_task_after_episode = int(self.status.get("episode", -1))
        self._publish_control(
            {
                "cmd": "set_task",
                "task_name": self.task_key,
                "task_config": str(self.init_state_id),
            }
        )
        self.phase = "waiting_for_ready"

    def _publish_control(self, command):
        msg = String()
        msg.data = json.dumps(command, separators=(",", ":"))
        self.control_pub.publish(msg)
        self.command_sent_at = time.monotonic()

    def _matches_target(self):
        return self.status.get("task_name") == self.task_key and str(self.status.get("task_config")) == str(self.init_state_id)

    def _matches_active_episode(self):
        return self._matches_target() and self.status.get("episode") == self.active_episode

    def _check_timeout(self, command):
        if self.command_sent_at is None:
            return
        if time.monotonic() - self.command_sent_at > self.command_timeout:
            self._abort(f"timed out waiting for {command!r} acknowledgement")

    def _record_episode(self):
        outcome = str(self.status["state"])
        result = {
            "task_suite_name": self.suite,
            "task_id": self.task_id,
            "init_state_id": self.init_state_id,
            "task_name": self.task_key,
            "episode": self.active_episode,
            "instruction": self.status.get("instruction", ""),
            "seed": self.status.get("seed"),
            "steps": int(self.status.get("episode_step", 0)),
            "outcome": outcome,
            "success": outcome == "success",
            "timestamp": time.time(),
        }
        self.results.append(result)
        with self.episodes_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._write_summary(complete=False)
        successes = sum(item["success"] for item in self.results)
        self.get_logger().info(f"[{len(self.results)}/{NUM_TASKS * NUM_INIT_STATES}] task={self.task_id:02d} init={self.init_state_id:02d} {outcome}; success={successes / len(self.results):.2%}")

    def _advance(self):
        self.init_state_id += 1
        if self.init_state_id == NUM_INIT_STATES:
            self.task_id += 1
            self.init_state_id = 0
        self.active_episode = None

    def _summary(self, complete):
        task_results = []
        for task_id in range(NUM_TASKS):
            records = [item for item in self.results if item["task_id"] == task_id]
            successes = sum(item["success"] for item in records)
            task_results.append(
                {
                    "task_id": task_id,
                    "episodes": len(records),
                    "successes": successes,
                    "success_rate": successes / len(records) if records else None,
                }
            )
        successes = sum(item["success"] for item in self.results)
        return {
            "protocol": "openpi_libero_official",
            "task_suite_name": self.suite,
            "task_order": "task_id_outer_init_state_id_inner",
            "expected_episodes": NUM_TASKS * NUM_INIT_STATES,
            "completed_episodes": len(self.results),
            "successes": successes,
            "success_rate": successes / len(self.results) if self.results else None,
            "complete": complete,
            "task_results": task_results,
            "started_at": self.started_at,
            "updated_at": time.time(),
        }

    def _write_summary(self, complete):
        temporary = self.summary_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self._summary(complete), stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.summary_path)

    def _finish(self):
        self.phase = "done"
        self._write_summary(complete=True)
        successes = sum(item["success"] for item in self.results)
        self.get_logger().info(f"suite complete: {successes}/{len(self.results)} ({successes / len(self.results):.2%}); summary={self.summary_path}")
        self.timer.cancel()
        rclpy.shutdown()

    def _abort(self, reason):
        self.phase = "failed"
        self.failure_reason = reason
        self.get_logger().error(reason)
        self.timer.cancel()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = LiberoSuiteEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if node.failure_reason:
        raise RuntimeError(node.failure_reason)


if __name__ == "__main__":
    main()
