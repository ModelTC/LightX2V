import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Thread

import cv2
import numpy as np
import rclpy
from common.contract import get_contract
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .page import render_index

# How long a stalled MJPEG stream waits before re-sending the last frame. This,
# together with the simulator re-publishing the final frame after success, keeps
# the page from going blank when no new frames arrive.
STREAM_KEEPALIVE_S = 2.0


class ImageHttpServer(ThreadingHTTPServer):
    daemon_threads = True


class FrameStore:
    def __init__(self, cameras):
        self.condition = Condition()
        self.frames = {name: (0, None) for name in cameras}
        self.task = ""
        self.status = {}

    def update(self, name, jpeg):
        with self.condition:
            seq, _ = self.frames[name]
            self.frames[name] = (seq + 1, jpeg)
            self.condition.notify_all()

    def wait_next(self, name, last_seq, timeout=STREAM_KEEPALIVE_S):
        with self.condition:
            self.condition.wait_for(lambda: self.frames[name][0] != last_seq, timeout=timeout)
            return self.frames[name]

    def update_task(self, task):
        with self.condition:
            self.task = task

    def get_task(self):
        with self.condition:
            return self.task

    def update_status(self, status):
        with self.condition:
            self.status = status

    def get_status(self):
        with self.condition:
            return self.status


class ImageWebViewerNode(Node):
    def __init__(self):
        super().__init__("image_web_viewer")

        self.declare_parameter("env", "libero")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8080)
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("cameras", [])
        self.declare_parameter("namespace", "")
        self.declare_parameter("task_topic", "")

        env = str(self.get_parameter("env").value).strip().lower()
        contract = get_contract(env)
        self.contract = contract

        cameras_param = list(self.get_parameter("cameras").value or [])
        self.cameras = cameras_param if cameras_param else list(contract.cameras)
        namespace = str(self.get_parameter("namespace").value).strip() or contract.namespace
        task_topic = str(self.get_parameter("task_topic").value).strip() or contract.task_topic

        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.frame_store = FrameStore(self.cameras)
        self.http_server = None
        self.http_thread = None

        for name in self.cameras:
            topic = f"{namespace}/{name}/image_raw"
            self.create_subscription(
                Image,
                topic,
                lambda msg, camera_name=name: self.on_image(camera_name, msg),
                10,
            )
        self.create_subscription(String, task_topic, self.on_task, 10)
        self.create_subscription(String, f"{namespace}/status", self.on_status, 10)
        self.control_pub = self.create_publisher(String, f"{namespace}/control", 10)

        self.start_http_server()

    def start_http_server(self):
        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        handler = make_handler(self.frame_store, self.cameras, self.contract.name, self.publish_control)
        self.http_server = ImageHttpServer((host, port), handler)
        self.http_thread = Thread(target=self.http_server.serve_forever, daemon=True)
        self.http_thread.start()
        self.get_logger().info(f"[{self.contract.name}] image web viewer on http://{host}:{port} cameras={self.cameras}")

    def on_image(self, name, msg):
        try:
            image = image_msg_to_bgr(msg)
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if ok:
                self.frame_store.update(name, encoded.tobytes())
        except Exception as exc:
            self.get_logger().error(f"failed to encode {name} image: {exc}")

    def on_task(self, msg):
        self.frame_store.update_task(msg.data)

    def on_status(self, msg):
        try:
            self.frame_store.update_status(json.loads(msg.data))
        except Exception as exc:
            self.get_logger().error(f"bad status message: {exc}")

    def publish_control(self, command: dict):
        msg = String()
        msg.data = json.dumps(command)
        self.control_pub.publish(msg)
        self.get_logger().info(f"forwarded control command: {msg.data}")

    def destroy_node(self):
        try:
            if self.http_server is not None:
                shutdown_thread = Thread(target=self.http_server.shutdown, daemon=True)
                shutdown_thread.start()
                shutdown_thread.join(timeout=1.0)
                self.http_server.server_close()
            if self.http_thread is not None:
                self.http_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            pass
        super().destroy_node()


def make_handler(frame_store, cameras, title, publish_control):
    camera_set = set(cameras)
    index_html = render_index(cameras, title=f"LightX2V ROS · {title}")

    class ImageWebViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in {"/", "/index.html"}:
                self.send_index()
                return
            if self.path == "/task.txt":
                self.send_task()
                return
            if self.path == "/status.json":
                self.send_status()
                return
            if self.path.endswith(".mjpg"):
                name = self.path[1 : -len(".mjpg")]
                if name in camera_set:
                    self.send_stream(name)
                    return
            self.send_error(404)

        def do_POST(self):
            if self.path != "/control":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                command = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(command, dict) or not command.get("cmd"):
                    raise ValueError("control payload must be a JSON object with a `cmd` field")
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, code=400)
                return
            publish_control(command)
            self.send_json({"ok": True})

        def send_json(self, payload, code=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_status(self):
            self.send_json(frame_store.get_status())

        def send_index(self):
            body = index_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_task(self):
            body = frame_store.get_task().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_stream(self, name):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            seq = 0
            while True:
                seq, frame = frame_store.wait_next(name, seq)
                if frame is None:
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break

        def log_message(self, *args):
            return

    return ImageWebViewerHandler


def image_msg_to_bgr(msg):
    encoding = msg.encoding.lower()
    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")

    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = row[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if encoding == "rgb8":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def main(args=None):
    rclpy.init(args=args)
    node = ImageWebViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
