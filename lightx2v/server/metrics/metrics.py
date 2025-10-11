# -*-coding=utf-8-*-
import threading
from typing import List, Tuple

from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from pydantic import BaseModel


class MetricsConfig(BaseModel):
    name: str
    desc: str
    type_: str
    labels: List[str] = []
    buckets: Tuple[float, ...] = (
        0.1,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
        600.0,
    )


METRICS_INFO = {
    "lightx2v_worker_request_count": MetricsConfig(
        name="lightx2v_worker_request_count",
        desc="The total number of requests",
        type_="counter",
    ),
    "lightx2v_worker_request_success": MetricsConfig(
        name="lightx2v_worker_request_success",
        desc="The number of successful requests",
        type_="counter",
    ),
    "lightx2v_worker_request_failure": MetricsConfig(
        name="lightx2v_worker_request_failure",
        desc="The number of failed requests",
        type_="counter",
        labels=["error_type"],
    ),
    "lightx2v_worker_request_duration": MetricsConfig(
        name="lightx2v_worker_request_duration",
        desc="Duration of the request (s)",
        type_="histogram",
        labels=["model_cls"],
    ),
    "lightx2v_input_audio_len": MetricsConfig(
        name="lightx2v_input_audio_len",
        desc="Length of the input audio",
        type_="histogram",
        buckets=(
            1.0,
            2.0,
            3.0,
            5.0,
            7.0,
            10.0,
            20.0,
            30.0,
            45.0,
            60.0,
            75.0,
            90.0,
            105.0,
            120.0,
        ),
    ),
    "lightx2v_input_image_len": MetricsConfig(
        name="lightx2v_input_image_len",
        desc="Length of the input image",
        type_="histogram",
    ),
    "lightx2v_input_prompt_len": MetricsConfig(
        name="lightx2v_input_prompt_len",
        desc="Length of the input prompt",
        type_="histogram",
    ),
    "lightx2v_load_model_duration": MetricsConfig(
        name="lightx2v_load_model_duration",
        desc="Duration of load model (s)",
        type_="histogram",
    ),
    "lightx2v_run_per_step_dit_duration": MetricsConfig(
        name="lightx2v_run_per_step_dit_duration",
        desc="Duration of run per step Dit (s)",
        type_="histogram",
        labels=["step_no", "total_steps"],
    ),
    "lightx2v_run_text_encode_duration": MetricsConfig(
        name="lightx2v_run_text_encode_duration",
        desc="Duration of run text encode (s)",
        type_="histogram",
        labels=["model_cls"],
    ),
    "lightx2v_run_img_encode_duration": MetricsConfig(
        name="lightx2v_run_img_encode_duration",
        desc="Duration of run img encode (s)",
        type_="histogram",
        labels=["model_cls"],
    ),
    "lightx2v_run_vae_encode_duration": MetricsConfig(
        name="lightx2v_run_vae_encode_duration",
        desc="Duration of run vae encode (s)",
        type_="histogram",
        labels=["model_cls"],
    ),
    "lightx2v_run_vae_decode_duration": MetricsConfig(
        name="lightx2v_run_vae_decode_duration",
        desc="Duration of run vae decode (s)",
        type_="histogram",
        labels=["model_cls"],
    ),
    "lightx2v_run_init_run_segment_duration": MetricsConfig(
        name="lightx2v_run_init_run_segment_duration",
        desc="Duration of run init_run_segment (s)",
        type_="histogram",
        labels=["model_cls"],
    ),
    "lightx2v_run_end_run_segment_duration": MetricsConfig(
        name="lightx2v_run_end_run_segment_duration",
        desc="Duration of run end_run_segment (s)",
        type_="histogram",
        labels=["model_cls"],
    ),
}


class MetricsClient:
    def __init__(self):
        self.init_metrics()

    def init_metrics(self):
        for metric_name, config in METRICS_INFO.items():
            if config.type_ == "counter":
                self.register_counter(config.name, config.desc, config.labels)
            elif config.type_ == "histogram":
                self.register_histogram(config.name, config.desc, config.labels, buckets=config.buckets)
            elif config.type_ == "gauge":
                self.register_gauge(config.name, config.desc, config.labels)
            else:
                logger.warning(f"Unsupported metric type: {config.type_} for {metric_name}")

    def register_counter(self, name, desc, labels):
        metric_instance = Counter(name, desc, labels)
        setattr(self, name, metric_instance)

    def register_histogram(self, name, desc, labels, buckets=None):
        buckets = buckets or (
            0.1,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            30.0,
            60.0,
            120.0,
            300.0,
            600.0,
        )
        metric_instance = Histogram(name, desc, labels, buckets=buckets)
        setattr(self, name, metric_instance)

    def register_gauge(self, name, desc, labels):
        metric_instance = Gauge(name, desc, labels)
        setattr(self, name, metric_instance)


class MetricsServer:
    def __init__(self, port=8000):
        self.port = port
        self.server_thread = None

    def start_server(self):
        def run_server():
            start_http_server(self.port)
            logger.info(f"Metrics server started on port {self.port}")

        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()


def server_process(metric_port=8001):
    metrics = MetricsServer(
        port=metric_port,
    )
    metrics.start_server()
