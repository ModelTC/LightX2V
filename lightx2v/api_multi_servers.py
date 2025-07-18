import argparse
import subprocess
import time
import socket
import os
from typing import List, Optional, Dict
import psutil
import requests
from loguru import logger
import concurrent.futures
from dataclasses import dataclass


@dataclass
class ServerConfig:
    port: int
    gpu_id: int
    model_cls: str
    task: str
    model_path: str
    config_json: str


def get_node_ip() -> str:
    """Get the IP address of the current node"""
    try:
        # Create a UDP socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to an external address (no actual connection needed)
        s.connect(("8.8.8.8", 80))
        # Get local IP
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.error(f"Failed to get IP address: {e}")
        return "localhost"


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def find_available_port(start_port: int) -> Optional[int]:
    """Find an available port starting from start_port"""
    port = start_port
    while port < start_port + 1000:  # Try up to 1000 ports
        if not is_port_in_use(port):
            return port
        port += 1
    return None


def start_server(config: ServerConfig) -> Optional[tuple[subprocess.Popen, str]]:
    """Start a single server instance"""
    try:
        # Set GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

        # Start server
        process = subprocess.Popen(
            [
                "python",
                "-m",
                "lightx2v.api_server",
                "--model_cls",
                config.model_cls,
                "--task",
                config.task,
                "--model_path",
                config.model_path,
                "--config_json",
                config.config_json,
                "--port",
                str(config.port),
            ],
            env=env,
        )

        # Wait for server to start, up to 600 seconds
        node_ip = get_node_ip()
        service_url = f"http://{node_ip}:{config.port}/v1/service/status"

        # Check once per second, up to 600 times
        for _ in range(600):
            try:
                response = requests.get(service_url, timeout=1)
                if response.status_code == 200:
                    return process, f"http://{node_ip}:{config.port}"
            except (requests.RequestException, ConnectionError) as e:
                pass
            time.sleep(1)

        # If timeout, terminate the process
        logger.error(f"Server startup timeout: port={config.port}, gpu={config.gpu_id}")
        process.terminate()
        return None

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument("--start_port", type=int, required=True, help="Starting port number")
    parser.add_argument("--model_cls", type=str, required=True, help="Model class")
    parser.add_argument("--task", type=str, required=True, help="Task type")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--config_json", type=str, required=True, help="Config file path")
    args = parser.parse_args()

    # Prepare configurations for all servers on this node
    server_configs = []
    current_port = args.start_port

    # Create configs for each GPU on this node
    for gpu in range(args.num_gpus):
        port = find_available_port(current_port)
        if port is None:
            logger.error(f"Cannot find available port starting from {current_port}")
            continue

        config = ServerConfig(port=port, gpu_id=gpu, model_cls=args.model_cls, task=args.task, model_path=args.model_path, config_json=args.config_json)
        server_configs.append(config)
        current_port = port + 1

    # Start all servers in parallel
    processes = []
    urls = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(server_configs)) as executor:
        future_to_config = {executor.submit(start_server, config): config for config in server_configs}
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                if result:
                    process, url = result
                    processes.append(process)
                    urls.append(url)
                    logger.info(f"Server started successfully: {url} (GPU: {config.gpu_id})")
                else:
                    logger.error(f"Failed to start server: port={config.port}, gpu={config.gpu_id}")
            except Exception as e:
                logger.error(f"Error occurred while starting server: {e}")

    # Print all server URLs
    logger.info("\nAll server URLs:")
    for url in urls:
        logger.info(url)

    # Print node information
    node_ip = get_node_ip()
    logger.info(f"\nCurrent node IP: {node_ip}")
    logger.info(f"Number of servers started: {len(urls)}")

    try:
        # Wait for all processes
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down all servers...")
        for process in processes:
            process.terminate()


if __name__ == "__main__":
    main()
