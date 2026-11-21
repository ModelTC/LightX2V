import argparse


def main():
    parser = argparse.ArgumentParser(description="LightX2V Server")

    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--model_cls", type=str, required=True, help="Model class name")
    parser.add_argument("--model-variant", type=str, default=None, help="Model-specific startup weight variant; MiniMax-H3 uses fl2av or ref2av.")
    parser.add_argument("--task", type=str, default=None, help="Startup task for models that require one. Multi-task models select task in each request.")
    parser.add_argument("--config_json", type=str, required=True, help="Path to startup config")
    parser.add_argument("--lora_dir", type=str, default=None, help="Directory containing LoRA files (.safetensors)")

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--metric_port", type=int, default=None, help="Metrics server port")
    parser.add_argument("--max_queue_size", type=int, default=10, help="Maximum active tasks (pending + processing)")

    args = parser.parse_args()

    from .main import run_server

    run_server(args)


if __name__ == "__main__":
    main()
