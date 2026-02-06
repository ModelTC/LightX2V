import argparse

from .main import run_server


def main():
    parser = argparse.ArgumentParser(description="LightX2V Server")

    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--model_cls", type=str, required=True, help="Model class name")
    parser.add_argument("--lora_dir", type=str, default=None, help="Directory containing LoRA files (.safetensors)")

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    args, unknown = parser.parse_known_args()

    for i in range(0, len(unknown), 2):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                value = unknown[i + 1]
                setattr(args, key, value)

    run_server(args)


if __name__ == "__main__":
    main()
