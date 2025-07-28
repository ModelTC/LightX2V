#!/usr/bin/env python3
import torch
import torch.multiprocessing as mp
import os
import sys
import random
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from lightx2v.infer import main as infer_main
import asyncio

def worker(rank, world_size, args):
    # Set distributed environment variables
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = str(args['master_port'])
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"Worker {rank} initialized on {device}")
    
    # Reconstruct the command line arguments
    sys.argv = [
        sys.argv[0],
        f"--model_cls={args['model_cls']}",
        f"--task={args['task']}",
        f"--model_path={args['model_path']}",
        f"--config_json={args['config_json']}",
        f"--prompt={args['prompt']}",
        f"--negative_prompt={args['negative_prompt']}",
        f"--image_path={args['image_path']}",
        f"--save_video_path={args['save_video_path']}"
    ]
    
    # Run the main function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(infer_main())

if __name__ == "__main__":
    torch.cuda.init()
    # Set paths and configurations (originally from bash variables)
    lightx2v_path = "/home/huangxinchi/workspace/temp/lightx2v"
    model_path = "/data/nvme0/yongyang/models/x2v_models/wan/Wan2.1-I2V-14B-480P"

    # Check CUDA devices
    cuda_devices = "2,4,6,7"
    print(f"Warn: CUDA_VISIBLE_DEVICES is not set, using default value: {cuda_devices}, change in script or set env variable.")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    # Validate paths
    if not lightx2v_path:
        print("Error: lightx2v_path is not set. Please set this variable first.")
        sys.exit(1)
    if not model_path:
        print("Error: model_path is not set. Please set this variable first.")
        sys.exit(1)

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONPATH"] = f"{lightx2v_path}:{os.environ.get('PYTHONPATH', '')}"
    os.environ["DTYPE"] = "BF16"
    os.environ["ENABLE_PROFILING_DEBUG"] = "true"
    os.environ["ENABLE_GRAPH_MODE"] = "false"
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1) 

    # Prepare arguments
    args = {
        "model_cls": "wan2.1",
        "task": "i2v",
        "model_path": model_path,
        "config_json": f"{lightx2v_path}/configs/wan_i2v_dist.json",
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard...",  # truncated for brevity
        "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清...",  # truncated
        "image_path": f"{lightx2v_path}/assets/inputs/imgs/img_0.jpg",
        "save_video_path": f"{lightx2v_path}/save_results/output_lightx2v_wan_i2v.mp4",
        "master_port": random.randint(20000, 29999)
    }

    print(f"Using MASTER_ADDR=127.0.0.1 and MASTER_PORT={args['master_port']}")
    
    world_size = 4
    mp.set_start_method("forkserver")
    mp.spawn(worker, args=(world_size, args), nprocs=world_size, join=True)