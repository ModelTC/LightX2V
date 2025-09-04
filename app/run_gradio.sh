#!/bin/bash

# Lightx2v Gradio Demo Startup Script
# Supports both Image-to-Video (i2v) and Text-to-Video (t2v) modes

# ==================== Configuration Area ====================
# ⚠️  Important: Please modify the following paths according to your actual environment

# 🚨 Storage Performance Tips 🚨
# 💾 Strongly recommend storing model files on SSD solid-state drives!
# 📈 SSD can significantly improve model loading speed and inference performance
# 🐌 Using mechanical hard drives (HDD) may cause slow model loading and affect overall experience


# Lightx2v project root directory path
# Example: /home/user/lightx2v or /data/video_gen/lightx2v
lightx2v_path=/data/video_gen/LightX2V
# Model path configuration
# Image-to-video model path (for i2v tasks)
# Example: /path/to/Wan2.1-I2V-14B-720P-Lightx2v
i2v_model_path=/path/to/Wan2.1-I2V-14B-480P-Lightx2v

# Text-to-video model path (for t2v tasks)
# Example: /path/to/Wan2.1-T2V-1.3B
t2v_model_path=/path/to/Wan2.1-T2V-1.3B

# Model size configuration
# Default model size (14b, 1.3b)
model_size="14b"

# Model class configuration
# Default model class (wan2.1, wan2.1_distill)
model_cls="wan2.1"

# Server configuration
server_name="0.0.0.0"
server_port=8032

# Output directory configuration
output_dir="./outputs"

# GPU configuration
gpu_id=0

# ==================== Environment Variables Setup ====================
export CUDA_VISIBLE_DEVICES=$gpu_id
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export PROFILING_DEBUG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==================== Parameter Parsing ====================
# Default task type
task="i2v"
# Default interface language
lang="zh"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            task="$2"
            shift 2
            ;;
        --lang)
            lang="$2"
            shift 2
            ;;
        --port)
            server_port="$2"
            shift 2
            ;;
        --gpu)
            gpu_id="$2"
            export CUDA_VISIBLE_DEVICES=$gpu_id
            shift 2
            ;;
        --model_size)
            model_size="$2"
            shift 2
            ;;
        --model_cls)
            model_cls="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --help)
            echo "🎬 Lightx2v Gradio Demo Startup Script"
            echo "=========================================="
            echo "Usage: $0 [options]"
            echo ""
            echo "📋 Available options:"
            echo "  --task i2v|t2v    Task type (default: i2v)"
            echo "                     i2v: Image-to-video generation"
            echo "                     t2v: Text-to-video generation"
            echo "  --lang zh|en      Interface language (default: zh)"
            echo "                     zh: Chinese interface"
            echo "                     en: English interface"
            echo "  --port PORT       Server port (default: 8032)"
            echo "  --gpu GPU_ID      GPU device ID (default: 0)"
            echo "  --model_size MODEL_SIZE"
            echo "                     Model size (default: 14b)"
            echo "                     14b: 14 billion parameters model"
            echo "                     1.3b: 1.3 billion parameters model"
                echo "  --model_cls MODEL_CLASS"
    echo "                     Model class (default: wan2.1)"
    echo "                     wan2.1: Standard model variant"
    echo "                     wan2.1_distill: Distilled model variant for faster inference"
    echo "  --output_dir OUTPUT_DIR"
    echo "                     Output video save directory (default: ./saved_videos)"
    echo "  --help            Show this help message"
            echo ""
            echo "🚀 Usage examples:"
            echo "  $0                                    # Default startup for image-to-video mode"
            echo "  $0 --task i2v --lang zh --port 8032   # Start with specified parameters"
            echo "  $0 --task t2v --lang en --port 7860   # Text-to-video with English interface"
            echo "  $0 --task i2v --gpu 1 --port 8032     # Use GPU 1"
                echo "  $0 --task t2v --model_size 1.3b       # Use 1.3B model"
    echo "  $0 --task i2v --model_size 14b        # Use 14B model"
    echo "  $0 --task i2v --model_cls wan2.1_distill  # Use distilled model"
    echo "  $0 --task i2v --output_dir ./custom_output  # Use custom output directory"
            echo ""
            echo "📝 Notes:"
            echo "  - Edit script to configure model paths before first use"
            echo "  - Ensure required Python dependencies are installed"
            echo "  - Recommended to use GPU with 8GB+ VRAM"
            echo "  - 🚨 Strongly recommend storing models on SSD for better performance"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help to see help information"
            exit 1
            ;;
    esac
done

# ==================== Parameter Validation ====================
if [[ "$task" != "i2v" && "$task" != "t2v" ]]; then
    echo "Error: Task type must be 'i2v' or 't2v'"
    exit 1
fi

if [[ "$lang" != "zh" && "$lang" != "en" ]]; then
    echo "Error: Language must be 'zh' or 'en'"
    exit 1
fi

# Validate model size
if [[ "$model_size" != "14b" && "$model_size" != "1.3b" ]]; then
    echo "Error: Model size must be '14b' or '1.3b'"
    exit 1
fi

# Validate model class
if [[ "$model_cls" != "wan2.1" && "$model_cls" != "wan2.1_distill" ]]; then
    echo "Error: Model class must be 'wan2.1' or 'wan2.1_distill'"
    exit 1
fi

# Select model path based on task type
if [[ "$task" == "i2v" ]]; then
    model_path=$i2v_model_path
    echo "🎬 Starting Image-to-Video mode"
else
    model_path=$t2v_model_path
    echo "🎬 Starting Text-to-Video mode"
fi

# Check if model path exists
if [[ ! -d "$model_path" ]]; then
    echo "❌ Error: Model path does not exist"
    echo "📁 Path: $model_path"
    echo "🔧 Solutions:"
    echo "  1. Check model path configuration in script"
    echo "  2. Ensure model files are properly downloaded"
    echo "  3. Verify path permissions are correct"
    echo "  4. 💾 Recommend storing models on SSD for faster loading"
    exit 1
fi

# Select demo file based on language
if [[ "$lang" == "zh" ]]; then
    demo_file="gradio_demo_zh.py"
    echo "🌏 Using Chinese interface"
else
    demo_file="gradio_demo.py"
    echo "🌏 Using English interface"
fi

# Check if demo file exists
if [[ ! -f "$demo_file" ]]; then
    echo "❌ Error: Demo file does not exist"
    echo "📄 File: $demo_file"
    echo "🔧 Solutions:"
    echo "  1. Ensure script is run in the correct directory"
    echo "  2. Check if file has been renamed or moved"
    echo "  3. Re-clone or download project files"
    exit 1
fi

# ==================== System Information Display ====================
echo "=========================================="
echo "🚀 Lightx2v Gradio Demo Starting..."
echo "=========================================="
echo "📁 Project path: $lightx2v_path"
echo "🤖 Model path: $model_path"
echo "🎯 Task type: $task"
echo "🤖 Model size: $model_size"
echo "🤖 Model class: $model_cls"
echo "🌏 Interface language: $lang"
echo "🖥️  GPU device: $gpu_id"
echo "🌐 Server address: $server_name:$server_port"
echo "📁 Output directory: $output_dir"
echo "=========================================="

# Display system resource information
echo "💻 System resource information:"
free -h | grep -E "Mem|Swap"
echo ""

# Display GPU information
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1
    echo ""
fi

# ==================== Start Demo ====================
echo "🎬 Starting Gradio demo..."
echo "📱 Please access in browser: http://$server_name:$server_port"
echo "⏹️  Press Ctrl+C to stop service"
echo "🔄 First startup may take several minutes to load resources..."
echo "=========================================="

# Start Python demo
python $demo_file \
    --model_path "$model_path" \
    --model_cls "$model_cls" \
    --task "$task" \
    --server_name "$server_name" \
    --server_port "$server_port" \
    --model_size "$model_size" \
    --output_dir "$output_dir"

# Display final system resource usage
echo ""
echo "=========================================="
echo "📊 Final system resource usage:"
free -h | grep -E "Mem|Swap"
