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
lightx2v_path=/path/to/LightX2V

# Model path configuration
# Example: /path/to/Wan2.1-I2V-14B-720P-Lightx2v
model_path=/path/to/models

# Server configuration
server_name="0.0.0.0"
server_port=8033

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
# Default interface language
lang="zh"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --model_path)
            model_path="$2"
            shift 2
            ;;
        --help)
            echo "🎬 Lightx2v Gradio Demo Startup Script"
            echo "=========================================="
            echo "Usage: $0 [options]"
            echo ""
            echo "📋 Available options:"
            echo "  --lang zh|en      Interface language (default: zh)"
            echo "                     zh: Chinese interface"
            echo "                     en: English interface"
            echo "  --port PORT       Server port (default: 8032)"
            echo "  --gpu GPU_ID      GPU device ID (default: 0)"
            echo "  --model_path PATH Model path (default: configured in script)"
            echo "  --output_dir DIR  Output video save directory (default: ./outputs)"
            echo "  --help            Show this help message"
            echo ""
            echo "📝 Notes:"
            echo "  - Task type (i2v/t2v) and model type are selected in the web UI"
            echo "  - Model class is auto-detected based on selected diffusion model"
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
if [[ "$lang" != "zh" && "$lang" != "en" ]]; then
    echo "Error: Language must be 'zh' or 'en'"
    exit 1
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
echo "🌏 Interface language: $lang"
echo "🖥️  GPU device: $gpu_id"
echo "🌐 Server address: $server_name:$server_port"
echo "📁 Output directory: $output_dir"
echo "📝 Note: Task type and model class are selected in web UI"
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
    --server_name "$server_name" \
    --server_port "$server_port" \
    --output_dir "$output_dir"

# Display final system resource usage
echo ""
echo "=========================================="
echo "📊 Final system resource usage:"
free -h | grep -E "Mem|Swap"
