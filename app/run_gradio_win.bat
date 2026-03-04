@echo off
chcp 65001 >nul
echo 🎬 LightX2V Gradio Windows Startup Script
echo ==========================================

REM ==================== Configuration Area ====================
REM ⚠️  Important: Please modify the following paths according to your actual environment

REM 🚨 Storage Performance Tips 🚨
REM 💾 Strongly recommend storing model files on SSD solid-state drives!
REM 📈 SSD can significantly improve model loading speed and inference performance
REM 🐌 Using mechanical hard drives (HDD) may cause slow model loading and affect overall experience

REM LightX2V project root directory path
REM Example: D:\LightX2V
set lightx2v_path=/path/to/LightX2V

REM Model path configuration
REM Model root directory path
REM Example: D:\models\LightX2V
set model_path=/path/to/LightX2V

REM Server configuration
set server_name=127.0.0.1
set server_port=8032

REM Output directory configuration
set output_dir=./outputs

REM GPU configuration
set gpu_id=0

REM ==================== Environment Variables Setup ====================
set CUDA_VISIBLE_DEVICES=%gpu_id%
set PYTHONPATH=%lightx2v_path%;%PYTHONPATH%
set PROFILING_DEBUG_LEVEL=2
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM ==================== Parameter Parsing ====================
REM Default interface language
set lang=zh

REM Parse command line arguments
:parse_args
if "%1"=="" goto :end_parse
if "%1"=="--lang" (
    set lang=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--port" (
    set server_port=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--gpu" (
    set gpu_id=%2
    set CUDA_VISIBLE_DEVICES=%gpu_id%
    shift
    shift
    goto :parse_args
)
if "%1"=="--output_dir" (
    set output_dir=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--help" (
    echo 🎬 LightX2V Gradio Windows Startup Script
    echo ==========================================
    echo Usage: %0 [options]
    echo.
    echo 📋 Available options:
    echo   --lang zh^|en      Interface language (default: zh)
    echo                      zh: Chinese interface
    echo                      en: English interface
    echo   --port PORT        Server port (default: 8032)
    echo   --gpu GPU_ID       GPU device ID (default: 0)
    echo   --output_dir OUTPUT_DIR
    echo                      Output video save directory (default: ./outputs)
    echo   --help             Show this help message
    echo.
    echo 🚀 Usage examples:
    echo   %0                                    # Default startup
    echo   %0 --lang zh --port 8032              # Start with specified parameters
    echo   %0 --lang en --port 7860              # English interface
    echo   %0 --gpu 1 --port 8032                # Use GPU 1
    echo   %0 --output_dir ./custom_output       # Use custom output directory
    echo.
    echo 📝 Notes:
    echo   - Edit script to configure model path before first use
    echo   - Ensure required Python dependencies are installed
    echo   - Recommended to use GPU with 8GB+ VRAM
    echo   - 🚨 Strongly recommend storing models on SSD for better performance
    pause
    exit /b 0
)
echo Unknown parameter: %1
echo Use --help to see help information
pause
exit /b 1

:end_parse

REM ==================== Parameter Validation ====================
if "%lang%"=="zh" goto :valid_lang
if "%lang%"=="en" goto :valid_lang
echo Error: Language must be 'zh' or 'en'
pause
exit /b 1

:valid_lang

REM Check if model path exists
if not exist "%model_path%" (
    echo ❌ Error: Model path does not exist
    echo 📁 Path: %model_path%
    echo 🔧 Solutions:
    echo   1. Check model path configuration in script
    echo   2. Ensure model files are properly downloaded
    echo   3. Verify path permissions are correct
    echo   4. 💾 Recommend storing models on SSD for faster loading
    pause
    exit /b 1
)

REM Select demo file based on language
if "%lang%"=="zh" (
    set demo_file=gradio_demo_zh.py
    echo 🌏 Using Chinese interface
) else (
    set demo_file=gradio_demo.py
    echo 🌏 Using English interface
)

REM Check if demo file exists
if not exist "%demo_file%" (
    echo ❌ Error: Demo file does not exist
    echo 📄 File: %demo_file%
    echo 🔧 Solutions:
    echo   1. Ensure script is run in the correct directory
    echo   2. Check if file has been renamed or moved
    echo   3. Re-clone or download project files
    pause
    exit /b 1
)

REM ==================== System Information Display ====================
echo ==========================================
echo 🚀 LightX2V Gradio Starting...
echo ==========================================
echo 📁 Project path: %lightx2v_path%
echo 🤖 Model path: %model_path%
echo 🌏 Interface language: %lang%
echo 🖥️  GPU device: %gpu_id%
echo 🌐 Server address: %server_name%:%server_port%
echo 📁 Output directory: %output_dir%
echo ==========================================

REM Display system resource information
echo 💻 System resource information:
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /format:table

REM Display GPU information
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>nul
if errorlevel 1 (
    echo 🎮 GPU information: Unable to get GPU info
) else (
    echo 🎮 GPU information:
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
)

REM ==================== Start Demo ====================
echo 🎬 Starting Gradio demo...
echo 📱 Please access in browser: http://%server_name%:%server_port%
echo ⏹️  Press Ctrl+C to stop service
echo 🔄 First startup may take several minutes to load resources...
echo ==========================================

REM Start Python demo
python %demo_file% ^
    --model_path "%model_path%" ^
    --server_name %server_name% ^
    --server_port %server_port% ^
    --output_dir "%output_dir%"

REM Display final system resource usage
echo.
echo ==========================================
echo 📊 Final system resource usage:
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /format:table

pause
