@echo off
chcp 65001 >nul
echo 启动LightX2V T2V推理...

:: 设置路径
set lightx2v_path=D:\LightX2V
set model_path=D:\models\Wan2.1-T2V-1.3B-Lightx2v

:: 检查CUDA_VISIBLE_DEVICES
if "%CUDA_VISIBLE_DEVICES%"=="" (
    set cuda_devices=0
    echo Warn: CUDA_VISIBLE_DEVICES is not set, using default value: %cuda_devices%, change at shell script or set env variable.
    set CUDA_VISIBLE_DEVICES=%cuda_devices%
)

:: 检查路径
if "%lightx2v_path%"=="" (
    echo Error: lightx2v_path is not set. Please set this variable first.
    exit /b 1
)

if "%model_path%"=="" (
    echo Error: model_path is not set. Please set this variable first.
    exit /b 1
)

:: 设置环境变量
set TOKENIZERS_PARALLELISM=false
set PYTHONPATH=%lightx2v_path%;%PYTHONPATH%
set PROFILING_DEBUG_LEVEL=2
set ENABLE_GRAPH_MODE=false
set DTYPE=BF16

echo 环境变量设置完成！
echo PYTHONPATH: %PYTHONPATH%
echo CUDA_VISIBLE_DEVICES: %CUDA_VISIBLE_DEVICES%
echo 模型路径: %model_path%

:: 切换到项目目录
cd /d %lightx2v_path%

:: 运行推理
python -m lightx2v.infer ^
--model_cls wan2.1 ^
--task t2v ^
--model_path %model_path% ^
--config_json %lightx2v_path%/configs/offload/block/wan_t2v_1_3b.json ^
--prompt "A beautiful sunset over a calm ocean, with golden rays of light reflecting on the water surface. The sky is painted with vibrant orange and pink clouds. A peaceful and serene atmosphere." ^
--negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" ^
--save_video_path %lightx2v_path%/save_results/output_lightx2v_wan_t2v.mp4

echo 推理完成！
pause
