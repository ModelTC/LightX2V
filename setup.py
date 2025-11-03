#!/usr/bin/env python3
"""
LightX2V: 轻量级视频生成推理框架
安装配置文件
"""

from setuptools import setup, find_packages
import os
import sys

# 确保Python版本兼容性
if sys.version_info < (3, 8):
    raise RuntimeError("LightX2V需要Python 3.8或更高版本")

# 读取版本信息
def get_version():
    """获取版本号"""
    version_file = os.path.join(os.path.dirname(__file__), "lightx2v", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# 读取README
def get_long_description():
    """读取长描述"""
    readme_file = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_file):
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return "LightX2V: 轻量级视频生成推理框架"

# 基础依赖
base_requirements = [
    "packaging",
    "ninja",
    "torch>=1.12.0",
    "torchvision>=0.13.0",
    "diffusers>=0.20.0",
    "transformers>=4.25.0",
    "tokenizers>=0.13.0",
    "tqdm",
    "accelerate",
    "safetensors",
    "opencv-python",
    "numpy",
    "imageio",
    "imageio-ffmpeg",
    "einops",
    "loguru",
    "qtorch",
    "ftfy",
    "gradio",
    "aiohttp",
    "pydantic",
]

# 可选依赖分组
extras_require = {
    # 服务部署相关
    "server": [
        "fastapi",
        "uvicorn",
        "PyJWT",
        "requests",
        "aio-pika",
        "asyncpg>=0.27.0",
        "aioboto3>=12.0.0",
        "alibabacloud_dypnsapi20170525==1.2.2",
        "redis==6.4.0",
        "tos",
    ],
    # 高性能推理相关
    "performance": [
        "vllm",
        "sgl-kernel",
    ],
    # 视频处理相关
    "video": [
        "decord",
        "av",
    ],
    # 开发工具
    "dev": [
        "ruff",
        "pre-commit",
    ],
    # 完整安装（包含所有可选依赖）
    "full": [
        "vllm",
        "sgl-kernel",
        "fastapi",
        "uvicorn",
        "PyJWT",
        "requests",
        "aio-pika",
        "asyncpg>=0.27.0",
        "aioboto3>=12.0.0",
        "alibabacloud_dypnsapi20170525==1.2.2",
        "redis==6.4.0",
        "tos",
        "decord",
        "av",
    ],
}

# 添加组合依赖
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="lightx2v",
    version=get_version(),
    author="LightX2V Contributors",
    author_email="",
    description="LightX2V: 轻量级视频生成推理框架",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/ModelTC/LightX2V",
    project_urls={
        "Documentation": "https://lightx2v-en.readthedocs.io/en/latest/",
        "Source": "https://github.com/ModelTC/LightX2V",
        "Tracker": "https://github.com/ModelTC/LightX2V/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=base_requirements,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "lightx2v": [
            "configs/*.json",
            "assets/**/*",
        ],
    },
    entry_points={
        "console_scripts": [
            "lightx2v-infer=lightx2v.cli:infer_main",
            "lightx2v-server=lightx2v.server.main:main",
        ],
    },
    zip_safe=False,
    keywords="video generation, AI, deep learning, diffusion models, text-to-video, image-to-video",
    license="Apache 2.0",
)