"""
LightX2V Setup Script
Minimal installation for VAE models only
"""

import os

from setuptools import find_packages, setup


# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Core dependencies for VAE models
vae_dependencies = [
    "torch>=2.0.0",
    "numpy>=1.20.0",
    "einops>=0.6.0",
    "loguru>=0.6.0",
]

# Full dependencies for complete LightX2V
full_dependencies = [
    "packaging",
    "ninja",
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "tokenizers",
    "tqdm",
    "accelerate",
    "safetensors",
    "opencv-python",
    "numpy",
    "imageio",
    "imageio-ffmpeg",
    "einops",
    "loguru",
    "ftfy",
    "gradio",
    "aiohttp",
    "pydantic",
    "fastapi",
    "uvicorn",
    "requests",
    "decord",
]

setup(
    name="lightx2v",
    version="1.0.0",
    author="LightX2V Team",
    author_email="",
    description="LightX2V: High-performance video generation models with optimized VAE",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ModelTC/LightX2V",
    packages=find_packages(include=["lightx2v", "lightx2v.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=vae_dependencies,
    extras_require={
        "full": full_dependencies,
        "vae": vae_dependencies,
    },
    include_package_data=True,
    zip_safe=False,
)
