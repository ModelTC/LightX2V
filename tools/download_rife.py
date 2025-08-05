#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests


def get_base_dir():
    """Get project root directory"""
    return Path(__file__).parent.parent


def download_file(url, save_path):
    """Download file"""
    print(f"Starting download: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded_size = 0

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    progress = (downloaded_size / total_size) * 100
                    print(f"\rDownload progress: {progress:.1f}%", end="", flush=True)

    print(f"\nDownload completed: {save_path}")


def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Starting extraction: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction completed: {extract_to}")


def find_flownet_pkl(extract_dir):
    """Find flownet.pkl file in extracted directory"""
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file == "flownet.pkl":
                return os.path.join(root, file)
    return None


def main():
    parser = argparse.ArgumentParser(description="Download RIFE model to specified directory")
    parser.add_argument("target_directory", help="Target directory path")

    args = parser.parse_args()

    target_dir = Path(args.target_directory)
    if not target_dir.is_absolute():
        target_dir = Path.cwd() / target_dir

    base_dir = get_base_dir()
    temp_dir = base_dir / "_temp"

    # Create temporary directory
    temp_dir.mkdir(exist_ok=True)

    target_dir.mkdir(parents=True, exist_ok=True)

    zip_url = "https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip"
    zip_path = temp_dir / "RIFEv4.26_0921.zip"

    try:
        # Download zip file
        download_file(zip_url, zip_path)

        # Extract file
        extract_zip(zip_path, temp_dir)

        # Find flownet.pkl file
        flownet_pkl = find_flownet_pkl(temp_dir)
        if flownet_pkl:
            # Copy flownet.pkl to target directory
            target_file = target_dir / "flownet.pkl"
            shutil.copy2(flownet_pkl, target_file)
            print(f"flownet.pkl copied to: {target_file}")
        else:
            print("Error: flownet.pkl file not found")
            return 1

        print("RIFE model download and installation completed!")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...")

        # Delete zip file if exists
        if zip_path.exists():
            try:
                zip_path.unlink()
                print(f"Deleted: {zip_path}")
            except Exception as e:
                print(f"Error deleting zip file: {e}")

        # Delete extracted folders
        for item in temp_dir.iterdir():
            if item.is_dir():
                try:
                    shutil.rmtree(item)
                    print(f"Deleted directory: {item}")
                except Exception as e:
                    print(f"Error deleting directory {item}: {e}")

        # Delete the temp directory itself if empty
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            try:
                temp_dir.rmdir()
                print(f"Deleted temp directory: {temp_dir}")
            except Exception as e:
                print(f"Error deleting temp directory: {e}")


if __name__ == "__main__":
    sys.exit(main())
