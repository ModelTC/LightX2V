"""
Unified weight loading and saving interface.

Handles various weight file formats and provides efficient I/O operations.
"""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger
from safetensors import safe_open
from safetensors import torch as st
from tqdm import tqdm


class WeightIO:
    """Unified interface for weight file I/O operations."""

    @staticmethod
    def load(
        path: Union[str, Path],
        device: str = "cpu",
        lazy: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Load weights from file(s).
        
        Args:
            path: Path to weight file or directory
            device: Device to load tensors on
            lazy: Use lazy loading for safetensors (memory efficient)
            
        Returns:
            Dictionary of weights
        """
        path = Path(path)
        
        if path.is_file():
            return WeightIO._load_single_file(path, device, lazy)
        elif path.is_dir():
            return WeightIO._load_directory(path, device, lazy)
        else:
            raise ValueError(f"Invalid path: {path}")

    @staticmethod
    def _load_single_file(
        file_path: Path,
        device: str,
        lazy: bool,
    ) -> Dict[str, torch.Tensor]:
        """Load weights from a single file."""
        logger.info(f"Loading weights from: {file_path}")
        
        if file_path.suffix in [".pt", ".pth"]:
            weights = torch.load(
                file_path, 
                map_location=device, 
                weights_only=True
            )
            # Handle nested structure (e.g., {"module": weights})
            if isinstance(weights, dict) and "module" in weights:
                weights = weights["module"]
            return weights
            
        elif file_path.suffix == ".safetensors":
            if lazy:
                # Lazy loading - more memory efficient
                with safe_open(file_path, framework="pt") as f:
                    weights = {}
                    keys = f.keys()
                    desc = f"Loading {file_path.name}"
                    for k in tqdm(keys, desc=desc, leave=False):
                        weights[k] = f.get_tensor(k).to(device)
                    return weights
            else:
                # Load all at once
                from safetensors.torch import load_file
                return load_file(str(file_path), device=device)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    @staticmethod
    def _load_directory(
        dir_path: Path,
        device: str,
        lazy: bool,
    ) -> Dict[str, torch.Tensor]:
        """Load and merge weights from all files in directory."""
        # Find all weight files
        weight_files = sorted(
            list(dir_path.glob("*.safetensors"))
            + list(dir_path.glob("*.pth"))
            + list(dir_path.glob("*.pt"))
        )
        
        if not weight_files:
            raise ValueError(f"No weight files found in: {dir_path}")
        
        logger.info(f"Found {len(weight_files)} weight files in {dir_path}")
        
        merged_weights = {}
        for file_path in tqdm(weight_files, desc="Loading weight files"):
            weights = WeightIO._load_single_file(file_path, device, lazy)
            
            # Check for duplicates
            duplicate_keys = set(weights.keys()) & set(merged_weights.keys())
            if duplicate_keys:
                raise ValueError(
                    f"Duplicate keys found in {file_path.name}: {duplicate_keys}"
                )
            
            merged_weights.update(weights)
            del weights
            gc.collect()
        
        logger.info(f"Loaded {len(merged_weights)} total weights")
        return merged_weights

    @staticmethod
    def save(
        weights: Dict[str, torch.Tensor],
        output_path: Path,
        output_name: str,
        save_format: str = "safetensors",
        layout: str = "single_file",
        chunk_size: int = 100,
    ) -> List[Path]:
        """
        Save weights to file(s).
        
        Args:
            weights: Weight dictionary to save
            output_path: Output directory
            output_name: Base name for output files
            save_format: "safetensors" or "pytorch"
            layout: "single_file", "by_block", or "chunked"
            chunk_size: Chunk size for chunked layout
            
        Returns:
            List of saved file paths
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_format == "pytorch":
            return WeightIO._save_pytorch(weights, output_path, output_name)
        elif save_format == "safetensors":
            if layout == "single_file":
                return WeightIO._save_single_file(weights, output_path, output_name)
            elif layout == "by_block":
                return WeightIO._save_by_block(weights, output_path)
            elif layout == "chunked":
                return WeightIO._save_chunked(weights, output_path, output_name, chunk_size)
            else:
                raise ValueError(f"Unknown layout: {layout}")
        else:
            raise ValueError(f"Unknown save format: {save_format}")

    @staticmethod
    def _save_pytorch(
        weights: Dict[str, torch.Tensor],
        output_path: Path,
        output_name: str,
    ) -> List[Path]:
        """Save as PyTorch .pth file."""
        output_file = output_path / f"{output_name}.pth"
        torch.save(weights, output_file)
        logger.info(f"Saved to: {output_file}")
        return [output_file]

    @staticmethod
    def _save_single_file(
        weights: Dict[str, torch.Tensor],
        output_path: Path,
        output_name: str,
    ) -> List[Path]:
        """Save as single safetensors file."""
        output_file = output_path / f"{output_name}.safetensors"
        
        # Calculate size
        total_size = sum(
            tensor.numel() * tensor.element_size() 
            for tensor in weights.values()
        )
        total_size_gb = total_size / (1024**3)
        
        if total_size_gb > 10:
            logger.warning(
                f"Model size is {total_size_gb:.2f}GB. "
                "Consider using by_block or chunked layout."
            )
        
        st.save_file(weights, str(output_file))
        logger.info(f"Saved to: {output_file} ({total_size_gb:.2f}GB)")
        return [output_file]

    @staticmethod
    def _save_by_block(
        weights: Dict[str, torch.Tensor],
        output_path: Path,
    ) -> List[Path]:
        """Save weights grouped by block index."""
        import re
        from collections import defaultdict
        
        block_groups = defaultdict(dict)
        non_block_weights = {}
        block_pattern = re.compile(r"blocks\.(\d+)\.")
        
        # Group weights by block
        for key, tensor in weights.items():
            match = block_pattern.search(key)
            if match:
                block_idx = match.group(1)
                block_groups[block_idx][key] = tensor
            else:
                non_block_weights[key] = tensor
        
        saved_files = []
        
        # Save each block
        for block_idx, block_weights in tqdm(
            block_groups.items(), desc="Saving blocks"
        ):
            output_file = output_path / f"block_{block_idx}.safetensors"
            st.save_file(block_weights, str(output_file))
            saved_files.append(output_file)
        
        # Save non-block weights
        if non_block_weights:
            output_file = output_path / "non_block.safetensors"
            st.save_file(non_block_weights, str(output_file))
            saved_files.append(output_file)
        
        # Save index file
        WeightIO._save_index(weights, saved_files, output_path)
        
        logger.info(f"Saved {len(saved_files)} files to: {output_path}")
        return saved_files

    @staticmethod
    def _save_chunked(
        weights: Dict[str, torch.Tensor],
        output_path: Path,
        output_name: str,
        chunk_size: int,
    ) -> List[Path]:
        """Save weights in chunks."""
        saved_files = []
        chunk_idx = 0
        current_chunk = {}
        
        for idx, (k, v) in tqdm(
            enumerate(weights.items()), 
            desc="Saving chunks",
            total=len(weights)
        ):
            current_chunk[k] = v
            
            if chunk_size > 0 and (idx + 1) % chunk_size == 0:
                output_file = output_path / f"{output_name}_part{chunk_idx}.safetensors"
                st.save_file(current_chunk, str(output_file))
                saved_files.append(output_file)
                current_chunk = {}
                chunk_idx += 1
        
        # Save remaining
        if current_chunk:
            output_file = output_path / f"{output_name}_part{chunk_idx}.safetensors"
            st.save_file(current_chunk, str(output_file))
            saved_files.append(output_file)
        
        # Save index file
        WeightIO._save_index(weights, saved_files, output_path)
        
        logger.info(f"Saved {len(saved_files)} chunks to: {output_path}")
        return saved_files

    @staticmethod
    def _save_index(
        weights: Dict[str, torch.Tensor],
        saved_files: List[Path],
        output_path: Path,
    ):
        """Save index file mapping keys to files."""
        import json
        
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {}
        }
        
        # Build weight map
        for file_path in saved_files:
            file_size = file_path.stat().st_size
            index["metadata"]["total_size"] += file_size
            
            # Load keys from this file
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    index["weight_map"][key] = file_path.name
        
        # Save index
        index_path = output_path / "model.safetensors.index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Index file saved to: {index_path}")

