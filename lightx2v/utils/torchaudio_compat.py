"""
Torchaudio compatibility layer using librosa and soundfile.
This module provides a drop-in replacement for basic torchaudio functionality
when torchaudio is not available or incompatible with the current PyTorch version.

Usage:
    Instead of:
        import torchaudio
    Use:
        from lightx2v.utils import torchaudio_compat as torchaudio
"""

import io
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from loguru import logger


def load(
    filepath: Union[str, io.BytesIO],
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file using soundfile, compatible with torchaudio.load() API.
    
    Args:
        filepath: Path to audio file or BytesIO object
        frame_offset: Number of frames to skip at the beginning
        num_frames: Number of frames to load (-1 loads all)
        normalize: Whether to normalize to [-1, 1] range
        channels_first: If True, return shape [channels, samples], else [samples, channels]
        format: Audio format (for BytesIO objects)
        
    Returns:
        Tuple of (waveform, sample_rate)
        - waveform: torch.Tensor of shape [channels, samples] if channels_first=True
        - sample_rate: int, sampling rate
    """
    try:
        # Handle BytesIO objects
        if isinstance(filepath, io.BytesIO):
            filepath.seek(0)
            waveform, sample_rate = sf.read(filepath, dtype='float32', always_2d=True)
        else:
            # Read with soundfile
            if num_frames > 0:
                waveform, sample_rate = sf.read(
                    filepath, 
                    start=frame_offset, 
                    frames=num_frames,
                    dtype='float32',
                    always_2d=True
                )
            else:
                waveform, sample_rate = sf.read(
                    filepath,
                    start=frame_offset,
                    dtype='float32',
                    always_2d=True
                )
        
        # Convert to torch tensor
        waveform = torch.from_numpy(waveform).float()
        
        # Normalize if needed
        if normalize and waveform.abs().max() > 1.0:
            waveform = waveform / waveform.abs().max()
        
        # soundfile returns [samples, channels], convert to [channels, samples]
        if channels_first:
            waveform = waveform.T
        
        return waveform, sample_rate
        
    except Exception as e:
        logger.error(f"Error loading audio file {filepath}: {e}")
        raise


def save(
    filepath: Union[str, io.BytesIO],
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
) -> None:
    """
    Save audio file using soundfile, compatible with torchaudio.save() API.
    
    Args:
        filepath: Output file path or BytesIO object
        src: Audio tensor
        sample_rate: Sampling rate
        channels_first: If True, input shape is [channels, samples], else [samples, channels]
        format: Audio format (e.g., 'WAV', 'FLAC', 'OGG')
        encoding: Encoding type
        bits_per_sample: Bits per sample
    """
    try:
        # Convert to numpy
        waveform = src.cpu().numpy()
        
        # Convert from [channels, samples] to [samples, channels] if needed
        if channels_first and waveform.ndim == 2:
            waveform = waveform.T
        elif waveform.ndim == 1:
            waveform = waveform.reshape(-1, 1)
        
        # Determine format
        if isinstance(filepath, io.BytesIO):
            if format is None:
                format = 'WAV'
            sf.write(filepath, waveform, sample_rate, format=format)
        else:
            sf.write(filepath, waveform, sample_rate, format=format)
            
    except Exception as e:
        logger.error(f"Error saving audio file {filepath}: {e}")
        raise


class Resample:
    """
    Resample transform compatible with torchaudio.transforms.Resample.
    Uses librosa for resampling.
    """
    
    def __init__(
        self,
        orig_freq: int,
        new_freq: int,
        resampling_method: str = "sinc_interp_kaiser",
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize resampler.
        
        Args:
            orig_freq: Original sampling frequency
            new_freq: Target sampling frequency
            resampling_method: Resampling method (ignored, librosa uses high-quality resampling)
            lowpass_filter_width: Lowpass filter width (ignored)
            rolloff: Rolloff frequency (ignored)
            dtype: Output dtype
        """
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.dtype = dtype
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Resample the waveform.
        
        Args:
            waveform: Input tensor of shape [..., time]
            
        Returns:
            Resampled tensor
        """
        if self.orig_freq == self.new_freq:
            return waveform
            
        # Store original shape and device
        original_shape = waveform.shape
        device = waveform.device
        
        # Flatten to 2D: [batch, time]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Move to CPU and convert to numpy for librosa
        waveform_np = waveform.cpu().numpy()
        
        # Resample each channel
        resampled_list = []
        for i in range(waveform_np.shape[0]):
            resampled = librosa.resample(
                waveform_np[i],
                orig_sr=self.orig_freq,
                target_sr=self.new_freq,
                res_type='kaiser_best'
            )
            resampled_list.append(resampled)
        
        # Stack and convert back to torch
        resampled_np = np.stack(resampled_list, axis=0)
        result = torch.from_numpy(resampled_np).to(self.dtype).to(device)
        
        # Restore original shape
        if squeeze_output:
            result = result.squeeze(0)
            
        return result


class transforms:
    """Transforms module compatible with torchaudio.transforms"""
    Resample = Resample


class functional:
    """Functional module compatible with torchaudio.functional"""
    
    @staticmethod
    def resample(
        waveform: torch.Tensor,
        orig_freq: int,
        new_freq: int,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        resampling_method: str = "sinc_interp_kaiser",
    ) -> torch.Tensor:
        """
        Resample waveform using librosa.
        
        Args:
            waveform: Input tensor
            orig_freq: Original sampling frequency
            new_freq: Target sampling frequency
            lowpass_filter_width: Lowpass filter width (ignored)
            rolloff: Rolloff frequency (ignored)
            resampling_method: Resampling method (ignored)
            
        Returns:
            Resampled tensor
        """
        resampler = Resample(orig_freq, new_freq)
        return resampler(waveform)


# Module-level info
__version__ = "0.1.0-compat"
logger.info(f"Using torchaudio compatibility layer (librosa + soundfile) v{__version__}")

