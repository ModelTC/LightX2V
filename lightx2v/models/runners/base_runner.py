from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from lightx2v.utils.utils import save_videos_grid


class TransformerModel(Protocol):
    """Protocol for transformer models"""

    def set_scheduler(self, scheduler: Any) -> None: ...
    def scheduler(self) -> Any: ...


class TextEncoderModel(Protocol):
    """Protocol for text encoder models"""

    def infer(self, texts: List[str], config: Dict[str, Any]) -> Any: ...


class ImageEncoderModel(Protocol):
    """Protocol for image encoder models"""

    def encode(self, image: Any) -> Any: ...


class VAEModel(Protocol):
    """Protocol for VAE models"""

    def encode(self, image: Any) -> Tuple[Any, Dict[str, Any]]: ...
    def decode(self, latents: Any, generator: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> Any: ...


class BaseRunner(ABC):
    """Abstract base class for all Runners

    Defines interface methods that all subclasses must implement
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def load_transformer(self) -> TransformerModel:
        """Load transformer model

        Returns:
            Loaded transformer model instance
        """
        pass

    @abstractmethod
    def load_text_encoder(self) -> Union[TextEncoderModel, List[TextEncoderModel]]:
        """Load text encoder

        Returns:
            Text encoder instance or list of text encoder instances
        """
        pass

    @abstractmethod
    def load_image_encoder(self) -> Optional[ImageEncoderModel]:
        """Load image encoder

        Returns:
            Image encoder instance or None if not needed
        """
        pass

    @abstractmethod
    def load_vae(self) -> Tuple[VAEModel, VAEModel]:
        """Load VAE encoder and decoder

        Returns:
            Tuple[vae_encoder, vae_decoder]: VAE encoder and decoder instances
        """
        pass

    @abstractmethod
    def run_image_encoder(self, img: Any) -> Any:
        """Run image encoder

        Args:
            img: Input image

        Returns:
            Image encoding result
        """
        pass

    @abstractmethod
    def run_vae_encoder(self, img: Any) -> Tuple[Any, Dict[str, Any]]:
        """Run VAE encoder

        Args:
            img: Input image

        Returns:
            Tuple of VAE encoding result and additional parameters
        """
        pass

    @abstractmethod
    def run_text_encoder(self, prompt: str, img: Optional[Any] = None) -> Any:
        """Run text encoder

        Args:
            prompt: Input text prompt
            img: Optional input image (for some models)

        Returns:
            Text encoding result
        """
        pass

    @abstractmethod
    def get_encoder_output_i2v(self, clip_encoder_out: Any, vae_encoder_out: Any, text_encoder_output: Any, img: Any) -> Dict[str, Any]:
        """Combine encoder outputs for i2v task

        Args:
            clip_encoder_out: CLIP encoder output
            vae_encoder_out: VAE encoder output
            text_encoder_output: Text encoder output
            img: Original image

        Returns:
            Combined encoder output dictionary
        """
        pass

    @abstractmethod
    def init_scheduler(self) -> None:
        """Initialize scheduler"""
        pass

    def set_target_shape(self) -> Dict[str, Any]:
        """Set target shape

        Subclasses can override this method to provide specific implementation

        Returns:
            Dictionary containing target shape information
        """
        return {}

    def save_video_func(self, images: Any) -> None:
        """Save video implementation

        Subclasses can override this method to customize save logic

        Args:
            images: Image sequence to save
        """
        save_videos_grid(images, self.config.get("save_video_path", "./output.mp4"), n_rows=1, fps=self.config.get("fps", 8))

    def load_vae_decoder(self) -> VAEModel:
        """Load VAE decoder

        Default implementation: get decoder from load_vae method
        Subclasses can override this method to provide different loading logic

        Returns:
            VAE decoder instance
        """
        if not hasattr(self, "vae_decoder") or self.vae_decoder is None:
            _, self.vae_decoder = self.load_vae()
        return self.vae_decoder
