"""
Device detection utility for GPU/CPU configuration.
Detects CUDA, MPS (Mac Metal), or falls back to CPU.
"""

import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class DeviceDetector:
    """Detects and configures the appropriate device for AI models."""

    @staticmethod
    def detect_device(preferred_device: str = "") -> Tuple[str, List[str]]:
        """
        Detect the best available device for running AI models.

        Args:
            preferred_device: Preferred device type ("cuda", "mps", "cpu", or empty for auto-detect)

        Returns:
            Tuple of (device_type, onnx_providers)
            - device_type: "cuda", "mps", or "cpu"
            - onnx_providers: List of ONNX Runtime providers for InsightFace

        Examples:
            >>> device, providers = DeviceDetector.detect_device()
            >>> print(f"Using device: {device}")
            Using device: cuda
        """
        device_type = "cpu"
        onnx_providers = ["CPUExecutionProvider"]

        # If preferred device is specified, try to use it
        if preferred_device:
            if preferred_device == "cuda":
                if DeviceDetector._check_cuda():
                    device_type = "cuda"
                    onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    logger.info("Using CUDA device (specified in config)")
                else:
                    logger.warning("CUDA requested but not available, falling back to CPU")
            elif preferred_device == "mps":
                if DeviceDetector._check_mps():
                    device_type = "mps"
                    onnx_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                    logger.info("Using MPS device (specified in config)")
                else:
                    logger.warning("MPS requested but not available, falling back to CPU")
            elif preferred_device == "cpu":
                logger.info("Using CPU device (specified in config)")
            else:
                logger.warning(f"Unknown device '{preferred_device}', using auto-detect")
                return DeviceDetector._auto_detect()
        else:
            # Auto-detect
            return DeviceDetector._auto_detect()

        return device_type, onnx_providers

    @staticmethod
    def _auto_detect() -> Tuple[str, List[str]]:
        """Auto-detect the best available device."""
        # Check CUDA first (highest priority)
        if DeviceDetector._check_cuda():
            logger.info("Auto-detected CUDA device")
            return "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Check MPS (Mac Metal)
        if DeviceDetector._check_mps():
            logger.info("Auto-detected MPS (Mac Metal) device")
            return "mps", ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        # Fallback to CPU
        logger.info("No GPU detected, using CPU")
        return "cpu", ["CPUExecutionProvider"]

    @staticmethod
    def _check_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.debug("PyTorch not installed, CUDA check skipped")
            return False
        except Exception as e:
            logger.debug(f"Error checking CUDA availability: {e}")
            return False

    @staticmethod
    def _check_mps() -> bool:
        """Check if MPS (Mac Metal) is available."""
        try:
            import torch
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except ImportError:
            logger.debug("PyTorch not installed, MPS check skipped")
            return False
        except Exception as e:
            logger.debug(f"Error checking MPS availability: {e}")
            return False

    @staticmethod
    def get_torch_device(device_type: str) -> str:
        """
        Get the appropriate torch device string.

        Args:
            device_type: Device type ("cuda", "mps", or "cpu")

        Returns:
            Device string for PyTorch models ("cuda", "mps", or "cpu")
        """
        return device_type

    @staticmethod
    def get_device_info(device_type: str) -> dict:
        """
        Get detailed information about the detected device.

        Args:
            device_type: Device type ("cuda", "mps", or "cpu")

        Returns:
            Dictionary with device information
        """
        info = {
            "device_type": device_type,
            "device_name": "CPU",
            "available_memory": None
        }

        if device_type == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    info["device_name"] = torch.cuda.get_device_name(0)
                    info["available_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            except Exception as e:
                logger.debug(f"Error getting CUDA device info: {e}")

        elif device_type == "mps":
            info["device_name"] = "Apple Metal (MPS)"
            # MPS doesn't provide easy memory info, so we'll leave it as None

        return info


# Convenience function for quick device detection
def get_device(preferred_device: str = "") -> Tuple[str, List[str]]:
    """
    Convenience function to detect device.

    Args:
        preferred_device: Preferred device type (empty for auto-detect)

    Returns:
        Tuple of (device_type, onnx_providers)
    """
    return DeviceDetector.detect_device(preferred_device)
