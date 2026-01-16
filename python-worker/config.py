"""
Configuration management for AI Worker.
Loads environment variables and provides centralized config.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # Project root directory
    BASE_DIR: Path = Path(__file__).resolve().parent

    # Device configuration
    DEVICE: str = os.getenv("DEVICE", "")  # auto-detect if empty

    # API Security
    API_KEY: str = os.getenv("API_KEY", "")
    ALLOWED_IPS: List[str] = os.getenv("ALLOWED_IPS", "127.0.0.1").split(",")

    # VectorDB configuration
    VECTORDB_PATH: str = os.getenv("VECTORDB_PATH", "./data/vectordb")

    # Thumbnail storage
    THUMBNAIL_PATH: str = os.getenv("THUMBNAIL_PATH", "./data/thumbnails")

    # Model settings
    FACE_SIMILARITY_THRESHOLD: float = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.6"))
    TAG_THRESHOLD: float = float(os.getenv("TAG_THRESHOLD", "0.25"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def __init__(self):
        """Initialize settings and create necessary directories."""
        # Ensure data directories exist
        Path(self.VECTORDB_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.THUMBNAIL_PATH).mkdir(parents=True, exist_ok=True)

        # Validate required settings
        if not self.API_KEY:
            raise ValueError("API_KEY must be set in environment variables")


# Global settings instance
settings = Settings()
