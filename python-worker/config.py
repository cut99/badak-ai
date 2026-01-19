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
    TAG_TOP_K: int = int(os.getenv("TAG_TOP_K", "10"))
    TAG_LANGUAGE: str = os.getenv("TAG_LANGUAGE", "id")  # "en" or "id"
    CONTEXT_MODE: str = os.getenv("CONTEXT_MODE", "comprehensive")  # "simple" or "comprehensive"

    # Job Queue settings
    JOB_QUEUE_MAX_WORKERS: int = int(os.getenv("JOB_QUEUE_MAX_WORKERS", "3"))
    JOB_RETENTION_HOURS: int = int(os.getenv("JOB_RETENTION_HOURS", "24"))
    JOB_QUEUE_MAX_SIZE: int = int(os.getenv("JOB_QUEUE_MAX_SIZE", "1000"))

    # Age Detection settings
    ENABLE_AGE_DETECTION: bool = os.getenv("ENABLE_AGE_DETECTION", "true").lower() == "true"

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
