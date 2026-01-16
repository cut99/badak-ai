"""
Image downloader service for fetching images from URLs.
Uses httpx async client for efficient image downloading.
"""

import logging
import httpx
from typing import Optional
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ImageDownloader:
    """Async image downloader using httpx."""

    def __init__(self, timeout: float = 30.0, max_size_mb: int = 50):
        """
        Initialize Image Downloader.

        Args:
            timeout: Request timeout in seconds (default: 30.0)
            max_size_mb: Maximum image size in MB (default: 50)
        """
        self.timeout = timeout
        self.max_size_bytes = max_size_mb * 1024 * 1024

    async def download_image(self, url: str) -> Image.Image:
        """
        Download image from URL asynchronously.

        Args:
            url: Image URL (supports http/https)

        Returns:
            PIL Image object

        Raises:
            httpx.HTTPError: If download fails
            ValueError: If image is too large or invalid format
            PIL.UnidentifiedImageError: If image format cannot be identified

        Example:
            >>> downloader = ImageDownloader()
            >>> image = await downloader.download_image("https://example.com/photo.jpg")
            >>> print(f"Downloaded image: {image.size}")
        """
        try:
            logger.debug(f"Downloading image from: {url}")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Check content length
                content_length = response.headers.get("content-length")
                if content_length:
                    size = int(content_length)
                    if size > self.max_size_bytes:
                        raise ValueError(
                            f"Image too large: {size / 1024 / 1024:.2f} MB "
                            f"(max: {self.max_size_bytes / 1024 / 1024} MB)"
                        )

                # Get image bytes
                image_bytes = response.content

                # Verify actual size
                if len(image_bytes) > self.max_size_bytes:
                    raise ValueError(
                        f"Image too large: {len(image_bytes) / 1024 / 1024:.2f} MB "
                        f"(max: {self.max_size_bytes / 1024 / 1024} MB)"
                    )

                # Load image
                image = Image.open(io.BytesIO(image_bytes))

                # Verify image loaded properly
                image.verify()

                # Reload image after verify (verify() closes the file)
                image = Image.open(io.BytesIO(image_bytes))

                logger.debug(f"Downloaded image: {image.size[0]}x{image.size[1]}, format: {image.format}")
                return image

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading image from {url}: {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error downloading image from {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            raise

    def download_image_sync(self, url: str) -> Image.Image:
        """
        Download image from URL synchronously.

        Args:
            url: Image URL (supports http/https)

        Returns:
            PIL Image object

        Raises:
            httpx.HTTPError: If download fails
            ValueError: If image is too large or invalid format
            PIL.UnidentifiedImageError: If image format cannot be identified

        Example:
            >>> downloader = ImageDownloader()
            >>> image = downloader.download_image_sync("https://example.com/photo.jpg")
            >>> print(f"Downloaded image: {image.size}")
        """
        try:
            logger.debug(f"Downloading image from: {url}")

            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                response.raise_for_status()

                # Check content length
                content_length = response.headers.get("content-length")
                if content_length:
                    size = int(content_length)
                    if size > self.max_size_bytes:
                        raise ValueError(
                            f"Image too large: {size / 1024 / 1024:.2f} MB "
                            f"(max: {self.max_size_bytes / 1024 / 1024} MB)"
                        )

                # Get image bytes
                image_bytes = response.content

                # Verify actual size
                if len(image_bytes) > self.max_size_bytes:
                    raise ValueError(
                        f"Image too large: {len(image_bytes) / 1024 / 1024:.2f} MB "
                        f"(max: {self.max_size_bytes / 1024 / 1024} MB)"
                    )

                # Load image
                image = Image.open(io.BytesIO(image_bytes))

                # Verify image loaded properly
                image.verify()

                # Reload image after verify (verify() closes the file)
                image = Image.open(io.BytesIO(image_bytes))

                logger.debug(f"Downloaded image: {image.size[0]}x{image.size[1]}, format: {image.format}")
                return image

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading image from {url}: {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error downloading image from {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            raise

    async def download_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Image.Image:
        """
        Download image with retry logic.

        Args:
            url: Image URL
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)

        Returns:
            PIL Image object

        Example:
            >>> downloader = ImageDownloader()
            >>> image = await downloader.download_with_retry("https://example.com/photo.jpg")
        """
        import asyncio

        last_error = None

        for attempt in range(max_retries):
            try:
                return await self.download_image(url)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Download attempt {attempt + 1}/{max_retries} failed for {url}: {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} download attempts failed for {url}")

        # If all retries failed, raise the last error
        raise last_error

    def is_valid_image_url(self, url: str) -> bool:
        """
        Check if URL appears to be a valid image URL.

        Args:
            url: URL to check

        Returns:
            True if URL looks like an image URL, False otherwise

        Example:
            >>> downloader = ImageDownloader()
            >>> if downloader.is_valid_image_url("https://example.com/photo.jpg"):
            ...     print("Valid image URL")
        """
        if not url:
            return False

        # Check protocol
        if not (url.startswith("http://") or url.startswith("https://")):
            return False

        # Check common image extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"]
        url_lower = url.lower()

        # Check if URL ends with image extension (before query params)
        url_path = url_lower.split("?")[0]
        has_extension = any(url_path.endswith(ext) for ext in image_extensions)

        # Also accept URLs without extension (presigned URLs, etc.)
        # If no clear extension, assume it might be valid
        return True

    def get_info(self) -> dict:
        """
        Get downloader configuration.

        Returns:
            Dictionary with configuration info

        Example:
            >>> downloader = ImageDownloader()
            >>> info = downloader.get_info()
            >>> print(info)
        """
        return {
            "timeout": self.timeout,
            "max_size_mb": self.max_size_bytes / 1024 / 1024,
            "supported_protocols": ["http", "https"]
        }
