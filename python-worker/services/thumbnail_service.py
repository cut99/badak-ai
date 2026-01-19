"""
Thumbnail service for storing and retrieving face crop images.
Saves representative face images for each cluster.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ThumbnailService:
    """Face thumbnail storage and retrieval service."""

    def __init__(self, thumbnail_path: str = "./data/thumbnails"):
        """
        Initialize Thumbnail service.

        Args:
            thumbnail_path: Directory path for storing thumbnails
        """
        self.thumbnail_path = Path(thumbnail_path)
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure thumbnail directory exists."""
        try:
            self.thumbnail_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Thumbnail directory ready at {self.thumbnail_path}")
        except Exception as e:
            logger.error(f"Failed to create thumbnail directory: {e}")
            raise

    def _get_thumbnail_filepath(self, cluster_id: str) -> Path:
        """
        Get the file path for a cluster thumbnail.

        Args:
            cluster_id: Cluster ID

        Returns:
            Path object for the thumbnail file
        """
        return self.thumbnail_path / f"{cluster_id}.jpg"

    def save_thumbnail(
        self,
        cluster_id: str,
        face_crop: Image.Image,
        overwrite: bool = False,
        quality: int = 85,
        max_size: int = 256
    ) -> bool:
        """
        Save a face crop as cluster thumbnail.

        Only saves if thumbnail doesn't exist (first face = representative),
        unless overwrite=True.

        Args:
            cluster_id: Cluster ID
            face_crop: PIL Image of cropped face
            overwrite: If True, overwrite existing thumbnail (default: False)
            quality: JPEG quality 1-100 (default: 85)
            max_size: Maximum width/height in pixels (default: 256)

        Returns:
            True if thumbnail was saved, False if skipped (already exists)

        Example:
            >>> thumbnail_service = ThumbnailService()
            >>> face_crop = Image.open("face.jpg")
            >>> saved = thumbnail_service.save_thumbnail("cluster-123", face_crop)
            >>> if saved:
            ...     print("Thumbnail saved")
            ... else:
            ...     print("Thumbnail already exists, skipped")
        """
        try:
            filepath = self._get_thumbnail_filepath(cluster_id)

            # Check if thumbnail already exists
            if filepath.exists() and not overwrite:
                logger.debug(f"Thumbnail for cluster {cluster_id} already exists, skipping")
                return False

            # Resize image to max_size while maintaining aspect ratio
            face_crop_resized = self._resize_image(face_crop, max_size)

            # Save as JPEG
            face_crop_resized.save(
                filepath,
                "JPEG",
                quality=quality,
                optimize=True
            )

            logger.debug(f"Saved thumbnail for cluster {cluster_id} at {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving thumbnail for cluster {cluster_id}: {e}")
            raise

    def _resize_image(self, image: Image.Image, max_size: int) -> Image.Image:
        """
        Resize image to fit within max_size while maintaining aspect ratio.

        Args:
            image: PIL Image to resize
            max_size: Maximum width or height

        Returns:
            Resized PIL Image
        """
        # Calculate new size maintaining aspect ratio
        width, height = image.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def get_thumbnail(self, cluster_id: str) -> Optional[bytes]:
        """
        Get thumbnail image as bytes.

        Args:
            cluster_id: Cluster ID

        Returns:
            Thumbnail image bytes (JPEG format) or None if not found

        Example:
            >>> thumbnail_service = ThumbnailService()
            >>> image_bytes = thumbnail_service.get_thumbnail("cluster-123")
            >>> if image_bytes:
            ...     # Save or return via API
            ...     with open("output.jpg", "wb") as f:
            ...         f.write(image_bytes)
        """
        try:
            filepath = self._get_thumbnail_filepath(cluster_id)

            if not filepath.exists():
                logger.debug(f"Thumbnail for cluster {cluster_id} not found")
                return None

            # Read and return file bytes
            with open(filepath, "rb") as f:
                image_bytes = f.read()

            logger.debug(f"Retrieved thumbnail for cluster {cluster_id}")
            return image_bytes

        except Exception as e:
            logger.error(f"Error getting thumbnail for cluster {cluster_id}: {e}")
            raise

    def get_thumbnail_as_image(self, cluster_id: str) -> Optional[Image.Image]:
        """
        Get thumbnail as PIL Image.

        Args:
            cluster_id: Cluster ID

        Returns:
            PIL Image object or None if not found

        Example:
            >>> thumbnail_service = ThumbnailService()
            >>> image = thumbnail_service.get_thumbnail_as_image("cluster-123")
            >>> if image:
            ...     image.show()
        """
        try:
            image_bytes = self.get_thumbnail(cluster_id)
            if image_bytes:
                return Image.open(io.BytesIO(image_bytes))
            return None

        except Exception as e:
            logger.error(f"Error loading thumbnail as image for cluster {cluster_id}: {e}")
            raise

    def delete_thumbnail(self, cluster_id: str) -> bool:
        """
        Delete thumbnail for a cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            True if deleted, False if not found

        Example:
            >>> thumbnail_service = ThumbnailService()
            >>> deleted = thumbnail_service.delete_thumbnail("cluster-123")
            >>> if deleted:
            ...     print("Thumbnail deleted")
        """
        try:
            filepath = self._get_thumbnail_filepath(cluster_id)

            if not filepath.exists():
                logger.debug(f"Thumbnail for cluster {cluster_id} not found, nothing to delete")
                return False

            # Delete file
            filepath.unlink()
            logger.debug(f"Deleted thumbnail for cluster {cluster_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting thumbnail for cluster {cluster_id}: {e}")
            raise

    def thumbnail_exists(self, cluster_id: str) -> bool:
        """
        Check if thumbnail exists for a cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            True if thumbnail exists, False otherwise

        Example:
            >>> thumbnail_service = ThumbnailService()
            >>> if thumbnail_service.thumbnail_exists("cluster-123"):
            ...     print("Thumbnail exists")
        """
        filepath = self._get_thumbnail_filepath(cluster_id)
        return filepath.exists()

    def get_thumbnail_count(self) -> int:
        """
        Get the total number of thumbnails stored.

        Returns:
            Number of thumbnail files

        Example:
            >>> thumbnail_service = ThumbnailService()
            >>> count = thumbnail_service.get_thumbnail_count()
            >>> print(f"Total thumbnails: {count}")
        """
        try:
            jpg_files = list(self.thumbnail_path.glob("*.jpg"))
            return len(jpg_files)
        except Exception as e:
            logger.error(f"Error counting thumbnails: {e}")
            raise

    def cleanup_orphaned_thumbnails(self, valid_cluster_ids: set) -> int:
        """
        Delete thumbnails that don't have corresponding clusters.

        Args:
            valid_cluster_ids: Set of valid cluster IDs

        Returns:
            Number of orphaned thumbnails deleted

        Example:
            >>> thumbnail_service = ThumbnailService()
            >>> valid_clusters = {"cluster-1", "cluster-2", "cluster-3"}
            >>> deleted = thumbnail_service.cleanup_orphaned_thumbnails(valid_clusters)
            >>> print(f"Deleted {deleted} orphaned thumbnails")
        """
        try:
            deleted_count = 0
            jpg_files = list(self.thumbnail_path.glob("*.jpg"))

            for filepath in jpg_files:
                # Extract cluster_id from filename (remove .jpg extension)
                cluster_id = filepath.stem

                # Delete if cluster_id not in valid set
                if cluster_id not in valid_cluster_ids:
                    filepath.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted orphaned thumbnail: {cluster_id}")

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} orphaned thumbnails")

            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up orphaned thumbnails: {e}")
            raise

    def get_info(self) -> dict:
        """
        Get thumbnail service information.

        Returns:
            Dictionary with service stats

        Example:
            >>> thumbnail_service = ThumbnailService()
            >>> info = thumbnail_service.get_info()
            >>> print(info)
        """
        return {
            "thumbnail_path": str(self.thumbnail_path),
            "total_thumbnails": self.get_thumbnail_count(),
            "path_exists": self.thumbnail_path.exists()
        }
