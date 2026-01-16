"""
Clustering service for face recognition and cluster management.
Handles finding or creating clusters based on face similarity.
"""

import logging
import numpy as np
from typing import Optional, List, Tuple
from uuid import uuid4
from services.vectordb import VectorDBService

logger = logging.getLogger(__name__)


class ClusteringService:
    """Face clustering logic using VectorDB similarity search."""

    def __init__(
        self,
        vectordb: VectorDBService,
        similarity_threshold: float = 0.6
    ):
        """
        Initialize Clustering service.

        Args:
            vectordb: VectorDBService instance
            similarity_threshold: Similarity threshold for clustering (default: 0.6)
                                 Faces with distance <= (1 - threshold) are considered same person
        """
        self.vectordb = vectordb
        self.similarity_threshold = similarity_threshold
        # Convert similarity threshold to distance threshold
        # Cosine similarity of 0.6 = cosine distance of 0.4
        self.distance_threshold = 1.0 - similarity_threshold

    def find_or_create_cluster(
        self,
        face_id: str,
        embedding: np.ndarray,
        file_id: str,
        bounding_box: List[int],
        metadata: Optional[dict] = None
    ) -> Tuple[str, bool]:
        """
        Find existing cluster for a face or create a new one.

        Algorithm:
        1. Search VectorDB for most similar face (top 1)
        2. If distance <= threshold (0.4), return existing cluster_id
        3. Else create new cluster, save to VectorDB, return new cluster_id

        Args:
            face_id: Unique face identifier
            embedding: 512-dim face embedding vector
            file_id: Original file/image identifier
            bounding_box: [x1, y1, x2, y2]
            metadata: Additional metadata (optional)

        Returns:
            Tuple of (cluster_id, is_new_cluster)

        Example:
            >>> clustering = ClusteringService(vectordb)
            >>> cluster_id, is_new = clustering.find_or_create_cluster(
            ...     "face-123", embedding, "file-456", [10, 20, 100, 120]
            ... )
            >>> if is_new:
            ...     print(f"Created new cluster: {cluster_id}")
            ... else:
            ...     print(f"Matched to existing cluster: {cluster_id}")
        """
        try:
            # Search for similar faces
            results = self.vectordb.search_similar(embedding, top_k=1)

            cluster_id = None
            is_new_cluster = True

            if results and results[0].distance <= self.distance_threshold:
                # Found similar face - use existing cluster
                cluster_id = results[0].cluster_id
                is_new_cluster = False
                logger.debug(
                    f"Face {face_id} matched to existing cluster {cluster_id} "
                    f"(distance: {results[0].distance:.3f})"
                )
            else:
                # No similar face found - create new cluster
                cluster_id = str(uuid4())
                is_new_cluster = True
                logger.debug(f"Face {face_id} assigned to new cluster {cluster_id}")

            # Add face to VectorDB
            self.vectordb.add_face(
                face_id=face_id,
                embedding=embedding,
                cluster_id=cluster_id,
                file_id=file_id,
                bounding_box=bounding_box,
                metadata=metadata
            )

            return cluster_id, is_new_cluster

        except Exception as e:
            logger.error(f"Error in find_or_create_cluster: {e}")
            raise

    def merge_clusters(
        self,
        source_cluster_ids: List[str],
        target_cluster_id: str,
        thumbnail_service=None
    ) -> int:
        """
        Merge multiple source clusters into a target cluster.

        Updates all faces from source clusters to target cluster_id.
        Optionally deletes source cluster thumbnails.

        Args:
            source_cluster_ids: List of source cluster IDs to merge
            target_cluster_id: Target cluster ID to merge into
            thumbnail_service: Optional ThumbnailService instance for cleanup

        Returns:
            Number of faces merged

        Example:
            >>> clustering = ClusteringService(vectordb)
            >>> merged_count = clustering.merge_clusters(
            ...     ["cluster-1", "cluster-2"],
            ...     "cluster-1",
            ...     thumbnail_service
            ... )
            >>> print(f"Merged {merged_count} faces")
        """
        try:
            total_merged = 0

            for source_id in source_cluster_ids:
                # Skip if source is same as target
                if source_id == target_cluster_id:
                    logger.debug(f"Skipping merge: source {source_id} is same as target")
                    continue

                # Get all faces in source cluster
                faces = self.vectordb.get_faces_by_cluster(source_id)

                if not faces:
                    logger.debug(f"Source cluster {source_id} has no faces, skipping")
                    continue

                # Update all faces to target cluster
                face_ids = [face["face_id"] for face in faces]
                self.vectordb.update_cluster(face_ids, target_cluster_id)

                total_merged += len(face_ids)
                logger.info(
                    f"Merged {len(face_ids)} faces from cluster {source_id} "
                    f"to cluster {target_cluster_id}"
                )

                # Delete source thumbnail if thumbnail_service provided
                if thumbnail_service:
                    try:
                        thumbnail_service.delete_thumbnail(source_id)
                        logger.debug(f"Deleted thumbnail for source cluster {source_id}")
                    except Exception as e:
                        logger.warning(f"Failed to delete thumbnail for {source_id}: {e}")

            logger.info(f"Merge complete: {total_merged} total faces merged to {target_cluster_id}")
            return total_merged

        except Exception as e:
            logger.error(f"Error merging clusters: {e}")
            raise

    def get_cluster_faces(self, cluster_id: str) -> List[dict]:
        """
        Get all faces in a cluster.

        Args:
            cluster_id: Cluster ID to query

        Returns:
            List of face dictionaries

        Example:
            >>> clustering = ClusteringService(vectordb)
            >>> faces = clustering.get_cluster_faces("cluster-123")
            >>> print(f"Cluster has {len(faces)} faces")
        """
        try:
            return self.vectordb.get_faces_by_cluster(cluster_id)
        except Exception as e:
            logger.error(f"Error getting cluster faces: {e}")
            raise

    def get_cluster_size(self, cluster_id: str) -> int:
        """
        Get the number of faces in a cluster.

        Args:
            cluster_id: Cluster ID to query

        Returns:
            Number of faces in the cluster

        Example:
            >>> clustering = ClusteringService(vectordb)
            >>> size = clustering.get_cluster_size("cluster-123")
            >>> print(f"Cluster size: {size}")
        """
        try:
            faces = self.vectordb.get_faces_by_cluster(cluster_id)
            return len(faces)
        except Exception as e:
            logger.error(f"Error getting cluster size: {e}")
            raise

    def delete_cluster(self, cluster_id: str, thumbnail_service=None) -> int:
        """
        Delete a cluster and all its faces.

        Args:
            cluster_id: Cluster ID to delete
            thumbnail_service: Optional ThumbnailService for thumbnail cleanup

        Returns:
            Number of faces deleted

        Example:
            >>> clustering = ClusteringService(vectordb)
            >>> deleted = clustering.delete_cluster("cluster-123", thumbnail_service)
            >>> print(f"Deleted cluster with {deleted} faces")
        """
        try:
            # Delete from VectorDB
            deleted_count = self.vectordb.delete_cluster(cluster_id)

            # Delete thumbnail if service provided
            if thumbnail_service:
                try:
                    thumbnail_service.delete_thumbnail(cluster_id)
                    logger.debug(f"Deleted thumbnail for cluster {cluster_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete thumbnail for {cluster_id}: {e}")

            logger.info(f"Deleted cluster {cluster_id} with {deleted_count} faces")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting cluster: {e}")
            raise

    def update_similarity_threshold(self, new_threshold: float):
        """
        Update the similarity threshold for clustering.

        Args:
            new_threshold: New similarity threshold (0.0 to 1.0)

        Example:
            >>> clustering = ClusteringService(vectordb)
            >>> clustering.update_similarity_threshold(0.7)
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")

        self.similarity_threshold = new_threshold
        self.distance_threshold = 1.0 - new_threshold
        logger.info(f"Updated similarity threshold to {new_threshold} (distance: {self.distance_threshold})")

    def get_statistics(self) -> dict:
        """
        Get clustering statistics.

        Returns:
            Dictionary with clustering stats

        Example:
            >>> clustering = ClusteringService(vectordb)
            >>> stats = clustering.get_statistics()
            >>> print(stats)
        """
        try:
            total_faces = self.vectordb.get_total_faces()
            total_clusters = self.vectordb.get_cluster_count()

            avg_cluster_size = total_faces / total_clusters if total_clusters > 0 else 0

            return {
                "total_faces": total_faces,
                "total_clusters": total_clusters,
                "average_cluster_size": round(avg_cluster_size, 2),
                "similarity_threshold": self.similarity_threshold,
                "distance_threshold": self.distance_threshold
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise

    def get_all_clusters_with_metadata(self) -> List[dict]:
        """
        Get all clusters with metadata.

        Returns:
            List of cluster dictionaries with metadata including:
            - cluster_id
            - face_count
            - file_ids (unique list of files containing this person)
            - created_at (timestamp of first face)

        Example:
            >>> clustering = ClusteringService(vectordb)
            >>> clusters = clustering.get_all_clusters_with_metadata()
            >>> for cluster in clusters:
            ...     print(f"{cluster['cluster_id']}: {cluster['face_count']} faces")
        """
        try:
            # Get all faces from VectorDB
            all_faces = self.vectordb.collection.get(include=["metadatas"])

            # Group by cluster_id
            clusters_dict = {}
            for i, metadata in enumerate(all_faces.get("metadatas", [])):
                cluster_id = metadata.get("cluster_id")
                if not cluster_id:
                    continue

                if cluster_id not in clusters_dict:
                    clusters_dict[cluster_id] = {
                        "cluster_id": cluster_id,
                        "face_count": 0,
                        "file_ids": [],
                        "created_at": metadata.get("created_at")
                    }

                clusters_dict[cluster_id]["face_count"] += 1
                file_id = metadata.get("file_id")
                if file_id and file_id not in clusters_dict[cluster_id]["file_ids"]:
                    clusters_dict[cluster_id]["file_ids"].append(file_id)

            logger.debug(f"Retrieved {len(clusters_dict)} clusters with metadata")
            return list(clusters_dict.values())

        except Exception as e:
            logger.error(f"Error getting all clusters: {e}")
            raise
