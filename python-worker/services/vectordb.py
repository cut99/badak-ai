"""
VectorDB service for face embedding storage and similarity search.
Uses ChromaDB with persistent storage and cosine distance.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from similarity search."""
    face_id: str
    cluster_id: str
    distance: float
    metadata: Dict[str, Any]


class VectorDBService:
    """Face embedding storage and similarity search using ChromaDB."""

    def __init__(self, persist_directory: str = "./data/vectordb"):
        """
        Initialize VectorDB service.

        Args:
            persist_directory: Directory for ChromaDB persistent storage
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB client and collection."""
        try:
            logger.info(f"Initializing ChromaDB at {self.persist_directory}")

            # Create ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection with cosine distance
            self.collection = self.client.get_or_create_collection(
                name="face_embeddings",
                metadata={"hnsw:space": "cosine"}
            )

            count = self.collection.count()
            logger.info(f"ChromaDB initialized successfully with {count} embeddings")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_face(
        self,
        face_id: str,
        embedding: np.ndarray,
        cluster_id: str,
        file_id: str,
        bounding_box: List[int],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a face embedding to the database.

        Args:
            face_id: Unique face identifier
            embedding: 512-dim face embedding vector
            cluster_id: Cluster ID this face belongs to
            file_id: Original file/image identifier
            bounding_box: [x1, y1, x2, y2]
            metadata: Additional metadata (optional)

        Example:
            >>> db = VectorDBService()
            >>> db.add_face("face-123", embedding, "cluster-456", "file-789", [10, 20, 100, 120])
        """
        if self.collection is None:
            raise RuntimeError("VectorDB not initialized")

        try:
            # Prepare metadata
            meta = {
                "cluster_id": cluster_id,
                "file_id": file_id,
                "bounding_box": str(bounding_box),
                "created_at": datetime.utcnow().isoformat()
            }

            # Add custom metadata if provided
            if metadata:
                meta.update(metadata)

            # Add to collection
            self.collection.add(
                ids=[face_id],
                embeddings=[embedding.tolist()],
                metadatas=[meta]
            )

            logger.debug(f"Added face {face_id} to cluster {cluster_id}")

        except Exception as e:
            logger.error(f"Error adding face to VectorDB: {e}")
            raise

    def search_similar(
        self,
        embedding: np.ndarray,
        top_k: int = 1,
        cluster_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for similar face embeddings.

        Args:
            embedding: 512-dim face embedding to search for
            top_k: Number of results to return (default: 1)
            cluster_filter: Optional cluster_id to filter results

        Returns:
            List of SearchResult objects sorted by similarity

        Example:
            >>> db = VectorDBService()
            >>> results = db.search_similar(embedding, top_k=5)
            >>> for result in results:
            ...     print(f"Found face {result.face_id} in cluster {result.cluster_id} (distance: {result.distance:.3f})")
        """
        if self.collection is None:
            raise RuntimeError("VectorDB not initialized")

        try:
            # Build query parameters
            query_params = {
                "query_embeddings": [embedding.tolist()],
                "n_results": top_k
            }

            # Add cluster filter if specified
            if cluster_filter:
                query_params["where"] = {"cluster_id": cluster_filter}

            # Search for similar embeddings
            results = self.collection.query(**query_params)

            # Convert to SearchResult objects
            search_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    face_id = results["ids"][0][i]
                    distance = results["distances"][0][i]
                    metadata = results["metadatas"][0][i]
                    cluster_id = metadata.get("cluster_id", "")

                    search_results.append(SearchResult(
                        face_id=face_id,
                        cluster_id=cluster_id,
                        distance=distance,
                        metadata=metadata
                    ))

            logger.debug(f"Found {len(search_results)} similar faces")
            return search_results

        except Exception as e:
            logger.error(f"Error searching VectorDB: {e}")
            raise

    def update_cluster(self, face_ids: List[str], new_cluster_id: str):
        """
        Update cluster_id for multiple faces.

        Args:
            face_ids: List of face IDs to update
            new_cluster_id: New cluster ID to assign

        Example:
            >>> db = VectorDBService()
            >>> db.update_cluster(["face-1", "face-2"], "cluster-new")
        """
        if self.collection is None:
            raise RuntimeError("VectorDB not initialized")

        try:
            for face_id in face_ids:
                # Get current metadata
                result = self.collection.get(ids=[face_id])
                if result["ids"]:
                    metadata = result["metadatas"][0]
                    metadata["cluster_id"] = new_cluster_id

                    # Update metadata
                    self.collection.update(
                        ids=[face_id],
                        metadatas=[metadata]
                    )

            logger.debug(f"Updated {len(face_ids)} faces to cluster {new_cluster_id}")

        except Exception as e:
            logger.error(f"Error updating cluster in VectorDB: {e}")
            raise

    def get_faces_by_cluster(self, cluster_id: str) -> List[Dict[str, Any]]:
        """
        Get all faces belonging to a cluster.

        Args:
            cluster_id: Cluster ID to query

        Returns:
            List of face dictionaries with id, embedding, and metadata

        Example:
            >>> db = VectorDBService()
            >>> faces = db.get_faces_by_cluster("cluster-123")
            >>> print(f"Cluster has {len(faces)} faces")
        """
        if self.collection is None:
            raise RuntimeError("VectorDB not initialized")

        try:
            # Query by cluster_id
            results = self.collection.get(
                where={"cluster_id": cluster_id},
                include=["embeddings", "metadatas"]
            )

            # Format results
            faces = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    faces.append({
                        "face_id": results["ids"][i],
                        "embedding": results["embeddings"][i] if results["embeddings"] else None,
                        "metadata": results["metadatas"][i] if results["metadatas"] else {}
                    })

            logger.debug(f"Found {len(faces)} faces in cluster {cluster_id}")
            return faces

        except Exception as e:
            logger.error(f"Error getting faces by cluster: {e}")
            raise

    def delete_cluster(self, cluster_id: str) -> int:
        """
        Delete all faces belonging to a cluster.

        Args:
            cluster_id: Cluster ID to delete

        Returns:
            Number of faces deleted

        Example:
            >>> db = VectorDBService()
            >>> deleted = db.delete_cluster("cluster-123")
            >>> print(f"Deleted {deleted} faces")
        """
        if self.collection is None:
            raise RuntimeError("VectorDB not initialized")

        try:
            # Get all faces in cluster
            faces = self.get_faces_by_cluster(cluster_id)
            face_ids = [face["face_id"] for face in faces]

            if face_ids:
                # Delete faces
                self.collection.delete(ids=face_ids)

            logger.debug(f"Deleted {len(face_ids)} faces from cluster {cluster_id}")
            return len(face_ids)

        except Exception as e:
            logger.error(f"Error deleting cluster: {e}")
            raise

    def get_cluster_count(self) -> int:
        """
        Get the total number of unique clusters.

        Returns:
            Number of unique clusters

        Example:
            >>> db = VectorDBService()
            >>> count = db.get_cluster_count()
            >>> print(f"Total clusters: {count}")
        """
        if self.collection is None:
            raise RuntimeError("VectorDB not initialized")

        try:
            # Get all metadatas
            results = self.collection.get(include=["metadatas"])

            # Extract unique cluster IDs
            cluster_ids = set()
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    if "cluster_id" in metadata:
                        cluster_ids.add(metadata["cluster_id"])

            return len(cluster_ids)

        except Exception as e:
            logger.error(f"Error getting cluster count: {e}")
            raise

    def get_total_faces(self) -> int:
        """
        Get the total number of faces in the database.

        Returns:
            Total number of faces

        Example:
            >>> db = VectorDBService()
            >>> total = db.get_total_faces()
            >>> print(f"Total faces: {total}")
        """
        if self.collection is None:
            raise RuntimeError("VectorDB not initialized")

        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting total faces: {e}")
            raise

    def reset_database(self):
        """
        Reset the entire database (DELETE ALL DATA).
        Use with caution!

        Example:
            >>> db = VectorDBService()
            >>> db.reset_database()
            WARNING: All data will be deleted!
        """
        if self.collection is None:
            raise RuntimeError("VectorDB not initialized")

        try:
            logger.warning("Resetting VectorDB - ALL DATA WILL BE DELETED")
            self.client.delete_collection("face_embeddings")
            self._initialize_db()
            logger.info("VectorDB reset complete")

        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            raise

    def get_info(self) -> Dict[str, Any]:
        """
        Get database information.

        Returns:
            Dictionary with database stats

        Example:
            >>> db = VectorDBService()
            >>> info = db.get_info()
            >>> print(info)
        """
        return {
            "persist_directory": self.persist_directory,
            "total_faces": self.get_total_faces(),
            "total_clusters": self.get_cluster_count(),
            "collection_name": "face_embeddings",
            "distance_metric": "cosine"
        }
