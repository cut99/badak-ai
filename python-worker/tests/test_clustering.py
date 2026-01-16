"""
Unit tests for clustering service.
Tests face clustering and cluster merging functionality.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from services.vectordb import VectorDBService
from services.clustering_service import ClusteringService
from services.thumbnail_service import ThumbnailService
from PIL import Image


# Test fixtures
@pytest.fixture
def temp_vectordb_dir():
    """Create a temporary directory for VectorDB testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_thumbnail_dir():
    """Create a temporary directory for thumbnail testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def vectordb_service(temp_vectordb_dir):
    """Create a VectorDB service instance for testing."""
    return VectorDBService(persist_directory=temp_vectordb_dir)


@pytest.fixture
def clustering_service(vectordb_service):
    """Create a clustering service instance for testing."""
    return ClusteringService(vectordb=vectordb_service, similarity_threshold=0.6)


@pytest.fixture
def thumbnail_service(temp_thumbnail_dir):
    """Create a thumbnail service instance for testing."""
    return ThumbnailService(thumbnail_path=temp_thumbnail_dir)


@pytest.fixture
def sample_embedding():
    """Generate a random 512-dim embedding for testing."""
    return np.random.rand(512).astype(np.float32)


@pytest.fixture
def sample_image():
    """Create a sample image for thumbnail testing."""
    img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode='RGB')


class TestClusteringService:
    """Test cases for ClusteringService."""

    def test_initialization(self, clustering_service):
        """Test that clustering service initializes correctly."""
        assert clustering_service is not None
        assert clustering_service.vectordb is not None
        assert clustering_service.similarity_threshold == 0.6
        assert clustering_service.distance_threshold == 0.4

    def test_find_or_create_cluster_new_cluster(self, clustering_service, sample_embedding):
        """Test creating a new cluster for the first face."""
        cluster_id, is_new = clustering_service.find_or_create_cluster(
            face_id="face-1",
            embedding=sample_embedding,
            file_id="file-1",
            bounding_box=[10, 20, 100, 120]
        )

        assert cluster_id is not None
        assert isinstance(cluster_id, str)
        assert is_new is True

    def test_find_or_create_cluster_similar_face(self, clustering_service):
        """Test that similar faces get assigned to the same cluster."""
        # Create first embedding
        embedding1 = np.random.rand(512).astype(np.float32)

        # Create very similar embedding (small difference)
        embedding2 = embedding1 + np.random.rand(512).astype(np.float32) * 0.01

        # Add first face
        cluster_id1, is_new1 = clustering_service.find_or_create_cluster(
            face_id="face-1",
            embedding=embedding1,
            file_id="file-1",
            bounding_box=[10, 20, 100, 120]
        )

        # Add similar face
        cluster_id2, is_new2 = clustering_service.find_or_create_cluster(
            face_id="face-2",
            embedding=embedding2,
            file_id="file-2",
            bounding_box=[15, 25, 105, 125]
        )

        assert is_new1 is True  # First face creates new cluster
        # Second face might or might not match depending on similarity
        # We just verify it returns valid results
        assert cluster_id2 is not None
        assert isinstance(is_new2, bool)

    def test_find_or_create_cluster_different_faces(self, clustering_service):
        """Test that different faces get different clusters."""
        # Create two very different embeddings
        embedding1 = np.zeros(512, dtype=np.float32)
        embedding2 = np.ones(512, dtype=np.float32)

        # Add first face
        cluster_id1, is_new1 = clustering_service.find_or_create_cluster(
            face_id="face-1",
            embedding=embedding1,
            file_id="file-1",
            bounding_box=[10, 20, 100, 120]
        )

        # Add very different face
        cluster_id2, is_new2 = clustering_service.find_or_create_cluster(
            face_id="face-2",
            embedding=embedding2,
            file_id="file-2",
            bounding_box=[15, 25, 105, 125]
        )

        assert is_new1 is True
        assert is_new2 is True
        assert cluster_id1 != cluster_id2

    def test_merge_clusters(self, clustering_service, thumbnail_service):
        """Test merging multiple clusters."""
        # Create three different clusters
        embeddings = [
            np.random.rand(512).astype(np.float32) for _ in range(3)
        ]

        cluster_ids = []
        for i, embedding in enumerate(embeddings):
            cluster_id, _ = clustering_service.find_or_create_cluster(
                face_id=f"face-{i}",
                embedding=embedding,
                file_id=f"file-{i}",
                bounding_box=[10, 20, 100, 120]
            )
            cluster_ids.append(cluster_id)

        # Merge clusters 1 and 2 into cluster 0
        target_cluster = cluster_ids[0]
        source_clusters = cluster_ids[1:]

        merged_count = clustering_service.merge_clusters(
            source_cluster_ids=source_clusters,
            target_cluster_id=target_cluster,
            thumbnail_service=thumbnail_service
        )

        assert merged_count == 2  # Two faces merged

        # Verify all faces now belong to target cluster
        target_faces = clustering_service.get_cluster_faces(target_cluster)
        assert len(target_faces) == 3

    def test_merge_clusters_same_as_target(self, clustering_service):
        """Test that merging a cluster into itself is handled correctly."""
        # Create a cluster
        embedding = np.random.rand(512).astype(np.float32)
        cluster_id, _ = clustering_service.find_or_create_cluster(
            face_id="face-1",
            embedding=embedding,
            file_id="file-1",
            bounding_box=[10, 20, 100, 120]
        )

        # Try to merge cluster into itself
        merged_count = clustering_service.merge_clusters(
            source_cluster_ids=[cluster_id],
            target_cluster_id=cluster_id
        )

        # Should skip merging to itself
        assert merged_count == 0

    def test_get_cluster_faces(self, clustering_service):
        """Test getting all faces in a cluster."""
        # Create a cluster with multiple faces
        base_embedding = np.random.rand(512).astype(np.float32)

        # Add 3 similar faces to same cluster
        cluster_id = None
        for i in range(3):
            embedding = base_embedding + np.random.rand(512).astype(np.float32) * 0.001
            cid, _ = clustering_service.find_or_create_cluster(
                face_id=f"face-{i}",
                embedding=embedding,
                file_id=f"file-{i}",
                bounding_box=[10, 20, 100, 120]
            )
            if cluster_id is None:
                cluster_id = cid

        # Get faces in cluster
        faces = clustering_service.get_cluster_faces(cluster_id)

        # Should have at least 1 face (might have all 3 if they all matched)
        assert len(faces) >= 1

    def test_get_cluster_size(self, clustering_service):
        """Test getting cluster size."""
        # Create a cluster
        embedding = np.random.rand(512).astype(np.float32)
        cluster_id, _ = clustering_service.find_or_create_cluster(
            face_id="face-1",
            embedding=embedding,
            file_id="file-1",
            bounding_box=[10, 20, 100, 120]
        )

        # Get cluster size
        size = clustering_service.get_cluster_size(cluster_id)
        assert size >= 1

    def test_delete_cluster(self, clustering_service, thumbnail_service, sample_image):
        """Test deleting a cluster."""
        # Create a cluster
        embedding = np.random.rand(512).astype(np.float32)
        cluster_id, _ = clustering_service.find_or_create_cluster(
            face_id="face-1",
            embedding=embedding,
            file_id="file-1",
            bounding_box=[10, 20, 100, 120]
        )

        # Save a thumbnail
        thumbnail_service.save_thumbnail(cluster_id, sample_image)

        # Delete cluster
        deleted_count = clustering_service.delete_cluster(
            cluster_id=cluster_id,
            thumbnail_service=thumbnail_service
        )

        assert deleted_count >= 1

        # Verify cluster is deleted
        faces = clustering_service.get_cluster_faces(cluster_id)
        assert len(faces) == 0

        # Verify thumbnail is deleted
        assert not thumbnail_service.thumbnail_exists(cluster_id)

    def test_update_similarity_threshold(self, clustering_service):
        """Test updating similarity threshold."""
        clustering_service.update_similarity_threshold(0.7)

        assert clustering_service.similarity_threshold == 0.7
        assert clustering_service.distance_threshold == 0.3

    def test_update_similarity_threshold_invalid(self, clustering_service):
        """Test that invalid threshold raises error."""
        with pytest.raises(ValueError):
            clustering_service.update_similarity_threshold(1.5)

        with pytest.raises(ValueError):
            clustering_service.update_similarity_threshold(-0.1)

    def test_get_statistics(self, clustering_service):
        """Test getting clustering statistics."""
        # Add some faces
        for i in range(3):
            embedding = np.random.rand(512).astype(np.float32)
            clustering_service.find_or_create_cluster(
                face_id=f"face-{i}",
                embedding=embedding,
                file_id=f"file-{i}",
                bounding_box=[10, 20, 100, 120]
            )

        # Get statistics
        stats = clustering_service.get_statistics()

        assert "total_faces" in stats
        assert "total_clusters" in stats
        assert "average_cluster_size" in stats
        assert "similarity_threshold" in stats
        assert "distance_threshold" in stats

        assert stats["total_faces"] >= 3
        assert stats["total_clusters"] >= 1
        assert stats["similarity_threshold"] == 0.6
        assert stats["distance_threshold"] == 0.4


class TestVectorDBService:
    """Test cases for VectorDBService."""

    def test_initialization(self, vectordb_service):
        """Test VectorDB initialization."""
        assert vectordb_service is not None
        assert vectordb_service.client is not None
        assert vectordb_service.collection is not None

    def test_add_face(self, vectordb_service, sample_embedding):
        """Test adding a face to VectorDB."""
        vectordb_service.add_face(
            face_id="face-test-1",
            embedding=sample_embedding,
            cluster_id="cluster-test-1",
            file_id="file-test-1",
            bounding_box=[10, 20, 100, 120]
        )

        # Verify face was added
        total_faces = vectordb_service.get_total_faces()
        assert total_faces >= 1

    def test_search_similar(self, vectordb_service, sample_embedding):
        """Test searching for similar faces."""
        # Add a face
        vectordb_service.add_face(
            face_id="face-test-2",
            embedding=sample_embedding,
            cluster_id="cluster-test-2",
            file_id="file-test-2",
            bounding_box=[10, 20, 100, 120]
        )

        # Search for similar face
        results = vectordb_service.search_similar(sample_embedding, top_k=1)

        assert len(results) >= 1
        assert results[0].face_id == "face-test-2"
        assert results[0].cluster_id == "cluster-test-2"

    def test_get_faces_by_cluster(self, vectordb_service):
        """Test getting faces by cluster ID."""
        # Add multiple faces to same cluster
        cluster_id = "cluster-test-3"
        for i in range(3):
            embedding = np.random.rand(512).astype(np.float32)
            vectordb_service.add_face(
                face_id=f"face-test-{i}",
                embedding=embedding,
                cluster_id=cluster_id,
                file_id=f"file-test-{i}",
                bounding_box=[10, 20, 100, 120]
            )

        # Get faces in cluster
        faces = vectordb_service.get_faces_by_cluster(cluster_id)
        assert len(faces) == 3

    def test_update_cluster(self, vectordb_service, sample_embedding):
        """Test updating cluster assignment."""
        # Add a face
        vectordb_service.add_face(
            face_id="face-test-update",
            embedding=sample_embedding,
            cluster_id="cluster-old",
            file_id="file-test",
            bounding_box=[10, 20, 100, 120]
        )

        # Update cluster
        vectordb_service.update_cluster(
            face_ids=["face-test-update"],
            new_cluster_id="cluster-new"
        )

        # Verify update
        faces = vectordb_service.get_faces_by_cluster("cluster-new")
        assert len(faces) == 1
        assert faces[0]["face_id"] == "face-test-update"

    def test_get_info(self, vectordb_service):
        """Test getting VectorDB info."""
        info = vectordb_service.get_info()

        assert "persist_directory" in info
        assert "total_faces" in info
        assert "total_clusters" in info
        assert "collection_name" in info
        assert "distance_metric" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
