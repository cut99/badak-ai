
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from services.vectordb import VectorDBService
from services.clustering_service import ClusteringService

# Test fixtures
@pytest.fixture
def temp_vectordb_dir():
    """Create a temporary directory for VectorDB testing."""
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

class TestClusterNaming:
    """Test cases for cluster naming functionality."""

    def test_update_cluster_name_success(self, clustering_service):
        """Test success when naming an existing cluster."""
        # Create a cluster first
        embedding = np.random.rand(512).astype(np.float32)
        cluster_id, _ = clustering_service.find_or_create_cluster(
            face_id="face-1",
            embedding=embedding,
            file_id="file-1",
            bounding_box=[10, 20, 100, 120]
        )
        
        # Update name
        new_name = "Jokowi"
        success = clustering_service.update_cluster_name(cluster_id, new_name)
        
        assert success is True
        
        # Verify name persists
        clusters = clustering_service.get_all_clusters_with_metadata()
        target_cluster = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
        
        assert target_cluster is not None
        assert target_cluster["name"] == new_name

    def test_update_cluster_name_not_found(self, clustering_service):
        """Test that naming a non-existent cluster fails."""
        success = clustering_service.update_cluster_name("non-existent-cluster", "Someone")
        assert success is False

    def test_get_all_clusters_defaults(self, clustering_service):
        """Test that clusters have None name by default."""
        # Create a cluster
        embedding = np.random.rand(512).astype(np.float32)
        cluster_id, _ = clustering_service.find_or_create_cluster(
            face_id="face-1",
            embedding=embedding,
            file_id="file-1",
            bounding_box=[10, 20, 100, 120]
        )
        
        clusters = clustering_service.get_all_clusters_with_metadata()
        target_cluster = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
        
        assert target_cluster is not None
        assert target_cluster["name"] is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
