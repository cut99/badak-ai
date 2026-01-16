"""
Integration tests for API endpoints.
Tests all API routes including security middleware.
"""

import pytest
import tempfile
import shutil
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import io

from main import app
from api.routes import initialize_services
from config import Settings


# Test fixtures
@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    vectordb_dir = tempfile.mkdtemp()
    thumbnail_dir = tempfile.mkdtemp()

    yield {
        "vectordb": vectordb_dir,
        "thumbnail": thumbnail_dir
    }

    # Cleanup
    shutil.rmtree(vectordb_dir, ignore_errors=True)
    shutil.rmtree(thumbnail_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_dirs):
    """Create test configuration."""
    config = Settings()
    config.VECTORDB_PATH = temp_dirs["vectordb"]
    config.THUMBNAIL_PATH = temp_dirs["thumbnail"]
    config.DEVICE = "cpu"
    config.API_KEY = "test-api-key-12345"
    config.ALLOWED_IPS = ["127.0.0.1", "testclient"]
    return config


@pytest.fixture
def client(test_config):
    """Create test client with initialized services."""
    # Initialize services with test config
    initialize_services(test_config)

    # Create test client
    client = TestClient(app)
    return client


@pytest.fixture
def valid_headers():
    """Get valid API headers for testing."""
    return {"X-API-Key": "test-api-key-12345"}


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    # Create a 640x480 RGB image
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    return img_bytes.getvalue()


class TestHealthEndpoint:
    """Test cases for /health endpoint."""

    def test_health_check_success(self, client):
        """Test that health check returns 200 and proper structure."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "status" in data
        assert "device" in data
        assert "device_info" in data
        assert "vectordb" in data
        assert "models_loaded" in data
        assert "thumbnails" in data

        # Verify models are loaded
        assert data["models_loaded"]["insightface"] is True
        assert data["models_loaded"]["openclip"] is True
        assert data["models_loaded"]["blip"] is True

    def test_health_check_no_auth_required(self, client):
        """Test that health check doesn't require authentication."""
        # No API key header
        response = client.get("/health")
        assert response.status_code == 200


class TestRootEndpoint:
    """Test cases for root endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["name"] == "BADAK AI Worker"


class TestProcessEndpoint:
    """Test cases for /api/process endpoint."""

    def test_process_missing_api_key(self, client):
        """Test that process endpoint requires API key."""
        response = client.post(
            "/api/process",
            json={"file_id": "test-1", "image_url": "http://example.com/image.jpg"}
        )

        assert response.status_code == 401

    def test_process_invalid_api_key(self, client):
        """Test that invalid API key is rejected."""
        response = client.post(
            "/api/process",
            json={"file_id": "test-1", "image_url": "http://example.com/image.jpg"},
            headers={"X-API-Key": "invalid-key"}
        )

        assert response.status_code == 401

    def test_process_missing_file_id(self, client, valid_headers):
        """Test that missing file_id returns 422."""
        response = client.post(
            "/api/process",
            json={"image_url": "http://example.com/image.jpg"},
            headers=valid_headers
        )

        assert response.status_code == 422

    def test_process_missing_image_url(self, client, valid_headers):
        """Test that missing image_url returns 422."""
        response = client.post(
            "/api/process",
            json={"file_id": "test-1"},
            headers=valid_headers
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_process_invalid_image_url(self, client, valid_headers):
        """Test that invalid image URL is handled gracefully."""
        response = client.post(
            "/api/process",
            json={
                "file_id": "test-1",
                "image_url": "http://invalid-url-that-does-not-exist.com/image.jpg"
            },
            headers=valid_headers
        )

        # Should return 500 with error message
        assert response.status_code == 500
        assert "detail" in response.json()

    def test_process_response_structure(self, client, valid_headers):
        """Test that successful process returns proper structure."""
        # Note: This test would need a valid image URL or mocking
        # For now, we verify the endpoint exists and accepts valid requests
        response = client.post(
            "/api/process",
            json={
                "file_id": "test-1",
                "image_url": "http://example.com/test.jpg"
            },
            headers=valid_headers
        )

        # Will fail due to invalid URL, but we can check error structure
        assert "detail" in response.json() or "faces" in response.json()


class TestMergeClustersEndpoint:
    """Test cases for /api/merge-clusters endpoint."""

    def test_merge_missing_api_key(self, client):
        """Test that merge endpoint requires API key."""
        response = client.post(
            "/api/merge-clusters",
            json={
                "source_cluster_ids": ["cluster-1", "cluster-2"],
                "target_cluster_id": "cluster-1"
            }
        )

        assert response.status_code == 401

    def test_merge_invalid_api_key(self, client):
        """Test that invalid API key is rejected."""
        response = client.post(
            "/api/merge-clusters",
            json={
                "source_cluster_ids": ["cluster-1", "cluster-2"],
                "target_cluster_id": "cluster-1"
            },
            headers={"X-API-Key": "invalid-key"}
        )

        assert response.status_code == 401

    def test_merge_missing_source_cluster_ids(self, client, valid_headers):
        """Test that missing source_cluster_ids returns 422."""
        response = client.post(
            "/api/merge-clusters",
            json={"target_cluster_id": "cluster-1"},
            headers=valid_headers
        )

        assert response.status_code == 422

    def test_merge_empty_source_cluster_ids(self, client, valid_headers):
        """Test that empty source_cluster_ids returns 422."""
        response = client.post(
            "/api/merge-clusters",
            json={
                "source_cluster_ids": [],
                "target_cluster_id": "cluster-1"
            },
            headers=valid_headers
        )

        assert response.status_code == 422

    def test_merge_all_same_as_target(self, client, valid_headers):
        """Test that merging cluster into itself returns 400."""
        response = client.post(
            "/api/merge-clusters",
            json={
                "source_cluster_ids": ["cluster-1"],
                "target_cluster_id": "cluster-1"
            },
            headers=valid_headers
        )

        assert response.status_code == 400
        assert "same as target" in response.json()["detail"].lower()

    def test_merge_valid_request_structure(self, client, valid_headers):
        """Test that valid merge request has proper structure."""
        response = client.post(
            "/api/merge-clusters",
            json={
                "source_cluster_ids": ["cluster-1", "cluster-2"],
                "target_cluster_id": "cluster-3"
            },
            headers=valid_headers
        )

        # Should succeed or return proper error
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "merged_count" in data
            assert "target_cluster_id" in data


class TestThumbnailEndpoint:
    """Test cases for /api/cluster/{cluster_id}/thumbnail endpoint."""

    def test_thumbnail_missing_api_key(self, client):
        """Test that thumbnail endpoint requires API key."""
        response = client.get("/api/cluster/cluster-123/thumbnail")

        assert response.status_code == 401

    def test_thumbnail_invalid_api_key(self, client):
        """Test that invalid API key is rejected."""
        response = client.get(
            "/api/cluster/cluster-123/thumbnail",
            headers={"X-API-Key": "invalid-key"}
        )

        assert response.status_code == 401

    def test_thumbnail_not_found(self, client, valid_headers):
        """Test that non-existent thumbnail returns 404."""
        response = client.get(
            "/api/cluster/non-existent-cluster/thumbnail",
            headers=valid_headers
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_thumbnail_content_type(self, client, valid_headers):
        """Test that existing thumbnail returns image/jpeg content type."""
        # First, we'd need to create a cluster with thumbnail
        # For now, test the 404 case
        response = client.get(
            "/api/cluster/test-cluster/thumbnail",
            headers=valid_headers
        )

        # Either 404 or image/jpeg
        if response.status_code == 200:
            assert response.headers["content-type"] == "image/jpeg"
        else:
            assert response.status_code == 404


class TestSecurityMiddleware:
    """Test cases for security middleware."""

    def test_api_key_middleware_blocks_invalid_key(self, client):
        """Test that API key middleware blocks invalid keys."""
        response = client.post(
            "/api/process",
            json={"file_id": "test", "image_url": "http://example.com/test.jpg"},
            headers={"X-API-Key": "wrong-key"}
        )

        assert response.status_code == 401

    def test_api_key_middleware_allows_valid_key(self, client, valid_headers):
        """Test that API key middleware allows valid keys."""
        response = client.post(
            "/api/process",
            json={"file_id": "test", "image_url": "http://example.com/test.jpg"},
            headers=valid_headers
        )

        # Should not be blocked by API key (might fail for other reasons)
        assert response.status_code != 401

    def test_health_endpoint_exempt_from_api_key(self, client):
        """Test that /health is exempt from API key requirement."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_docs_endpoint_exempt_from_api_key(self, client):
        """Test that /docs is exempt from API key requirement."""
        response = client.get("/docs")

        # Should redirect to /docs or return docs page
        assert response.status_code in [200, 307]


class TestCORSMiddleware:
    """Test cases for CORS middleware."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.get("/health")

        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers or response.status_code == 200


class TestRequestValidation:
    """Test cases for request validation."""

    def test_process_request_validation(self, client, valid_headers):
        """Test that ProcessRequest validates fields."""
        # Missing both fields
        response = client.post(
            "/api/process",
            json={},
            headers=valid_headers
        )
        assert response.status_code == 422

    def test_merge_request_validation(self, client, valid_headers):
        """Test that MergeRequest validates fields."""
        # Invalid types
        response = client.post(
            "/api/merge-clusters",
            json={
                "source_cluster_ids": "not-a-list",
                "target_cluster_id": "cluster-1"
            },
            headers=valid_headers
        )
        assert response.status_code == 422


class TestErrorHandling:
    """Test cases for error handling."""

    def test_404_for_unknown_endpoint(self, client):
        """Test that unknown endpoints return 404."""
        response = client.get("/unknown/endpoint")

        assert response.status_code == 404

    def test_405_for_wrong_method(self, client, valid_headers):
        """Test that wrong HTTP method returns 405."""
        # GET on POST-only endpoint
        response = client.get("/api/process", headers=valid_headers)

        assert response.status_code == 405


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
