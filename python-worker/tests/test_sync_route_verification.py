
import pytest
import shutil
import tempfile
import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from datetime import datetime

# Adjust import path if necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app
from api.schemas import ProcessResponse
from config import settings

@pytest.fixture
def clean_app_environment():
    """Create temporary directories for app services."""
    # Create temp dirs
    temp_vectordb = tempfile.mkdtemp()
    temp_thumbnails = tempfile.mkdtemp()
    
    # Store original paths
    original_vectordb = settings.VECTORDB_PATH
    original_thumbnails = settings.THUMBNAIL_PATH
    
    # Override settings
    settings.VECTORDB_PATH = temp_vectordb
    settings.THUMBNAIL_PATH = temp_thumbnails
    
    yield
    
    # Restore settings
    settings.VECTORDB_PATH = original_vectordb
    settings.THUMBNAIL_PATH = original_thumbnails
    
    # Cleanup
    shutil.rmtree(temp_vectordb, ignore_errors=True)
    shutil.rmtree(temp_thumbnails, ignore_errors=True)

@pytest.mark.asyncio
async def test_process_sync_route(clean_app_environment):
    client = TestClient(app)
    
    # Mock data
    mock_files_id = "test-file-123"
    mock_result = {
        "file_id": mock_files_id,
        "faces": [],
        "tags": ["test"],
        "objects": [],
        "context": "test context",
        "context_detail": {
            "english_caption": "test",
            "indonesian_phrase": "test",
            "indonesian_description": "test"
        }
    }

    # Patch the process_image_handler in api.routes
    with patch('api.routes.process_image_handler', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = mock_result

        response = client.post(
            "/api/process-sync",
            json={
                "file_id": mock_files_id,
                "image_url": "http://example.com/image.jpg"
            }
        )

        assert response.status_code == 200
        data = response.json()
        
        assert data["job_id"] is not None
        assert data["job_type"] == "process_sync"
        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert data["result"]["file_id"] == mock_files_id
        assert data["result"]["context"] == "test context"
        
        # Verify name is present (even if None in this mock case)
        # Note: If faces were present in mock_result, we would verify their name field
        if data["result"].get("faces"):
            for face in data["result"]["faces"]:
                assert "name" in face
                assert "cluster_name" in face
        
        # Verify handler was called
        mock_handler.assert_called_once()
