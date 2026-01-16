"""
API request and response schemas using Pydantic.
Defines data models for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ProcessRequest(BaseModel):
    """Request schema for image processing."""
    file_id: str = Field(..., description="Unique file identifier")
    image_url: str = Field(..., description="Presigned URL or accessible image URL")

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "file-uuid-123",
                "image_url": "https://minio.example.com/bucket/image.jpg?presigned=..."
            }
        }


class FaceResult(BaseModel):
    """Face detection result with cluster assignment."""
    face_id: str = Field(..., description="Unique face identifier")
    cluster_id: str = Field(..., description="Cluster ID (person identifier)")
    bounding_box: List[int] = Field(..., description="Face bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence (0-1)", ge=0.0, le=1.0)
    is_new_cluster: bool = Field(..., description="True if this is a new cluster/person")

    class Config:
        json_schema_extra = {
            "example": {
                "face_id": "face-uuid-456",
                "cluster_id": "cluster-uuid-789",
                "bounding_box": [100, 150, 300, 400],
                "confidence": 0.98,
                "is_new_cluster": False
            }
        }


class ProcessResponse(BaseModel):
    """Response schema for image processing."""
    file_id: str = Field(..., description="Original file identifier")
    faces: List[FaceResult] = Field(..., description="List of detected faces")
    tags: List[str] = Field(..., description="Image tags from OpenCLIP")
    context: str = Field(..., description="Indonesian context phrase from BLIP")

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "file-uuid-123",
                "faces": [
                    {
                        "face_id": "face-uuid-456",
                        "cluster_id": "cluster-uuid-789",
                        "bounding_box": [100, 150, 300, 400],
                        "confidence": 0.98,
                        "is_new_cluster": False
                    }
                ],
                "tags": ["outdoor", "formal", "group photo"],
                "context": "sedang bersalaman"
            }
        }


class MergeRequest(BaseModel):
    """Request schema for merging clusters."""
    source_cluster_ids: List[str] = Field(
        ...,
        description="List of source cluster IDs to merge",
        min_length=1
    )
    target_cluster_id: str = Field(..., description="Target cluster ID to merge into")

    class Config:
        json_schema_extra = {
            "example": {
                "source_cluster_ids": ["cluster-uuid-1", "cluster-uuid-2"],
                "target_cluster_id": "cluster-uuid-1"
            }
        }


class MergeResponse(BaseModel):
    """Response schema for cluster merge operation."""
    success: bool = Field(..., description="Whether merge was successful")
    merged_count: int = Field(..., description="Number of faces merged")
    target_cluster_id: str = Field(..., description="Target cluster ID")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "merged_count": 45,
                "target_cluster_id": "cluster-uuid-1"
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Health status")
    device: str = Field(..., description="Device type (cuda/mps/cpu)")
    device_info: dict = Field(..., description="Detailed device information")
    vectordb: dict = Field(..., description="VectorDB statistics")
    models_loaded: dict = Field(..., description="Model loading status")
    thumbnails: dict = Field(..., description="Thumbnail storage info")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "device": "cuda",
                "device_info": {
                    "device_type": "cuda",
                    "device_name": "NVIDIA RTX 3060",
                    "available_memory": "12.00 GB"
                },
                "vectordb": {
                    "total_faces": 1500,
                    "total_clusters": 250,
                    "collection_name": "face_embeddings"
                },
                "models_loaded": {
                    "insightface": True,
                    "openclip": True,
                    "blip": True
                },
                "thumbnails": {
                    "total_thumbnails": 250,
                    "path": "./data/thumbnails"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Generic error response schema."""
    detail: str = Field(..., description="Error message")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Image download failed: HTTP 404"
            }
        }
