"""
API request and response schemas using Pydantic.
Defines data models for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Union


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
    thumbnail_base64: Optional[str] = Field(None, description="Base64-encoded thumbnail image (JPEG)")

    class Config:
        json_schema_extra = {
            "example": {
                "face_id": "face-uuid-456",
                "cluster_id": "cluster-uuid-789",
                "bounding_box": [100, 150, 300, 400],
                "confidence": 0.98,
                "is_new_cluster": False,
                "thumbnail_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
            }
        }


class ContextDetail(BaseModel):
    """Detailed context information (simplified)."""
    english_caption: str = Field(..., description="Original English caption from BLIP")
    indonesian_phrase: str = Field(..., description="Short Indonesian phrase")
    indonesian_description: str = Field(..., description="Detailed Indonesian description")


class ProcessResponse(BaseModel):
    """Response schema for image processing."""
    file_id: str = Field(..., description="Original file identifier")
    faces: List[FaceResult] = Field(..., description="List of detected faces")
    tags: List[str] = Field(..., description="Image tags from OpenCLIP (now includes context elements)")
    objects: List[str] = Field(default_factory=list, description="Visible objects in English")
    context: str = Field(..., description="Indonesian context phrase from BLIP")
    context_detail: Optional[ContextDetail] = Field(None, description="Detailed context information")

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
                        "is_new_cluster": False,
                        "thumbnail_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
                    }
                ],
                "tags": ["dalam ruangan", "formal", "rapat", "foto bersama"],
                "context": "sedang bersalaman",
                "context_detail": {
                    "english_caption": "two government officials shaking hands in an office",
                    "indonesian_phrase": "sedang bersalaman",
                    "indonesian_description": "Dua orang sedang bersalaman di ruang kantor formal",
                    "elements": {
                        "people": {"count": 2, "count_indonesian": "dua orang"},
                        "activity": {"english": "handshake", "indonesian": "bersalaman"},
                        "setting": {"english": "office", "indonesian": "ruang kantor"},
                        "mood": "formal"
                    }
                }
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


class BatchProcessRequest(BaseModel):
    """Request schema for batch image processing."""
    images: List[ProcessRequest] = Field(
        ...,
        description="List of images to process",
        min_length=1,
        max_length=50
    )

    class Config:
        json_schema_extra = {
            "example": {
                "images": [
                    {
                        "file_id": "file-uuid-1",
                        "image_url": "https://example.com/image1.jpg"
                    },
                    {
                        "file_id": "file-uuid-2",
                        "image_url": "https://example.com/image2.jpg"
                    }
                ]
            }
        }


class BatchProcessResult(BaseModel):
    """Individual result in batch processing."""
    file_id: str = Field(..., description="File identifier")
    success: bool = Field(..., description="Whether processing succeeded")
    data: Optional[ProcessResponse] = Field(None, description="Process result if successful")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "file-uuid-1",
                "success": True,
                "data": {
                    "file_id": "file-uuid-1",
                    "faces": [],
                    "tags": ["indoor", "formal"],
                    "context": "sedang rapat"
                },
                "error": None
            }
        }


class BatchProcessResponse(BaseModel):
    """Response schema for batch processing."""
    total: int = Field(..., description="Total number of images processed")
    successful: int = Field(..., description="Number of successful processing")
    failed: int = Field(..., description="Number of failed processing")
    results: List[BatchProcessResult] = Field(..., description="Individual results")

    class Config:
        json_schema_extra = {
            "example": {
                "total": 10,
                "successful": 9,
                "failed": 1,
                "results": []
            }
        }


class ClusterInfo(BaseModel):
    """Information about a face cluster."""
    cluster_id: str = Field(..., description="Cluster identifier")
    face_count: int = Field(..., description="Number of faces in this cluster")
    thumbnail_base64: Optional[str] = Field(None, description="Base64 thumbnail")
    thumbnail_url: str = Field(..., description="Thumbnail URL endpoint")
    created_at: Optional[str] = Field(None, description="First face timestamp")
    file_ids: List[str] = Field(default_factory=list, description="Files containing this person")

    class Config:
        json_schema_extra = {
            "example": {
                "cluster_id": "cluster-uuid-789",
                "face_count": 15,
                "thumbnail_base64": None,
                "thumbnail_url": "/api/cluster/cluster-uuid-789/thumbnail",
                "created_at": "2024-01-15T10:30:00",
                "file_ids": ["file-1", "file-2", "file-3"]
            }
        }


class ClusterGalleryResponse(BaseModel):
    """Response for cluster gallery."""
    total_clusters: int = Field(..., description="Total number of clusters")
    clusters: List[ClusterInfo] = Field(..., description="List of cluster information")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(50, description="Number of clusters per page")
    has_more: bool = Field(False, description="Whether there are more pages")

    class Config:
        json_schema_extra = {
            "example": {
                "total_clusters": 150,
                "clusters": [],
                "page": 1,
                "page_size": 50,
                "has_more": True
            }
        }


class JobSubmitResponse(BaseModel):
    """Response when submitting a job for async processing."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (always 'queued' initially)")
    estimated_time: float = Field(..., description="Estimated processing time in seconds")
    created_at: str = Field(..., description="Job creation timestamp (ISO format)")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-uuid-123",
                "status": "queued",
                "estimated_time": 15.5,
                "created_at": "2024-01-16T10:30:00.123456"
            }
        }


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str = Field(..., description="Job identifier")
    job_type: str = Field(..., description="Type of job (process, batch_process, merge_clusters)")
    status: str = Field(..., description="Job status (queued, processing, completed, failed)")
    progress: int = Field(..., description="Progress percentage (0-100)", ge=0, le=100)
    created_at: str = Field(..., description="Job creation timestamp")
    started_at: Optional[str] = Field(None, description="Job start timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp")
    estimated_time: Optional[float] = Field(None, description="Estimated duration in seconds")
    result: Optional[Union[ProcessResponse, BatchProcessResponse, MergeResponse]] = Field(
        None, description="Job result (only if status is completed)"
    )
    error: Optional[str] = Field(None, description="Error message (only if status is failed)")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-uuid-123",
                "job_type": "process",
                "status": "completed",
                "progress": 100,
                "created_at": "2024-01-16T10:30:00.123456",
                "started_at": "2024-01-16T10:30:05.123456",
                "completed_at": "2024-01-16T10:30:20.123456",
                "estimated_time": 15.0,
                "result": {
                    "file_id": "file-uuid-123",
                    "faces": [],
                    "tags": ["dalam ruangan", "formal"],
                    "context": "sedang rapat"
                },
                "error": None
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
