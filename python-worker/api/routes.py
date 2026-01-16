"""
API routes for the AI Worker.
Handles image processing, cluster merging, and thumbnail retrieval.
"""

import logging
from uuid import uuid4
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Response
from fastapi.responses import StreamingResponse
import io

from api.schemas import (
    ProcessRequest,
    ProcessResponse,
    FaceResult,
    MergeRequest,
    MergeResponse,
    HealthResponse,
    ErrorResponse
)
from services.image_downloader import ImageDownloader
from services.vectordb import VectorDBService
from services.clustering_service import ClusteringService
from services.thumbnail_service import ThumbnailService
from models.insightface_model import InsightFaceModel
from models.openclip_model import OpenCLIPModel
from models.blip_model import BLIPModel
from utils.device_detector import DeviceDetector

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global services and models (will be initialized on startup)
image_downloader: Optional[ImageDownloader] = None
vectordb_service: Optional[VectorDBService] = None
clustering_service: Optional[ClusteringService] = None
thumbnail_service: Optional[ThumbnailService] = None
insightface_model: Optional[InsightFaceModel] = None
openclip_model: Optional[OpenCLIPModel] = None
blip_model: Optional[BLIPModel] = None
device_type: Optional[str] = None
onnx_providers: Optional[list] = None


def initialize_services(config):
    """
    Initialize all services and models.
    Called during application startup.

    Args:
        config: Application configuration object
    """
    global image_downloader, vectordb_service, clustering_service, thumbnail_service
    global insightface_model, openclip_model, blip_model, device_type, onnx_providers

    logger.info("Initializing services and models...")

    # Detect device
    device_type, onnx_providers = DeviceDetector.detect_device(config.DEVICE)
    device_info = DeviceDetector.get_device_info(device_type)
    logger.info(f"Device detected: {device_type} - {device_info['device_name']}")

    # Initialize services
    image_downloader = ImageDownloader()
    vectordb_service = VectorDBService(persist_directory=config.VECTORDB_PATH)
    clustering_service = ClusteringService(
        vectordb=vectordb_service,
        similarity_threshold=config.FACE_SIMILARITY_THRESHOLD
    )
    thumbnail_service = ThumbnailService(thumbnail_path=config.THUMBNAIL_PATH)

    # Initialize AI models
    logger.info("Loading AI models (this may take a few minutes)...")
    insightface_model = InsightFaceModel(device_type=device_type, onnx_providers=onnx_providers)
    openclip_model = OpenCLIPModel(device_type=device_type, threshold=config.TAG_THRESHOLD)
    blip_model = BLIPModel(device_type=device_type)

    logger.info("All services and models initialized successfully")


@router.post(
    "/api/process",
    response_model=ProcessResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Process image and extract faces, tags, and context",
    description="Downloads image, detects faces, assigns clusters, and extracts tags and context"
)
async def process_image(request: ProcessRequest):
    """
    Process an image to detect faces and extract metadata.

    Workflow:
    1. Download image from URL
    2. Run InsightFace to detect faces and extract embeddings
    3. For each face, find or create cluster (person identification)
    4. Crop face and save thumbnail for new clusters
    5. Run OpenCLIP to extract tags
    6. Run BLIP to generate Indonesian context phrase
    7. Return results

    Args:
        request: ProcessRequest with file_id and image_url

    Returns:
        ProcessResponse with faces, tags, and context
    """
    try:
        logger.info(f"Processing image for file_id: {request.file_id}")

        # 1. Download image
        image = await image_downloader.download_image(request.image_url)
        logger.debug(f"Downloaded image: {image.size}")

        # 2. Detect faces with InsightFace
        detected_faces = insightface_model.detect_faces(image)
        logger.info(f"Detected {len(detected_faces)} faces")

        # 3. Process each face - find or create cluster
        face_results = []
        for face in detected_faces:
            # Generate unique face ID
            face_id = str(uuid4())

            # Find or create cluster
            cluster_id, is_new_cluster = clustering_service.find_or_create_cluster(
                face_id=face_id,
                embedding=face.embedding,
                file_id=request.file_id,
                bounding_box=face.bounding_box
            )

            # 4. Save thumbnail if new cluster
            if is_new_cluster:
                face_crop = insightface_model.crop_face(image, face.bounding_box)
                thumbnail_service.save_thumbnail(cluster_id, face_crop)
                logger.debug(f"Saved thumbnail for new cluster: {cluster_id}")

            # Add to results
            face_results.append(FaceResult(
                face_id=face_id,
                cluster_id=cluster_id,
                bounding_box=face.bounding_box,
                confidence=face.confidence,
                is_new_cluster=is_new_cluster
            ))

        # 5. Get tags from OpenCLIP
        tags = openclip_model.get_tags(image)
        logger.debug(f"Extracted tags: {tags}")

        # 6. Get context from BLIP
        context = blip_model.get_context(image)
        logger.debug(f"Generated context: {context}")

        # 7. Return response
        response = ProcessResponse(
            file_id=request.file_id,
            faces=face_results,
            tags=tags,
            context=context
        )

        logger.info(f"Successfully processed file_id: {request.file_id}")
        return response

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image processing failed: {str(e)}"
        )


@router.post(
    "/api/merge-clusters",
    response_model=MergeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Merge multiple face clusters into one",
    description="Merges source clusters into target cluster and deletes source thumbnails"
)
async def merge_clusters(request: MergeRequest):
    """
    Merge multiple face clusters into a single cluster.

    This is used when the user identifies that multiple clusters represent the same person.

    Args:
        request: MergeRequest with source_cluster_ids and target_cluster_id

    Returns:
        MergeResponse with success status and merged count
    """
    try:
        logger.info(
            f"Merging clusters: {request.source_cluster_ids} -> {request.target_cluster_id}"
        )

        # Validate that source and target are not all the same
        if all(src == request.target_cluster_id for src in request.source_cluster_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All source clusters are the same as target cluster"
            )

        # Perform merge
        merged_count = clustering_service.merge_clusters(
            source_cluster_ids=request.source_cluster_ids,
            target_cluster_id=request.target_cluster_id,
            thumbnail_service=thumbnail_service
        )

        logger.info(
            f"Successfully merged {merged_count} faces to cluster {request.target_cluster_id}"
        )

        return MergeResponse(
            success=True,
            merged_count=merged_count,
            target_cluster_id=request.target_cluster_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging clusters: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cluster merge failed: {str(e)}"
        )


@router.get(
    "/api/cluster/{cluster_id}/thumbnail",
    response_class=Response,
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Thumbnail image"
        },
        404: {"model": ErrorResponse}
    },
    summary="Get cluster thumbnail image",
    description="Returns the representative face thumbnail for a cluster"
)
async def get_cluster_thumbnail(cluster_id: str):
    """
    Get the thumbnail image for a face cluster.

    Args:
        cluster_id: Cluster ID

    Returns:
        JPEG image (binary)
    """
    try:
        # Get thumbnail bytes
        thumbnail_bytes = thumbnail_service.get_thumbnail(cluster_id)

        if thumbnail_bytes is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thumbnail not found for cluster: {cluster_id}"
            )

        # Return image
        return Response(
            content=thumbnail_bytes,
            media_type="image/jpeg"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thumbnail for cluster {cluster_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve thumbnail: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Returns service health status and statistics"
)
async def health_check():
    """
    Health check endpoint.

    Returns system status, device info, model status, and database statistics.

    Returns:
        HealthResponse with system health information
    """
    try:
        # Get device info
        device_info = DeviceDetector.get_device_info(device_type) if device_type else {}

        # Get VectorDB info
        vectordb_info = vectordb_service.get_info() if vectordb_service else {}

        # Get thumbnail info
        thumbnail_info = thumbnail_service.get_info() if thumbnail_service else {}

        # Check model loading status
        models_loaded = {
            "insightface": insightface_model is not None,
            "openclip": openclip_model is not None,
            "blip": blip_model is not None
        }

        return HealthResponse(
            status="healthy",
            device=device_type or "unknown",
            device_info=device_info,
            vectordb=vectordb_info,
            models_loaded=models_loaded,
            thumbnails=thumbnail_info
        )

    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )
