"""
API routes for the AI Worker.
Handles image processing, cluster merging, and thumbnail retrieval.
"""

import logging
import asyncio
import base64
from uuid import uuid4
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Response, Query
from fastapi.responses import StreamingResponse
import io

from api.schemas import (
    ProcessRequest,
    ProcessResponse,
    FaceResult,
    ContextDetail,
    MergeRequest,
    MergeResponse,
    BatchProcessRequest,
    BatchProcessResult,
    BatchProcessResponse,
    ClusterInfo,
    ClusterGalleryResponse,
    JobSubmitResponse,
    JobStatusResponse,
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
job_queue_service = None  # Will be set by register_job_handlers


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
    openclip_model = OpenCLIPModel(
        device_type=device_type,
        threshold=config.TAG_THRESHOLD,
        top_k=config.TAG_TOP_K,
        language=config.TAG_LANGUAGE
    )
    blip_model = BLIPModel(device_type=device_type)

    logger.info("All services and models initialized successfully")


def register_job_handlers(job_queue_svc):
    """
    Register job handlers with the job queue service.

    Args:
        job_queue_svc: JobQueueService instance
    """
    global job_queue_service

    logger.info("Registering job handlers...")
    job_queue_service = job_queue_svc
    job_queue_service.register_handler("process", process_image_handler)
    job_queue_service.register_handler("batch_process", batch_process_handler)
    job_queue_service.register_handler("merge_clusters", merge_clusters_handler)
    logger.info("All job handlers registered successfully")


# ==================== JOB HANDLERS ====================
# These are standalone handler functions that will be registered with JobQueueService

async def process_image_handler(request_data: dict, progress_callback) -> dict:
    """
    Job handler for single image processing.

    Args:
        request_data: Dictionary with 'file_id' and 'image_url'
        progress_callback: Async function to update progress

    Returns:
        ProcessResponse as dict
    """
    try:
        file_id = request_data["file_id"]
        image_url = request_data["image_url"]

        logger.info(f"Processing image for file_id: {file_id}")
        await progress_callback(10)

        # 1. Download image
        image = await image_downloader.download_image(image_url)
        logger.debug(f"Downloaded image: {image.size}")
        await progress_callback(20)

        # 2. Detect faces with InsightFace
        detected_faces = insightface_model.detect_faces(image)
        logger.info(f"Detected {len(detected_faces)} faces")
        await progress_callback(40)

        # 3. Process each face - find or create cluster
        face_results = []
        for face in detected_faces:
            # Generate unique face ID
            face_id = str(uuid4())

            # Find or create cluster
            cluster_id, is_new_cluster = clustering_service.find_or_create_cluster(
                face_id=face_id,
                embedding=face.embedding,
                file_id=file_id,
                bounding_box=face.bounding_box
            )

            # 4. Save thumbnail if new cluster
            if is_new_cluster:
                face_crop = insightface_model.crop_face(image, face.bounding_box)
                thumbnail_service.save_thumbnail(cluster_id, face_crop)
                logger.debug(f"Saved thumbnail for new cluster: {cluster_id}")

            # Add to results
            face_results.append({
                "face_id": face_id,
                "cluster_id": cluster_id,
                "bounding_box": face.bounding_box,
                "confidence": face.confidence,
                "is_new_cluster": is_new_cluster
            })

        await progress_callback(60)

        # 5. Fetch thumbnails for all faces
        for face_result in face_results:
            thumbnail_bytes = thumbnail_service.get_thumbnail(face_result["cluster_id"])
            if thumbnail_bytes:
                thumbnail_base64 = base64.b64encode(thumbnail_bytes).decode('utf-8')
                face_result["thumbnail_base64"] = thumbnail_base64
                logger.debug(f"Added thumbnail for cluster: {face_result['cluster_id']}")

        await progress_callback(70)

        # 6. Get tags from OpenCLIP (Indonesian by default)
        tags = openclip_model.get_tags(image)
        logger.debug(f"Extracted tags: {tags}")
        await progress_callback(80)

        # 7. Get context from BLIP (comprehensive mode)
        context_comprehensive = blip_model.get_context_comprehensive(image)
        context = context_comprehensive["indonesian_phrase"]
        logger.debug(f"Generated context: {context}")
        await progress_callback(90)

        # 7.5 Detect school age (SD/SMP/SMA) using hybrid approach
        english_caption = context_comprehensive["english_caption"]
        face_ages = [face.age for face in detected_faces if face.age is not None]
        school_age_tag = blip_model.detect_school_age(english_caption, face_ages)

        # 8. Merge context elements to tags (Indonesian)
        elements = context_comprehensive["elements"]

        # Add people count
        if elements.get("people") and elements["people"].get("count_indonesian"):
            tags.append(elements["people"]["count_indonesian"])

        # Add activity
        if elements.get("activity") and elements["activity"].get("indonesian"):
            tags.append(elements["activity"]["indonesian"])

        # Add setting
        if elements.get("setting") and elements["setting"].get("indonesian"):
            tags.append(elements["setting"]["indonesian"])

        # Add mood
        if elements.get("mood"):
            tags.append(elements["mood"])

        # Add school age if detected
        if school_age_tag:
            tags.append(school_age_tag)
            logger.debug(f"School age tag added: {school_age_tag}")

        # Extract objects using OpenCLIP (extensive list) and combine with BLIP
        objects = []
        
        # 1. Get from OpenCLIP
        try:
            openclip_objects = openclip_model.get_objects(image)
            logger.debug(f"Detected objects with OpenCLIP: {openclip_objects}")
            objects.extend(openclip_objects)
        except Exception as e:
            logger.error(f"Failed to get objects from OpenCLIP: {e}")

        # 2. Get from BLIP (already extracted in step 7)
        blip_objects = elements.get("objects", {}).get("english", [])
        if blip_objects:
            logger.debug(f"Detected objects with BLIP: {blip_objects}")
            objects.extend(blip_objects)
        
        # 3. Deduplicate
        objects = list(set(objects))

        # Build simplified context_detail (for backward compatibility, but simplified)
        context_detail = {
            "english_caption": context_comprehensive["english_caption"],
            "indonesian_phrase": context_comprehensive["indonesian_phrase"],
            "indonesian_description": context_comprehensive["indonesian_description"]
        }

        # 9. Return response as dict
        result = {
            "file_id": file_id,
            "faces": face_results,
            "tags": tags,  # Now includes context elements
            "objects": objects,  # New field: English objects array
            "context": context,
            "context_detail": context_detail  # Simplified, no elements
        }

        logger.info(f"Successfully processed file_id: {file_id}")
        return result

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise


async def batch_process_handler(request_data: dict, progress_callback) -> dict:
    """
    Job handler for batch image processing.

    Args:
        request_data: Dictionary with 'images' list
        progress_callback: Async function to update progress

    Returns:
        BatchProcessResponse as dict
    """
    async def process_single_image(image_request: dict) -> dict:
        """Process single image and return result or error."""
        try:
            result = await process_image_handler(image_request, lambda p: asyncio.sleep(0))
            return {
                "file_id": image_request["file_id"],
                "success": True,
                "data": result,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error processing {image_request['file_id']}: {e}")
            return {
                "file_id": image_request["file_id"],
                "success": False,
                "data": None,
                "error": str(e)
            }

    try:
        images = request_data["images"]
        logger.info(f"Batch processing {len(images)} images")
        await progress_callback(10)

        # Use semaphore to limit concurrent processing (max 5 concurrent)
        semaphore = asyncio.Semaphore(5)

        async def process_with_limit(img_req):
            async with semaphore:
                return await process_single_image(img_req)

        # Process all images concurrently with limit
        tasks = [process_with_limit(img_req) for img_req in images]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        await progress_callback(90)

        # Calculate statistics
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")

        return {
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "results": results
        }

    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        raise


async def merge_clusters_handler(request_data: dict, progress_callback) -> dict:
    """
    Job handler for merging clusters.

    Args:
        request_data: Dictionary with 'source_cluster_ids' and 'target_cluster_id'
        progress_callback: Async function to update progress

    Returns:
        MergeResponse as dict
    """
    try:
        source_cluster_ids = request_data["source_cluster_ids"]
        target_cluster_id = request_data["target_cluster_id"]

        logger.info(f"Merging clusters: {source_cluster_ids} -> {target_cluster_id}")
        await progress_callback(20)

        # Validate that source and target are not all the same
        if all(src == target_cluster_id for src in source_cluster_ids):
            raise ValueError("All source clusters are the same as target cluster")

        await progress_callback(40)

        # Perform merge
        merged_count = clustering_service.merge_clusters(
            source_cluster_ids=source_cluster_ids,
            target_cluster_id=target_cluster_id,
            thumbnail_service=thumbnail_service
        )

        await progress_callback(80)

        logger.info(f"Successfully merged {merged_count} faces to cluster {target_cluster_id}")

        return {
            "success": True,
            "merged_count": merged_count,
            "target_cluster_id": target_cluster_id
        }

    except Exception as e:
        logger.error(f"Error merging clusters: {e}", exc_info=True)
        raise


# ==================== API ENDPOINTS ====================

@router.post(
    "/api/process",
    response_model=JobSubmitResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Submit image processing job",
    description="Submits image for async processing. Returns job_id to check status later."
)
async def process_image(request: ProcessRequest):
    """
    Submit an image processing job.

    Instead of processing immediately, this endpoint submits the job to a queue
    and returns a job_id. Use GET /api/jobs/{job_id} to check status and retrieve results.

    Args:
        request: ProcessRequest with file_id and image_url

    Returns:
        JobSubmitResponse with job_id, status, and estimated_time
    """
    try:
        logger.info(f"Submitting process job for file_id: {request.file_id}")

        # Submit job to queue
        job_id = job_queue_service.submit_job(
            job_type="process",
            request_data=request.model_dump()
        )

        # Get job details
        job = job_queue_service.get_job(job_id)

        return JobSubmitResponse(
            job_id=job.job_id,
            status=job.status.value,
            estimated_time=job.estimated_duration or 0.0,
            created_at=job.created_at.isoformat()
        )

    except Exception as e:
        logger.error(f"Error submitting process job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit job: {str(e)}"
        )


@router.post(
    "/api/merge-clusters",
    response_model=JobSubmitResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Submit cluster merge job",
    description="Submits cluster merge job for async processing"
)
async def merge_clusters(request: MergeRequest):
    """
    Submit a cluster merge job.

    Instead of processing immediately, this endpoint submits the job to a queue
    and returns a job_id. Use GET /api/jobs/{job_id} to check status and retrieve results.

    Args:
        request: MergeRequest with source_cluster_ids and target_cluster_id

    Returns:
        JobSubmitResponse with job_id, status, and estimated_time
    """
    try:
        logger.info(
            f"Submitting merge job: {request.source_cluster_ids} -> {request.target_cluster_id}"
        )

        # Validate that source and target are not all the same
        if all(src == request.target_cluster_id for src in request.source_cluster_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All source clusters are the same as target cluster"
            )

        # Submit job to queue
        job_id = job_queue_service.submit_job(
            job_type="merge_clusters",
            request_data=request.model_dump()
        )

        # Get job details
        job = job_queue_service.get_job(job_id)

        return JobSubmitResponse(
            job_id=job.job_id,
            status=job.status.value,
            estimated_time=job.estimated_duration or 0.0,
            created_at=job.created_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting merge job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit job: {str(e)}"
        )


@router.post(
    "/api/batch-process",
    response_model=JobSubmitResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Submit batch processing job",
    description="Submits batch of images for async processing"
)
async def batch_process_images(request: BatchProcessRequest):
    """
    Submit a batch image processing job.

    Instead of processing immediately, this endpoint submits the job to a queue
    and returns a job_id. Use GET /api/jobs/{job_id} to check status and retrieve results.

    Args:
        request: BatchProcessRequest with list of images to process (max 50)

    Returns:
        JobSubmitResponse with job_id, status, and estimated_time
    """
    try:
        logger.info(f"Submitting batch process job with {len(request.images)} images")

        # Submit job to queue
        job_id = job_queue_service.submit_job(
            job_type="batch_process",
            request_data=request.model_dump()
        )

        # Get job details
        job = job_queue_service.get_job(job_id)

        return JobSubmitResponse(
            job_id=job.job_id,
            status=job.status.value,
            estimated_time=job.estimated_duration or 0.0,
            created_at=job.created_at.isoformat()
        )

    except Exception as e:
        logger.error(f"Error submitting batch process job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit job: {str(e)}"
        )


@router.get(
    "/api/jobs/{job_id}",
    response_model=JobStatusResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Get job status and result",
    description="Retrieve status, progress, and result of a submitted job"
)
async def get_job_status(job_id: str):
    """
    Get job status and result.

    Use this endpoint to check the status of a job that was submitted via
    POST /api/process, /api/batch-process, or /api/merge-clusters.

    Args:
        job_id: Job identifier returned from job submission

    Returns:
        JobStatusResponse with status, progress, and result (if completed)
    """
    try:
        logger.debug(f"Getting status for job: {job_id}")

        # Get job from queue service
        job = job_queue_service.get_job(job_id)

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}"
            )

        # Build response
        response = JobStatusResponse(
            job_id=job.job_id,
            job_type=job.job_type,
            status=job.status.value,
            progress=job.progress,
            created_at=job.created_at.isoformat() if job.created_at else None,
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            estimated_time=job.estimated_duration,
            result=job.result if job.status.value == "completed" else None,
            error=job.error if job.status.value == "failed" else None
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job status: {str(e)}"
        )


@router.get(
    "/api/clusters",
    response_model=ClusterGalleryResponse,
    responses={
        500: {"model": ErrorResponse}
    },
    summary="Get all clusters (persons) with thumbnails",
    description="Retrieve gallery of all detected persons with metadata and pagination support"
)
async def get_cluster_gallery(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Number of clusters per page"),
    include_thumbnails: bool = Query(True, description="Whether to include base64 thumbnails"),
    sort_by: str = Query("face_count", regex="^(face_count|created_at)$", description="Sort order")
):
    """
    Get cluster gallery with pagination.

    Returns all face clusters (unique persons) with their metadata including
    face count, file IDs, and optionally base64-encoded thumbnails.

    Args:
        page: Page number (1-indexed)
        page_size: Number of clusters per page (max 200)
        include_thumbnails: Whether to include base64 thumbnails (default: True)
        sort_by: Sort order - "face_count" or "created_at" (default: "face_count")

    Returns:
        ClusterGalleryResponse with paginated cluster information
    """
    try:
        logger.info(f"Getting cluster gallery: page={page}, page_size={page_size}, sort_by={sort_by}")

        # Get all clusters from ClusteringService
        all_clusters = clustering_service.get_all_clusters_with_metadata()

        # Sort clusters
        if sort_by == "face_count":
            all_clusters.sort(key=lambda x: x["face_count"], reverse=True)
        elif sort_by == "created_at":
            all_clusters.sort(key=lambda x: x.get("created_at") or "", reverse=True)

        # Pagination
        total = len(all_clusters)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_clusters = all_clusters[start_idx:end_idx]

        # Build response
        cluster_infos = []
        for cluster in page_clusters:
            # Get thumbnail
            thumbnail_base64 = None
            if include_thumbnails:
                thumbnail_bytes = thumbnail_service.get_thumbnail(cluster["cluster_id"])
                if thumbnail_bytes:
                    thumbnail_base64 = base64.b64encode(thumbnail_bytes).decode('utf-8')

            cluster_infos.append(ClusterInfo(
                cluster_id=cluster["cluster_id"],
                face_count=cluster["face_count"],
                thumbnail_base64=thumbnail_base64,
                thumbnail_url=f"/api/cluster/{cluster['cluster_id']}/thumbnail",
                created_at=cluster.get("created_at"),
                file_ids=cluster.get("file_ids", [])
            ))

        logger.info(f"Returning {len(cluster_infos)} clusters (page {page}/{(total + page_size - 1) // page_size})")

        return ClusterGalleryResponse(
            total_clusters=total,
            clusters=cluster_infos,
            page=page,
            page_size=page_size,
            has_more=end_idx < total
        )

    except Exception as e:
        logger.error(f"Error getting cluster gallery: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve clusters: {str(e)}"
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
