"""
BADAK AI Worker - Main Application
FastAPI application for face recognition, tagging, and captioning.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from config import settings
from api.routes import router, initialize_services, register_job_handlers
from services.job_queue import JobQueueService
from middleware.security import (
    APIKeyMiddleware,
    IPWhitelistMiddleware,
    RequestLoggingMiddleware
)
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(
    name="ai_worker",
    level=settings.LOG_LEVEL,
    log_file=None  # Can be configured to write to file
)

# Global job queue service
job_queue_service: JobQueueService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Handles:
    - Loading AI models on startup
    - Initializing services
    - Starting job queue workers
    - Cleanup on shutdown
    """
    global job_queue_service

    # Startup
    logger.info("=" * 60)
    logger.info("BADAK AI Worker Starting...")
    logger.info("=" * 60)

    try:
        # Initialize all services and models
        initialize_services(settings)
        logger.info("Application startup complete")

        # Initialize and start job queue service
        logger.info("Initializing job queue service...")
        job_queue_service = JobQueueService(
            max_workers=settings.JOB_QUEUE_MAX_WORKERS,
            job_retention_hours=settings.JOB_RETENTION_HOURS,
            max_queue_size=settings.JOB_QUEUE_MAX_SIZE
        )
        await job_queue_service.start()
        logger.info(f"Job queue service started: {settings.JOB_QUEUE_MAX_WORKERS} workers, {settings.JOB_QUEUE_MAX_SIZE} max queue size")

        # Register job handlers
        register_job_handlers(job_queue_service)

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("=" * 60)
    logger.info("BADAK AI Worker Shutting Down...")
    logger.info("=" * 60)

    # Stop job queue service
    if job_queue_service:
        logger.info("Stopping job queue service...")
        await job_queue_service.stop()
        logger.info("Job queue service stopped")


# Define API Key security scheme for OpenAPI docs
# auto_error=False because the actual validation is handled by middleware
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


# Create FastAPI application
app = FastAPI(
    title="BADAK AI Worker",
    description="Python AI Worker for face recognition, tagging, and captioning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    dependencies=[Depends(api_key_scheme)]
)

# Configure CORS (internal only - adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5000",
        # Add your C# backend URLs here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middlewares
# Note: Order matters - logging should be first, then IP whitelist, then API key

# 1. Request logging (logs all requests)
app.add_middleware(RequestLoggingMiddleware)

# 2. IP whitelist (checks IP before API key)
app.add_middleware(
    IPWhitelistMiddleware,
    allowed_ips=settings.ALLOWED_IPS,
    exempt_paths=["/health", "/docs", "/redoc", "/openapi.json"]
)

# 3. API key verification (checks after IP is validated)
app.add_middleware(
    APIKeyMiddleware,
    api_key=settings.API_KEY,
    exempt_paths=["/health", "/docs", "/redoc", "/openapi.json"]
)

# Include API routes
app.include_router(router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information.
    """
    return {
        "name": "BADAK AI Worker",
        "version": "1.0.0",
        "description": "Python AI Worker for face recognition, tagging, and captioning",
        "endpoints": {
            "health": "/health",
            "process": "POST /api/process",
            "merge_clusters": "POST /api/merge-clusters",
            "get_thumbnail": "GET /api/cluster/{cluster_id}/thumbnail"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn

    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )
