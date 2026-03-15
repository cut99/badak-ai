"""
BADAK AI Worker - Main Application
FastAPI application for face recognition, tagging, and captioning.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
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
    log_file=None
)

# Global job queue service
job_queue_service: JobQueueService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    global job_queue_service

    logger.info("=" * 60)
    logger.info("BADAK AI Worker Starting...")
    logger.info("=" * 60)

    try:
        initialize_services(settings)
        logger.info("Application startup complete")

        logger.info("Initializing job queue service...")
        job_queue_service = JobQueueService(
            max_workers=settings.JOB_QUEUE_MAX_WORKERS,
            job_retention_hours=settings.JOB_RETENTION_HOURS,
            max_queue_size=settings.JOB_QUEUE_MAX_SIZE
        )
        await job_queue_service.start()
        logger.info(f"Job queue service started: {settings.JOB_QUEUE_MAX_WORKERS} workers")

        register_job_handlers(job_queue_service)

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        raise

    yield

    logger.info("=" * 60)
    logger.info("BADAK AI Worker Shutting Down...")
    logger.info("=" * 60)

    if job_queue_service:
        logger.info("Stopping job queue service...")
        await job_queue_service.stop()
        logger.info("Job queue service stopped")


api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(
    title="BADAK AI Worker",
    description="Python AI Worker for face recognition, tagging, and captioning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    dependencies=[Depends(api_key_scheme)]
)

# ============================================================
# MIDDLEWARE ORDER IS CRITICAL!
# FastAPI executes middlewares in REVERSE order of addition.
# Last added = First executed
# 
# Execution order will be:
# 1. CORS (added last, runs first) - handles preflight
# 2. Request Logging
# 3. IP Whitelist
# 4. API Key
# ============================================================

# Add security middlewares FIRST (they will run AFTER CORS)

# 4. API key verification (runs last in security chain)
app.add_middleware(
    APIKeyMiddleware,
    api_key=settings.API_KEY,
    exempt_paths=["/health", "/docs", "/redoc", "/openapi.json"]
)

# 3. IP whitelist
app.add_middleware(
    IPWhitelistMiddleware,
    allowed_ips=settings.ALLOWED_IPS,
    exempt_paths=["/health", "/docs", "/redoc", "/openapi.json"]
)

# 2. Request logging
app.add_middleware(RequestLoggingMiddleware)

# 1. CORS - ADD LAST so it runs FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:4200",
        "http://localhost:5000",
        "https://localhost:4200",
        # Add your production URLs here
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight for 10 minutes
)

# Include API routes
app.include_router(router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "BADAK AI Worker",
        "version": "1.0.0",
        "description": "Python AI Worker for face recognition, tagging, and captioning",
        "endpoints": {
            "health": "GET /health",
            "process_sync": "POST /api/process-sync",
            "process": "POST /api/process",
            "batch_process": "POST /api/batch-process",
            "get_job": "GET /api/jobs/{job_id}",
            "merge_clusters": "POST /api/merge-clusters",
            "get_clusters": "GET /api/clusters",
            "get_thumbnail": "GET /api/cluster/{cluster_id}/thumbnail",
            "update_cluster_name": "POST /api/cluster/name"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )