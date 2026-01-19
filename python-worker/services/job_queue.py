"""
Job Queue Service for asynchronous background task processing.
Implements in-memory job queue with worker pool management.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Any, Callable, List
from uuid import uuid4
import statistics

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Job data model for tracking background tasks."""
    job_id: str = field(default_factory=lambda: str(uuid4()))
    job_type: str = ""  # "process", "batch_process", "merge_clusters"
    status: JobStatus = JobStatus.QUEUED
    request_data: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: int = 0  # 0-100%
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[float] = None  # seconds

    def to_dict(self) -> dict:
        """Convert job to dictionary for API response."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_duration": self.estimated_duration,
            "result": self.result,
            "error": self.error
        }


class JobQueueService:
    """
    In-memory job queue service with background worker pool.

    Features:
    - FIFO queue with configurable worker pool
    - Job status tracking and progress monitoring
    - Automatic duration estimation based on history
    - TTL-based job cleanup
    """

    def __init__(
        self,
        max_workers: int = 3,
        job_retention_hours: int = 24,
        max_queue_size: int = 1000
    ):
        """
        Initialize job queue service.

        Args:
            max_workers: Maximum concurrent workers (default: 3)
            job_retention_hours: Hours to retain completed jobs (default: 24)
            max_queue_size: Maximum queue size (default: 1000)
        """
        self.max_workers = max_workers
        self.job_retention_hours = job_retention_hours
        self.max_queue_size = max_queue_size

        self.jobs: Dict[str, Job] = {}  # job_id -> Job
        self.queue: deque = deque()  # FIFO queue of job_ids
        self.active_workers = 0
        self.processing_history: Dict[str, List[float]] = {}  # job_type -> [durations]

        self.worker_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.handlers: Dict[str, Callable] = {}  # job_type -> handler function

        self._lock = asyncio.Lock()  # For thread-safe operations

    async def start(self):
        """Start background workers and cleanup task."""
        logger.info(f"Starting job queue service with {self.max_workers} workers")
        self.worker_task = asyncio.create_task(self._process_queue())
        self.cleanup_task = asyncio.create_task(self._cleanup_old_jobs())

    async def stop(self):
        """Stop background workers gracefully."""
        logger.info("Stopping job queue service")
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    def register_handler(self, job_type: str, handler: Callable):
        """
        Register a handler function for a job type.

        Args:
            job_type: Type of job (e.g., "process", "batch_process")
            handler: Async function to execute job
        """
        self.handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")

    def submit_job(self, job_type: str, request_data: dict) -> str:
        """
        Submit a new job to the queue.

        Args:
            job_type: Type of job to execute
            request_data: Request data for the job

        Returns:
            job_id: Unique job identifier

        Raises:
            ValueError: If queue is full or job_type not registered
        """
        if len(self.queue) >= self.max_queue_size:
            raise ValueError(f"Queue is full (max: {self.max_queue_size})")

        if job_type not in self.handlers:
            raise ValueError(f"No handler registered for job type: {job_type}")

        job = Job(
            job_type=job_type,
            request_data=request_data,
            estimated_duration=self._estimate_duration(job_type, request_data)
        )

        self.jobs[job.job_id] = job
        self.queue.append(job.job_id)

        logger.info(
            f"Job submitted: {job.job_id} (type: {job_type}, "
            f"estimated: {job.estimated_duration:.1f}s, queue: {len(self.queue)})"
        )

        return job.job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job object or None if not found
        """
        return self.jobs.get(job_id)

    def get_all_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """
        Get all jobs, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of jobs
        """
        if status:
            return [job for job in self.jobs.values() if job.status == status]
        return list(self.jobs.values())

    def get_queue_stats(self) -> dict:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue stats
        """
        return {
            "total_jobs": len(self.jobs),
            "queued": len([j for j in self.jobs.values() if j.status == JobStatus.QUEUED]),
            "processing": len([j for j in self.jobs.values() if j.status == JobStatus.PROCESSING]),
            "completed": len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED]),
            "failed": len([j for j in self.jobs.values() if j.status == JobStatus.FAILED]),
            "active_workers": self.active_workers,
            "max_workers": self.max_workers,
            "queue_length": len(self.queue)
        }

    def _estimate_duration(self, job_type: str, request_data: dict) -> float:
        """
        Estimate job duration based on history and queue.

        Args:
            job_type: Type of job
            request_data: Request data (used for batch job calculations)

        Returns:
            Estimated duration in seconds
        """
        # Calculate average processing time for this job type
        history = self.processing_history.get(job_type, [])
        if history:
            # Use exponential moving average (more weight on recent jobs)
            recent_history = history[-10:]
            avg_time = statistics.mean(recent_history)
        else:
            # Default estimates per job type
            defaults = {
                "process": 5.0,
                "batch_process": 10.0,
                "merge_clusters": 2.0
            }
            avg_time = defaults.get(job_type, 5.0)

        # For batch jobs, multiply by number of images
        if job_type == "batch_process":
            num_images = len(request_data.get("images", []))
            avg_time = avg_time * num_images / 5  # Adjust for concurrent processing

        # Add estimated time of all queued jobs ahead
        queued_time = sum(
            self.jobs[jid].estimated_duration or 5.0
            for jid in list(self.queue)
            if jid in self.jobs and self.jobs[jid].status == JobStatus.QUEUED
        )

        return avg_time + queued_time

    async def _process_queue(self):
        """Background worker to process jobs from queue."""
        logger.info("Job queue worker started")

        while True:
            try:
                # Check if we can process more jobs
                if self.active_workers < self.max_workers and self.queue:
                    job_id = self.queue.popleft()

                    # Start job processing in background
                    asyncio.create_task(self._execute_job(job_id))

                await asyncio.sleep(0.1)  # Small delay to prevent busy loop

            except Exception as e:
                logger.error(f"Error in queue worker: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _execute_job(self, job_id: str):
        """
        Execute a single job.

        Args:
            job_id: Job identifier
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            return

        async with self._lock:
            self.active_workers += 1

        try:
            logger.info(f"Starting job: {job_id} (type: {job.job_type})")

            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow()
            job.progress = 10

            # Get handler for this job type
            handler = self.handlers.get(job.job_type)
            if not handler:
                raise ValueError(f"No handler for job type: {job.job_type}")

            # Execute handler with progress callback
            async def progress_callback(progress: int):
                job.progress = min(progress, 99)  # Never reach 100 until complete

            result = await handler(job.request_data, progress_callback)

            job.result = result
            job.status = JobStatus.COMPLETED
            job.progress = 100

            # Record duration for future estimation
            duration = (datetime.utcnow() - job.started_at).total_seconds()
            if job.job_type not in self.processing_history:
                self.processing_history[job.job_type] = []
            self.processing_history[job.job_type].append(duration)

            # Keep only last 50 durations per type
            if len(self.processing_history[job.job_type]) > 50:
                self.processing_history[job.job_type] = self.processing_history[job.job_type][-50:]

            logger.info(
                f"Job completed: {job_id} (duration: {duration:.1f}s, "
                f"avg: {statistics.mean(self.processing_history[job.job_type]):.1f}s)"
            )

        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
            job.progress = 0
            logger.error(f"Job failed: {job_id} - {e}", exc_info=True)

        finally:
            job.completed_at = datetime.utcnow()
            async with self._lock:
                self.active_workers -= 1

    async def _cleanup_old_jobs(self):
        """Background task to cleanup old completed/failed jobs."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                cutoff_time = datetime.utcnow() - timedelta(hours=self.job_retention_hours)
                jobs_to_delete = []

                for job_id, job in self.jobs.items():
                    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                        if job.completed_at and job.completed_at < cutoff_time:
                            jobs_to_delete.append(job_id)

                for job_id in jobs_to_delete:
                    del self.jobs[job_id]

                if jobs_to_delete:
                    logger.info(f"Cleaned up {len(jobs_to_delete)} old jobs")

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}", exc_info=True)
