from fastapi import FastAPI, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware import Middleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated, Dict, Any, Optional
import uuid
import asyncio
import logging
import time
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.mastermind import Mastermind
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_PROMPT_LENGTH = 500
MIN_PROMPT_LENGTH = 10
RATE_LIMIT = "10/minute"

# Type aliases
PromptString = Annotated[
    str,
    StringConstraints(
        min_length=MIN_PROMPT_LENGTH,
        max_length=MAX_PROMPT_LENGTH,
        strip_whitespace=True
    )
]

class GenerateRequest(BaseModel):
    """Request model for video generation"""
    input_prompt: PromptString = Field(
        ...,
        example="A beautiful sunset over mountains",
        description="Text prompt for video generation (10-500 characters)"
    )

class JobStatusResponse(BaseModel):
    """Response model for job status"""
    status: str = Field(..., example="pending")
    job_id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174000")
    input_prompt: Optional[str] = Field(None)
    result: Optional[Dict[str, Any]] = Field(None)
    error: Optional[str] = Field(None)

# Thread-safe job store
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = Lock()
executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(
    title="Video Generation API",
    description="API for generating videos from text prompts",
    version="1.0.0",
    middleware=[
        Middleware(TrustedHostMiddleware, allowed_hosts=["*"]),
        Middleware(GZipMiddleware)
    ],
    responses={
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: JSONResponse(
    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
    content={"detail": "Rate limit exceeded"}
))

@app.post("/generate", response_model=JobStatusResponse)
@limiter.limit(RATE_LIMIT)
async def generate_video(
    request: Request,
    payload: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """Start a new video generation job"""
    job_id = str(uuid.uuid4())
    
    # Initialize job status with thread safety
    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "input_prompt": payload.input_prompt,
            "start_time": time.time(),
            "result": None,
            "error": None
        }
    
    # Run generation in background
    background_tasks.add_task(run_generation, job_id, payload.input_prompt)
    
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "pending",
            "job_id": job_id,
            "input_prompt": payload.input_prompt
        }
    )

async def run_generation(job_id: str, input_prompt: str):
    """Background task to run video generation"""
    try:
        logger.info(f"Starting generation for job {job_id}")
        loop = asyncio.get_running_loop()
        mastermind = Mastermind()
        
        # Run blocking operations in thread pool
        result = await loop.run_in_executor(
            executor,
            functools.partial(mastermind.generate_video, input_prompt)
        )
        
        # Update job status
        with jobs_lock:
            jobs[job_id].update({
                "status": "completed",
                "result": result,
                "end_time": time.time()
            })
        logger.info(f"Successfully completed job {job_id}")
        
    except Exception as e:
        error_msg = f"Generation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        with jobs_lock:
            jobs[job_id].update({
                "status": "failed",
                "error": error_msg,
                "end_time": time.time()
            })

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """Check status of a generation job"""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = jobs[job_id]
        return {
            "status": job["status"],
            "job_id": job_id,
            "input_prompt": job.get("input_prompt"),
            "result": job.get("result"),
            "error": job.get("error")
        }

@app.delete("/cleanup")
async def cleanup_jobs(max_age_hours: int = 24):
    """Clean up old completed/failed jobs from memory"""
    cutoff = time.time() - (max_age_hours * 3600)
    with jobs_lock:
        to_delete = [
            job_id for job_id, job in jobs.items()
            if job["status"] in ("completed", "failed") 
            and job.get("end_time", 0) < cutoff
        ]
        for job_id in to_delete:
            del jobs[job_id]
        return {"cleaned": len(to_delete)}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Get result of a completed generation job"""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="Job not yet completed"
        )
    
    if not job.get("result"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result data not found"
        )
    
    return job["result"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        limit_concurrency=20,
        timeout_keep_alive=30
    )
