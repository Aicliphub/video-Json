from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, constr, Field
from typing import Dict, Any
import uuid
import asyncio
import json
import logging
import os
import time
from fastapi.middleware import Middleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.mastermind import Mastermind

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerateRequest(BaseModel):
    input_prompt: str = Field(..., min_length=10, max_length=500)

app = FastAPI(
    title="Video Generation API",
    middleware=[
        Middleware(TrustedHostMiddleware, allowed_hosts=["*"]),
        Middleware(GZipMiddleware)
    ]
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apply rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: JSONResponse(
    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
    content={"detail": "Rate limit exceeded"}
))

from threading import Lock

# Thread-safe job store
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = Lock()

@app.post("/generate")
@limiter.limit("10/minute")
async def generate_video(request: Request, payload: GenerateRequest):
    """Start a new video generation job"""
    job_id = str(uuid.uuid4())
    
    # Initialize job status with thread safety
    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "input_prompt": payload.input_prompt,
            "result": None,
            "error": None
        }
    
    # Run generation in background with new instance
    asyncio.create_task(run_generation(job_id, payload.input_prompt))
    
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"job_id": job_id, "status": "pending"}
    )

from concurrent.futures import ThreadPoolExecutor
import functools

# Global thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

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
        
        # Store the JSON data directly in the result
        with jobs_lock:
            jobs[job_id].update({
                "status": "completed",
                "result": result,
                "end_time": time.time()
            })
        logger.info(f"Successfully completed job {job_id}")
        
    except Exception as e:
        error_msg = f"Job {job_id} failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        with jobs_lock:
            jobs[job_id].update({
                "status": "failed",
                "error": error_msg,
                "end_time": time.time()
            })

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Check status of a generation job"""
    with jobs_lock:
        if job_id not in jobs:
            return {
                "status": "not_found",
                "job_id": job_id,
                "error": "Job not found"
            }
        
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
    """Clean up old completed/failed jobs from memory (does not delete generated JSON files)"""
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
    """Get result of a completed generation job including JSON assets"""
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
    
    # Return the JSON data directly
    json_data = job["result"].get("json_data")
    if not json_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="JSON data not found in result"
        )
    
    return json_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,  # Match available CPU cores
        limit_concurrency=20,  # More realistic concurrency limit
        timeout_keep_alive=30  # Prevent hung connections
    )
