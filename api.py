from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import uuid
import asyncio
import json
import logging
from src.mastermind import Mastermind

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerateRequest(BaseModel):
    input_prompt: str

app = FastAPI(title="Video Generation API")

from threading import Lock

# Thread-safe job store
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = Lock()

@app.post("/generate")
async def generate_video(request: GenerateRequest):
    """Start a new video generation job"""
    job_id = str(uuid.uuid4())
    
    # Initialize job status with thread safety
    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "input_prompt": request.input_prompt,
            "result": None,
            "error": None
        }
    
    # Run generation in background with new instance
    asyncio.create_task(run_generation(job_id, request.input_prompt))
    
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
        loop = asyncio.get_running_loop()
        mastermind = Mastermind()
        
        # Run blocking operations in thread pool
        result = await loop.run_in_executor(
            executor,
            functools.partial(mastermind.generate_video, input_prompt)
        )
        
        with jobs_lock:
            jobs[job_id].update({
                "status": "completed",
                "result": result
            })
    except Exception as e:
        with jobs_lock:
            jobs[job_id].update({
                "status": "failed",
                "error": str(e)
            })

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Check status of a generation job"""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        return {
            "job_id": job_id,
            "status": jobs[job_id]["status"]
        }

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
    
    # Read the generated JSON file
    json_path = job["result"].get("json_path")
    if not json_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="JSON assets file not found"
        )
    
    try:
        with open(json_path, 'r') as f:
            json_content = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read JSON file: {str(e)}"
        )
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "assets": {
            "json_path": json_path,
            "content": json_content
        }
    }

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
