"""Background job status.

The UI polls these while files are being processed. There is no websocket: one
local user polling a loopback endpoint every second costs nothing, and it keeps
the Tauri/React port (phase 6) simpler.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import JobOut
from app.tasks.queue import task_queue

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobOut])
def list_jobs(limit: int = Query(50, ge=1, le=200)) -> list[JobOut]:
    return [JobOut(**job.as_dict()) for job in task_queue.recent(limit)]


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str) -> JobOut:
    job = task_queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job with id {job_id}")
    return JobOut(**job.as_dict())
