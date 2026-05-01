import threading
import time
import uuid

from fastapi import FastAPI

from runpod_app.worker import handler, job_log_path


app = FastAPI()
jobs: dict[str, dict] = {}


def _log_tail(job_id: str, max_chars: int = 6000) -> str:
    path = job_log_path(job_id)
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_chars * 2))
        return handle.read().decode("utf-8", errors="replace")[-max_chars:]


def _run_job(job_id: str, payload: dict) -> None:
    started = time.time()
    jobs[job_id] = {"id": job_id, "status": "IN_PROGRESS", "delayTime": 0}
    try:
        output = handler({"id": job_id, "input": payload.get("input") or {}})
        execution_ms = int((time.time() - started) * 1000)
        if isinstance(output, dict) and output.get("error"):
            jobs[job_id] = {
                "id": job_id,
                "status": "FAILED",
                "executionTime": execution_ms,
                "output": output,
                "error": output.get("error"),
            }
            return
        jobs[job_id] = {
            "id": job_id,
            "status": "COMPLETED",
            "executionTime": execution_ms,
            "output": output,
        }
    except Exception as exc:
        jobs[job_id] = {
            "id": job_id,
            "status": "FAILED",
            "executionTime": int((time.time() - started) * 1000),
            "error": "pod_server_exception",
            "output": {"error": "pod_server_exception", "message": str(exc)[-6000:]},
        }


@app.get("/health")
def health():
    counts = {"completed": 0, "failed": 0, "inProgress": 0, "inQueue": 0, "retried": 0}
    for job in jobs.values():
        status = job.get("status")
        if status == "COMPLETED":
            counts["completed"] += 1
        elif status == "FAILED":
            counts["failed"] += 1
        elif status == "IN_PROGRESS":
            counts["inProgress"] += 1
    return {"jobs": counts, "workers": {"idle": 1, "ready": 1, "running": counts["inProgress"]}}


@app.post("/run")
def run(payload: dict):
    job_id = f"{uuid.uuid4()}-pod"
    jobs[job_id] = {"id": job_id, "status": "IN_QUEUE"}
    thread = threading.Thread(target=_run_job, args=(job_id, payload), daemon=True)
    thread.start()
    return {"id": job_id, "status": "IN_QUEUE"}


@app.post("/runsync")
def runsync(payload: dict):
    job_id = f"sync-{uuid.uuid4()}-pod"
    output = handler({"id": job_id, "input": payload.get("input") or {}})
    if isinstance(output, dict) and output.get("error"):
        return {"id": job_id, "status": "FAILED", "error": output.get("error"), "output": output}
    return {"id": job_id, "status": "COMPLETED", "output": output}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id) or {"id": job_id, "status": "NOT_FOUND"}
    if job.get("status") in {"IN_PROGRESS", "IN_QUEUE"}:
        return {**job, "log_tail": _log_tail(job_id)}
    return job
