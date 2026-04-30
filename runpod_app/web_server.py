import base64
import os
import time
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_BASE = "https://api.runpod.ai/v2"
ROOT = Path(__file__).resolve().parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def _headers():
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        raise HTTPException(status_code=500, detail="RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set.")
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}"}


async def _upload_to_b64(upload: UploadFile) -> str:
    data = await upload.read()
    limit_mb = int(os.getenv("MAX_UPLOAD_MB", "60"))
    if len(data) > limit_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Upload is over {limit_mb}MB. Use object storage URLs for larger files.")
    return base64.b64encode(data).decode("utf-8")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/jobs")
async def create_job(
    source: UploadFile = File(...),
    audio: UploadFile = File(...),
    prompt: str = Form("A person is talking naturally."),
    sample_steps: int = Form(8),
    size: str = Form("infinitetalk-480"),
):
    payload = {
        "input": {
            "source": await _upload_to_b64(source),
            "audio": await _upload_to_b64(audio),
            "prompt": prompt,
            "sample_steps": sample_steps,
            "size": size,
        }
    }
    response = requests.post(f"{RUNPOD_BASE}/{RUNPOD_ENDPOINT_ID}/run", json=payload, headers=_headers(), timeout=120)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    response = requests.get(f"{RUNPOD_BASE}/{RUNPOD_ENDPOINT_ID}/status/{job_id}", headers=_headers(), timeout=120)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    body = response.json()
    body["checked_at"] = int(time.time())
    return body
