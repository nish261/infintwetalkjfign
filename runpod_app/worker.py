import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod


ROOT = Path(os.getenv("INFINITETALK_ROOT", "/workspace/InfiniteTalk"))
WEIGHTS = Path(os.getenv("INFINITETALK_WEIGHTS", str(ROOT / "weights")))
OUTPUT_DIR = Path(os.getenv("INFINITETALK_OUTPUT_DIR", "/workspace/outputs"))
DEFAULT_SIZE = os.getenv("INFINITETALK_SIZE", "infinitetalk-480")
DEFAULT_STEPS = int(os.getenv("INFINITETALK_STEPS", "8"))
DEFAULT_TIMEOUT = int(os.getenv("INFINITETALK_TIMEOUT", "3600"))
WEIGHTS_READY = False


def _run(cmd: list[str], cwd: Path = ROOT, timeout: int = 3600) -> None:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout[-6000:])


def _ensure_weights() -> None:
    global WEIGHTS_READY
    if WEIGHTS_READY:
        return

    required = [
        WEIGHTS / "Wan2.1-I2V-14B-480P",
        WEIGHTS / "chinese-wav2vec2-base",
        WEIGHTS / "InfiniteTalk/single/infinitetalk.safetensors",
    ]
    if all(path.exists() for path in required):
        WEIGHTS_READY = True
        return

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    lock_dir = WEIGHTS.parent / ".download-lock"
    while True:
        try:
            lock_dir.mkdir(parents=True)
            break
        except FileExistsError:
            time.sleep(10)
            if all(path.exists() for path in required):
                WEIGHTS_READY = True
                return

    try:
        _run(["hf", "download", "Wan-AI/Wan2.1-I2V-14B-480P", "--local-dir", str(WEIGHTS / "Wan2.1-I2V-14B-480P")], timeout=7200)
        _run(["hf", "download", "TencentGameMate/chinese-wav2vec2-base", "--local-dir", str(WEIGHTS / "chinese-wav2vec2-base")], timeout=1800)
        _run(["hf", "download", "TencentGameMate/chinese-wav2vec2-base", "model.safetensors", "--revision", "refs/pr/1", "--local-dir", str(WEIGHTS / "chinese-wav2vec2-base")], timeout=1800)
        _run(["hf", "download", "MeiGen-AI/InfiniteTalk", "--local-dir", str(WEIGHTS / "InfiniteTalk")], timeout=3600)
        WEIGHTS_READY = True
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def _write_b64(data: str, dest: Path) -> Path:
    if "," in data and data[:64].startswith("data:"):
        data = data.split(",", 1)[1]
    dest.write_bytes(base64.b64decode(data))
    return dest


def _download(url: str, dest: Path) -> Path:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return dest


def _materialize_file(value: str, dest_dir: Path, fallback_name: str) -> Path:
    parsed = urlparse(value)
    suffix = Path(parsed.path).suffix if parsed.scheme in {"http", "https"} else ""
    dest = dest_dir / (fallback_name + (suffix or ""))
    if parsed.scheme in {"http", "https"}:
        return _download(value, dest)
    return _write_b64(value, dest)


def _maybe_upload(output_path: Path) -> dict:
    upload_url = os.getenv("RESULT_UPLOAD_URL")
    if upload_url:
        content_type = mimetypes.guess_type(output_path.name)[0] or "video/mp4"
        with output_path.open("rb") as handle:
            response = requests.put(upload_url, data=handle, headers={"Content-Type": content_type}, timeout=300)
        response.raise_for_status()
        public_url = os.getenv("RESULT_PUBLIC_URL")
        return {"url": public_url or upload_url.split("?", 1)[0]}

    max_inline_mb = int(os.getenv("MAX_INLINE_RESULT_MB", "40"))
    if output_path.stat().st_size > max_inline_mb * 1024 * 1024:
        return {
            "error": "result_too_large_for_inline_return",
            "path": str(output_path),
            "message": "Set RESULT_UPLOAD_URL/RESULT_PUBLIC_URL or mount persistent storage for large videos.",
        }
    return {
        "filename": output_path.name,
        "mime_type": "video/mp4",
        "base64": base64.b64encode(output_path.read_bytes()).decode("utf-8"),
    }


def handler(job):
    payload = job.get("input") or {}
    if not payload.get("source") or not payload.get("audio"):
        return {"error": "source and audio are required"}

    job_id = str(job.get("id") or uuid.uuid4())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"infinitetalk-{job_id}-"))

    try:
        _ensure_weights()
        source = _materialize_file(payload["source"], work_dir, "source")
        audio = _materialize_file(payload["audio"], work_dir, "audio.wav")
        prompt = payload.get("prompt") or "A person is talking naturally."
        run_name = f"infinitetalk-{job_id}-{int(time.time())}"
        input_json = work_dir / "input.json"
        save_file = OUTPUT_DIR / run_name

        input_json.write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "cond_video": str(source),
                    "cond_audio": {"person1": str(audio)},
                }
            ),
            encoding="utf-8",
        )

        cmd = [
            "python",
            "generate_infinitetalk.py",
            "--ckpt_dir",
            str(WEIGHTS / "Wan2.1-I2V-14B-480P"),
            "--wav2vec_dir",
            str(WEIGHTS / "chinese-wav2vec2-base"),
            "--infinitetalk_dir",
            str(WEIGHTS / "InfiniteTalk/single/infinitetalk.safetensors"),
            "--input_json",
            str(input_json),
            "--size",
            payload.get("size", DEFAULT_SIZE),
            "--sample_steps",
            str(int(payload.get("sample_steps", DEFAULT_STEPS))),
            "--mode",
            payload.get("mode", "streaming"),
            "--motion_frame",
            str(int(payload.get("motion_frame", 9))),
            "--num_persistent_param_in_dit",
            str(int(payload.get("num_persistent_param_in_dit", 0))),
            "--save_file",
            str(save_file),
        ]

        if payload.get("quant", True):
            quant_path = WEIGHTS / "InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors"
            if quant_path.exists():
                cmd.extend(["--quant", "fp8", "--quant_dir", str(quant_path)])

        if payload.get("lora_dir"):
            cmd.extend(["--lora_dir", str(payload["lora_dir"])])
            cmd.extend(["--lora_scale", str(float(payload.get("lora_scale", 1.0)))])
            cmd.extend(["--sample_text_guide_scale", str(float(payload.get("sample_text_guide_scale", 1.0)))])
            cmd.extend(["--sample_audio_guide_scale", str(float(payload.get("sample_audio_guide_scale", 2.0)))])
            cmd.extend(["--sample_shift", str(float(payload.get("sample_shift", 2.0)))])

        result = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(payload.get("timeout", DEFAULT_TIMEOUT)),
            check=False,
        )

        output_path = save_file.with_suffix(".mp4")
        if result.returncode != 0 or not output_path.exists():
            return {"error": "generation_failed", "returncode": result.returncode, "log_tail": result.stdout[-6000:]}

        upload_result = _maybe_upload(output_path)
        return {"ok": True, "result": upload_result, "log_tail": result.stdout[-2000:]}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
