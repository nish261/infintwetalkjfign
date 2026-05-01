import base64
import json
import math
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import urlparse

import requests
import runpod


ROOT = Path(os.getenv("INFINITETALK_ROOT", "/workspace/InfiniteTalk"))
WEIGHTS = Path(os.getenv("INFINITETALK_WEIGHTS", str(ROOT / "weights")))
OUTPUT_DIR = Path(os.getenv("INFINITETALK_OUTPUT_DIR", "/workspace/outputs"))
DEFAULT_SIZE = os.getenv("INFINITETALK_SIZE", "infinitetalk-480")
DEFAULT_STEPS = int(os.getenv("INFINITETALK_STEPS", "8"))
DEFAULT_TIMEOUT = int(os.getenv("INFINITETALK_TIMEOUT", "3600"))
DEFAULT_MODE = os.getenv("INFINITETALK_MODE", "streaming")
DEFAULT_FRAME_NUM = int(os.getenv("INFINITETALK_FRAME_NUM", "81"))
DEFAULT_OFFLOAD_MODEL = os.getenv("INFINITETALK_OFFLOAD_MODEL", "false")
WEIGHTS_READY = False


_HF_ENV = {**os.environ, "HF_HUB_DISABLE_XET": "1"}


def _run(cmd: list[str], cwd: Path = ROOT, timeout: int = 3600, env: dict | None = None) -> None:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout[-6000:])


def _hf_download(args: list[str], local_dir: Path, timeout: int) -> None:
    cmd = ["hf", "download", *args, "--local-dir", str(local_dir)]
    try:
        _run(cmd, timeout=timeout, env=_HF_ENV)
    except RuntimeError as exc:
        message = str(exc)
        if "Invalid metadata file" not in message and "Disk quota exceeded" not in message:
            raise
        shutil.rmtree(local_dir / ".cache", ignore_errors=True)
        for partial in local_dir.rglob("*.incomplete"):
            partial.unlink(missing_ok=True)
        _run(cmd, timeout=timeout, env=_HF_ENV)


def _weights_valid() -> bool:
    required = [
        WEIGHTS / "Wan2.1-I2V-14B-480P",
        WEIGHTS / "chinese-wav2vec2-base",
        WEIGHTS / "InfiniteTalk/single/infinitetalk.safetensors",
    ]
    if not all(path.exists() for path in required):
        return False
    # Validate that JSON config files aren't empty/corrupt (can happen after a crashed XET download)
    for json_path in (WEIGHTS / "Wan2.1-I2V-14B-480P").rglob("*.json"):
        try:
            if json_path.stat().st_size == 0:
                return False
        except OSError:
            return False
    return True


def _ensure_weights() -> None:
    global WEIGHTS_READY
    if WEIGHTS_READY:
        return

    if _weights_valid():
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
        _hf_download(["Wan-AI/Wan2.1-I2V-14B-480P"], WEIGHTS / "Wan2.1-I2V-14B-480P", timeout=7200)
        _hf_download(["TencentGameMate/chinese-wav2vec2-base"], WEIGHTS / "chinese-wav2vec2-base", timeout=1800)
        _hf_download(["TencentGameMate/chinese-wav2vec2-base", "model.safetensors", "--revision", "refs/pr/1"], WEIGHTS / "chinese-wav2vec2-base", timeout=1800)
        _hf_download(["MeiGen-AI/InfiniteTalk"], WEIGHTS / "InfiniteTalk", timeout=3600)
        WEIGHTS_READY = True
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def _suffix_from_data_url(data: str) -> str:
    if not data.startswith("data:") or "," not in data:
        return ""
    mime_type = data[5:].split(";", 1)[0]
    suffix = mimetypes.guess_extension(mime_type) or ""
    if suffix == ".jpe":
        return ".jpg"
    return suffix


def _write_b64(data: str, dest: Path) -> Path:
    if "," in data and data[:128].startswith("data:"):
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


def _safe_filename(name: str) -> str:
    clean = Path(unquote(name)).name
    keep = []
    for char in clean:
        keep.append(char if char.isalnum() or char in ".-_" else "_")
    clean = "".join(keep).strip("._")
    return clean or "upload"


def _materialize_file(value: str, dest_dir: Path, fallback_name: str, original_name: str | None = None) -> Path:
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        suffix = Path(parsed.path).suffix
        dest = dest_dir / (fallback_name + (suffix or ""))
        return _download(value, dest)

    if original_name:
        dest = dest_dir / _safe_filename(original_name)
    else:
        suffix = _suffix_from_data_url(value) or Path(fallback_name).suffix
        base = Path(fallback_name).stem
        dest = dest_dir / f"{base}{suffix}"
    return _write_b64(value, dest)


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _wan_frame_count(value: int) -> int:
    # InfiniteTalk expects frame counts in the Wan 4n+1 cadence.
    value = max(5, int(value))
    remainder = value % 4
    return value if remainder == 1 else value + ((1 - remainder) % 4)


def job_log_path(job_id: str) -> Path:
    return OUTPUT_DIR / f"{_safe_filename(job_id)}.log"


def _tail(path: Path, max_chars: int = 6000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_chars * 2))
        return handle.read().decode("utf-8", errors="replace")[-max_chars:]


def handler(job):
    payload = job.get("input") or {}
    source_val = payload.get("source") or payload.get("image_url")
    audio_val = payload.get("audio") or payload.get("wav_url")
    if not source_val or not audio_val:
        return {"error": "source/image_url and audio/wav_url are required"}

    job_id = str(job.get("id") or uuid.uuid4())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"infinitetalk-{job_id}-"))

    try:
        _ensure_weights()
        source = _materialize_file(source_val, work_dir, "source", payload.get("source_name"))
        audio = _materialize_file(audio_val, work_dir, "audio.wav", payload.get("audio_name"))
        prompt = payload.get("prompt") or "A person is talking naturally."
        run_name = f"infinitetalk-{job_id}-{int(time.time())}"
        input_json = work_dir / "input.json"
        save_file = OUTPUT_DIR / run_name
        log_path = job_log_path(job_id)

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

        mode = str(payload.get("mode") or DEFAULT_MODE)
        frame_num = _wan_frame_count(int(payload.get("frame_num", DEFAULT_FRAME_NUM)))
        duration_seconds = payload.get("duration_seconds")
        max_frame_num = payload.get("max_frame_num")
        if max_frame_num is None and duration_seconds:
            max_frame_num = _wan_frame_count(math.ceil(float(duration_seconds) * 16))

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
            mode,
            "--motion_frame",
            str(int(payload.get("motion_frame", 9))),
            "--frame_num",
            str(frame_num),
            "--save_file",
            str(save_file),
            "--t5_cpu",
        ]

        if max_frame_num is not None:
            cmd.extend(["--max_frame_num", str(_wan_frame_count(int(max_frame_num)))])

        offload_model = payload.get("offload_model", DEFAULT_OFFLOAD_MODEL)
        if str(offload_model).strip().lower() != "auto":
            cmd.extend(["--offload_model", "True" if _truthy(offload_model) else "False"])

        if payload.get("num_persistent_param_in_dit") is not None:
            cmd.extend([
                "--num_persistent_param_in_dit",
                str(int(payload["num_persistent_param_in_dit"])),
            ])

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

        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND " + json.dumps(cmd) + "\n")
            log.flush()
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=int(payload.get("timeout", DEFAULT_TIMEOUT)),
                check=False,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

        output_path = save_file.with_suffix(".mp4")
        log_tail = _tail(log_path)
        if result.returncode != 0 or not output_path.exists():
            return {"error": "generation_failed", "returncode": result.returncode, "log_tail": log_tail}

        upload_url = os.getenv("RESULT_UPLOAD_URL")
        if upload_url:
            content_type = "video/mp4"
            with output_path.open("rb") as handle:
                resp = requests.put(upload_url, data=handle, headers={"Content-Type": content_type}, timeout=300)
            resp.raise_for_status()
            public_url = os.getenv("RESULT_PUBLIC_URL") or upload_url.split("?", 1)[0]
            return {
                "ok": True,
                "video_url": public_url,
                "result": {"url": public_url, "mime_type": "video/mp4", "filename": output_path.name},
                "log_tail": log_tail[-2000:],
            }

        video = base64.b64encode(output_path.read_bytes()).decode("utf-8")
        return {
            "ok": True,
            "video": video,
            "result": {"base64": video, "mime_type": "video/mp4", "filename": output_path.name},
            "log_tail": log_tail[-2000:],
        }
    except Exception as exc:
        return {"error": "worker_exception", "message": str(exc)[-6000:]}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
