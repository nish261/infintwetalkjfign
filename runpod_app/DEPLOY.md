# InfiniteTalk RunPod App Deployment

This setup has two parts:

1. A RunPod Serverless endpoint that runs `runpod_app/worker.py` on a GPU.
2. A small FastAPI web app that keeps your RunPod API key server-side and gives you an upload UI.

## Cheapest Practical RunPod Settings

- Endpoint type: Serverless
- Worker type: Flex
- Minimum workers: `0`
- Maximum workers: `1` to prevent surprise spend
- GPU: start with 48GB VRAM. The Hub template uses `AMPERE_48,ADA_48_PRO,AMPERE_80,ADA_80_PRO`.
- Default app settings: `480p`, `8` steps. Increase steps only when quality is worth the cost.

## Runtime And Cost Estimate

Measured live SaaS smoke test:

- Output: 2 second image-to-video MP4
- Resolution: `512x512`
- Steps: `6`
- Execution time: `84.478` seconds
- Observed worker price: `$0.77/hr`

Linear estimate for the current workflow:

```text
estimated_runtime_seconds =
  84.478
  * (target_video_seconds / 2)
  * ((target_width * target_height) / (512 * 512))
  * (target_steps / 6)
```

For the app's `infinitetalk-720` setting, the SaaS sends `720x720`, not `1280x720`.

| Target | Estimated one-worker runtime | Estimated compute cost at $0.77/hr | Workers for ~10 min wall time |
| --- | ---: | ---: | ---: |
| 60s at 512x512, 6 steps | 42.2 min | $0.54 | 5 |
| 60s at 720x720, 6 steps | 83.5 min | $1.07 | 9 |
| 60s at 1280x720, 6 steps | 148.5 min | $1.91 | 15 |

Workers reduce wall time only if the request is split into chunks and stitched after rendering. More workers do not speed up one monolithic ComfyUI job.

GPU upgrade break-even:

| GPU class | RunPod serverless list price | Must be faster than the $0.77/hr worker by |
| --- | ---: | ---: |
| A6000/A40 48GB | $1.22/hr | 1.58x |
| L40/L40S/6000 Ada 48GB | $1.90/hr | 2.47x |
| A100 80GB | $2.72/hr | 3.53x |
| H100 80GB | $4.18/hr | 5.43x |

If a GPU is slower than its break-even multiple, it may still finish sooner but it costs more per completed video.

## Build Worker Image

From the repo root:

```bash
docker build -f runpod_app/Dockerfile -t your-dockerhub/infinitetalk-runpod:latest .
docker push your-dockerhub/infinitetalk-runpod:latest
```

Use that image in a RunPod Serverless endpoint.

The Docker build downloads the model weights into the image. This makes cold starts larger but avoids paying for model download every job. If the image is too large for your registry/workflow, move weights to a RunPod network volume and set `INFINITETALK_WEIGHTS`.

## Run Web App

On a cheap VPS:

```bash
cd infinitetalk-runpod-src
python3 -m venv .venv-web
. .venv-web/bin/activate
pip install -r runpod_app/web_requirements.txt
export RUNPOD_API_KEY="your_runpod_key"
export RUNPOD_ENDPOINT_ID="your_endpoint_id"
fastapi run runpod_app/web_server.py --host 0.0.0.0 --port 8000
```

Then open `http://your-vps-ip:8000`.

## API Payload

The worker accepts:

```json
{
  "input": {
    "source": "base64 or https url for image/video",
    "audio": "base64 or https url for audio",
    "prompt": "A person is talking naturally.",
    "sample_steps": 8,
    "size": "infinitetalk-480"
  }
}
```

For production, use object storage URLs for uploads and outputs. Inline base64 is convenient but not ideal for large videos.
