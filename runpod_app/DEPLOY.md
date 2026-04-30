# InfiniteTalk RunPod App Deployment

This setup has two parts:

1. A RunPod Serverless endpoint that runs `runpod_app/worker.py` on a GPU.
2. A small FastAPI web app that keeps your RunPod API key server-side and gives you an upload UI.

## Cheapest Practical RunPod Settings

- Endpoint type: Serverless
- Worker type: Flex
- Minimum workers: `0`
- Maximum workers: `1` to prevent surprise spend
- GPU: start with 48GB VRAM such as A40/A6000. Try 24GB only with fp8 quantization and `num_persistent_param_in_dit=0`.
- Default app settings: `480p`, `8` steps. Increase steps only when quality is worth the cost.

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
