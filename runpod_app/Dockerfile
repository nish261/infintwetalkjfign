FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

WORKDIR /workspace/InfiniteTalk

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs libgl1 libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/InfiniteTalk/requirements.txt

RUN printf '%s\n' \
    'diffusers==0.33.1' \
    'xfuser==0.4.1' \
    'transformers>=4.49.0,<5' \
    'tokenizers>=0.20.3,<0.22' \
    > /tmp/infinitetalk-constraints.txt

RUN pip install --no-cache-dir --upgrade pip wheel packaging psutil ninja \
    && pip install --no-cache-dir xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir flash_attn==2.7.4.post1 --no-build-isolation || echo "flash_attn build failed, xformers fallback will be used" \
    && pip install --no-cache-dir misaki[en] librosa runpod requests huggingface_hub fastapi uvicorn[standard] \
    && pip install --no-cache-dir -c /tmp/infinitetalk-constraints.txt -r requirements.txt

COPY . /workspace/InfiniteTalk

ENV INFINITETALK_ROOT=/workspace/InfiniteTalk \
    INFINITETALK_WEIGHTS=/runpod-volume/infinitetalk/weights \
    INFINITETALK_STEPS=8 \
    INFINITETALK_SIZE=infinitetalk-480 \
    HF_HUB_DISABLE_XET=1

CMD ["python", "runpod_app/worker.py"]
