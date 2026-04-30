FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

WORKDIR /workspace/InfiniteTalk

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs libgl1 libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/InfiniteTalk/requirements.txt

RUN pip install --no-cache-dir --upgrade pip wheel packaging psutil ninja \
    && pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir flash_attn==2.7.4.post1 --no-build-isolation \
    && pip install --no-cache-dir misaki[en] librosa runpod requests huggingface_hub \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "transformers>=4.49.0,<5" "tokenizers>=0.20.3,<0.22"

COPY . /workspace/InfiniteTalk

ENV INFINITETALK_ROOT=/workspace/InfiniteTalk \
    INFINITETALK_WEIGHTS=/workspace/InfiniteTalk/weights \
    INFINITETALK_STEPS=8 \
    INFINITETALK_SIZE=infinitetalk-480

RUN hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P \
    && hf download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base \
    && hf download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base \
    && hf download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk

CMD ["python", "runpod_app/worker.py"]
