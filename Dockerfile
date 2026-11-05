# syntax=docker/dockerfile:1
# CPU inference image (Fargate-friendly). GPU serving needs a different base and drivers.
#
# Build with BuildKit so cache mounts work (saves bandwidth on rebuilds; caches are NOT in the image):
#   DOCKER_BUILDKIT=1 docker build -t ai-detection-api .
#
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
COPY scripts/prefetch_hf_models.py scripts/prefetch_hf_models.py

# Pip cache: on-host BuildKit cache — speeds rebuilds, does not increase image size.
# torch: PyTorch CPU index wheels are typically smaller than default PyPI manylinux bundles.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-api.txt \
    && python -m spacy download en_core_web_sm

# Hugging Face hub cache: on-host BuildKit cache between builds; copy into image for offline runtime.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    HF_HOME=/root/.cache/huggingface \
    python scripts/prefetch_hf_models.py \
    && mkdir -p /app/.cache/huggingface \
    && cp -a /root/.cache/huggingface/. /app/.cache/huggingface/

ENV TRANSFORMERS_OFFLINE=1

COPY app.py classifier_mlp.py detector.py features.py segment_scoring.py text_normalize.py ./
COPY scripts/sync_model_from_s3.py scripts/sync_model_from_s3.py
# CI/local: optional baked-in weights; production often uses MODEL_S3_URI at startup instead.
COPY model/ ./
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh \
    && useradd --create-home --no-log-init --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
