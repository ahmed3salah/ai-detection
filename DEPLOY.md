# Container deployment (Docker / ECS)

## What the image contains

- FastAPI app on port **8000** (`uvicorn app:app`).
- **spaCy** `en_core_web_sm` and **Hugging Face** caches for `gpt2` and `allenai/scibert_scivocab_uncased` baked at build time (same models as local inference — prediction quality unchanged).
- Dependencies come from **[`requirements-api.txt`](requirements-api.txt)** (drops training-only packages like `datasets` / `openai` that the `/detect` path does not use). **PyTorch** is installed from the official **CPU wheel index** (typically smaller than the default PyPI bundle).
- **`TRANSFORMERS_OFFLINE=1`** at runtime so inference does not depend on huggingface.co.
- Optional **baked-in** weights: copy into [`model/`](model/) before build (see [`model/ARTIFACTS.md`](model/ARTIFACTS.md)); the Dockerfile copies `model/*` into `/app`. If you use **S3 sync** at startup, those files are overwritten or supplemented by whatever is downloaded from S3.

## S3 model sync (recommended for production)

On container start, [`docker-entrypoint.sh`](docker-entrypoint.sh) runs [`scripts/sync_model_from_s3.py`](scripts/sync_model_from_s3.py) when **`MODEL_S3_URI`**, **`MODEL_S3_BUCKET`** or **`PARAGRAPH_MODEL_S3_URI`** is set. It downloads into **`MODEL_SYNC_DEST`** (default: current working directory, **`/app`** in the image).

| Variable | Meaning |
|----------|---------|
| `MODEL_S3_URI` | e.g. `s3://my-bucket/inference/prod` — lists all objects under that prefix and downloads allowlisted basenames (`model.pt`, `model.pkl`, `model_config.pkl`, `feature_scaler.pkl`, `tfidf_vectorizer.pkl`). |
| `MODEL_S3_BUCKET` + `MODEL_S3_PREFIX` | Alternative to URI; prefix may be **empty** for keys at the **bucket root** with those basenames only (no full-bucket listing). |
| `MODEL_S3_REGION` | Optional region for the S3 client. |
| `MODEL_S3_EXTRA_FILES` | Comma-separated extra basenames to allow under a prefix. |
| `MODEL_S3_ALLOW_ANY` | Set to `1` to download every object under a **non-empty** prefix (basename → `/app`; collisions overwrite). Not allowed with an empty prefix. |
| `MODEL_SYNC_DEST` | Override download directory (tests / custom layouts). |
| `PARAGRAPH_MODEL_S3_URI` | The paragraph model `/v2/score-segments` serves: the prefix of one `train_transformer.py` artifacts directory, e.g. `s3://my-bucket/inference/paragraph/scibert-full-1108`. Every object under it is downloaded with its folder layout into `PARAGRAPH_MODEL_SYNC_DEST` (default `paragraph_model`, which the app reads; override the app side with `DETECTION_SEGMENT_MODEL_DIR`). It replaces the old directory only when every file arrived. Without it, v2 falls back to the v1 model with provisional bands (`calibrated: false`). |

**IAM (ECS task role):** grant `s3:ListBucket` on the bucket (scoped by prefix in the policy `Condition` if possible) and `s3:GetObject` on `arn:...:bucket/name/prefix/*`. For bucket-root mode (empty prefix), grant `GetObject` on `arn:...:bucket/name/model.pkl` etc.  
**Credentials:** the sync script uses the standard AWS SDK chain. On **ECS/Fargate**, attach a **task role** with those S3 actions (no access keys in the task). The error `Unable to locate credentials` means nothing in the environment provided keys or a role.

**Updating weights:** upload new objects to the same prefix (same names), then **start new ECS tasks** (rolling deployment / force new deployment) so the entrypoint runs again and re-downloads. In-process hot reload is not implemented.

**Local testing** (host shell — uses your `~/.aws` or SSO):

```bash
export MODEL_S3_URI=s3://your-bucket/path/to/models
export MODEL_SYNC_DEST=/tmp/modelsync
python scripts/sync_model_from_s3.py
```

**Local Docker** with S3 (the image runs as `appuser`, `HOME=/home/appuser`; it does **not** include your laptop’s `~/.aws` unless you mount it):

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_S3_URI=s3://your-bucket/path/to/models \
  -e AWS_PROFILE=default \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -v "$HOME/.aws:/home/appuser/.aws:ro" \
  ai-detection-api
```

Adjust `AWS_PROFILE` / region to match your account. For SSO profiles, ensure you’ve run `aws sso login` on the host before starting the container.

## Local build and run

Use **BuildKit** (`DOCKER_BUILDKIT=1`) so pip and Hugging Face use **persistent build caches** on your machine (fewer re-downloads on later builds). Those caches are **not** shipped inside the image.

From the repo root:

```bash
export DOCKER_BUILDKIT=1
# Optional: baked-in weights (omit if using S3 only)
cp model.pt model_config.pkl model/ 2>/dev/null || true
cp feature_scaler.pkl tfidf_vectorizer.pkl model/ 2>/dev/null || true

docker build -t ai-detection-api .

docker run --rm -p 8000:8000 \
  -e MODEL_S3_URI=s3://your-bucket/path/to/models \
  ai-detection-api
```

Docker Desktop usually has BuildKit on; on Linux CI set `DOCKER_BUILDKIT=1` (or use `docker buildx build`).

Optional API key (matches production):

```bash
docker run --rm -p 8000:8000 -e DETECTION_API_KEY="your-secret" ai-detection-api
```

Callers must send `Authorization: Bearer your-secret` or `X-Api-Key: your-secret` on `POST /detect`. If `DETECTION_API_KEY` is unset, `/detect` stays open (local dev).

Health checks:

- **`GET /health`** — liveness (always 200 if the process responds).
- **`GET /ready`** — readiness (503 until `model.pt` / `model.pkl` + config load and pass validation).

## CI: artifacts before `docker build` (without S3 at runtime)

1. Restore or download `model.pt`, `model_config.pkl`, and optional `feature_scaler.pkl`, `tfidf_vectorizer.pkl` (for example from S3 or CI artifacts).
2. Copy them into **`model/`** in the build context (not only repo root; root copies are ignored by `.dockerignore` to avoid accidental huge contexts).
3. Run **`DOCKER_BUILDKIT=1 docker build`** (or `docker buildx build`) and push to ECR so pip/HF build caches can speed repeat pipelines where the runner supports it.

If the task sets `MODEL_S3_URI` in production, you can ship an image **without** large files under `model/` (only `model/ARTIFACTS.md` in git) and rely entirely on S3.

## ECS / Fargate checklist

- **Image**: Push to ECR; task definition uses the image and container port **8000**.
- **Entrypoint**: image uses [`docker-entrypoint.sh`](docker-entrypoint.sh) (S3 sync when configured, then `uvicorn`).
- **Health**: Target group health check **`GET /ready`** (HTTP 200); use **`/health`** only if you want liveness without loading the classifier.
- **CPU / memory**: PyTorch + transformers + spaCy need a generous task size (for example **2 vCPU / 4–8 GiB** as a starting point; tune with load tests).
- **Networking**: Prefer tasks in **private** subnets; expose **ALB** with **HTTPS (ACM)**. If the bucket is only reachable privately, ensure **VPC S3 endpoint** or routing allows `GetObject`/`ListBucket` from the task subnets.
- **Secrets**: Set **`DETECTION_API_KEY`** from **AWS Secrets Manager** (or SSM) in the task definition; do not bake secrets into the image.
- **Environment**: Do not unset `HF_HOME` or `TRANSFORMERS_OFFLINE` unless you intentionally want runtime Hub access.
- **Concurrency**: each process runs one inference at a time; a request that waits longer than `DETECTION_INFERENCE_WAIT_SECONDS` (default 30) gets a **503**, which callers retry. To serve more at once, raise `WEB_CONCURRENCY` (uvicorn worker processes). Each process loads its own models, so size task memory for it, and set `OMP_NUM_THREADS` to about vCPUs / workers so they don't fight over cores.
- **Paragraph scoring**: `POST /v2/score-segments` takes at most `DETECTION_MAX_SEGMENTS` segments and `DETECTION_MAX_TOTAL_CHARS` characters (413 above that). `GET /v2/model-info` reports the serving model and the limits. `calibrated: false` means the bands are provisional.

## GPU

This Dockerfile targets **CPU** (typical Fargate). GPU on ECS requires a different base image, NVIDIA runtime, and task definition GPU settings.
