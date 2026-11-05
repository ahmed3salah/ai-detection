# Model files for Docker / production

## Option A: Bake into the image (build time)

Before `docker build`, copy trained artifacts into this directory so they end up in the image root (`/app`):

- `model.pt` and `model_config.pkl` (PyTorch training path), **or** `model.pkl` (legacy)
- `feature_scaler.pkl` if training used `use_feature_scaler`
- `tfidf_vectorizer.pkl` if training used TF-IDF

Example (from repo root after training):

```bash
cp model.pt model_config.pkl feature_scaler.pkl model/ 2>/dev/null || true
# omit optional files if you do not use them
docker build -t ai-detection-api .
```

## Option B: Pull from private S3 (runtime)

Do **not** rely on copying large files into `model/` for production if you prefer a single image tag and **updates from S3**.

1. Upload the same basenames under a prefix, e.g. `s3://my-bucket/prod/model.pkl`, `.../model_config.pkl`, etc.
2. Set **`MODEL_S3_URI=s3://my-bucket/prod`** (or `MODEL_S3_BUCKET` + `MODEL_S3_PREFIX`) on the ECS task. The container entrypoint downloads them into `/app` before Uvicorn starts.

See **[DEPLOY.md](../DEPLOY.md)** for IAM, env vars, and rolling deployments when you publish new weights.

CI can still build with only `model/ARTIFACTS.md` in the image if every environment sets `MODEL_S3_URI`.
