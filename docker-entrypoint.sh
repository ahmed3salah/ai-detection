#!/bin/sh
set -e
if [ -n "${MODEL_S3_URI:-}" ] || [ -n "${MODEL_S3_BUCKET:-}" ]; then
  python /app/scripts/sync_model_from_s3.py
fi
exec "$@"
