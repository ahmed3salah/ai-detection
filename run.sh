#!/usr/bin/env bash
# Run the API (loads model.pt or model.pkl from this directory). Uses venv if present.
cd "$(dirname "$0")"
if [ -x "./venv/bin/python" ]; then
  exec ./venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
fi
exec python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
