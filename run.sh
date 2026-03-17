#!/usr/bin/env bash
# Run the app using the project's venv so all dependencies (fastapi, etc.) are found.
cd "$(dirname "$0")"
exec ./venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
