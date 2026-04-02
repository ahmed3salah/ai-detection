@echo off
REM Run the API from the project root; loads model.pt (+ model_config.pkl) or model.pkl.
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" -m uvicorn app:app --host 0.0.0.0 --port 8000
) else (
  python -m uvicorn app:app --host 0.0.0.0 --port 8000
)
