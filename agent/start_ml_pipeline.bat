@echo off
REM Start the ML pipeline API (Windows)
REM Usage: start_ml_pipeline.bat
REM Runs on http://localhost:8001
REM
REM Required by n8n workflow for churn prediction + segmentation

cd /d "%~dp0.."

if not exist ".venv" (
    echo Creating virtual environment...
    uv venv .venv
    uv pip install fastapi uvicorn pydantic aiosqlite python-dotenv pandas scikit-learn joblib matplotlib scipy
)

echo Starting ML Pipeline API on http://localhost:8001
call .venv\Scripts\activate
uvicorn ml_pipeline_api.main:app --host 0.0.0.0 --port 8001 --reload
