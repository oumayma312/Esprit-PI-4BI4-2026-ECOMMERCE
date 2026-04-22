@echo off
REM Start the artisan materials agent backend (Windows)
REM Usage: start_backend.bat
REM
REM Edit agent/.env to configure the LLM endpoint, model, and SearXNG host.

cd /d "%~dp0.."

if not exist ".venv" (
    echo Creating virtual environment...
    uv venv .venv
    uv pip install fastapi uvicorn pydantic aiosqlite python-dotenv resend langchain langchain-openai langchain-community langgraph
)

echo Config loaded from agent/.env
echo.
echo Starting FastAPI backend on http://localhost:8000
call .venv\Scripts\activate
uvicorn agent.backend.main:app --host 0.0.0.0 --port 8000 --reload
