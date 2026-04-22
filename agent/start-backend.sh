#!/bin/bash
# Start the artisan materials agent backend
# Usage: ./start-backend.sh
#
# Edit agent/.env to configure the LLM endpoint, model, and SearXNG host.

set -e

# Go to project root (koudos/)
cd "$(dirname "$0")/.."

# Use existing venv or create one
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
    uv pip install langchain langchain-openai langchain-community langgraph fastapi uvicorn pydantic aiosqlite python-dotenv resend
fi

echo "Config loaded from agent/.env"
echo ""
echo "Starting FastAPI backend on http://localhost:8000"
source .venv/bin/activate
uvicorn agent.backend.main:app --host 0.0.0.0 --port 8000 --reload
