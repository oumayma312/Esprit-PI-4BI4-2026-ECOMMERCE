#!/bin/bash
# Start the Angular frontend
# Usage: ./start-frontend.sh
# Note: Backend must be running on port 8000 first

set -e
cd "$(dirname "$0")/artisan-materials"

echo "Starting Angular frontend on http://localhost:4200"
echo "Make sure the backend is running on port 8000"
echo ""
npx ng serve --host 0.0.0.0 --port 4200
