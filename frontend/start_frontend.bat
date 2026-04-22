@echo off
REM Start the Angular frontend (Windows)
REM Usage: start_frontend.bat
REM Note: Backend must be running on port 8000 first

cd /d "%~dp0artisan-materials"

echo Starting Angular frontend on http://localhost:4200
echo Make sure the backend is running on port 8000
echo.
npx ng serve --host 0.0.0.0 --port 4200
