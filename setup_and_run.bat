@echo off
REM Koudos — Unified setup and launch script (Windows)
REM Double-click to run, or: setup_and_run.bat

echo.
echo ============================================================
echo   Koudos - Atelier Materials + ML Integration (Windows)
echo ============================================================
echo.

REM Check if python is in PATH
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not in PATH.
    echo Please install Python 3.11+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check if uv is available
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo uv found.
) else (
    echo uv not found. Installing via pip...
    python -m pip install uv --quiet
)

REM Run the Python script
python "%~dp0setup_and_run.py" %*
pause
