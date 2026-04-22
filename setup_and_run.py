#!/usr/bin/env python3
"""
Koudos — Unified setup and launch script.

Cross-platform (Linux / macOS / Windows) script that:
  1. Installs Python dependencies (via uv)
  2. Installs Angular CLI if needed
  3. Imports ML data if not already imported
  4. Launches FastAPI backend (port 8000) and Angular frontend (port 4200)
  5. Stays running until Ctrl+C, then cleanly shuts down both services.

Usage:
    python setup_and_run.py          # Linux / macOS
    python setup_and_run.py          # Windows (with python in PATH)
    ./setup_and_run.py               # Linux / macOS (after chmod +x)
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import platform
import shutil
import time
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
AGENT_DIR = PROJECT_ROOT / "agent"
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "artisan-materials"
DB_PATH = AGENT_DIR / "backend" / "materials.db"
PYTHON_EXE = sys.executable
UV_AVAILABLE = shutil.which("uv") is not None

# Ports
BACKEND_PORT = 8000
FRONTEND_PORT = 4200

# Backend package path for uvicorn
BACKEND_MODULE = "agent.backend.main:app"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[koudos] {msg}", flush=True)


def error(msg: str) -> None:
    print(f"[koudos] ERROR: {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    log(f"  Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd, cwd=cwd, check=check,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )


def which(cmd: str) -> str | None:
    """Check if a command is available."""
    return shutil.which(cmd)


# ─── Step 1: Install Python dependencies ─────────────────────────────────────

def install_python_deps() -> None:
    """Install all Python dependencies using uv (or pip as fallback)."""
    log("Checking Python dependencies...")

    # Create venv if it doesn't exist
    if not VENV_DIR.exists() or not (VENV_DIR / "pyvenv.cfg").exists():
        log("Creating virtual environment...")
        if UV_AVAILABLE:
            run(["uv", "venv", str(VENV_DIR)])
        else:
            run([PYTHON_EXE, "-m", "venv", str(VENV_DIR)])

    # Determine the venv's python
    if platform.system() == "Windows":
        venv_python = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_python = VENV_DIR / "bin" / "python"

    if not venv_python.exists():
        error(f"Virtual environment python not found at {venv_python}")
        sys.exit(1)

    # Install packages
    packages = [
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.34.0",
        "langchain>=0.3.0",
        "langchain-openai>=0.3.0",
        "langchain-community>=0.3.0",
        "langgraph>=0.2.0",
        "pydantic>=2.0.0",
        "aiosqlite>=0.20.0",
        "python-dotenv>=1.0.0",
        "resend>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0,<2.0",
        "scikit-learn>=1.3.0,<1.14",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0,<1.14",
    ]

    if UV_AVAILABLE:
        log("Installing packages with uv...")
        run(["uv", "pip", "install", "-p", str(venv_python), *packages])
    else:
        pip = str(venv_python).replace("python", "pip")
        log("Installing packages with pip...")
        run(["python", "-m", "pip", "install", "--upgrade", "pip"])
        run(["python", "-m", "pip", "install", *packages])

    log("Python dependencies installed.")


# ─── Step 2: Install Angular CLI ─────────────────────────────────────────────

def install_angular_cli() -> None:
    """Install Angular CLI globally via npm if not present."""
    log("Checking Angular CLI...")

    ng_cmd = which("ng.cmd") or which("ng")
    if ng_cmd:
        # Use cmd /c on Windows for .cmd/.bat files
        if platform.system() == "Windows":
            ng_version = run(["cmd", "/c", "ng", "--version"], check=False).stdout.strip()
        else:
            ng_version = run(["ng", "--version"], check=False).stdout.strip()
        log(f"  Angular CLI already installed: {ng_version.split(chr(10))[0]}")
        return

    npm = which("npm")
    if not npm:
        error("npm is not installed. Please install Node.js and npm first.")
        error("  Linux:   sudo apt install nodejs npm")
        error("  macOS:   brew install node")
        error("  Windows: Download from https://nodejs.org/")
        sys.exit(1)

    log("Installing Angular CLI globally...")
    run([npm, "install", "-g", "@angular/cli@^21.0.0"])
    log("Angular CLI installed.")


# ─── Step 3: Install frontend dependencies ───────────────────────────────────

def install_frontend_deps() -> None:
    """Install npm dependencies for the Angular app."""
    log("Checking frontend dependencies...")

    node_modules = FRONTEND_DIR / "node_modules"
    if node_modules.exists():
        log("  Frontend dependencies already installed.")
        return

    npm = which("npm")
    if not npm:
        error("npm is not installed. See step 2.")
        sys.exit(1)

    log("Installing frontend dependencies...")
    run([npm, "install"], cwd=FRONTEND_DIR)
    log("Frontend dependencies installed.")


# ─── Step 4: Import ML data ─────────────────────────────────────────────────

def import_data() -> None:
    """Import customer data and run ML predictions if not already done."""
    log("Checking ML data import...")

    if not DB_PATH.exists():
        log("Database not found. Running full import...")
        _do_import()
        return

    # Check if customers table has data
    venv_python = _get_venv_python()
    result = run([
        venv_python, "-c",
        f"import sys; sys.path.insert(0, '{AGENT_DIR}'); "
        f"from backend import db; db.init_db(); "
        f"rows = db.list_customers(limit=1); "
        f"print(len(rows))"
    ], cwd=PROJECT_ROOT, check=False)

    try:
        count = int(result.stdout.strip())
    except (ValueError, AttributeError):
        count = 0

    if count > 0:
        log(f"  Database already has {count} customers. Skipping import.")
    else:
        log("Importing ML data...")
        _do_import()


def _do_import() -> None:
    """Run the actual data import."""
    venv_python = _get_venv_python()
    import_script = AGENT_DIR / "backend" / "import_data.py"

    if not import_script.exists():
        error(f"Import script not found: {import_script}")
        sys.exit(1)

    log(f"  Running: {import_script}")
    result = run([venv_python, str(import_script)], cwd=PROJECT_ROOT, check=False)
    print(result.stdout, end="")
    if result.returncode != 0:
        error("Data import failed. Check the output above.")
        sys.exit(1)


def _get_venv_python() -> str:
    """Get the path to the venv's Python executable."""
    if platform.system() == "Windows":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")


# ─── Step 5: Launch services ─────────────────────────────────────────────────

class ServiceManager:
    """Manages backend and frontend subprocess lifecycles."""

    def __init__(self):
        self.backend_proc: subprocess.Popen | None = None
        self.frontend_proc: subprocess.Popen | None = None
        self._original_sigint = None
        self._original_sigterm = None

    def start(self) -> None:
        """Start both backend and frontend."""
        log("Starting services...")
        log("")

        # Start backend
        log(f"  Backend  → http://localhost:{BACKEND_PORT}")
        self.backend_proc = self._start_backend()

        # Wait for backend to be ready
        time.sleep(3)

        # Start frontend
        log(f"  Frontend → http://localhost:{FRONTEND_PORT}")
        self.frontend_proc = self._start_frontend()

        log("")
        log("=" * 60)
        log("  Services are running!")
        log(f"  Frontend: http://localhost:{FRONTEND_PORT}")
        log(f"  Backend:  http://localhost:{BACKEND_PORT}")
        log(f"  API Docs: http://localhost:{BACKEND_PORT}/docs")
        log("=" * 60)
        log("")
        log("Press Ctrl+C to stop all services.")
        log("")

    def _start_backend(self) -> subprocess.Popen:
        """Start the FastAPI backend."""
        venv_python = _get_venv_python()
        # --reload doesn't work well on Windows (needs Unix signals)
        reload_flag = "--reload" if platform.system() != "Windows" else ""
        cmd = [venv_python, "-m", "uvicorn", BACKEND_MODULE,
               "--host", "0.0.0.0", "--port", str(BACKEND_PORT)]
        if reload_flag:
            cmd.append(reload_flag)
        return subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def _start_frontend(self) -> subprocess.Popen:
        npm = which("npm") or which("npm.cmd")
        if not npm:
            error("npm is not found. Please ensure Node.js is installed and in PATH.")
            sys.exit(1)
        # Use cmd /c on Windows for .cmd/.bat files
        if platform.system() == "Windows":
            return subprocess.Popen(
                ["cmd", "/c", "npm", "run", "start", "--", "--host", "0.0.0.0", "--port", str(FRONTEND_PORT)],
                cwd=FRONTEND_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        return subprocess.Popen(
            [npm, "run", "start", "--", "--host", "0.0.0.0", "--port", str(FRONTEND_PORT)],
            cwd=FRONTEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    def stop(self) -> None:
        """Stop both backend and frontend gracefully."""
        log("")
        log("Shutting down services...")

        if self.frontend_proc and self.frontend_proc.poll() is None:
            log("  Stopping frontend...")
            self._terminate(self.frontend_proc)
            self.frontend_proc = None

        if self.backend_proc and self.backend_proc.poll() is None:
            log("  Stopping backend...")
            self._terminate(self.backend_proc)
            self.backend_proc = None

        log("All services stopped.")

    def _terminate(self, proc: subprocess.Popen) -> None:
        """Terminate a process tree."""
        try:
            if platform.system() == "Windows":
                proc.terminate()
            else:
                # On Unix, terminate the process group
                import os
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                if platform.system() == "Windows":
                    proc.kill()
                else:
                    import os
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
            except Exception:
                pass
        except Exception:
            pass


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point."""
    print("")
    print("=" * 60)
    print("  Koudos — Atelier Materials + ML Integration")
    print("=" * 60)
    print("")

    # Determine OS info
    system = platform.system()
    python_ver = platform.python_version()
    log(f"OS: {system} | Python: {python_ver}")
    log(f"Project: {PROJECT_ROOT}")
    log("")

    # Check prerequisites
    if not which("python3") and not which("python"):
        error("Python 3.11+ is required.")
        sys.exit(1)

    # Step 1: Install Python deps
    install_python_deps()

    # Step 2: Install Angular CLI
    install_angular_cli()

    # Step 3: Install frontend deps
    install_frontend_deps()

    # Step 4: Import data
    import_data()

    # Step 5: Launch services
    manager = ServiceManager()

    # Set up signal handlers for clean shutdown
    def signal_handler(sig, frame):
        manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        manager.start()
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()


if __name__ == "__main__":
    main()
