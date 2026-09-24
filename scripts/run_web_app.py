#!/usr/bin/env python3
"""Launch the React + FastAPI web dashboard.

Development (hot reload for the React UI):

    ./.venv/bin/python scripts/run_web_app.py --dev

Production-style (build React once, serve everything from FastAPI):

    ./.venv/bin/python scripts/run_web_app.py

Then open http://127.0.0.1:8000 (prod) or http://127.0.0.1:5173 (dev).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "app" / "web"
DIST_DIR = WEB_DIR / "dist"


def _npm() -> str:
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("npm not found. Install Node.js to build the React frontend.")
    return npm


def build_frontend() -> None:
    if not (WEB_DIR / "node_modules").exists():
        subprocess.run([_npm(), "install"], cwd=WEB_DIR, check=True)
    subprocess.run([_npm(), "run", "build"], cwd=WEB_DIR, check=True)


def run_backend(host: str, port: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.backend.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.call(cmd, cwd=REPO_ROOT)


def run_dev(host: str, port: int) -> int:
    if not (WEB_DIR / "node_modules").exists():
        subprocess.run([_npm(), "install"], cwd=WEB_DIR, check=True)

    backend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.backend.main:app",
            "--host",
            host,
            "--port",
            str(port),
            "--reload",
        ],
        cwd=REPO_ROOT,
    )
    frontend = subprocess.Popen(
        [_npm(), "run", "dev", "--", "--host", host, "--port", "5173"],
        cwd=WEB_DIR,
    )

    print(f"Backend : http://{host}:{port}")
    print(f"Frontend: http://{host}:5173")
    print("Press Ctrl+C to stop both servers.")

    try:
        while True:
            if backend.poll() is not None or frontend.poll() is not None:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (frontend, backend):
            if proc.poll() is None:
                proc.terminate()
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ASL web dashboard")
    parser.add_argument("--dev", action="store_true", help="Run Vite dev server + API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-build", action="store_true", help="Skip npm build in prod mode")
    args = parser.parse_args()

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Web dependencies missing. Install with:\n"
            "  ./.venv/bin/pip install -r app/requirements-web.txt"
        ) from exc

    if args.dev:
        raise SystemExit(run_dev(args.host, args.port))

    if not args.skip_build:
        build_frontend()

    raise SystemExit(run_backend(args.host, args.port))


if __name__ == "__main__":
    main()
