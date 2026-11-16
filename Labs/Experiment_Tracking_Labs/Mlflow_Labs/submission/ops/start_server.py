"""Start the MLflow tracking server in the background and wait until it is healthy.

    python -m ops.start_server

Why these flags (each one fixed a real problem):
  --backend-store-uri sqlite:///mlflow.db   SQL backend: required by the Model Registry
  --artifacts-destination ./mlartifacts     where models/plots are stored (served via the server)
  --workers 1                               several workers sharing one SQLite file stalled run creation
"""
import os
import subprocess
import sys
import time
import urllib.request

import config

PID_FILE = config.ROOT / "ops" / ".server.pid"
LOG_FILE = config.ROOT / "logs" / "mlflow_server.log"
HOST, PORT = "127.0.0.1", 5000


def healthy() -> bool:
    try:
        return urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=2).status == 200
    except Exception:
        return False


def main(timeout_s: int = 120) -> None:
    if healthy():
        print(f"MLflow already running at http://{HOST}:{PORT}")
        return
    LOG_FILE.parent.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", f"sqlite:///{config.ROOT / 'mlflow.db'}",
        "--artifacts-destination", str(config.ROOT / "mlartifacts"),
        "--host", HOST, "--port", str(PORT), "--workers", "1",
    ]
    flags = 0
    if os.name == "nt":
        # Outlive this script, no visible window. NOT DETACHED_PROCESS: with no console at all,
        # MLflow's uvicorn child died immediately with 0xC000013A (console-control exit).
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    with open(LOG_FILE, "ab") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=log, cwd=config.ROOT, creationflags=flags,
            env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
        )
    PID_FILE.write_text(str(proc.pid))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if healthy():
            print(f"MLflow up at http://{HOST}:{PORT} (pid {proc.pid}); log: {LOG_FILE}")
            return
        if proc.poll() is not None:
            sys.exit(f"server exited early (code {proc.returncode}); see {LOG_FILE}")
        time.sleep(1)
    sys.exit(f"server not healthy after {timeout_s}s; see {LOG_FILE}")


if __name__ == "__main__":
    main()
