"""Stop the MLflow server started by ops.start_server (kills the whole process tree).

    python -m ops.stop_server
"""
import os
import subprocess
import sys

from ops.start_server import PID_FILE, healthy


def main() -> None:
    if not PID_FILE.exists():
        print("no pid file; server was not started by ops.start_server" + (" (but something answers on :5000)" if healthy() else ""))
        return
    pid = int(PID_FILE.read_text())
    if os.name == "nt":
        # /T kills child processes too (uvicorn workers); /F forces
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.kill(pid, 15)
    PID_FILE.unlink(missing_ok=True)
    print(f"stopped MLflow server (pid {pid}); port 5000 {'STILL BUSY' if healthy() else 'free'}")


if __name__ == "__main__":
    sys.exit(main())
