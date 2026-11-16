"""Shared helpers for Stage 3: run a model server, talk to it over HTTP, build request payloads.

The model server is a DIFFERENT process from the tracking server (port 5000):
  tracking server  :5000  runs/models/registry metadata; needed to resolve models:/ URIs
  model server     :5001  `mlflow models serve`; answers POST /invocations
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

import config
import data

MODEL_PORT = 5001
LOG_DIR = config.ROOT / "logs"


def url_for(port: int = MODEL_PORT) -> str:
    return f"http://127.0.0.1:{port}"


def _pid_file(port: int):
    return LOG_DIR / f".model_server_{port}.pid"


def get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    """GET returning (status, body); HTTP errors are returned, not raised."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def wait_healthy(base_url: str, timeout_s: float = 180, proc=None) -> float:
    """Poll /health until 200. Returns seconds waited. Exits early if the process died."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if get(f"{base_url}/health", 2)[0] == 200:
            return time.time() - t0
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server process exited early with code {proc.returncode}")
        time.sleep(0.5)
    raise TimeoutError(f"{base_url}/health not OK after {timeout_s}s")


def start_model_server(model_uri: str, port: int = MODEL_PORT, workers: int = 1, timeout_s: float = 180) -> dict:
    """Start `mlflow models serve` detached. models:/ URIs need the tracking server (:5000) up.
    --env-manager local = use THIS python environment (the default builds a fresh virtualenv, slow)."""
    if get(f"{url_for(port)}/health", 2)[0] == 200:
        raise RuntimeError(f"something is already serving on port {port}; stop it first")
    LOG_DIR.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlflow", "models", "serve", "-m", model_uri,
        "-h", "127.0.0.1", "-p", str(port), "--workers", str(workers), "--env-manager", "local",
    ]
    flags = 0
    if os.name == "nt":  # NOT DETACHED_PROCESS (see ops/start_server.py): hidden console instead
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    log = open(LOG_DIR / f"model_server_{port}.log", "wb")
    t0 = time.time()
    # `mlflow models serve` runs a BARE `uvicorn ...` shell command. Without an activated venv that
    # resolves to whichever uvicorn is first on PATH: here the GLOBAL Python install, i.e. the wrong
    # environment (it crashed on import). Put this interpreter's Scripts dir first so the venv wins.
    scripts = os.path.dirname(sys.executable)
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=log, cwd=config.ROOT, creationflags=flags,
        env={**os.environ, "PATH": scripts + os.pathsep + os.environ.get("PATH", ""),
             "MLFLOW_TRACKING_URI": config.TRACKING_URI, "MLFLOW_DISABLE_AGENT_HINT": "1"},
    )
    _pid_file(port).write_text(str(proc.pid))
    try:
        wait_healthy(url_for(port), timeout_s, proc)
    except Exception:
        stop_model_server(port)
        raise RuntimeError(f"model server failed to become healthy; see {LOG_DIR / f'model_server_{port}.log'}")
    return {"pid": proc.pid, "port": port, "seconds_to_healthy": time.time() - t0}


def stop_model_server(port: int = MODEL_PORT) -> bool:
    pf = _pid_file(port)
    if not pf.exists():
        return False
    pid = int(pf.read_text())
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)  # /T = whole tree
    else:
        os.kill(pid, 15)
    pf.unlink(missing_ok=True)
    time.sleep(1)
    return True


# ------------------------------------------------------------------ requests ---
def post(base_url: str, body: bytes | str, content_type: str = "application/json", timeout: float = 60.0):
    """POST to /invocations. Returns (status, response_text, seconds). Never raises on HTTP errors."""
    if isinstance(body, str):
        body = body.encode()
    req = urllib.request.Request(
        f"{base_url}/invocations", data=body, headers={"Content-Type": content_type}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), time.perf_counter() - t0
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}", time.perf_counter() - t0


def split_payload(df: pd.DataFrame) -> dict:
    """dataframe_split payload. double_precision=15 matters: pandas' default of 10 digits would round
    the inputs and make a server-vs-local comparison fail for the wrong reason."""
    return {"dataframe_split": json.loads(df.to_json(orient="split", index=False, double_precision=15))}


def records_payload(df: pd.DataFrame) -> dict:
    return {"dataframe_records": json.loads(df.to_json(orient="records", double_precision=15))}


def predictions(response_text: str):
    """The model's output array from a /invocations response ({"predictions": [...]})."""
    import numpy as np
    return np.asarray(json.loads(response_text)["predictions"], dtype="float64")


def sample_features(n: int, seed: int = 0) -> pd.DataFrame:
    """n raw feature rows from the VALIDATION split (the test split stays untouched until Stage 4)."""
    _, _, _, val, _ = data.get_splits()
    X, _ = data.xy(val)
    return X.sample(n=min(n, len(X)), random_state=seed).reset_index(drop=True)
