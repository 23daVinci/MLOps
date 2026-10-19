from __future__ import annotations

import os
import pendulum
import requests
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from flask import Flask, render_template

# ---------- Config ----------
# Ensure WEBSERVER points to 'airflow-apiserver' as defined in your compose
WEBSERVER = os.getenv("AIRFLOW_WEBSERVER", "http://airflow-apiserver:8080")
AF_USER   = os.getenv("AIRFLOW_USERNAME", "airflow")
AF_PASS   = os.getenv("AIRFLOW_PASSWORD", "airflow")

# Pipeline stages shown on the status page, in order: (dag_id, label)
STAGES = [
    ("Airflow_Lab2", "Data preparation"),
    ("Airflow_Lab2_Train", "Model training"),
]

# Set up robust paths for Docker Workers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# ---------- Flask app ----------
app = Flask(__name__, template_folder=TEMPLATE_DIR)


def get_token():
    """Airflow 3 uses JWT auth: exchange username/password for a short-lived token."""
    resp = requests.post(
        f"{WEBSERVER}/auth/token",
        json={"username": AF_USER, "password": AF_PASS},
        timeout=10,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Auth Error {resp.status_code}")
    return resp.json()["access_token"]


def get_stage_info(dag_id, label, headers):
    """Latest run of one DAG, reduced to a status page row."""
    # Airflow 3 prefers ordering by 'run_after' for the most recent activity
    url = f"{WEBSERVER}/api/v2/dags/{dag_id}/dagRuns?order_by=-run_after&limit=1"
    row = {
        "dag_id": dag_id,
        "label": label,
        "state": "unknown",
        "status": "unknown",   # success | in_progress | failed | unknown
        "run_id": "N/A",
        "logical_date": "N/A",
        "start_date": "N/A",
        "end_date": "N/A",
        "note": "",
    }

    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        row["note"] = f"API Error {r.status_code}"
        return row

    runs = r.json().get("dag_runs", [])
    if not runs:
        row["status"] = "in_progress"
        row["state"] = "no runs yet"
        row["note"] = "Waiting for a DAG run to appear..."
        return row

    run = runs[0]
    state = run.get("state", "unknown")
    row.update({
        "state": state,
        "run_id": run.get("dag_run_id", "N/A"),
        "logical_date": run.get("logical_date") or run.get("run_after") or "N/A",
        "start_date": run.get("start_date") or "N/A",
        "end_date": run.get("end_date") or "N/A",
    })

    if state == "success":
        row["status"] = "success"
    elif state in ("queued", "running"):
        row["status"] = "in_progress"
    elif state == "failed":
        row["status"] = "failed"
    return row


def get_pipeline_status():
    """
    Combined verdict over all stages:
      any stage failed or unreadable -> failed
      else any stage unfinished      -> in_progress
      else all stages succeeded      -> success
    """
    try:
        headers = {"Accept": "application/json", "Authorization": f"Bearer {get_token()}"}
        stages = [get_stage_info(dag_id, label, headers) for dag_id, label in STAGES]
    except Exception as e:
        return {"overall": "failed", "stages": [], "headline": "Cannot reach Airflow",
                "note": f"Network issue: {e}", "failed_stage": None}

    statuses = [s["status"] for s in stages]
    failed = next((s for s in stages if s["status"] in ("failed", "unknown")), None)

    if failed:
        overall = "failed"
        headline = f"{failed['label']} failed" if failed["status"] == "failed" \
            else f"{failed['label']} status unavailable"
    elif "in_progress" in statuses:
        overall = "in_progress"
        headline = "Pipeline in progress"
    else:
        overall = "success"
        headline = "Pipeline succeeded"

    return {"overall": overall, "stages": stages, "headline": headline,
            "note": "", "failed_stage": failed["label"] if failed else None}


@app.route("/")
def index():
    return render_template("status.html", **get_pipeline_status())

@app.route("/health")
def health():
    return "ok", 200

def start_flask_app():
    """
    Starts the Flask server.
    Note: Blocks the worker slot. Disable reloader for Docker.
    """
    print(f"Starting Flask on 0.0.0.0:5555 checking {WEBSERVER}...", flush=True)
    app.run(host="0.0.0.0", port=5555, debug=False, use_reloader=False)

# ---------- DAG Definition ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

dag = DAG(
    dag_id="Airflow_Lab2_Flask",
    default_args=default_args,
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
)

PythonOperator(
    task_id="start_Flask_API",
    python_callable=start_flask_app,
    dag=dag,
)
