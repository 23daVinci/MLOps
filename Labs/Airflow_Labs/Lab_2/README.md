# Airflow Lab 2: Multi-DAG ML Pipeline with Model Selection and a Status Page

This lab builds a small machine learning pipeline in Apache Airflow 3 that is split across **three DAGs**:

```
 Airflow_Lab2                      Airflow_Lab2_Train                           Airflow_Lab2_Flask
 (data preparation, @daily)        (model selection, triggered)                 (status web page)

 owner task                        resolve input path
   -> load data                      -> evaluate 5 models in parallel
   -> preprocess                     -> pick the best, save it as model.sav
   -> separate outputs               -> verify the saved model
   -> trigger Train DAG  ---------->    -> send e-mail   (to a local Mailpit inbox)
                                        -> trigger Flask DAG  ------------>   one task that serves
                                                                              http://localhost:5555
```

The model predicts whether a user clicked on an ad (`dags/data/advertising.csv`, 1000 rows, 10 columns).
The Flask page at `http://localhost:5555` combines the state of the first two DAGs into one verdict.

Original lab tutorial video: [Airflow Lab2](https://youtu.be/LwBFOyfN5TY). This README describes the **modified** version of the lab.

---

## Contents

1. [What you need to install](#1-what-you-need-to-install)
2. [Get the code and prepare the folder](#2-get-the-code-and-prepare-the-folder)
3. [Start Airflow](#3-start-airflow)
4. [Run the pipeline](#4-run-the-pipeline)
5. [What a successful run looks like](#5-what-a-successful-run-looks-like)
6. [Try a failure](#6-try-a-failure-optional)
7. [Stop and clean up](#7-stop-and-clean-up)
8. [What I changed in the lab, and why](#8-what-I-changed-in-the-lab-and-why)
9. [How the pieces fit together](#9-how-the-pieces-fit-together)
10. [Using real Gmail instead of Mailpit (optional)](#10-using-real-gmail-instead-of-mailpit-optional)\
11. [Troubleshooting](#12-troubleshooting)
12. [File map](#13-file-map)

---

## 1. What you need to install

You do **not** need Python, Airflow, pandas or scikit-learn on your own machine. Everything runs inside Docker containers. You only need:

| Requirement | Version | Notes |
|---|---|---|
| **Docker Desktop** (Windows, macOS) or **Docker Engine + Compose plugin** (Linux) | Docker Engine 28.x and Docker Compose v2.39 were used for testing. Any Compose **v2** (the `docker compose` command, with a space) should work. The old `docker-compose` v1 is not supported. | On Windows, Docker Desktop's default **WSL 2** backend is required. The installer sets it up. |
| **Git** | any recent version | To clone the repository. |
| **A web browser** | any | For the Airflow UI, the status page and the mail inbox. |
| **Internet access** | | The first start downloads several GB of container images (Airflow alone is over 1 GB compressed) and some Python packages. |

Machine resources (Docker Desktop: Settings > Resources):

- **RAM for Docker:** at least 4 GB, **8 GB recommended**.
- **CPUs:** at least 2.
- **Free disk:** about 10 GB.
- **Free ports on your machine:** `8080` (Airflow UI), `5555` (status page), `8025` (mail inbox). If another program uses one of them, stop it first (see [Troubleshooting](#12-troubleshooting)). The lab binds these ports to `127.0.0.1`, so they are reachable **only from your own machine**, not from other devices on your network. This matters because the UI uses the default login `airflow` / `airflow`.

Install Docker Desktop from <https://docs.docker.com/desktop/>, start it, and wait until it reports that the engine is running. Then check:

```
docker --version
docker compose version
```

Both must print a version. If `docker compose version` fails, your Docker is too old or the Compose plugin is missing.

### Versions that run inside the containers

These come from the container images, so you do not install them. They are listed so results can be reproduced:

| Component | Version |
|---|---|
| Apache Airflow | 3.2.2 (`apache/airflow:3.2.2`, Python 3.13) |
| Airflow executor | CeleryExecutor (workers pick up tasks from a Redis queue) |
| PostgreSQL (Airflow metadata) | 16 |
| Redis (task queue) | 7.2 |
| Mailpit (local mail catcher) | v1.31.6 |
| pandas / scikit-learn / Flask | 2.3.3 / 1.8.0 / 3.1.3 (installed when the containers start; see [Known limitations](#11-known-limitations)) |

---

## 2. Get the code and prepare the folder

All commands below are run from the `Lab_2` folder.

**1. Clone the repository and move into the lab folder.**

```
git clone https://github.com/23daVinci/MLOps.git
cd MLOps/Labs/Airflow_Labs/Lab_2
```

**2. Create the folders that Airflow mounts into the containers.** Git does not store empty folders, so a fresh clone does not have them.

PowerShell (Windows):
```powershell
mkdir config, logs, plugins, working_data, model -Force
```
macOS / Linux / Git Bash:
```bash
mkdir -p config logs plugins working_data model
```

The `config` folder starts empty. Airflow creates its own `airflow.cfg` inside it the first time it starts (section 3). That file contains keys Airflow generates for your installation, so it is deliberately **not** stored in the repository.

**3. Create the `.env` file** from the template. It sets the user ID the Airflow containers run as, and holds optional e-mail settings.

PowerShell:
```powershell
Copy-Item .env.example .env
```
macOS / Linux / Git Bash:
```bash
cp .env.example .env
```
On **Linux**, also replace the user ID with your own: `echo "AIRFLOW_UID=$(id -u)" > .env`. On Windows and macOS, the default `50000` is fine.

**4. Create the Compose file.** The repository keeps it as `docker-compose.lab.yaml` because `docker-compose.yaml` is listed in the repository's `.gitignore`. Copy it so that plain `docker compose` commands find it:

PowerShell:
```powershell
Copy-Item docker-compose.lab.yaml docker-compose.yaml
```
macOS / Linux / Git Bash:
```bash
cp docker-compose.lab.yaml docker-compose.yaml
```

> **Do not run `setup.sh`** from this folder. It deletes `./config`, `./logs` and `./plugins`, deletes `.env`, and runs `docker compose down -v` (which wipes the database). It is a leftover from the original lab and is not needed.
>
> `requirements.txt` is also not used. It is for installing Airflow directly on a machine, which this lab does not do.

---

## 3. Start Airflow

**1. Initialize Airflow** (run once). This creates the database tables and the admin user, then exits.

```
docker compose up airflow-init
```

The first run downloads images, so it takes several minutes. It is finished when the last lines include `airflow-init-1 exited with code 0`. A warning that `FERNET_KEY` is not set is expected and harmless for this lab.

**2. Start all services** in the background:

```
docker compose up -d
```

The first start also installs pandas, scikit-learn and Flask inside each Airflow container (the `_PIP_ADDITIONAL_REQUIREMENTS` setting), so allow **2 to 4 minutes** until everything is healthy.

**3. Check that everything is up:**

```
docker compose ps
```

Expected: every service shows `Up ... (healthy)` except `airflow-init`, which shows `Exited (0)`. The services are:

| Service | Role |
|---|---|
| `postgres` | Airflow's metadata database (run history, task states, XCom values) |
| `redis` | Task queue between the scheduler and the worker |
| `airflow-apiserver` | Web UI and REST API, published on port **8080** |
| `airflow-scheduler` | Decides which tasks are ready and queues them |
| `airflow-dag-processor` | Reads the files in `dags/` and registers the DAGs |
| `airflow-worker` | Runs the task code; also hosts the Flask server, published on port **5555** |
| `airflow-triggerer` | Handles deferred tasks (not used by this lab) |
| `mailpit` | Local mail catcher, web inbox published on port **8025** |

---

## 4. Run the pipeline

**1. Open the Airflow UI:** <http://localhost:8080>. Log in with username `airflow` and password `airflow`.

**2. Check the DAG list.** You should see exactly three DAGs: `Airflow_Lab2`, `Airflow_Lab2_Train` and `Airflow_Lab2_Flask`, and **no red "DAG Import Errors" banner**. (If a DAG is missing, wait a minute and refresh; see [Troubleshooting](#12-troubleshooting).)

**3. Unpause `Airflow_Lab2`.** Use the toggle next to its name. It starts **paused**; the other two do not.

> **Expect one extra run.** `Airflow_Lab2` is scheduled `@daily` with `catchup=False`. Unpausing makes Airflow create **one** run for the most recent missed daily slot (a `scheduled__...` run), and that run behaves like any other. Triggering it yourself in the next step then creates a second run. Both are normal.

**4. Trigger it.** Click the play button on `Airflow_Lab2` and choose **Trigger**.

Command-line alternative (steps 3 and 4):
```
docker compose exec airflow-scheduler airflow dags unpause Airflow_Lab2
docker compose exec airflow-scheduler airflow dags trigger Airflow_Lab2
```

**5. Watch it.** Open each DAG and use the **Grid** or **Graph** view. The full chain takes under two minutes once the containers are warm:

1. `Airflow_Lab2` runs its 5 tasks, the last one (`trigger_train_dag`) starts `Airflow_Lab2_Train`.
2. `Airflow_Lab2_Train` runs, and its last task (`my_trigger_task`) starts `Airflow_Lab2_Flask`.
3. `Airflow_Lab2_Flask` starts a web server and **stays in the `running` state on purpose**.

**6. Open the results:**

- Status page: <http://localhost:5555>
- Mail inbox: <http://localhost:8025>

---

## 5. What a successful run looks like

Use this as a checklist.

**Airflow UI (<http://localhost:8080>)**

| DAG | Expected result |
|---|---|
| `Airflow_Lab2` | Run state **success**. 5 green tasks: `task_using_linked_owner`, `load_data_task`, `data_preprocessing_task`, `separate_data_outputs_task`, `trigger_train_dag`. |
| `Airflow_Lab2_Train` | A run created by `Airflow_Lab2`, state **success**. 10 green task instances: `resolve_input_path`, **5** `evaluate_candidate` instances (mapped, indexes 0 to 4), `select_best_model_task`, `load_model_task`, `send_email`, `my_trigger_task`. |
| `Airflow_Lab2_Flask` | A run in state **running**. This is correct, because the task is a web server that never exits. |

**Log lines.** Open `Airflow_Lab2_Train` > the run > a task > **Logs**:

- Each `evaluate_candidate` instance prints one score, for example `logistic_regression: cv_accuracy=0.9714`.
- `select_best_model_task` prints `Best model: logistic_regression (cv=0.9714, test=0.9700)`.
- `load_model_task` prints `Model score on test data: 0.97`.

Expected cross-validation scores (5-fold, on the training split). They are deterministic with the pinned seeds, so you should see these values with the library versions above:

| Model | CV accuracy |
|---|---|
| **logistic_regression** (winner) | **0.9714** |
| knn | 0.9643 |
| svc | 0.9629 |
| gradient_boosting | 0.9600 |
| random_forest | 0.9586 |

The winning model scores **0.97 accuracy on the held-out test set**.

**Files created on your machine** (in the `Lab_2` folder, through the mounted folders):

| File | What it is |
|---|---|
| `working_data/raw.pkl` | The raw CSV as a pickled DataFrame (about 129 KB) |
| `working_data/preprocessed.pkl` | Train/test split, scaled (about 88 KB) |
| `model/model.sav` | The winning model, a `LogisticRegression` (under 1 KB) |

**Status page (<http://localhost:5555>)**: green, headline **"Pipeline succeeded"**, with two stage cards:
`1. Data preparation -> success` and `2. Model training -> success`, each showing its run ID and times.
While a run is in progress, the page is amber ("Pipeline in progress") and refreshes itself every 10 seconds.

**Mail inbox (<http://localhost:8025>)**: one message per training run, with the subject **`Best model: logistic_regression`**. The body lists the model, its cross-validation accuracy and its test accuracy. (If you triggered `Airflow_Lab2` twice, as step 3 of section 4 describes, there are two messages.)

---

## 6. Try a failure (optional)

This shows the status page reporting a real failure. Run the command from **Git Bash, WSL, macOS or Linux**. It starts the training DAG with a file path that does not exist:

```bash
docker compose exec airflow-scheduler airflow dags trigger Airflow_Lab2_Train --conf '{"preprocessed_path": "/nope/missing.pkl"}'
```

Within about 30 seconds, `Airflow_Lab2_Train` shows a **failed** run (its `evaluate_candidate` tasks fail), and <http://localhost:5555> turns red with the headline **"Model training failed"**. The next successful run turns it green again. (The same input can also be given in the UI's trigger dialog as a JSON config.)

---

## 7. Stop and clean up

Stop and remove the containers, keeping the database (run history, admin user):
```
docker compose down
```

Remove everything, including the database volume (the next start behaves like a first start):
```
docker compose down -v
```

`working_data/`, `model/` and `logs/` are normal folders on your machine; delete them by hand if you want.

---

## 8. What I changed in the lab, and why

The original Lab 2 had two DAGs and was written for an older Airflow. These are the changes, each with the reason.

### 8.1 One training DAG became a separate DAG that compares models

| | Original | Now |
|---|---|---|
| DAGs | 2 (`Airflow_Lab2`, `Airflow_Lab2_Flask`) | **3** (`Airflow_Lab2`, **`Airflow_Lab2_Train`**, `Airflow_Lab2_Flask`) |
| Model | One `LogisticRegression`, trained in `build_save_model_task` | **Five candidates**, compared; the best one is saved |
| Where training lives | Inside `Airflow_Lab2` | In its own DAG, `dags/train_models.py` |

How the training DAG works:

1. **`resolve_input_path`** finds the preprocessed data file. It uses the path sent by the upstream DAG, or a default location if you trigger the DAG by hand.
2. **`evaluate_candidate`** is *one task definition expanded into five parallel task instances* (Airflow's **dynamic task mapping**), one for each of: logistic regression, random forest, gradient boosting, SVC and k-nearest neighbours. Each is scored with **5-fold cross-validation on the training split only**. Adding a model to `CANDIDATES` in `dags/src/model_development.py` adds a parallel task automatically.
3. **`select_best_model_task`** picks the highest cross-validation score, refits that model on the full training split, saves it to `model/model.sav`, and reports its accuracy on the **test** split. The test split is never used to *choose* the model, so the reported test accuracy is an honest estimate.
4. **`load_model_task`** reloads the saved file and prints its test score, proving the saved model works.
5. **`send_email`** and **`my_trigger_task`** run in parallel at the end.

### 8.2 The DAGs are chained by triggers, passing data through `conf`

- `Airflow_Lab2` ends with `trigger_train_dag`, a `TriggerDagRunOperator`. It passes the path of the preprocessed file in `conf`, so the training DAG knows exactly which file to use. It uses the **default trigger rule**, so if data preparation fails, training does **not** run on stale data.
- `Airflow_Lab2_Train` ends with `my_trigger_task`, which starts the Flask DAG. It uses `trigger_rule=ALL_DONE`, so the status page is launched **even if training failed**, which is when you most need it.
- Both triggers use `wait_for_completion=False` (fire and forget). This is essential for the Flask trigger, because the Flask task never finishes.

### 8.3 E-mail notification, enabled and made testable

- The `send_email` task (previously commented out) is active, in the training DAG. Its subject names the winning model, and its body shows the scores (rendered from the previous task's output).
- **Why Mailpit:** the original instructions use a Gmail app password and `airflow.cfg`. Airflow 3's `EmailOperator` reads an **Airflow connection** (`smtp_default`) instead. I added a `mailpit` container (a fake mail server that catches mail and shows it at <http://localhost:8025>) so e-mail can be tested with **no real credentials** and nothing leaving your machine.
- The connection is defined in `docker-compose.lab.yaml` as the environment variable `AIRFLOW_CONN_SMTP_DEFAULT`, built from `SMTP_*` variables that default to Mailpit. Switching to Gmail needs only values in `.env` ([section 10](#10-using-real-gmail-instead-of-mailpit-optional)).
- The recipient is set in `dags/train_models.py` (`to=`). With Mailpit nothing is delivered anywhere.

### 8.4 The status page now reports both pipeline stages (and was repaired)

- **Bug fix:** the original Flask code authenticated to Airflow's REST API with HTTP **Basic** auth. Airflow 3 rejects that (`401`), so the original page always showed "failed". The code now requests a short-lived **JWT token** from `/auth/token` and sends it as a `Bearer` token.
- **Combined page:** `http://localhost:5555` now shows **both** `Airflow_Lab2` and `Airflow_Lab2_Train`, one card each, and one overall verdict, using this rule:
  - any stage **failed** (or its status cannot be read) -> **failed**, naming the stage;
  - otherwise any stage **running, queued or not yet started** -> **in progress**;
  - otherwise -> **success**.
- The old `/success` and `/failure` pages were replaced by one page, `dags/templates/status.html`. (`success.html` and `failure.html` are still in the folder but are no longer used.)
- The page lists the stages in the `STAGES` list near the top of `dags/Flask_API.py`.

### 8.5 Smaller changes

- **Airflow 3 setup:** `docker-compose.lab.yaml` is the official Airflow 3.2.2 Compose file with these edits: example DAGs off, pandas/scikit-learn/Flask installed at start-up, the `working_data` and `model` folders mounted into the containers, port `5555` published on the worker (for Flask), the SMTP connection, and the `mailpit` service.
- **Owner:** the linked owner in `dags/main.py` is now `Naman Lazarus` (the key in `owner_links` must match the task's `owner` exactly for the link to appear in the UI).
- **Removed:** `build_save_model_task` and the old single-model `build_model` function (replaced by section 8.1).
- **Reproducibility:** Mailpit is pinned to `v1.31.6` and the Airflow image to `3.2.2`; random seeds are fixed for the models.
- **README corrections:** the original README described `notify_success` / `notify_failure` callbacks and a Flask `check_dag_status()` function that do not exist in the code. They are not described here.

---

## 9. How the pieces fit together

**Where each task runs.** You write DAG files; the *dag-processor* reads them; the *scheduler* decides which task is ready and puts it on the Redis queue; a *worker* takes it and runs your Python function; results and states are stored in Postgres; the *API server* shows it all.

**How data moves between tasks.** Tasks do not share memory, so each one saves its result to a file under `working_data/` and returns only the **file path**. Airflow stores that returned path as an **XCom** (a small value in Postgres) and gives it to the next task. Passing paths, not whole datasets, keeps the database small. It works here because the folder is mounted into the worker; with several workers on different machines you would need shared storage such as S3.

**How the DAGs are connected.** `trigger_train_dag` starts the training DAG and sends it `conf={"preprocessed_path": ...}`; the training DAG's first task reads it. Nothing else links the DAGs.

**How the status page gets its data.** Flask runs inside the worker container. On each page load it asks the Airflow API (`http://airflow-apiserver:8080`, a container name that works only on Docker's internal network) for the latest run of each stage, using a fresh token.

**Why a task that never finishes is acceptable.** `start_Flask_API` calls `app.run(...)`, which serves forever, so its run stays `running` and holds one worker slot. That is what keeps the page alive.

---

## 10. Using real Gmail instead of Mailpit (optional)

Commented out for security reasons. Gmail needs an **app password** (a 16-character password you create in your Google account after turning on 2-Step Verification). 

1. Open `.env` and uncomment the `SMTP_*` lines, replacing the placeholders with your address and app password. **Never commit `.env`** (it is listed in the repository's `.gitignore`) and never paste the password anywhere else.
2. In `dags/train_models.py`, change `to=` from the placeholder `you@example.com` to an address you control.
3. Recreate the containers so they read the new values: `docker compose up -d`.
4. Trigger a run and check your inbox and spam folder.

If sending fails, the log of the `send_email` task shows the SMTP error; `535` means the login was rejected. Some networks block outgoing SMTP on port 587.

---

## 11. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `docker compose version` fails | Docker too old, or Compose v2 missing. Update Docker Desktop. |
| `port is already allocated` on 8080, 5555 or 8025 | Another program or another Airflow stack is using the port. Stop it (for example `docker compose down` in the other project) and run `docker compose up -d` again. |
| `airflow-init` ends with a non-zero code | Read its log: `docker compose logs airflow-init`. The most common cause is too little memory for Docker. |
| A service stays `starting` for several minutes | The first start installs packages. Check `docker compose logs airflow-worker`. |
| A DAG run stays `queued` and no task starts | The DAG is **paused**. Unpause it with the toggle. (A triggered run of a paused DAG waits.) |
| A task is `queued` with the DAG unpaused | The worker is busy or not healthy. Check `docker compose ps`. |
| Red "DAG Import Errors" banner | Open it to see the traceback. `ModuleNotFoundError` for pandas or sklearn right after the first start means packages are still installing; wait and refresh. |
| A DAG is missing from the list | New files are found within 5 minutes; force with `airflow dags reserialize` (section 11). |
| <http://localhost:5555> does not load | The Flask DAG has not started yet. Trigger `Airflow_Lab2_Flask`, or finish a training run. |
| Status page says "Cannot reach Airflow" | The API server is down or unhealthy. Check `docker compose ps`. |
| Status page shows old behaviour after an edit | Restart the worker and re-trigger the Flask DAG (section 11). |
| No e-mail in the inbox | Check that the `send_email` task succeeded, that `.env` has the `SMTP_*` lines commented out, and that you are looking at <http://localhost:8025>. |
| Permission errors writing `model/` or `working_data/` (Linux) | Set `AIRFLOW_UID` in `.env` to `$(id -u)` and recreate the folders. |

---

## 12. File map

```
Lab_2/
  README.md                    this file
  docker-compose.lab.yaml      the Compose file (copy it to docker-compose.yaml, section 2)
  .env.example                 template for .env (user ID and optional SMTP settings)
  config/airflow.cfg           generated by Airflow on first start (not in git: it holds generated keys)
  dags/
    main.py                    DAG 1: Airflow_Lab2 (data preparation, triggers training)
    train_models.py            DAG 2: Airflow_Lab2_Train (compare models, save best, e-mail)
    Flask_API.py               DAG 3: Airflow_Lab2_Flask (status page server)
    src/model_development.py   the task functions: load, preprocess, candidates, select best
    templates/status.html      the combined status page
    data/advertising.csv       the dataset
  working_data/                created at run time: raw.pkl, preprocessed.pkl (not in git)
  model/                       created at run time: model.sav (not in git)
  logs/, plugins/              Airflow logs and plugins folders (not in git)
```

Not used by this version of the lab: `setup.sh`, `requirements.txt`, `cookies.txt`, `templates/success.html`, `templates/failure.html`.
