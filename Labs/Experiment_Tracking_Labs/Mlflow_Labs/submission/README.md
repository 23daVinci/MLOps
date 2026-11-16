# MLflow end-to-end pipeline (Adult Census Income)

Stages: **Tracking -> Registry -> Deployment -> Evaluation -> Monitoring**, first with ML models, then a DL model.
Everything runs from this folder as modules (`python -m <package>.<script>`).

| Stage | Status |
|---|---|
| 0. Foundations (layout, data validation, run hygiene) | done |
| 1. Tracking (baseline, GBM, PyTorch MLP, searches, lab-parity demos) | done |
| 2. Registry (versions, aliases, gated promotion, rollback) | base done; extensions planned |
| 3. Deployment (serve, Docker, batch scoring) | planned |
| 4. Evaluation (final test eval, comparison, importance) | planned |
| 5. Monitoring (Evidently, closed loop) | planned |

## Setup (Windows, from this folder)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-dl.txt
```

## Server

```bash
.venv/Scripts/python -m ops.start_server    # starts detached, waits for /health, UI at http://127.0.0.1:5000
.venv/Scripts/python -m ops.stop_server     # stops the whole process tree
```

Flags it uses, each of which fixed a real problem: `--backend-store-uri sqlite:///mlflow.db` (SQL backend,
required by the registry), `--artifacts-destination ./mlartifacts`, `--workers 1` (several workers sharing one
SQLite file stalled run creation). Backends: the old `mlruns/` file store has no registry and is deprecated in
MLflow 3; SQLite + server is the minimum for anything registry-related.

## Stage 1: tracking

```bash
python -m s1_tracking.train_baseline                          # logistic regression
python -m s1_tracking.train_gbm --sampler tpe --trials 12     # gradient boosting, Optuna TPE
python -m s1_tracking.train_gbm --sampler random --trials 12  # same budget, random search (the control)
python -m s1_tracking.train_mlp --sampler tpe --trials 10     # PyTorch MLP, models-from-code pyfunc
python -m s1_tracking.compare_runs                            # leaderboard + sampler plot in reports/
```

Lab-parity demos (each uses its own experiment so the leaderboard stays clean):

```bash
python -m s1_tracking.eda                   # Lab 2 steps 4-7: target, distributions, missingness (TRAIN split only)
python -m s1_tracking.autolog_demo          # Lab 1: autolog vs manual, measured
python -m s1_tracking.env_requirements_demo # Lab 1 serving.py: pip_requirements / extra / file / constraints
python -m s1_tracking.experiment_admin      # course notes: rename, tag, delete, restore, gc (throwaway experiment)
```

## Stage 2: registry

```bash
python -m s2_registry.register_models      # 'adult-income-classifier' v1 (baseline), v2 (gbm)
python -m s2_registry.promote 1            # bootstrap champion (contract gate only)
python -m s2_registry.promote 2            # must pass contract + beat champion by 0.005 val AUC
python -m s2_registry.promote --rollback   # champion <-> previous-champion (run twice to undo)
```

Consumers load `models:/adult-income-classifier@champion` (follows promotions) or
`models:/adult-income-classifier/2` (pinned). Every model is served through pyfunc and returns
`[P(<=50K), P(>50K)]`; without `pyfunc_predict_fn="predict_proba"` sklearn models return hard labels
(autolog models do exactly that, see `reports/autolog_vs_manual.md`).

## Layout

| Path | Role |
|---|---|
| `config.py` | seed, split sizes, tracking URI, experiment and registered-model names |
| `data.py` | load, clean, **validate**, fingerprint, stratified 60/20/20 split |
| `features.py` | sklearn preprocessing (explicit column lists) |
| `metrics.py` | metrics and plots shared by all models |
| `ops/` | start/stop the MLflow server |
| `s1_tracking/` | `common.py` (git_dirty, data profile, system metrics, timing), `search.py` (random/TPE runner), training scripts, `mlp_pyfunc.py`, demos |
| `s2_registry/` | register, promote, rollback |

The test split is never touched during tracking or registry work. It is only used in the final evaluation stage.

## Not included (by choice)

Spark (`spark_udf`, `SparkTrials`; replaced by a pandas batch-scoring job in Stage 3), GitHub Actions / pytest CI,
deep fairness-slice analysis, GenAI tracing, cloud endpoints, Kubernetes.
