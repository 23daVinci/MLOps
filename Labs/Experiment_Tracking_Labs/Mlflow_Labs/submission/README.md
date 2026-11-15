# MLflow end-to-end pipeline (Adult Census Income)

Stages: **Tracking** -> Registry -> Deployment -> Evaluation, first with an ML model, then a DL model.

## Setup (Windows, from this folder)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## 1. Start the tracking server (keep it running in its own terminal)

```bash
.venv/Scripts/python -m mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 --port 5000 --workers 1
```

- `sqlite:///mlflow.db` is the **backend store** (runs, params, metrics, registry). A SQL backend is what the Model Registry requires.
- `./mlartifacts` is the **artifact store** (models, plots).
- `--workers 1`: several workers sharing one SQLite file stalled run creation on this machine.
- UI: http://127.0.0.1:5000

## 2. Baseline run

```bash
.venv/Scripts/python train_baseline.py
```

## 3. Better model (random search, nested runs)

```bash
.venv/Scripts/python train_gbm.py      # ~5 min: 12 trials + refit of the best
```

## 4. Registry

```bash
.venv/Scripts/python register_models.py   # creates 'adult-income-classifier' v1 (baseline), v2 (gbm)
.venv/Scripts/python promote.py 1         # bootstrap champion (contract gate only)
.venv/Scripts/python promote.py 2         # must pass contract + beat champion by 0.005 val AUC
.venv/Scripts/python promote.py --rollback  # champion <-> previous-champion (run twice to undo)
```

Consumers load `models:/adult-income-classifier@champion` (follows promotions) or
`models:/adult-income-classifier/2` (pinned). Models are served through pyfunc, which returns
`[P(<=50K), P(>50K)]` (`pyfunc_predict_fn="predict_proba"`); without that, pyfunc returns hard labels.

## Files

| File | Role |
|---|---|
| `config.py` | seed, split sizes, tracking URI, experiment name |
| `data.py` | load, clean, fingerprint, stratified 60/20/20 split |
| `evaluation.py` | preprocessing pipeline, metrics, plots (shared by all models) |
| `train_baseline.py` | logistic regression baseline, fully tracked |
| `train_gbm.py` | gradient boosting with a tracked random search (parent + nested trial runs) |
| `register_models.py` | registers candidates as versions with model card, tags, lineage |
| `promote.py` | gated alias promotion (`champion`) and rollback |

The test split is never touched during tracking. It is only used in the final evaluation stage.
