# MLflow end-to-end pipeline (Adult Census Income)

Stages: **Tracking -> Registry -> Deployment -> Evaluation -> Monitoring**, first with ML models, then a DL model.
Everything runs from this folder as modules (`python -m <package>.<script>`).

| Stage | Status |
|---|---|
| 0. Foundations (layout, data validation, run hygiene) | done |
| 1. Tracking (baseline, GBM, PyTorch MLP, searches, lab-parity demos) | done |
| 2. Registry (versions, aliases, 3 gates, challenger, deprecate, rollback, prod copy, lineage) | done |
| 3. Deployment (serve, edge cases, load test, alias experiment, shadow, Docker, batch scoring) | done |
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
python -m s2_registry.register_models [--dry-run]   # best run per family; skips near-duplicates
python -m s2_registry.promote 1                      # bootstrap champion
python -m s2_registry.promote 2                      # 3 gates: contract, budgets, performance
python -m s2_registry.promote --challenge 3          # mark a sound-but-not-better model as challenger
python -m s2_registry.promote --check 2              # audit gates 1-2, change no alias (backfill measurements)
python -m s2_registry.promote --deprecate 3 --reason "..."   # retire, never delete
python -m s2_registry.promote --rollback             # champion <-> previous-champion (twice = undo)
python -m s2_registry.promote_to_prod                # copy the champion into adult-income-classifier-prod
python -m s2_registry.registry_report                # lineage table + findings (reports/registry_report.md)
```

Lab-parity demos: `python -m s2_registry.legacy_stage_demo` (Lab 2's stage API on a throwaway model) and
`python -m s2_registry.uri_tour` (every way to load a model, and which version each really returns).

Gates, in order (cheapest evidence first): **contract** (valid probabilities, reproduces logged AUC),
**budgets** (single-row p95 latency <= 50 ms, size <= 50 MB; env-overridable), **performance**
(beats the champion by 0.005 val AUC). Failures never rewrite history: a version that already
passed keeps `passed` and gets a `last_promotion_attempt` note instead.

Aliases: `champion` (serving), `previous-champion` (rollback target), `challenger` (under evaluation).
Consumers load `models:/adult-income-classifier@champion` (floats with promotions) or
`models:/adult-income-classifier/2` (pinned). **Never serve `models:/.../latest`**: it is the highest
version number, not the best model (the tour shows it resolving to the failed MLP).
Every model returns `[P(<=50K), P(>50K)]` through pyfunc; without `pyfunc_predict_fn="predict_proba"`
sklearn models return hard labels (autolog models do, see `reports/autolog_vs_manual.md`).

## Stage 3: deployment

The model server (`mlflow models serve`, port 5001) is a different process from the tracking server (port 5000).
It needs the tracking server up, because `models:/` URIs are resolved through the registry.

```bash
python -m ops.start_server                                  # tracking server first
python -m s3_deployment.serve_local start                   # serve @champion on :5001 (also: status, stop)
python -m s3_deployment.smoke_test                          # run after EVERY deploy
python -m s3_deployment.parity_test                         # served output == in-process output?
python -m s3_deployment.edge_cases                          # 18 requests, predictions written BEFORE running
python -m s3_deployment.load_test [--workers-compare]       # p50/p95/p99, throughput, 1 vs 2 workers
python -m s3_deployment.alias_experiment                    # proves a running server ignores alias changes
python -m s3_deployment.shadow_compare                      # champion vs challenger on the same traffic
python -m s3_deployment.batch_score make-sample && python -m s3_deployment.batch_score run
python -m s3_deployment.docker_build [--version N]          # build + run + verify a container image
```

Request format (`POST /invocations`, `Content-Type: application/json`):

```json
{"dataframe_split": {"columns": ["age", "workclass", "..."], "data": [[39.0, "State-gov", "..."]]}}
```
Response: `{"predictions": [[P(<=50K), P(>50K)], ...]}`. Lab 1's `{"columns", "instances"}` payload is rejected.

What was measured (numbers from this laptop; client and server share the CPU, so read them as relative):

| Finding | Evidence |
|---|---|
| Served model = in-process model | bit-identical on 2000 rows locally; max diff 1.1e-16 (one float ulp) in the Linux container |
| A request costs ~25 ms regardless of size | 1 row: 32 rows/s. 1000 rows per request: ~21,000 rows/s. Batch your calls |
| One worker serializes requests | 4 concurrent clients: throughput 32 -> 41 req/s, p50 25 -> 95 ms |
| Two workers scale cleanly | same load: 74 req/s, p50 53 ms; batch=1000 reaches ~40,000 rows/s |
| An alias is resolved once, at startup | after a rollback the running server still answered as v2 until restarted |
| Shadow: champion v2 vs challenger v3 | 5.1% of decisions would flip (206 vs 205, symmetric), rank correlation 0.94, mean score 0.236 vs 0.234 |
| Docker image | `docker image inspect` 297 MB (compressed), `docker images` 1.3 GB (unpacked); cold start 12 s; runs as uid 10001; build 227 s |

The model server does NOT validate semantics: `age=250` is scored. Row validation lives in `batch_score.py`
(quarantine file + tripwire) and would live in a gateway for online traffic.

## Known issues (found by Stage 3 testing)

- **v1-v3 carry a wrong signature.** `infer_signature` marked `workclass`, `occupation`, `native-country` as optional
  (they had NaNs), so a request that OMITS one passes MLflow's check and crashes inside the sklearn pipeline with an
  opaque 400. Fixed for new models by `strict_signature` (all columns required, null values still allowed; verified
  against a control). The registered versions predate the fix. Note the promotion gate (+0.005 AUC) would reject an
  equal-quality retrain: shipping a contract-only fix needs an explicit override that does not exist yet.
- **Empty batches fail** (sklearn cannot transform 0 rows): callers must not send them; `batch_score` skips empty chunks.
- **Numeric-looking strings are accepted** (`"age": "39"` is coerced to 39.0): the contract is looser than the types suggest.
- **v3 was trained from an uncommitted working tree**, so its recorded commit cannot reproduce it (see `registry_report`).

## Layout

| Path | Role |
|---|---|
| `config.py` | seed, split sizes, tracking URI, experiment and registered-model names |
| `data.py` | load, clean, **validate**, fingerprint, stratified 60/20/20 split |
| `features.py` | sklearn preprocessing (explicit column lists) |
| `metrics.py` | metrics and plots shared by all models |
| `ops/` | start/stop the MLflow server |
| `s1_tracking/` | `common.py` (git_dirty, data profile, system metrics, timing), `search.py` (random/TPE runner), training scripts, `mlp_pyfunc.py`, demos |
| `s3_deployment/` | `common.py` (start/stop/query a model server), serve, tests, load, shadow, batch scoring, `Dockerfile`, `docker_build.py` |
| `s2_registry/` | `common.py`, register, promote (gates/challenger/deprecate/rollback), promote_to_prod, registry_report, demos |

The test split is never touched during tracking or registry work. It is only used in the final evaluation stage.

## Not included (by choice)

Spark (`spark_udf`, `SparkTrials`; replaced by a pandas batch-scoring job in Stage 3), GitHub Actions / pytest CI,
deep fairness-slice analysis, GenAI tracing, cloud endpoints, Kubernetes.
