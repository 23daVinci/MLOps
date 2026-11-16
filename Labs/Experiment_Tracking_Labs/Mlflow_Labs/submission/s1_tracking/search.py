"""Reusable hyperparameter-search runner with MLflow nested runs.

Lab 2 used hyperopt (TPE) with a parent run and one child run per trial. hyperopt is unmaintained and
breaks on NumPy 2, so we use Optuna, which has the same algorithms. Two samplers on purpose:
  random : the honest reference. Any smarter method must beat this to justify itself.
  tpe    : Tree-structured Parzen Estimator (what hyperopt's tpe.suggest is); learns from past trials.

Usage (call INSIDE an active parent run, so the trials become its children):

    with mlflow.start_run(run_name="gbm-search"):
        result = run_search(space, fit_eval, n_trials=12, sampler="tpe", seed=42)
"""
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SAMPLERS = ("random", "tpe")


@dataclass
class SearchResult:
    best_params: dict
    best_score: float
    scores: list          # score of each trial, in order
    best_so_far: list     # running maximum: the "did the search keep improving?" curve


def n_startup_trials(n_trials: int) -> int:
    """TPE needs some random trials to learn from first. Optuna's default is 10, which on a
    12-trial budget makes TPE ~83% random (and 100% random at 10 trials): we saw TPE and random
    return IDENTICAL results for exactly that reason. Use a third of the budget instead."""
    return max(3, n_trials // 3)


def make_sampler(name: str, seed: int, n_trials: int):
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials(n_trials))
    raise ValueError(f"sampler must be one of {SAMPLERS}, got {name!r}")


def run_search(
    space: Callable,      # space(trial) -> params dict, using trial.suggest_*
    fit_eval: Callable,   # fit_eval(params) -> {"val_auc": ..., ...}; may log extra things to the child run
    n_trials: int,
    sampler: str = "tpe",
    seed: int = 42,
    metric: str = "val_auc",
) -> SearchResult:
    study = optuna.create_study(direction="maximize", sampler=make_sampler(sampler, seed, n_trials))
    if sampler == "tpe":
        mlflow.log_param("tpe_startup_trials", n_startup_trials(n_trials))  # trials before TPE starts guiding
    scores, best_so_far = [], []

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
            mlflow.set_tag("stage", "trial")
            mlflow.log_params(params)
            result = fit_eval(params)
            mlflow.log_metrics(result)
        score = result[metric]
        scores.append(score)
        best_so_far.append(max(scores))
        print(f"trial-{trial.number:02d} [{sampler}] {metric}={score:.4f}  {params}")
        return score

    study.optimize(objective, n_trials=n_trials)

    # Parent-run artifact: did the search converge, or was it still improving at the end?
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scores, "o", alpha=0.5, label="trial")
    ax.plot(best_so_far, "-", label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel(metric)
    ax.set_title(f"search progress ({sampler})")
    ax.legend()
    mlflow.log_figure(fig, "search_progress.png")
    plt.close(fig)
    mlflow.log_metric("search_best_score", study.best_value)

    return SearchResult(dict(study.best_params), study.best_value, scores, best_so_far)
