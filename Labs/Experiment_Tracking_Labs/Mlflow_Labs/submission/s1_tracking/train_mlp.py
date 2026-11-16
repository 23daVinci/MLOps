"""PyTorch MLP (embeddings for categoricals) with a tracked hyperparameter search.

Same structure as the GBM so the two are directly comparable in the UI:
    mlp-search-<sampler> (parent) -> trial-00 .. trial-N (children, each with per-epoch curves)

    python -m s1_tracking.train_mlp --sampler tpe --trials 10
"""
import argparse
import copy
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

import config
import data
from metrics import compute_metrics, make_plots
from s1_tracking.common import log_run_context, setup_mlflow, strict_signature, timed
from s1_tracking.mlp_pyfunc import MLPNet, TabularEncoder, net_config
from s1_tracking.search import SAMPLERS, run_search

SELECTION_METRIC = "val_auc"
MAX_EPOCHS = 30
PATIENCE = 4          # stop when val AUC has not improved for this many epochs
MODEL_CODE = Path(__file__).parent / "mlp_pyfunc.py"


def space(trial) -> dict:
    return {
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "width": trial.suggest_categorical("width", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "emb_cap": trial.suggest_categorical("emb_cap", [4, 8, 16]),
    }


def predict_proba(net: nn.Module, x_num, x_cat) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(x_num, x_cat)).numpy().astype("float64")


def train_one(params: dict, encoder: TabularEncoder, tensors: dict, y_train, y_val):
    """Train with early stopping on val AUC. Logs a per-epoch curve (step=epoch) to the ACTIVE run.
    Returns (best_net, history, best_epoch)."""
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    cfg = net_config(encoder, [params["width"]] * params["n_layers"], params["dropout"], params["emb_cap"])
    net = MLPNet(**cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    xn, xc, yt = tensors["train_num"], tensors["train_cat"], torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    n, bs = len(yt), params["batch_size"]

    best_auc, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(net(xn[idx], xc[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_proba = predict_proba(net, tensors["val_num"], tensors["val_cat"])
        val_auc = roc_auc_score(y_val, val_proba)
        val_loss = float(-np.mean(y_val * np.log(val_proba + 1e-12) + (1 - y_val) * np.log(1 - val_proba + 1e-12)))
        row = {"train_loss": total / n, "val_loss": val_loss, "val_auc_epoch": val_auc}
        history.append(row)
        mlflow.log_metrics(row, step=epoch)  # time series: shows overfitting / convergence in the UI
        if val_auc > best_auc:
            best_auc, best_epoch, best_state = val_auc, epoch, copy.deepcopy(net.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break
    net.load_state_dict(best_state)  # restore the best epoch, not the last
    return net, history, best_epoch


def main(sampler: str, n_trials: int):
    setup_mlflow()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    df, data_version, train, val, _test = data.get_splits()
    X_train, y_train = data.xy(train)
    X_val, y_val = data.xy(val)
    numeric_cols, categorical_cols = data.feature_columns(X_train)

    # Encoder is fit on TRAIN only (vocab, medians, means), exactly like the sklearn preprocessors.
    encoder = TabularEncoder.fit(X_train, numeric_cols, categorical_cols)
    tr_num, tr_cat = encoder.transform(X_train)
    va_num, va_cat = encoder.transform(X_val)
    tensors = {"train_num": tr_num, "train_cat": tr_cat, "val_num": va_num, "val_cat": va_cat}
    y_val_np = y_val.to_numpy()

    best = {"score": -1.0, "net": None, "params": None, "history": None, "epoch": 0}

    def fit_eval(params: dict) -> dict:
        with timed("trial_train_seconds"):
            net, history, best_epoch = train_one(params, encoder, tensors, y_train, y_val_np)
        train_proba = predict_proba(net, tr_num, tr_cat)
        val_proba = predict_proba(net, va_num, va_cat)
        out = {}
        for name, y, proba in [("train", y_train.to_numpy(), train_proba), ("val", y_val_np, val_proba)]:
            out.update({f"{name}_{k}": v for k, v in compute_metrics(y, proba).items()})
        out["best_epoch"] = best_epoch
        out["n_parameters"] = sum(p.numel() for p in net.parameters())
        if out[SELECTION_METRIC] > best["score"]:  # keep the best trial's weights: no refit needed
            best.update(score=out[SELECTION_METRIC], net=net, params=params, history=history, epoch=best_epoch)
        return out

    with mlflow.start_run(run_name=f"mlp-search-{sampler}") as parent:
        mlflow.set_tags({"stage": "candidate", "model_family": "neural_net", "sampler": sampler})
        log_run_context(data_version, train, val)
        mlflow.log_params(
            {
                "n_trials": n_trials, "selection_metric": SELECTION_METRIC, "search_strategy": sampler,
                "seed": config.SEED, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                "n_train": len(train), "n_val": len(val), "torch_version": torch.__version__,
                "numeric_cols": ",".join(numeric_cols), "categorical_cols": ",".join(categorical_cols),
            }
        )
        mlflow.log_input(mlflow.data.from_pandas(train, name="adult-train", targets=config.TARGET), context="training")
        mlflow.log_input(mlflow.data.from_pandas(val, name="adult-val", targets=config.TARGET), context="validation")

        result = run_search(space, fit_eval, n_trials, sampler=sampler, seed=config.SEED)

        # ---- the best trial's model, with its training curve --------------------
        net, params = best["net"], best["params"]
        mlflow.log_params({f"best__{k}": v for k, v in params.items()})
        train_proba = predict_proba(net, tr_num, tr_cat)
        val_proba = predict_proba(net, va_num, va_cat)
        final_metrics = {}
        for name, y, proba in [("train", y_train.to_numpy(), train_proba), ("val", y_val_np, val_proba)]:
            final_metrics.update({f"{name}_{k}": v for k, v in compute_metrics(y, proba).items()})
        mlflow.log_metrics(final_metrics)
        mlflow.log_metric("best_epoch", best["epoch"])

        hist = pd.DataFrame(best["history"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist["train_loss"], label="train loss")
        ax.plot(hist["val_loss"], label="val loss")
        ax.axvline(best["epoch"], ls="--", c="gray", label="best epoch")
        ax.set_xlabel("epoch"); ax.set_ylabel("log loss"); ax.legend(); ax.set_title("best trial: loss curves")
        mlflow.log_figure(fig, "loss_curves.png"); plt.close(fig)
        for filename, fig in make_plots(y_val_np, val_proba).items():
            mlflow.log_figure(fig, f"val_plots/{filename}"); plt.close(fig)

        # ---- package: plain-data artifacts + models-from-code pyfunc -------------
        cfg = net_config(encoder, [params["width"]] * params["n_layers"], params["dropout"], params["emb_cap"])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "encoder.json").write_text(json.dumps(encoder.to_dict()))
            (tmp / "config.json").write_text(json.dumps(cfg))
            torch.save(net.state_dict(), tmp / "weights.pt")
            signature = strict_signature(X_train, np.zeros((len(X_train), 2)))  # 13 inputs -> (n, 2) float64
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=str(MODEL_CODE),  # a PATH = models-from-code (no pickled classes)
                artifacts={"encoder": str(tmp / "encoder.json"), "config": str(tmp / "config.json"), "weights": str(tmp / "weights.pt")},
                signature=signature,
                input_example=X_train.head(5),
                pip_requirements=[
                    f"mlflow=={mlflow.__version__}",
                    f"torch=={torch.__version__.split('+')[0]}",
                    f"numpy=={np.__version__}",
                    f"pandas=={pd.__version__}",
                ],
            )

        # ---- round-trip self-test: what we LOADED must equal what we TRAINED -----
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        reloaded = np.asarray(loaded.predict(X_val))
        diff = float(np.abs(reloaded[:, 1] - val_proba).max())
        mlflow.log_metric("reload_max_abs_diff", diff)
        assert reloaded.shape == (len(X_val), 2), f"bad output shape {reloaded.shape}"
        assert diff < 1e-5, f"reloaded model differs from trained model by {diff}"

        print(f"\nparent run_id: {parent.info.run_id}")
        print(f"sampler:       {sampler}")
        print(f"best params:   {params}  (best epoch {best['epoch']})")
        print(f"val_auc:       {final_metrics['val_auc']:.4f}   train_auc: {final_metrics['train_auc']:.4f}")
        print(f"reload check:  max |loaded - trained| = {diff:.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sampler", choices=SAMPLERS, default="tpe")
    p.add_argument("--trials", type=int, default=10)
    args = p.parse_args()
    main(args.sampler, args.trials)
