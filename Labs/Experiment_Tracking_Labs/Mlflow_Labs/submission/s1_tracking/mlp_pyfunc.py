"""PyTorch tabular MLP packaged as an MLflow pyfunc "model from code".

Lab 2 wrapped a model in a PythonModel subclass so that serving returns probabilities. This is the
modern version of that idea:
  * models-from-code: MLflow stores THIS FILE and executes it at load time. No cloudpickle of Python
    objects, so loading never runs arbitrary pickled code.
  * everything the model needs is plain data: encoder.json (vocab/means), config.json (architecture)
    and weights.pt (a state_dict, loaded with weights_only=True).
  * this file must stay self-contained (no imports from our project); training imports the same
    classes from here, so training and serving run identical code.

Contract (same as the sklearn models, so registry gates and serving work unchanged):
    input : DataFrame with the 13 raw Adult features
    output: float64 array (n, 2) = [P(income <= 50K), P(income > 50K)]
"""
import json

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.pyfunc import PythonModel
from torch import nn


class TabularEncoder:
    """Turns raw rows into (numeric float32 matrix, categorical int64 index matrix).
    Numeric: median-impute then standardize. Categorical: vocabulary index, 0 = unknown/missing."""

    def __init__(self, numeric_cols, categorical_cols, medians, means, stds, vocab):
        self.numeric_cols, self.categorical_cols = list(numeric_cols), list(categorical_cols)
        self.medians, self.means, self.stds, self.vocab = medians, means, stds, vocab
        self._index = {c: {v: i + 1 for i, v in enumerate(vocab[c])} for c in self.categorical_cols}

    @classmethod
    def fit(cls, df: pd.DataFrame, numeric_cols, categorical_cols) -> "TabularEncoder":
        num = df[list(numeric_cols)].apply(pd.to_numeric, errors="coerce")
        medians = num.median()
        filled = num.fillna(medians)
        stds = filled.std(ddof=0).replace(0, 1.0)
        return cls(
            numeric_cols, categorical_cols,
            medians.to_dict(), filled.mean().to_dict(), stds.to_dict(),
            {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in categorical_cols},
        )

    def to_dict(self) -> dict:
        return {
            "numeric_cols": self.numeric_cols, "categorical_cols": self.categorical_cols,
            "medians": self.medians, "means": self.means, "stds": self.stds, "vocab": self.vocab,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TabularEncoder":
        return cls(d["numeric_cols"], d["categorical_cols"], d["medians"], d["means"], d["stds"], d["vocab"])

    def _col(self, df, c):
        # A column absent from the request is treated as missing, not as an error.
        return df[c] if c in df.columns else pd.Series(np.nan, index=df.index)

    def transform(self, df: pd.DataFrame):
        num = np.column_stack(
            [
                ((pd.to_numeric(self._col(df, c), errors="coerce").fillna(self.medians[c]) - self.means[c]) / self.stds[c]).to_numpy()
                for c in self.numeric_cols
            ]
        ).astype(np.float32)
        cat = np.column_stack(
            [
                self._col(df, c).map(lambda v: str(v) if pd.notna(v) else None).map(self._index[c]).fillna(0).astype(np.int64).to_numpy()
                for c in self.categorical_cols
            ]
        )
        return torch.from_numpy(num), torch.from_numpy(cat)


class MLPNet(nn.Module):
    def __init__(self, n_numeric: int, cardinalities: list, emb_dims: list, hidden: list, dropout: float):
        super().__init__()
        # +1 row per embedding for the "unknown / missing" index 0
        self.embs = nn.ModuleList([nn.Embedding(card + 1, d) for card, d in zip(cardinalities, emb_dims)])
        in_dim = n_numeric + sum(emb_dims)
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        parts = [x_num] + [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.mlp(torch.cat(parts, dim=1)).squeeze(1)  # one logit per row


def net_config(encoder: TabularEncoder, hidden, dropout, emb_cap) -> dict:
    cards = [len(encoder.vocab[c]) for c in encoder.categorical_cols]
    return {
        "n_numeric": len(encoder.numeric_cols),
        "cardinalities": cards,
        "emb_dims": [min(emb_cap, max(2, (c + 1) // 2)) for c in cards],
        "hidden": list(hidden),
        "dropout": float(dropout),
    }


class MLPModel(PythonModel):
    def load_context(self, context):
        with open(context.artifacts["encoder"]) as f:
            self.encoder = TabularEncoder.from_dict(json.load(f))
        with open(context.artifacts["config"]) as f:
            self.net = MLPNet(**json.load(f))
        # weights_only=True: only tensors are deserialized, never arbitrary Python objects
        self.net.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu", weights_only=True))
        self.net.eval()

    def predict(self, model_input, params=None):
        x_num, x_cat = self.encoder.transform(pd.DataFrame(model_input))
        with torch.no_grad():
            p = torch.sigmoid(self.net(x_num, x_cat)).numpy().astype("float64")
        return np.column_stack([1.0 - p, p])


mlflow.models.set_model(MLPModel())
