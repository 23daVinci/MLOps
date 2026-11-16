"""Preprocessing shared by every sklearn-based model."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = True,
    dense: bool = False,
) -> ColumnTransformer:
    """Numeric: impute median (+ scale). Categorical: impute mode + one-hot.
    Lives inside the model Pipeline so serving applies the exact same steps.
    Columns are passed explicitly (not via a dtype selector) so the feature contract
    is visible, loggable, and the pipeline serializes safely with skops."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        # linear models accept sparse; sklearn's histogram GBM needs dense
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
