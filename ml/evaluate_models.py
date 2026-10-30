
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_models(X, y, models, groups=None):
    """
    Compare regression models without splitting individual frequency points.

    Current data:
        groups=None -> Leave-One-Curve-Out CV.

    Future repeated-sweep data:
        groups=array of experimental run IDs
        -> GroupKFold CV.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    if len(y) < 3:
        raise ValueError("At least 3 independent samples are required.")

    if groups is None:
        cv = LeaveOneOut()
        evaluation_name = "LOOCV exploratory"
    else:
        groups = np.asarray(groups)

        if len(groups) != len(y):
            raise ValueError("groups must have one ID per sample.")

        unique_groups = np.unique(groups)

        if len(unique_groups) < 2:
            raise ValueError(
                "Group CV requires at least two distinct experimental groups."
            )

        n_splits = min(5, len(unique_groups))
        cv = GroupKFold(n_splits=n_splits)
        evaluation_name = f"GroupKFold ({n_splits} folds)"

    rows = []

    for name, model in models.items():
        predictions = cross_val_predict(
            clone(model),
            X,
            y,
            cv=cv,
            groups=groups,
            n_jobs=1,
        )

        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))

        # R² is undefined for a single test observation.
        # LOOCV has one test observation per fold, so calculate
        # R² over all out-of-fold predictions instead.
        r2 = (
            r2_score(y, predictions)
            if len(y) >= 2
            else float("nan")
        )

        rows.append({
            "model": name,
            "samples": len(y),
            "MAE": mae,
            "RMSE": rmse,
            "R2_OOF": r2,
            "evaluation": evaluation_name,
        })

    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
