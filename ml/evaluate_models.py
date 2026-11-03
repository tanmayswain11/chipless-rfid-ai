
"""
Model evaluation for chipless RFID multiparameter sensing.

Evaluation strategy:
1. Leave-One-Out Cross-Validation (LOOCV).
2. Compare original-only training against augmented training.
3. Generate synthetic curves only from training folds.
4. Evaluate exclusively on untouched original curves.

Results are exploratory because the number of independent
measured curves is small.
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.feature_extraction import build_features
from ml.preprocessing import filter_labelled_samples
from ml.model_zoo import get_models
from ml.augmentation import augment_curve


def _build_xy(samples):
    """Convert curve samples into feature and target arrays."""

    X = np.vstack([
        build_features(
            sample["frequency_hz"],
            sample["magnitude_db"]
        )
        for sample in samples
    ])

    y = np.asarray(
        [float(sample["target"]) for sample in samples],
        dtype=float
    )

    return X, y


def _build_augmented_training_data(
    training_samples,
    copies_per_curve=2,
    random_state=42
):
    """
    Generate synthetic copies of training curves only.

    Original training samples are retained.
    Validation samples are never passed to this function.
    """

    rng = np.random.default_rng(random_state)

    augmented_samples = list(training_samples)

    for sample in training_samples:

        for copy_number in range(copies_per_curve):

            seed = int(rng.integers(0, 2**31 - 1))

            result = augment_curve(
                sample["frequency_hz"],
                sample["magnitude_db"],
                random_state=seed
            )

            augmented_samples.append({
                **sample,
                "frequency_hz": result["frequency_hz"],
                "magnitude_db": result["augmented_magnitude_db"],
                "is_synthetic": True,
                "augmentation_copy": copy_number + 1
            })

    return _build_xy(augmented_samples)


def evaluate_models(
    samples,
    copies_per_curve=2,
    random_state=42
):
    """
    Compare regression models using LOOCV.

    Parameters
    ----------
    samples : list
        Raw curve samples from data_loader.
    copies_per_curve : int
        Synthetic copies generated per training curve.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    results_df : pandas.DataFrame
        Model comparison metrics.
    predictions_df : pandas.DataFrame
        Out-of-fold predictions.
    """

    labelled_samples, unlabelled_samples = filter_labelled_samples(
        samples
    )

    n_samples = len(labelled_samples)

    if n_samples < 4:
        raise ValueError(
            "At least 4 independent labelled curves are required "
            "for this exploratory LOOCV evaluation."
        )

    if copies_per_curve < 0:
        raise ValueError("copies_per_curve cannot be negative.")

    X, y = _build_xy(labelled_samples)

    models = get_models(n_samples=n_samples)

    loo = LeaveOneOut()

    prediction_records = []

    print("\nLOOCV Evaluation")
    print("----------------")
    print("Independent labelled curves:", n_samples)
    print("Unlabelled curves excluded:", len(unlabelled_samples))
    print("Original feature matrix:", X.shape)
    print("Models:", len(models))
    print("Synthetic copies per training curve:", copies_per_curve)

    for fold, (train_idx, test_idx) in enumerate(
        loo.split(X),
        start=1
    ):

        training_samples = [
            labelled_samples[i] for i in train_idx
        ]

        validation_sample = labelled_samples[test_idx[0]]

        X_train_original = X[train_idx]
        y_train_original = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        # Generate augmentation using training samples only.
        if copies_per_curve > 0:
            X_train_augmented, y_train_augmented = (
                _build_augmented_training_data(
                    training_samples,
                    copies_per_curve=copies_per_curve,
                    random_state=random_state + fold
                )
            )
        else:
            X_train_augmented = X_train_original
            y_train_augmented = y_train_original

        for model_name, estimator in models.items():

            # Experiment A: original training data only.
            baseline_model = clone(estimator)

            baseline_model.fit(
                X_train_original,
                y_train_original
            )

            baseline_prediction = float(
                baseline_model.predict(X_test)[0]
            )

            prediction_records.append({
                "fold": fold,
                "curve": validation_sample.get("curve", "unknown"),
                "model": model_name,
                "experiment": "Original Only",
                "actual": float(y_test[0]),
                "predicted": baseline_prediction,
                "absolute_error": abs(
                    float(y_test[0]) - baseline_prediction
                )
            })

            # Experiment B: original + augmented training data.
            if copies_per_curve > 0:

                augmented_model = clone(estimator)

                augmented_model.fit(
                    X_train_augmented,
                    y_train_augmented
                )

                augmented_prediction = float(
                    augmented_model.predict(X_test)[0]
                )

                prediction_records.append({
                    "fold": fold,
                    "curve": validation_sample.get("curve", "unknown"),
                    "model": model_name,
                    "experiment": "Augmented Training",
                    "actual": float(y_test[0]),
                    "predicted": augmented_prediction,
                    "absolute_error": abs(
                        float(y_test[0]) - augmented_prediction
                    )
                })

        print(f"Completed fold {fold}/{n_samples}")

    predictions_df = pd.DataFrame(prediction_records)

    results = []

    for (model_name, experiment), group in predictions_df.groupby(
        ["model", "experiment"]
    ):

        actual = group["actual"].to_numpy()
        predicted = group["predicted"].to_numpy()

        mae = mean_absolute_error(actual, predicted)

        rmse = np.sqrt(
            mean_squared_error(actual, predicted)
        )

        r2 = r2_score(actual, predicted)

        results.append({
            "model": model_name,
            "experiment": experiment,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "independent_samples": n_samples
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        ["experiment", "MAE"]
    ).reset_index(drop=True)

    print("\nModel Comparison Results")
    print("------------------------")
    print(results_df.to_string(index=False))

    return results_df, predictions_df
