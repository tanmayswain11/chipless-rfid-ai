
"""
Preprocessing utilities for chipless RFID multiparameter sensing.

Each complete frequency-response curve is one independent sample.
Only curves with known target labels are used for supervised learning.
"""

import numpy as np

from ml.feature_extraction import build_features


def validate_xy(X, y, labels=None):
    """
    Validate feature matrix X and target vector y.

    Parameters
    ----------
    X : array-like
        Feature matrix (n_samples, n_features).
    y : array-like
        Target values (n_samples,).
    labels : list, optional
        Original curve names.

    Returns
    -------
    X, y, labels
        Validated NumPy arrays and labels.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")

    if len(X) != len(y):
        raise ValueError(
            f"Sample mismatch: X has {len(X)} samples, "
            f"but y has {len(y)} targets."
        )

    if len(X) == 0:
        raise ValueError("No labelled samples available.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")

    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or infinite values.")

    if labels is None:
        labels = [f"curve_{i + 1}" for i in range(len(y))]

    if len(labels) != len(y):
        raise ValueError("Number of labels does not match sample count.")

    return X, y, list(labels)


def filter_labelled_samples(samples):
    """
    Keep only samples having valid numerical target labels.

    Unlabelled curves remain available for visualization,
    but are excluded from supervised model training.
    """

    labelled = []
    unlabelled = []

    for sample in samples:
        target = sample.get("target")

        if target is None:
            unlabelled.append(sample)
            continue

        try:
            target = float(target)
        except (TypeError, ValueError):
            unlabelled.append(sample)
            continue

        if not np.isfinite(target):
            unlabelled.append(sample)
            continue

        labelled.append(sample)

    return labelled, unlabelled


def samples_to_xy(samples):
    """
    Convert labelled RF curves into machine-learning arrays.

    Returns
    -------
    X : ndarray
        Feature matrix.
    y : ndarray
        Target values.
    labels : list
        Original curve column names.
    labelled_samples : list
        Samples used to create X and y.
    """

    labelled_samples, unlabelled_samples = filter_labelled_samples(
        samples
    )

    if len(labelled_samples) < 2:
        raise ValueError(
            "At least two labelled curves are required "
            "to construct a supervised dataset."
        )

    X = []
    y = []
    labels = []

    for sample in labelled_samples:

        features = build_features(
            sample["frequency_hz"],
            sample["magnitude_db"]
        )

        X.append(features)
        y.append(float(sample["target"]))

        labels.append(
            sample.get("curve", "unknown_curve")
        )

    X, y, labels = validate_xy(X, y, labels)

    print("\nPreprocessing report")
    print("--------------------")
    print("Total input curves:", len(samples))
    print("Labelled curves:", len(labelled_samples))
    print("Unlabelled curves:", len(unlabelled_samples))
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("All feature values finite:", np.isfinite(X).all())

    return X, y, labels, labelled_samples
