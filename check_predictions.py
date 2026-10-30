
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train_all_models import load_samples, ROOT

MODEL_DIR = ROOT / "ml" / "saved_models"


def inspect_predictions(csv_path, target_type, prefix, model_name):
    # Load the sample dictionaries
    samples = load_samples(
        csv_path=csv_path,
        target_type=target_type,
    )

    if not samples:
        print(f"No samples found for {target_type}.")
        return

    # Extract features, actual targets, and curve names
    X = np.vstack([sample["features"] for sample in samples])
    y = np.array([sample["target"] for sample in samples], dtype=float)
    labels = [sample["curve"] for sample in samples]

    # Load the trained model
    model_path = MODEL_DIR / f"{prefix}_{model_name}.joblib"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    model = joblib.load(model_path)

    # Generate predictions
    predictions = model.predict(X)

    result = pd.DataFrame({
        "curve": labels,
        "actual": y,
        "predicted": predictions,
        "absolute_error": np.abs(y - predictions),
    })

    print(f"\n{'=' * 90}")
    print(f"Target: {target_type}")
    print(f"Model: {model_name}")
    print(f"Samples: {len(samples)}")
    print(f"Feature matrix shape: {X.shape}")
    print("WARNING: These are in-sample predictions, not cross-validation results.\n")

    print(result.round(3).to_string(index=False))

    print(f"\nMean absolute error: {np.mean(np.abs(y - predictions)):.3f}")
    print(f"Maximum absolute error: {np.max(np.abs(y - predictions)):.3f}")


if __name__ == "__main__":
    inspect_predictions(
        ROOT / "data" / "raw" / "plot1_rcs.csv",
        "temperature",
        "temperature",
        "extra_trees",
    )

    inspect_predictions(
        ROOT / "data" / "raw" / "moisture1.csv",
        "water_volume",
        "water_volume",
        "extra_trees",
    )
