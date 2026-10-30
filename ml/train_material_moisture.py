
import argparse
import re
from pathlib import Path

import joblib

from ml.preprocessing import read_csv
from ml.feature_extraction import wide_csv_to_curve_samples
from ml.evaluate_models import benchmark_models
from ml.model_zoo import get_models


def parse_water_ml(column_name):
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*ml\b",
        column_name,
        flags=re.IGNORECASE
    )

    return float(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/raw/moisture1.csv"
    )

    parser.add_argument(
        "--frequency-col",
        default="Frequency"
    )

    args = parser.parse_args()

    df = read_csv(args.data)

    X, y, curve_names = wide_csv_to_curve_samples(
        df,
        args.frequency_col,
        parse_water_ml
    )

    print("Independent water-volume curves:", len(y))
    print("Curve labels:", list(zip(curve_names, y)))

    metrics = benchmark_models(X, y)

    Path("data/processed").mkdir(
        parents=True, exist_ok=True
    )

    metrics.to_csv(
        "data/processed/material_moisture_model_metrics.csv",
        index=False
    )

    print(metrics.to_string(index=False))

    valid = metrics.dropna(subset=["MAE"])

    if valid.empty:
        raise RuntimeError("All model evaluations failed.")

    best_name = valid.iloc[0]["model"]
    best_model = get_models()[best_name]
    best_model.fit(X, y)

    Path("ml/saved_models").mkdir(
        parents=True, exist_ok=True
    )

    joblib.dump({
        "model": best_model,
        "target": "estimated_added_water_ml",
        "feature_count": X.shape[1],
        "training_curve_names": curve_names,
        "warning": (
            "Predicts added water volume for this experimental "
            "setup, not relative humidity or general moisture."
        ),
    }, "ml/saved_models/material_moisture_model.joblib")

    print("Saved model:", best_name)


if __name__ == "__main__":
    main()
