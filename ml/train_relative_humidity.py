
import argparse
import re
from pathlib import Path

import joblib

from ml.preprocessing import read_csv
from ml.feature_extraction import wide_csv_to_curve_samples
from ml.evaluate_models import benchmark_models
from ml.model_zoo import get_models


def parse_relative_humidity(column_name):
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:%|\s*percent)\s*RH\b",
        column_name,
        flags=re.IGNORECASE
    )

    return float(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/raw/relative_humidity.csv"
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
        parse_relative_humidity
    )

    if len(y) < 2:
        raise ValueError(
            "Need at least two RF curves with reference %RH labels."
        )

    metrics = benchmark_models(X, y)

    Path("data/processed").mkdir(
        parents=True, exist_ok=True
    )

    metrics.to_csv(
        "data/processed/relative_humidity_metrics.csv",
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
        "target": "relative_humidity_percent",
        "feature_count": X.shape[1],
        "training_curve_names": curve_names,
        "warning": (
            "Exploratory only. Validate against independent "
            "calibrated humidity measurements."
        ),
    }, "ml/saved_models/relative_humidity_model.joblib")

    print("Saved model:", best_name)


if __name__ == "__main__":
    main()
