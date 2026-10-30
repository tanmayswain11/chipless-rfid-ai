
from pathlib import Path
import re
import json

import joblib
import numpy as np
import pandas as pd

from ml.feature_extraction import (
    build_features,
    FREQ_MIN_HZ,
    FREQ_MAX_HZ,
    GRID_POINTS,
)
from ml.model_zoo import get_models
from ml.evaluate_models import evaluate_models


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "ml" / "saved_models"


# --------------------------------------------------
# 2. Find the frequency column
# --------------------------------------------------

def find_frequency_column(df):
    for col in df.columns:
        if "freq" in str(col).lower():
            return col

    raise ValueError(
        f"Frequency column not found. Available columns: {list(df.columns)}"
    )


# --------------------------------------------------
# 3. Convert frequency to Hz
# --------------------------------------------------

def convert_frequency_to_hz(series, column_name):
    values = pd.to_numeric(series, errors="coerce")
    name = str(column_name).lower()

    if "ghz" in name:
        return values * 1e9

    if "mhz" in name:
        return values * 1e6

    if "khz" in name:
        return values * 1e3

    median = values.dropna().median()

    if pd.isna(median):
        raise ValueError("Frequency column contains no numeric values.")

    # Fallback for headers without explicit units.
    if median < 10:
        return values * 1e9

    if median < 10000:
        return values * 1e6

    return values


# --------------------------------------------------
# 4. Extract temperature or water-volume labels
# --------------------------------------------------

def parse_target(column, target_type):
    label = str(column).lower()

    if target_type == "temperature":
        patterns = [
            r"(\d+(?:\.\d+)?)\s*degree",
            r"tempreture\s*(\d+(?:\.\d+)?)",
            r"temperature\s*(\d+(?:\.\d+)?)",
        ]

    elif target_type == "water_volume":
        patterns = [
            r"(\d+(?:\.\d+)?)\s*ml\b",
        ]

    else:
        raise ValueError(
            f"Unknown target type: {target_type}"
        )

    for pattern in patterns:
        match = re.search(pattern, label)

        if match:
            return float(match.group(1))

    return None


# --------------------------------------------------
# 5. Load complete curves as independent samples
# --------------------------------------------------

def load_samples(csv_path, target_type):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    frequency_col = find_frequency_column(df)

    frequency_hz = convert_frequency_to_hz(
        df[frequency_col],
        frequency_col,
    )

    samples = []

    for col in df.columns:
        if col == frequency_col:
            continue

        target = parse_target(col, target_type)

        if target is None:
            continue

        magnitude = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        valid = frequency_hz.notna() & magnitude.notna()

        if valid.sum() < 20:
            print(
                f"WARNING: Skipping {col}: "
                f"only {valid.sum()} valid measurements."
            )
            continue

        features = build_features(
            frequency_hz[valid].to_numpy(),
            magnitude[valid].to_numpy(),
        )

        samples.append({
            "curve": str(col),
            "target": target,
            "features": features,
        })

    if not samples:
        raise ValueError(
            f"No target curves found in {csv_path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    return samples


# --------------------------------------------------
# 6. Train, cross-validate, and save every algorithm
# --------------------------------------------------

def train_dataset(csv_path, target_type, output_prefix):
    samples = load_samples(csv_path, target_type)

    X = np.vstack([
        sample["features"]
        for sample in samples
    ])

    y = np.array([
        sample["target"]
        for sample in samples
    ], dtype=float)

    labels = [
        sample["curve"]
        for sample in samples
    ]

    print("\n" + "=" * 70)
    print("Target:", target_type)
    print("CSV:", csv_path.name)
    print("Independent curves:", len(samples))
    print("Feature matrix shape:", X.shape)
    print("Feature count per curve:", X.shape[1])
    print("Curve labels:", list(zip(labels, y)))

    if len(samples) < 3:
        raise ValueError(
            "At least three independent curves are required "
            "for exploratory cross-validation."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # A. Evaluate all seven models using LOOCV
    # --------------------------------------------------

    models = get_models()

    results = evaluate_models(
        X=X,
        y=y,
        models=models,
        groups=None,
    )

    results_path = (
        OUTPUT_DIR / f"{output_prefix}_comparison.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    print("\nCross-validation results:")
    print(results.to_string(index=False))
    print("\nComparison saved:", results_path)

    # --------------------------------------------------
    # B. Save every model fitted on all available curves
    # --------------------------------------------------

    for name, model in models.items():
        fitted_model = model.fit(X, y)

        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        model_path = OUTPUT_DIR / (
            f"{output_prefix}_{safe_name}.joblib"
        )

        joblib.dump(
            fitted_model,
            model_path,
        )

        metadata = {
            "target_type": target_type,
            "model_name": name,
            "curve_samples": len(samples),
            "feature_count": int(X.shape[1]),
            "grid_points": GRID_POINTS,
            "frequency_min_hz": FREQ_MIN_HZ,
            "frequency_max_hz": FREQ_MAX_HZ,
            "input_curves": labels,
            "training_targets": y.tolist(),
            "feature_definition": (
                "Interpolated RF magnitudes plus curve summary features"
            ),
            "validation": "Leave-one-curve-out cross-validation",
            "evaluation_status": "Exploratory only",
        }

        metadata_path = model_path.with_suffix(".json")

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print("Saved model:", model_path.name)
        print("Saved metadata:", metadata_path.name)

    print("\nFinished:", target_type)


# --------------------------------------------------
# 7. Run temperature and water-volume experiments
# --------------------------------------------------

if __name__ == "__main__":

    train_dataset(
        ROOT / "data" / "raw" / "plot1_rcs.csv",
        "temperature",
        "temperature",
    )

    train_dataset(
        ROOT / "data" / "raw" / "moisture1.csv",
        "water_volume",
        "water_volume",
    )
