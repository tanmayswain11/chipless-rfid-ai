
import argparse
import json
import re

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.data_loader import (
    load_curve_samples,
    default_dataset_paths,
)

from ml.preprocessing import samples_to_xy

from ml.model_zoo import (
    get_models,
    get_rh_models,
)

from ml.evaluate_models import evaluate_models


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "saved_models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET CONFIGURATION
# ============================================================

TARGET_NAMES = {
    "temperature": "Temperature",
    "water_volume": "Water Volume",
    "relative_humidity": "Relative Humidity",
}


# ============================================================
# PROVISIONAL RH LABELS
# ============================================================

# These values are assumptions for experimentation.
# They are NOT actual measured RH values.
#
# Assignment follows the order of the three response
# columns in realhumid cf.csv.

PROVISIONAL_RH_LABELS = [30.0, 50.0, 70.0]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(name):
    """
    Convert model names into safe filenames.
    """

    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)

    return name.strip("_")


def get_labelled_samples(samples):
    """
    Keep samples with valid numerical target labels.
    """

    labelled_samples = []

    for sample in samples:

        target = sample.get("target")

        if target is None:
            continue

        try:
            target = float(target)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(target):
            continue

        sample["target"] = target
        labelled_samples.append(sample)

    return labelled_samples


# ============================================================
# NORMAL MODEL TRAINING
# TEMPERATURE AND WATER VOLUME
# ============================================================

def train_target(
    target_type,
    csv_path,
    copies_per_curve=2,
    random_state=42
):
    """
    Train and evaluate normal supervised regression targets.

    Used for:
        1. Temperature
        2. Water volume

    Uses LOOCV through evaluate_models().
    """

    print("\n" + "=" * 65)
    print(f"TRAINING: {TARGET_NAMES[target_type].upper()}")
    print("=" * 65)

    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        return None

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    samples = load_curve_samples(
        csv_path,
        target_type
    )

    labelled_samples = get_labelled_samples(samples)

    print(f"Total curves: {len(samples)}")
    print(f"Labelled curves: {len(labelled_samples)}")

    # --------------------------------------------------------
    # MINIMUM SAMPLE CHECK
    # --------------------------------------------------------

    if len(labelled_samples) < 4:

        print(
            f"Skipping {target_type}: "
            "at least 4 labelled curves are required."
        )

        return None

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    X, y, labels, labelled_samples = samples_to_xy(
        labelled_samples
    )

    print("Feature matrix:", X.shape)
    print("Target shape:", y.shape)

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    results_df, predictions_df = evaluate_models(
        labelled_samples,
        copies_per_curve=copies_per_curve,
        random_state=random_state
    )

    if results_df.empty:
        print("No evaluation results generated.")
        return None

    # --------------------------------------------------------
    # SAVE COMPARISON RESULTS
    # --------------------------------------------------------

    results_path = (
        PROCESSED_DIR /
        f"{target_type}_comparison.csv"
    )

    predictions_path = (
        PROCESSED_DIR /
        f"{target_type}_predictions.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------------

    # This selection is exploratory because it uses
    # the same LOOCV results for model comparison.

    best_row = results_df.loc[
        results_df["MAE"].idxmin()
    ]

    best_model_name = best_row["model"]

    print("\nBest exploratory model:")
    print(best_model_name)

    print("MAE:", best_row["MAE"])
    print("RMSE:", best_row["RMSE"])
    print("R2:", best_row["R2"])

    # --------------------------------------------------------
    # FIT FINAL MODEL ON ORIGINAL CURVES
    # --------------------------------------------------------

    models = get_models(
        n_samples=len(y)
    )

    if best_model_name not in models:
        raise KeyError(
            f"Model '{best_model_name}' not found "
            "in model_zoo.py."
        )

    final_model = models[best_model_name]

    # Fit only on original measured curves.
    # Synthetic augmentation is not used here.

    final_model.fit(X, y)

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    target_model_dir = MODEL_DIR / target_type
    target_model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_filename = (
        f"{target_type}_"
        f"{safe_filename(best_model_name)}.joblib"
    )

    model_path = target_model_dir / model_filename

    model_package = {

        "model": final_model,

        "target_type": target_type,

        "target_name": TARGET_NAMES[target_type],

        "model_name": best_model_name,

        "feature_count": int(X.shape[1]),

        "training_samples": int(len(y)),

        "training_labels": y.tolist(),

        "curve_labels": labels,

        "selection_experiment": best_row["experiment"],

        "evaluation_metrics": {
            "MAE": float(best_row["MAE"]),
            "RMSE": float(best_row["RMSE"]),
            "R2": float(best_row["R2"]),
        },

        "evaluation_method": "LOOCV",

        "label_status": "dataset_labels",

        "experimental": False,
    }

    joblib.dump(
        model_package,
        model_path
    )

    print("\nModel saved:", model_path)

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary = {

        "target": target_type,

        "target_name": TARGET_NAMES[target_type],

        "best_model": best_model_name,

        "independent_samples": int(len(y)),

        "feature_count": int(X.shape[1]),

        "MAE": float(best_row["MAE"]),

        "RMSE": float(best_row["RMSE"]),

        "R2": float(best_row["R2"]),

        "model_path": str(model_path),

        "experimental": False,
    }

    summary_path = (
        PROCESSED_DIR /
        f"{target_type}_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    return summary


# ============================================================
# SEPARATE RELATIVE HUMIDITY TRAINING
# ============================================================

def train_relative_humidity(
    csv_path,
    random_state=42
):
    """
    Separate experimental RH training function.

    Uses three independent RF response curves.

    Provisional RH labels:
        Curve 1 -> 30%
        Curve 2 -> 50%
        Curve 3 -> 70%

    Does NOT perform LOOCV model selection.
    Does NOT claim validation accuracy.

    All six models are fitted using the three original curves.
    """

    print("\n" + "=" * 65)
    print("EXPERIMENTAL RELATIVE HUMIDITY TRAINING")
    print("=" * 65)

    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"RH dataset not found: {csv_path}")
        return None

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    samples = load_curve_samples(
        csv_path,
        "relative_humidity"
    )

    print("Total RH response curves:", len(samples))

    # --------------------------------------------------------
    # ASSIGN PROVISIONAL LABELS
    # --------------------------------------------------------

    if len(samples) != 3:

        raise ValueError(
            "Expected exactly 3 RH response curves in "
            f"{csv_path.name}, but found {len(samples)}. "
            "Check the CSV columns and curve order."
        )

    for sample, rh_value in zip(
        samples,
        PROVISIONAL_RH_LABELS
    ):

        sample["target"] = float(rh_value)

        sample["target_source"] = (
            "provisional_assumption"
        )

        sample["is_verified_label"] = False

    print("\nProvisional RH labels:")

    for sample in samples:

        print(
            f"Curve: {sample['curve']} | "
            f"Assigned RH: {sample['target']}% | "
            f"Verified: {sample['is_verified_label']}"
        )

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    X, y, labels, labelled_samples = samples_to_xy(
        samples
    )

    if len(y) != 3:

        raise ValueError(
            "RH feature extraction did not return "
            "exactly three labelled samples."
        )

    print("\nRH feature matrix:", X.shape)
    print("RH target shape:", y.shape)

    # --------------------------------------------------------
    # GET SEPARATE RH MODELS
    # --------------------------------------------------------

    models = get_rh_models(
        n_samples=len(y)
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    rh_model_dir = MODEL_DIR / "relative_humidity"

    rh_model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # FIT AND SAVE ALL RH MODELS
    # --------------------------------------------------------

    training_predictions = []

    for model_name, model in models.items():

        print("\nTraining:", model_name)

        model.fit(X, y)

        predictions = model.predict(X)

        model_filename = (
            "relative_humidity_"
            f"{safe_filename(model_name)}.joblib"
        )

        model_path = (
            rh_model_dir / model_filename
        )

        model_package = {

            "model": model,

            "target_type": "relative_humidity",

            "target_name": "Relative Humidity",

            "model_name": model_name,

            "feature_count": int(X.shape[1]),

            "training_samples": int(len(y)),

            "training_labels": y.tolist(),

            "curve_labels": labels,

            "provisional_labels": PROVISIONAL_RH_LABELS,

            "label_status": "assumed_not_measured",

            "experimental": True,

            "evaluation_method": (
                "Training fit only - no independent validation"
            ),

            "warning": (
                "RH labels are provisional assumptions. "
                "Predictions are not validated real-world "
                "humidity measurements."
            ),
        }

        joblib.dump(
            model_package,
            model_path
        )

        print("Saved:", model_path)

        # ----------------------------------------------------
        # RECORD TRAINING FIT PREDICTIONS
        # ----------------------------------------------------

        for index in range(len(y)):

            training_predictions.append({

                "model": model_name,

                "curve": labels[index],

                "assumed_actual_rh": float(y[index]),

                "training_prediction_rh": float(
                    predictions[index]
                ),

                "prediction_error": float(
                    predictions[index] - y[index]
                ),

                "label_status": "provisional_assumption",

                "independent_validation": False,
            })

    # --------------------------------------------------------
    # SAVE RH TRAINING PREDICTIONS
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(
        training_predictions
    )

    predictions_path = (
        PROCESSED_DIR /
        "relative_humidity_training_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE RH TRAINING SUMMARY
    # --------------------------------------------------------

    summary = {

        "target": "relative_humidity",

        "target_name": "Relative Humidity",

        "training_method": "Experimental model fitting",

        "independent_samples": 3,

        "feature_count": int(X.shape[1]),

        "models_trained": list(models.keys()),

        "provisional_rh_labels": PROVISIONAL_RH_LABELS,

        "label_status": "assumed_not_measured",

        "validation_performed": False,

        "training_predictions_file": str(
            predictions_path
        ),

        "model_directory": str(
            rh_model_dir
        ),

        "experimental": True,

        "warning": (
            "The three RH labels are assumptions, not "
            "measured humidity values. Training fit "
            "predictions are not independent validation."
        ),
    }

    summary_path = (
        PROCESSED_DIR /
        "relative_humidity_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("RH EXPERIMENTAL TRAINING COMPLETED")
    print("=" * 65)

    print("Independent curves:", len(y))
    print("Models trained:", len(models))
    print("RH labels:", PROVISIONAL_RH_LABELS)

    print("Predictions saved:", predictions_path)
    print("Summary saved:", summary_path)

    return summary


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train chipless RFID multiparameter "
            "regression models."
        )
    )

    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=[
            "all",
            "temperature",
            "water_volume",
            "relative_humidity",
        ],
        help="Select the target dataset."
    )

    parser.add_argument(
        "--copies",
        type=int,
        default=2,
        help="Synthetic training copies per curve."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()

    dataset_paths = default_dataset_paths()

    summaries = []

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    if args.target in ["all", "temperature"]:

        summary = train_target(
            target_type="temperature",
            csv_path=dataset_paths["temperature"],
            copies_per_curve=args.copies,
            random_state=args.seed
        )

        if summary is not None:
            summaries.append(summary)

    # --------------------------------------------------------
    # WATER VOLUME
    # --------------------------------------------------------

    if args.target in ["all", "water_volume"]:

        summary = train_target(
            target_type="water_volume",
            csv_path=dataset_paths["water_volume"],
            copies_per_curve=args.copies,
            random_state=args.seed
        )

        if summary is not None:
            summaries.append(summary)

    # --------------------------------------------------------
    # RELATIVE HUMIDITY
    # --------------------------------------------------------

    if args.target in ["all", "relative_humidity"]:

        summary = train_relative_humidity(
            csv_path=dataset_paths["relative_humidity"],
            random_state=args.seed
        )

        if summary is not None:
            summaries.append(summary)

    # --------------------------------------------------------
    # SAVE OVERALL TRAINING SUMMARY
    # --------------------------------------------------------

    if summaries:

        summary_df = pd.DataFrame(summaries)

        summary_path = (
            PROCESSED_DIR /
            "training_summary.csv"
        )

        summary_df.to_csv(
            summary_path,
            index=False
        )

        print("\nOverall training summary saved:")
        print(summary_path)

    print("\nTraining pipeline finished.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
