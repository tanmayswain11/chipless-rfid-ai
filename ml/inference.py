
from pathlib import Path

import joblib
import numpy as np

from ml.feature_extraction import build_feature_matrix


class MultiparameterPredictor:
    """
    Unified inference engine for chipless RFID multiparameter sensing.

    Inputs:
        frequency_hz: 1D frequency array
        magnitude_db: 1D magnitude response array

    Outputs:
        Temperature estimate (°C)
        Water-volume estimate (mL)
        Experimental relative-humidity estimate (%)

    Important:
        RH labels are assumed, not experimentally verified.
        Water-volume model has poor exploratory performance.
    """

    def __init__(self, project_root=None):

        # --------------------------------------------------
        # PROJECT ROOT
        # --------------------------------------------------

        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent

        self.project_root = Path(project_root)

        # --------------------------------------------------
        # MODEL PATHS
        # --------------------------------------------------

        self.model_paths = {
            "temperature": (
                self.project_root
                / "saved_models"
                / "temperature"
                / "temperature_extra_trees.joblib"
            ),

            "water_volume": (
                self.project_root
                / "saved_models"
                / "water_volume"
                / "water_volume_svr_rbf.joblib"
            ),

            "relative_humidity": (
                self.project_root
                / "saved_models"
                / "relative_humidity"
                / "relative_humidity_extra_trees.joblib"
            ),
        }

        self.artifacts = {}
        self.models = {}

        self._load_models()

    # ==================================================
    # LOAD TRAINED MODELS
    # ==================================================

    def _load_models(self):

        for target, path in self.model_paths.items():

            if not path.exists():
                raise FileNotFoundError(
                    f"Model not found for {target}: {path}"
                )

            artifact = joblib.load(path)

            if not isinstance(artifact, dict):
                raise TypeError(
                    f"Invalid model artifact for {target}. "
                    "Expected a dictionary."
                )

            model = artifact.get("model")

            if model is None:
                raise ValueError(
                    f"Estimator missing in artifact: {path}"
                )

            expected_features = artifact.get("feature_count")

            if expected_features != 31:
                raise ValueError(
                    f"Unexpected feature count for {target}: "
                    f"{expected_features}. Expected 31."
                )

            if "model_name" not in artifact:
                raise ValueError(
                    f"Model name missing in artifact: {path}"
                )

            self.artifacts[target] = artifact
            self.models[target] = model

        print("All multiparameter models loaded successfully.")

    # ==================================================
    # FEATURE EXTRACTION
    # ==================================================

    def _extract_features(self, frequency_hz, magnitude_db):

        frequency_hz = np.asarray(
            frequency_hz,
            dtype=float
        )

        magnitude_db = np.asarray(
            magnitude_db,
            dtype=float
        )

        # --------------------------------------------------
        # INPUT VALIDATION
        # --------------------------------------------------

        if frequency_hz.ndim != 1 or magnitude_db.ndim != 1:
            raise ValueError(
                "Input arrays must be one-dimensional."
            )

        if len(frequency_hz) != len(magnitude_db):
            raise ValueError(
                "Frequency and magnitude arrays must have equal lengths."
            )

        if len(frequency_hz) < 2:
            raise ValueError(
                "At least two frequency points are required."
            )

        if not np.isfinite(frequency_hz).all():
            raise ValueError(
                "Frequency contains invalid values."
            )

        if not np.isfinite(magnitude_db).all():
            raise ValueError(
                "Magnitude contains invalid values."
            )

        if np.any(np.diff(frequency_hz) <= 0):
            raise ValueError(
                "Frequency values must be strictly increasing."
            )

        # --------------------------------------------------
        # BUILD SENSOR SAMPLE
        # --------------------------------------------------

        sample = {
            "curve": "live_measurement",
            "frequency_hz": frequency_hz,
            "magnitude_db": magnitude_db,
            "target": None,
        }

        # --------------------------------------------------
        # EXTRACT FEATURES
        # --------------------------------------------------

        X = build_feature_matrix([sample])

        X = np.asarray(X, dtype=float)

        # --------------------------------------------------
        # HANDLE BOTH POSSIBLE OUTPUT SHAPES
        # --------------------------------------------------

        # Case 1: Feature extractor returns (31,)
        if X.shape == (31,):

            X = X.reshape(1, 31)

        # Case 2: Feature extractor already returns (1, 31)
        elif X.shape == (1, 31):

            pass

        else:

            raise ValueError(
                f"Expected 31 extracted features, "
                f"received shape {X.shape}."
            )

        # --------------------------------------------------
        # FEATURE VALIDATION
        # --------------------------------------------------

        if not np.isfinite(X).all():
            raise ValueError(
                "Extracted features contain invalid values."
            )

        if X.shape != (1, 31):
            raise ValueError(
                f"Expected model input shape (1, 31), "
                f"received {X.shape}."
            )

        return X

    # ==================================================
    # MULTIPARAMETER PREDICTION
    # ==================================================

    def predict(self, frequency_hz, magnitude_db):

        # Extract features
        X = self._extract_features(
            frequency_hz,
            magnitude_db
        )

        results = {}

        # --------------------------------------------------
        # PREDICT EACH PARAMETER
        # --------------------------------------------------

        for target, model in self.models.items():

            prediction = float(
                model.predict(X)[0]
            )

            artifact = self.artifacts[target]

            results[target] = {
                "prediction": prediction,

                "unit": {
                    "temperature": "°C",
                    "water_volume": "mL",
                    "relative_humidity": "%",
                }[target],

                "model_name": artifact["model_name"],

                "experimental": artifact.get(
                    "experimental",
                    False
                ),

                "label_status": artifact.get(
                    "label_status"
                ),

                "training_samples": artifact.get(
                    "training_samples"
                ),
            }

        # --------------------------------------------------
        # FINAL RESPONSE
        # --------------------------------------------------

        return {
            "status": "success",

            "frequency_points": len(frequency_hz),

            "feature_count": X.shape[1],

            "input_shape": X.shape,

            "predictions": results,

            "warnings": [
                (
                    "Water-volume prediction has poor exploratory "
                    "LOOCV performance (negative R2)."
                ),

                (
                    "Relative-humidity labels are assumed, "
                    "not experimentally verified."
                ),

                (
                    "Training-set compatibility does not establish "
                    "real-world prediction accuracy."
                ),

                (
                    "All three models are applied to the same input "
                    "RF curve. Validate each target using appropriate "
                    "experimental measurements."
                ),
            ],
        }
