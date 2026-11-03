
from pathlib import Path
import joblib
import numpy as np

from ml.data_loader import load_curve_samples
from ml.feature_extraction import build_feature_matrix

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

model_dir = Path("saved_models/relative_humidity")
csv_path = Path("data/raw/realhumid cf.csv")

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

samples = load_curve_samples(
    csv_path,
    "relative_humidity"
)

# Provisional labels: NOT experimentally verified
rh_labels = np.array([30.0, 50.0, 70.0])

# --------------------------------------------------
# 3. FEATURE EXTRACTION
# --------------------------------------------------

X = build_feature_matrix(samples)

print("\nRH MODEL VERIFICATION")
print("=" * 60)

print("Feature matrix shape:", X.shape)
print("Feature values finite:", np.isfinite(X).all())

assert X.shape == (3, 31), "Unexpected feature matrix shape"
assert np.isfinite(X).all(), "Features contain NaN or infinity"

# --------------------------------------------------
# 4. MODEL FILES
# --------------------------------------------------

model_files = [
    "relative_humidity_extra_trees.joblib",
    "relative_humidity_random_forest.joblib",
    "relative_humidity_gradient_boosting.joblib",
    "relative_humidity_decision_tree.joblib",
    "relative_humidity_svr_rbf.joblib",
    "relative_humidity_ridge_regression.joblib",
]

# --------------------------------------------------
# 5. LOAD AND TEST EACH MODEL
# --------------------------------------------------

for filename in model_files:

    model_path = model_dir / filename

    assert model_path.exists(), f"Missing model: {filename}"

    # Load complete model artifact
    artifact = joblib.load(model_path)

    assert isinstance(artifact, dict), "Invalid model artifact"

    # Extract actual estimator
    model = artifact["model"]

    print("\n" + "-" * 60)
    print("Model:", artifact["model_name"])
    print("Training samples:", artifact["training_samples"])
    print("Feature count:", artifact["feature_count"])
    print("Experimental:", artifact["experimental"])
    print("Label status:", artifact["label_status"])

    # Predict
    predictions = model.predict(X)

    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()

    print("\nPredictions:")
    for i, prediction in enumerate(predictions):
        print(
            f"Curve {i + 1}: "
            f"Assigned RH = {rh_labels[i]}% | "
            f"Predicted RH = {prediction:.2f}%"
        )

print("\n" + "=" * 60)
print("ALL RH MODEL VERIFICATION CHECKS PASSED!")
print("=" * 60)
