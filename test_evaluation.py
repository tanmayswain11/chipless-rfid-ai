
from ml.data_loader import (
    load_curve_samples,
    default_dataset_paths
)

from ml.evaluate_models import evaluate_models

paths = default_dataset_paths()

samples = load_curve_samples(
    paths["temperature"],
    "temperature"
)

results, predictions = evaluate_models(
    samples,
    copies_per_curve=2,
    random_state=42
)

print("\nFinal Evaluation Verification")
print("-----------------------------")

print("Results shape:", results.shape)
print("Predictions shape:", predictions.shape)

print("\nExperiments:")
print(results["experiment"].unique())

print("\nModels:")
print(results["model"].unique())

# Save temporary test outputs
results.to_csv(
    "temperature_evaluation_test.csv",
    index=False
)

predictions.to_csv(
    "temperature_predictions_test.csv",
    index=False
)

print("\nEvaluation files saved successfully.")
