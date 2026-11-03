
from pathlib import Path
from ml.data_loader import load_curve_samples
from ml.inference import MultiparameterPredictor

# Load one temperature response curve for an initial
# software compatibility test.
samples = load_curve_samples(
    Path("data/raw/plot1_rcs.csv"),
    "temperature"
)

sample = samples[0]

predictor = MultiparameterPredictor()

result = predictor.predict(
    frequency_hz=sample["frequency_hz"],
    magnitude_db=sample["magnitude_db"],
)

print("\nMULTIPARAMETER INFERENCE TEST")
print("=" * 55)

print("Status:", result["status"])
print("Frequency points:", result["frequency_points"])
print("Feature count:", result["feature_count"])

for target, details in result["predictions"].items():

    print(f"\n{target.upper()}")
    print("Prediction:", round(details["prediction"], 3))
    print("Unit:", details["unit"])
    print("Model:", details["model_name"])
    print("Experimental:", details["experimental"])

print("\nWARNINGS")
for warning in result["warnings"]:
    print("-", warning)
