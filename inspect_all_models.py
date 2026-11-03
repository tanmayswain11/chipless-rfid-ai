
from pathlib import Path
import joblib

MODEL_ROOT = Path("saved_models")

print("\nMULTIPARAMETER MODEL INSPECTION")
print("=" * 65)

model_files = sorted(MODEL_ROOT.rglob("*.joblib"))

if not model_files:
    print("No trained model files found.")
    raise SystemExit(1)

for path in model_files:

    print("\n" + "-" * 65)
    print("FILE:", path)
    
    try:
        artifact = joblib.load(path)

        print("Artifact type:", type(artifact))

        if isinstance(artifact, dict):

            print("Dictionary keys:", list(artifact.keys()))

            for key, value in artifact.items():

                if key == "model":
                    print("Actual estimator:", type(value))
                    print(
                        "Expected feature count:",
                        getattr(value, "n_features_in_", "Not available")
                    )

                elif isinstance(value, (list, tuple)):
                    print(f"{key}: {value}")

                else:
                    print(f"{key}: {value}")

        else:
            print("Estimator:", type(artifact))
            print(
                "Expected feature count:",
                getattr(artifact, "n_features_in_", "Not available")
            )

    except Exception as error:
        print("ERROR loading model:", error)

print("\n" + "=" * 65)
print("MODEL INSPECTION COMPLETED")
