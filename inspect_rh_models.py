
import joblib
from pathlib import Path

model_dir = Path("saved_models/relative_humidity")

for path in sorted(model_dir.glob("*.joblib")):
    print("\n" + "=" * 55)
    print("FILE:", path.name)

    artifact = joblib.load(path)

    print("Object type:", type(artifact))

    if isinstance(artifact, dict):
        print("Dictionary keys:", list(artifact.keys()))

        for key, value in artifact.items():
            print(f"  {key}: {type(value)}")
    else:
        print("Object:", artifact)
