
from pathlib import Path
import pandas as pd

results_dir = Path("ml/saved_models")

for filename in [
    "temperature_comparison.csv",
    "water_volume_comparison.csv",
]:
    path = results_dir / filename

    print(f"\n{'=' * 60}")
    print(filename)

    if not path.exists():
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path)
    print(df.to_string(index=False))

    print("\nColumns:", df.columns.tolist())
