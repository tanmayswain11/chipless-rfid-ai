
from ml.data_loader import load_curve_samples
from pathlib import Path
import numpy as np

csv_path = Path("data/raw/realhumid cf.csv")

samples = load_curve_samples(
    csv_path,
    "relative_humidity"
)

print("\nRH DATASET VALIDATION")
print("-" * 45)

print("Total curves:", len(samples))

for sample in samples:
    print("\nCurve:", sample["curve"])
    print("Frequency points:", len(sample["frequency_hz"]))
    print("First frequency:", sample["frequency_hz"][0])
    print("First magnitude:", sample["magnitude_db"][0])
    print("Last frequency:", sample["frequency_hz"][-1])
    print("Target:", sample["target"])

# 1. Check total curves
assert len(samples) == 3, "Expected exactly 3 RH curves"

# 2. Check frequency and magnitude lengths
assert all(
    len(sample["frequency_hz"]) == 801
    and len(sample["magnitude_db"]) == 801
    for sample in samples
), "Each RH curve must contain 801 frequency and magnitude values"

# 3. Check frequency range
assert all(
    np.isclose(sample["frequency_hz"][0], 800_000_000)
    and np.isclose(sample["frequency_hz"][-1], 5_000_000_000)
    for sample in samples
), "Incorrect frequency range"

# 4. Check missing targets
assert all(
    sample["target"] is None
    for sample in samples
), "RH targets should remain None before provisional labeling"

# 5. Check finite values
assert all(
    np.isfinite(sample["frequency_hz"]).all()
    and np.isfinite(sample["magnitude_db"]).all()
    for sample in samples
), "Dataset contains NaN or infinite values"

print("\nAll RH loader tests passed!")
