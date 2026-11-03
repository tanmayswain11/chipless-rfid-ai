
from pathlib import Path
from ml.data_loader import load_curve_samples
from ml.feature_extraction import build_feature_matrix

samples = load_curve_samples(
    Path("data/raw/realhumid cf.csv"),
    "relative_humidity"
)

for sample, label in zip(samples, [30.0, 50.0, 70.0]):
    sample["target"] = label

result = build_feature_matrix(samples)

print("\nFEATURE EXTRACTION DIAGNOSTICS")
print("=" * 55)
print("Return type:", type(result))
print("Number of returned values:", len(result))

for i, value in enumerate(result):
    print(f"\nReturn value {i}:")
    print("Type:", type(value))

    if hasattr(value, "shape"):
        print("Shape:", value.shape)
    else:
        print("Value:", value)
