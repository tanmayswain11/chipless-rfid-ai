
from ml.data_loader import load_curve_samples, default_dataset_paths
from ml.preprocessing import samples_to_xy

paths = default_dataset_paths()

samples = load_curve_samples(
    paths["temperature"],
    "temperature"
)

X, y, labels, labelled_samples = samples_to_xy(samples)

print("\nFinal verification")
print("------------------")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Curve labels:", labels)
print("Temperature targets:", y)
