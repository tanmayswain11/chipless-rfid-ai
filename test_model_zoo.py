
from ml.data_loader import (
    load_curve_samples,
    default_dataset_paths
)

from ml.preprocessing import samples_to_xy
from ml.model_zoo import get_models

paths = default_dataset_paths()

samples = load_curve_samples(
    paths["temperature"],
    "temperature"
)

X, y, labels, labelled_samples = samples_to_xy(samples)

models = get_models(n_samples=len(y))

print("\nModel Zoo Verification")
print("----------------------")

print("Feature matrix:", X.shape)
print("Target vector:", y.shape)

print("\nAvailable models:")

for name, model in models.items():
    print(f"{name}: {type(model).__name__}")

print("\nTotal models:", len(models))
