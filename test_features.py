
from ml.data_loader import (
    load_curve_samples,
    default_dataset_paths
)

from ml.feature_extraction import (
    build_features,
    build_feature_matrix
)

import numpy as np


# Load temperature dataset
paths = default_dataset_paths()

samples = load_curve_samples(
    paths["temperature"],
    "temperature"
)

print("Number of curves:", len(samples))


# Extract one curve
first_sample = samples[0]

features = build_features(
    first_sample["frequency_hz"],
    first_sample["magnitude_db"]
)

print("Feature vector shape:", features.shape)
print("Feature values:")
print(features)


# Extract all curves
X = build_feature_matrix(samples)

print("\nFeature matrix shape:", X.shape)

print(
    "All features finite:",
    np.isfinite(X).all()
)
