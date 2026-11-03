
import numpy as np
import matplotlib.pyplot as plt

from ml.data_loader import (
    load_curve_samples,
    default_dataset_paths
)

from ml.augmentation import (
    augment_curve,
    generate_augmented_samples
)

paths = default_dataset_paths()

samples = load_curve_samples(
    paths["temperature"],
    "temperature"
)

original = samples[0]

result = augment_curve(
    original["frequency_hz"],
    original["magnitude_db"],
    random_state=42
)

print("Original curves:", len(samples))

augmented_samples = generate_augmented_samples(
    samples,
    copies_per_curve=3,
    random_state=42
)

print("Synthetic curves:", len(augmented_samples))
print(
    "Original curve unchanged:",
    np.array_equal(
        original["magnitude_db"],
        samples[0]["magnitude_db"]
    )
)

plt.figure(figsize=(12, 5))

plt.plot(
    result["frequency_hz"] / 1e9,
    result["original_magnitude_db"],
    label="Original curve"
)

plt.plot(
    result["frequency_hz"] / 1e9,
    result["augmented_magnitude_db"],
    label="Augmented curve",
    alpha=0.8
)

plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.title("Original vs Synthetic RFID Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
