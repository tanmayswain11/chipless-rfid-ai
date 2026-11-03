
"""
Controlled augmentation for chipless RFID RF curves.

IMPORTANT:
- Original measured curves are never modified.
- Generated curves are synthetic simulations.
- Noise levels are assumptions until measured repeatability
  data are available.
"""

import numpy as np


def _validate_curve(frequency_hz, magnitude_db):
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    magnitude_db = np.asarray(magnitude_db, dtype=float)

    if frequency_hz.ndim != 1 or magnitude_db.ndim != 1:
        raise ValueError("Frequency and magnitude must be 1D arrays.")

    if len(frequency_hz) != len(magnitude_db):
        raise ValueError("Frequency and magnitude lengths differ.")

    if len(frequency_hz) < 5:
        raise ValueError("At least five frequency points are required.")

    if not np.isfinite(frequency_hz).all():
        raise ValueError("Frequency contains invalid values.")

    if not np.isfinite(magnitude_db).all():
        raise ValueError("Magnitude contains invalid values.")

    if np.any(np.diff(frequency_hz) <= 0):
        raise ValueError("Frequency must be strictly increasing.")

    return frequency_hz, magnitude_db


def correlated_noise(
    magnitude_db,
    sigma_db=0.15,
    correlation=0.90,
    random_state=None
):
    """
    Generate standardized AR(1)-style correlated noise.

    sigma_db is an assumed simulation parameter, not a
    measured PNA/VNA noise specification.
    """

    magnitude_db = np.asarray(magnitude_db, dtype=float)

    if sigma_db < 0:
        raise ValueError("sigma_db cannot be negative.")

    if not 0 <= correlation < 1:
        raise ValueError("correlation must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    innovations = rng.normal(size=len(magnitude_db))
    noise = np.zeros(len(magnitude_db), dtype=float)

    noise[0] = innovations[0]

    for i in range(1, len(noise)):
        noise[i] = (
            correlation * noise[i - 1]
            + np.sqrt(1 - correlation**2) * innovations[i]
        )

    noise_std = np.std(noise)

    if noise_std > 0:
        noise = noise / noise_std

    return magnitude_db + sigma_db * noise


def baseline_drift(
    magnitude_db,
    drift_db=0.20
):
    """
    Add a smooth linear baseline drift.

    drift_db is the maximum absolute drift at the curve edges.
    """

    magnitude_db = np.asarray(magnitude_db, dtype=float)

    if drift_db < 0:
        raise ValueError("drift_db cannot be negative.")

    x = np.linspace(-1, 1, len(magnitude_db))
    return magnitude_db + drift_db * x


def amplitude_variation(
    magnitude_db,
    scale_range=(0.995, 1.005),
    offset_range=(-0.10, 0.10),
    random_state=None
):
    """
    Apply small amplitude scaling and offset.

    These ranges are illustrative simulation assumptions.
    """

    magnitude_db = np.asarray(magnitude_db, dtype=float)

    rng = np.random.default_rng(random_state)

    scale = rng.uniform(*scale_range)
    offset = rng.uniform(*offset_range)

    return scale * magnitude_db + offset


def frequency_shift(
    frequency_hz,
    magnitude_db,
    shift_hz=1e6
):
    """
    Simulate a small frequency-axis calibration shift.

    The output is interpolated back onto the original frequency grid.
    """

    frequency_hz, magnitude_db = _validate_curve(
        frequency_hz,
        magnitude_db
    )

    if shift_hz < 0:
        raise ValueError("shift_hz cannot be negative.")

    shifted_frequency = frequency_hz + shift_hz

    shifted_magnitude = np.interp(
        frequency_hz,
        shifted_frequency,
        magnitude_db,
        left=magnitude_db[0],
        right=magnitude_db[-1]
    )

    return shifted_magnitude


def augment_curve(
    frequency_hz,
    magnitude_db,
    random_state=None,
    noise_sigma_db=0.15,
    drift_db=0.20,
    apply_frequency_shift=True
):
    """
    Generate one synthetic variation of an RF curve.

    The original curve is not modified.

    Returns
    -------
    dict containing original and augmented curve data.
    """

    frequency_hz, magnitude_db = _validate_curve(
        frequency_hz,
        magnitude_db
    )

    rng = np.random.default_rng(random_state)

    seed_noise = int(rng.integers(0, 2**31 - 1))
    seed_amplitude = int(rng.integers(0, 2**31 - 1))

    augmented = magnitude_db.copy()

    augmented = correlated_noise(
        augmented,
        sigma_db=noise_sigma_db,
        correlation=0.90,
        random_state=seed_noise
    )

    augmented = baseline_drift(
        augmented,
        drift_db=drift_db
    )

    augmented = amplitude_variation(
        augmented,
        random_state=seed_amplitude
    )

    if apply_frequency_shift:
        shift = rng.uniform(0, 1e6)

        augmented = frequency_shift(
            frequency_hz,
            augmented,
            shift_hz=shift
        )

    return {
        "frequency_hz": frequency_hz.copy(),
        "original_magnitude_db": magnitude_db.copy(),
        "augmented_magnitude_db": augmented,
        "augmentation_type": "combined_simulation",
        "noise_sigma_db_assumed": noise_sigma_db,
        "baseline_drift_db_assumed": drift_db
    }


def generate_augmented_samples(
    samples,
    copies_per_curve=3,
    random_state=42
):
    """
    Generate synthetic copies while retaining original sample labels.

    Use these copies only in training data after splitting original
    independent curves into training and validation sets.
    """

    if copies_per_curve < 1:
        raise ValueError("copies_per_curve must be at least 1.")

    rng = np.random.default_rng(random_state)
    augmented_samples = []

    for sample in samples:

        for copy_number in range(copies_per_curve):

            seed = int(rng.integers(0, 2**31 - 1))

            result = augment_curve(
                sample["frequency_hz"],
                sample["magnitude_db"],
                random_state=seed
            )

            augmented_samples.append({
                **sample,
                "frequency_hz": result["frequency_hz"],
                "magnitude_db": result["augmented_magnitude_db"],
                "original_curve": sample.get("curve", "unknown"),
                "augmentation_copy": copy_number + 1,
                "is_synthetic": True
            })

    return augmented_samples
