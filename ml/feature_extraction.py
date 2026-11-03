
from __future__ import annotations

import numpy as np


# ==================================================
# CONFIGURATION
# ==================================================

FREQ_MIN_HZ = 800e6
FREQ_MAX_HZ = 5e9

GRID_POINTS = 128

N_LOCAL_EXTREMA = 5


# ==================================================
# 1. LOCAL PEAK AND DIP DETECTION
# ==================================================

def detect_local_extrema(
    signal: np.ndarray,
    find_peaks: bool = True
) -> np.ndarray:

    signal = np.asarray(signal, dtype=float)

    if len(signal) < 3:
        return np.array([], dtype=int)

    if find_peaks:

        indices = np.where(
            (signal[1:-1] > signal[:-2])
            & (signal[1:-1] >= signal[2:])
        )[0] + 1

    else:

        indices = np.where(
            (signal[1:-1] < signal[:-2])
            & (signal[1:-1] <= signal[2:])
        )[0] + 1

    return indices


# ==================================================
# 2. CLEAN AND INTERPOLATE RF SIGNAL
# ==================================================

def preprocess_signal(
    frequency_hz,
    magnitude_db,
    freq_min_hz=FREQ_MIN_HZ,
    freq_max_hz=FREQ_MAX_HZ,
    grid_points=GRID_POINTS
):

    frequency = np.asarray(
        frequency_hz,
        dtype=float
    )

    magnitude = np.asarray(
        magnitude_db,
        dtype=float
    )

    if len(frequency) != len(magnitude):

        raise ValueError(
            "Frequency and magnitude lengths do not match."
        )

    # Remove invalid values.
    valid = (
        np.isfinite(frequency)
        & np.isfinite(magnitude)
    )

    frequency = frequency[valid]
    magnitude = magnitude[valid]

    # Select operating frequency range.
    band = (
        (frequency >= freq_min_hz)
        & (frequency <= freq_max_hz)
    )

    frequency = frequency[band]
    magnitude = magnitude[band]

    if len(frequency) < 10:

        raise ValueError(
            "Insufficient valid RF points in 800 MHz–5 GHz band."
        )

    # Sort frequency.
    order = np.argsort(frequency)

    frequency = frequency[order]
    magnitude = magnitude[order]

    # Average duplicate frequency points.
    unique_frequency, inverse = np.unique(
        frequency,
        return_inverse=True
    )

    if len(unique_frequency) != len(frequency):

        sums = np.zeros(
            len(unique_frequency),
            dtype=float
        )

        counts = np.zeros(
            len(unique_frequency),
            dtype=float
        )

        np.add.at(
            sums,
            inverse,
            magnitude
        )

        np.add.at(
            counts,
            inverse,
            1
        )

        frequency = unique_frequency
        magnitude = sums / counts

    # Interpolate on a consistent grid.
    # Use measured endpoints to avoid extrapolating beyond data.
    grid = np.linspace(
        frequency[0],
        frequency[-1],
        grid_points
    )

    interpolated_signal = np.interp(
        grid,
        frequency,
        magnitude
    )

    return grid, interpolated_signal


# ==================================================
# 3. SUMMARY FEATURES
# ==================================================

def extract_summary_features(
    frequency,
    signal
):

    features = [

        np.mean(signal),

        np.std(signal),

        np.min(signal),

        np.max(signal),

        np.ptp(signal),

        np.median(signal),

        np.percentile(signal, 25),

        np.percentile(signal, 75),

        frequency[np.argmax(signal)] / 1e9,

        frequency[np.argmin(signal)] / 1e9,

        np.var(signal)

    ]

    return features


# ==================================================
# 4. LOCAL PEAK FEATURES
# ==================================================

def extract_peak_features(
    frequency,
    signal
):

    indices = detect_local_extrema(
        signal,
        find_peaks=True
    )

    # Sort strongest peaks first.
    if len(indices) > 0:

        indices = indices[
            np.argsort(signal[indices])[::-1]
        ]

    indices = indices[:N_LOCAL_EXTREMA]

    features = []

    for i in range(N_LOCAL_EXTREMA):

        if i < len(indices):

            index = indices[i]

            peak_frequency = (
                frequency[index] / 1e9
            )

            peak_magnitude = signal[index]

        else:

            peak_frequency = 0.0
            peak_magnitude = 0.0

        features.extend([
            peak_frequency,
            peak_magnitude
        ])

    return features


# ==================================================
# 5. LOCAL DIP FEATURES
# ==================================================

def extract_dip_features(
    frequency,
    signal
):

    indices = detect_local_extrema(
        signal,
        find_peaks=False
    )

    # Sort deepest dips first.
    if len(indices) > 0:

        indices = indices[
            np.argsort(signal[indices])
        ]

    indices = indices[:N_LOCAL_EXTREMA]

    features = []

    for i in range(N_LOCAL_EXTREMA):

        if i < len(indices):

            index = indices[i]

            dip_frequency = (
                frequency[index] / 1e9
            )

            dip_magnitude = signal[index]

        else:

            dip_frequency = 0.0
            dip_magnitude = 0.0

        features.extend([
            dip_frequency,
            dip_magnitude
        ])

    return features


# ==================================================
# 6. MAIN FEATURE EXTRACTION FUNCTION
# ==================================================

def build_features(
    frequency_hz,
    magnitude_db
):

    frequency, signal = preprocess_signal(
        frequency_hz,
        magnitude_db
    )

    features = []

    # Summary features: 11
    features.extend(
        extract_summary_features(
            frequency,
            signal
        )
    )

    # Peak features: 10
    features.extend(
        extract_peak_features(
            frequency,
            signal
        )
    )

    # Dip features: 10
    features.extend(
        extract_dip_features(
            frequency,
            signal
        )
    )

    feature_vector = np.asarray(
        features,
        dtype=float
    )

    if not np.isfinite(feature_vector).all():

        raise ValueError(
            "Feature vector contains NaN or infinity."
        )

    return feature_vector


# ==================================================
# 7. BUILD FEATURE MATRIX
# ==================================================

def build_feature_matrix(samples):

    if not samples:

        raise ValueError(
            "No RF samples supplied."
        )

    X = np.vstack([

        build_features(
            sample["frequency_hz"],
            sample["magnitude_db"]
        )

        for sample in samples

    ])

    return X
