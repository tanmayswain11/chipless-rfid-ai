
import numpy as np

# Sensor operating band
FREQ_MIN_HZ = 800e6
FREQ_MAX_HZ = 5e9
GRID_POINTS = 128
N_RESONANCES = 5


def build_features(frequency_hz, magnitude_db):
    """
    Convert one measured frequency-response curve into a fixed-length
    feature vector.

    Input:
        frequency_hz: frequency values in Hz
        magnitude_db: measured magnitude/RCS values in dB

    Output:
        128 interpolated magnitudes
        10 summary features
        20 resonance features (5 peaks and 5 dips, each with
        frequency and amplitude)
    """

    freq = np.asarray(frequency_hz, dtype=float).ravel()
    mag = np.asarray(magnitude_db, dtype=float).ravel()

    if freq.size != mag.size:
        raise ValueError(
            "Frequency and magnitude arrays must have equal lengths."
        )

    # Remove NaN and infinite values.
    valid = np.isfinite(freq) & np.isfinite(mag)
    freq = freq[valid]
    mag = mag[valid]

    if freq.size < 20:
        raise ValueError(
            "Not enough valid measurements to extract features."
        )

    # Keep the sensor's specified frequency band.
    band = (
        (freq >= FREQ_MIN_HZ)
        & (freq <= FREQ_MAX_HZ)
    )
    freq = freq[band]
    mag = mag[band]

    if freq.size < 20:
        raise ValueError(
            "Fewer than 20 valid measurements in the 800 MHz–5 GHz band."
        )

    # Sort by frequency.
    order = np.argsort(freq)
    freq = freq[order]
    mag = mag[order]

    # Average duplicate-frequency readings.
    unique_freq, inverse = np.unique(
        freq, return_inverse=True
    )
    sums = np.bincount(inverse, weights=mag)
    counts = np.bincount(inverse)
    unique_mag = sums / counts

    freq = unique_freq
    mag = unique_mag

    # Require coverage of the full operating band.
    if freq[0] > FREQ_MIN_HZ or freq[-1] < FREQ_MAX_HZ:
        raise ValueError(
            "Curve does not cover the full 800 MHz–5 GHz band. "
            "Check the original measurement file."
        )

    # Interpolate onto a common frequency grid.
    grid = np.linspace(
        FREQ_MIN_HZ, FREQ_MAX_HZ, GRID_POINTS
    )
    interpolated = np.interp(grid, freq, mag)

    # Summary statistics.
    mean_mag = np.mean(interpolated)
    std_mag = np.std(interpolated)
    min_mag = np.min(interpolated)
    max_mag = np.max(interpolated)
    range_mag = max_mag - min_mag
    median_mag = np.median(interpolated)
    p25 = np.percentile(interpolated, 25)
    p75 = np.percentile(interpolated, 75)
    variance_mag = np.var(interpolated)

    peak_idx = int(np.argmax(interpolated))
    dip_idx = int(np.argmin(interpolated))

    summary = np.array([
        mean_mag,
        std_mag,
        min_mag,
        max_mag,
        range_mag,
        median_mag,
        p25,
        p75,
        grid[peak_idx] / 1e9,
        grid[dip_idx] / 1e9,
    ])

    # Detect local maxima and minima on the common grid.
    # A peak is a local maximum; a dip is a local minimum.
    peaks = np.where(
        (interpolated[1:-1] > interpolated[:-2])
        & (interpolated[1:-1] >= interpolated[2:])
    )[0] + 1

    dips = np.where(
        (interpolated[1:-1] < interpolated[:-2])
        & (interpolated[1:-1] <= interpolated[2:])
    )[0] + 1

    # Keep the strongest peaks and deepest dips.
    peaks = sorted(
        peaks,
        key=lambda i: interpolated[i],
        reverse=True,
    )[:N_RESONANCES]

    dips = sorted(
        dips,
        key=lambda i: interpolated[i],
    )[:N_RESONANCES]

    # Fixed-size features: frequency in GHz and amplitude in dB.
    # Missing resonance slots are padded with zero.
    peak_features = np.zeros(N_RESONANCES * 2)
    dip_features = np.zeros(N_RESONANCES * 2)

    for j, idx in enumerate(peaks):
        peak_features[2 * j] = grid[idx] / 1e9
        peak_features[2 * j + 1] = interpolated[idx]

    for j, idx in enumerate(dips):
        dip_features[2 * j] = grid[idx] / 1e9
        dip_features[2 * j + 1] = interpolated[idx]

    # Added variance feature.
    variance_feature = np.array([variance_mag])

    features = np.concatenate([
        interpolated,       # 128 features
        summary,            # 10 features
        variance_feature,   # 1 feature
        peak_features,      # 10 features
        dip_features,       # 10 features
    ])

    if not np.all(np.isfinite(features)):
        raise ValueError("Feature extraction produced invalid values.")

    return features
