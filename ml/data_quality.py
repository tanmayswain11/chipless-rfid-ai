
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "temperature": ROOT / "data" / "raw" / "plot1_rcs.csv",
    "water_volume": ROOT / "data" / "raw" / "moisture1.csv",
}


def find_frequency_column(df):
    for column in df.columns:
        if "freq" in str(column).lower():
            return column

    raise ValueError(f"No frequency column found: {list(df.columns)}")


def frequency_to_hz(series, column_name):
    values = pd.to_numeric(series, errors="coerce")
    name = str(column_name).lower()

    if "ghz" in name:
        return values * 1e9
    if "mhz" in name:
        return values * 1e6
    if "khz" in name:
        return values * 1e3

    median = values.dropna().median()

    if median < 10:
        return values * 1e9
    if median < 10000:
        return values * 1e6

    return values


def audit_csv(name, path):
    print("\n" + "=" * 65)
    print(f"DATASET: {name}")
    print(f"FILE: {path}")

    if not path.exists():
        print("ERROR: CSV file not found.")
        return

    df = pd.read_csv(path)
    freq_col = find_frequency_column(df)
    freq = frequency_to_hz(df[freq_col], freq_col)

    print("Shape:", df.shape)
    print("Frequency column:", freq_col)
    print("Missing frequency values:", freq.isna().sum())

    valid_freq = freq.dropna()

    if len(valid_freq):
        print(
            "Frequency range (GHz):",
            valid_freq.min() / 1e9,
            "to",
            valid_freq.max() / 1e9,
        )
        print(
            "Duplicate frequency values:",
            valid_freq.duplicated().sum(),
        )

        if (
            valid_freq.min() > 800e6
            or valid_freq.max() < 5e9
        ):
            print(
                "WARNING: The CSV may not cover the full "
                "800 MHz–5 GHz range."
            )

    print("\nCurve checks:")

    for column in df.columns:
        if column == freq_col:
            continue

        magnitude = pd.to_numeric(df[column], errors="coerce")
        valid = magnitude.dropna()

        print(f"\n{column}")
        print("Missing/non-numeric:", magnitude.isna().sum())
        print("Valid values:", len(valid))

        if len(valid):
            print("Minimum:", valid.min())
            print("Maximum:", valid.max())

            suspicious = valid[
                (~np.isfinite(valid))
                | (valid < -300)
                | (valid > 100)
            ]

            if len(suspicious):
                print(
                    "WARNING:",
                    len(suspicious),
                    "suspicious values; verify the original "
                    "instrument export.",
                )


if __name__ == "__main__":
    for name, path in FILES.items():
        audit_csv(name, path)
