
from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

LABEL_CONFIG = ROOT / "config" / "target_labels.json"


# --------------------------------------------------
# 1. IDENTIFY FREQUENCY COLUMN
# --------------------------------------------------

def find_frequency_column(df: pd.DataFrame) -> str:

    aliases = {
        "frequency_hz",
        "freq_hz",
        "frequency",
        "freq",
        "freq. [hz]",
        "freq [hz]",
        "frequency (hz)"
    }

    for col in df.columns:

        if str(col).strip().lower() in aliases:
            return col

    if len(df.columns) == 0:
        raise ValueError("CSV contains no columns.")

    first_col = df.columns[0]

    values = pd.to_numeric(
        df[first_col],
        errors="coerce"
    ).dropna()

    if (
        len(values) >= 10
        and values.is_monotonic_increasing
    ):
        return first_col

    raise ValueError(
        "Frequency column not identified. "
        "Rename the frequency column to frequency_hz."
    )


# --------------------------------------------------
# 2. EXTRACT TARGET FROM CURVE NAME
# --------------------------------------------------

def parse_target(
    column_name: str,
    target_type: str
) -> Optional[float]:

    text = str(column_name).lower()

    patterns = {

        "temperature": [
            r"([-+]?\d+(?:\.\d+)?)\s*(?:°\s*c|degree|degrees|celsius)",
            r"temperature[_\s-]*([-+]?\d+(?:\.\d+)?)"
        ],

        "water_volume": [
            r"([-+]?\d+(?:\.\d+)?)\s*(?:ml|milliliter|milliliters)",
            r"(?:water|moisture)[_\s-]*([-+]?\d+(?:\.\d+)?)"
        ],

        "relative_humidity": [
            r"([-+]?\d+(?:\.\d+)?)\s*%",
            r"(?:rh|humidity)[_\s-]*([-+]?\d+(?:\.\d+)?)"
        ]
    }

    for pattern in patterns.get(target_type, []):

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:
            return float(match.group(1))

    # Generic numeric fallback for temperature/water labels.
    if target_type in {"temperature", "water_volume"}:

        match = re.search(
            r"[-+]?\d+(?:\.\d+)?",
            text
        )

        if match:
            return float(match.group(0))

    return None


# --------------------------------------------------
# 3. LOAD MANUAL TARGET LABELS
# --------------------------------------------------

def load_label_overrides() -> dict:

    if not LABEL_CONFIG.exists():
        return {}

    with open(
        LABEL_CONFIG,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# --------------------------------------------------
# 4. LOAD COMPLETE RF CURVES
# --------------------------------------------------

def load_curve_samples(
    csv_path: str | Path,
    target_type: str,
    frequency_col: str | None = None
) -> list[dict]:

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # --------------------------------------------------
    # READ CSV CORRECTLY
    # --------------------------------------------------

    if target_type == "relative_humidity":

        # RH dataset has no header row.
        # Therefore, preserve its first measurement row.

        df = pd.read_csv(
            csv_path,
            header=None
        )

        if df.empty:
            raise ValueError("RH CSV file is empty.")

        if df.shape[1] != 4:
            raise ValueError(
                "Expected exactly 4 columns in RH dataset: "
                "1 frequency column and 3 response curves. "
                f"Found {df.shape[1]} columns."
            )

        df.columns = [
            "frequency_hz",
            "rh_curve_1",
            "rh_curve_2",
            "rh_curve_3"
        ]

        frequency_col = "frequency_hz"

    else:

        # Temperature and water-volume datasets
        # continue using their existing headers.

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError("CSV file is empty.")

        frequency_col = (
            frequency_col
            or find_frequency_column(df)
        )

    # --------------------------------------------------
    # VALIDATE FREQUENCY COLUMN
    # --------------------------------------------------

    if frequency_col not in df.columns:
        raise ValueError(
            f"Frequency column '{frequency_col}' "
            "does not exist."
        )

    frequency = pd.to_numeric(
        df[frequency_col],
        errors="coerce"
    ).to_numpy(dtype=float)

    # --------------------------------------------------
    # LOAD MANUAL LABEL OVERRIDES
    # --------------------------------------------------

    overrides = load_label_overrides().get(
        target_type,
        {}
    )

    samples = []

    response_columns = [
        col for col in df.columns
        if col != frequency_col
    ]

    # --------------------------------------------------
    # EXTRACT EACH INDEPENDENT RF CURVE
    # --------------------------------------------------

    for column in response_columns:

        magnitude = pd.to_numeric(
            df[column],
            errors="coerce"
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(frequency)
            & np.isfinite(magnitude)
        )

        if valid.sum() < 10:
            continue

        f = frequency[valid]
        m = magnitude[valid]

        # Sort frequency values.
        order = np.argsort(f)

        f = f[order]
        m = m[order]

        # --------------------------------------------------
        # HANDLE DUPLICATE FREQUENCY POINTS
        # --------------------------------------------------

        curve_df = pd.DataFrame({

            "frequency_hz": f,

            "magnitude_db": m

        })

        curve_df = curve_df.groupby(
            "frequency_hz",
            as_index=False
        )["magnitude_db"].mean()

        f = curve_df["frequency_hz"].to_numpy(
            dtype=float
        )

        m = curve_df["magnitude_db"].to_numpy(
            dtype=float
        )

        # --------------------------------------------------
        # EXTRACT TARGET LABEL
        # --------------------------------------------------

        if target_type == "relative_humidity":

            # RH labels are assigned separately in
            # train_relative_humidity().
            # Never infer RH from RF magnitude values.

            target = None

        else:

            # First check manual configuration.
            target = overrides.get(str(column))

            # Otherwise parse the column name.
            if target is None:

                target = parse_target(
                    str(column),
                    target_type
                )

        # --------------------------------------------------
        # CREATE SAMPLE
        # --------------------------------------------------

        samples.append({

            "source_file": str(csv_path),

            "curve": str(column),

            "target": (
                None if target is None
                else float(target)
            ),

            "frequency_hz": f,

            "magnitude_db": m

        })

    if not samples:

        raise ValueError(
            f"No valid RF curves found in {csv_path}"
        )

    return samples


# --------------------------------------------------
# 5. DEFAULT DATASET LOCATIONS
# --------------------------------------------------

def default_dataset_paths() -> dict:

    return {

        "temperature":
            RAW_DIR / "plot1_rcs.csv",

        "water_volume":
            RAW_DIR / "moisture1.csv",

        "relative_humidity":
            RAW_DIR / "realhumid cf.csv"

    }
