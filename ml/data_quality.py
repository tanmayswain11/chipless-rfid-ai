
"""
Dataset quality inspection for chipless RFID sensing.

Inspects original CSV files without modifying them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_loader import (
    default_dataset_paths,
    find_frequency_column
)


def inspect_csv(csv_path, dataset_name="Dataset"):
    """Inspect one original CSV file."""

    csv_path = Path(csv_path)

    print("\n" + "=" * 65)
    print(f"DATA QUALITY REPORT: {dataset_name}")
    print("=" * 65)

    if not csv_path.exists():
        print("File not found:", csv_path)
        return None

    df = pd.read_csv(csv_path)

    frequency_col = find_frequency_column(df)

    print("File:", csv_path.name)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Frequency column:", frequency_col)

    print("\nColumn names:")
    for column in df.columns:
        print(" -", column)

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    numeric_df = df.select_dtypes(include=[np.number])

    print("\nInfinite values:")
    print(
        np.isinf(numeric_df.to_numpy()).sum()
        if not numeric_df.empty else 0
    )

    print("\nDuplicate complete rows:", df.duplicated().sum())

    if frequency_col is not None:

        frequency = pd.to_numeric(
            df[frequency_col],
            errors="coerce"
        ).dropna().to_numpy()

        if len(frequency) > 0:

            print("\nFrequency information:")
            print("Minimum:", frequency.min(), "Hz")
            print("Maximum:", frequency.max(), "Hz")
            print("Frequency points:", len(frequency))

            unique_frequency = np.unique(frequency)

            print(
                "Duplicate frequency values:",
                len(frequency) - len(unique_frequency)
            )

            if len(unique_frequency) > 1:
                differences = np.diff(unique_frequency)

                print(
                    "Minimum frequency spacing:",
                    differences.min(),
                    "Hz"
                )

                print(
                    "Maximum frequency spacing:",
                    differences.max(),
                    "Hz"
                )

    print("\nCurve information:")

    if frequency_col is not None:

        curve_columns = [
            col for col in df.columns
            if col != frequency_col
        ]

        print("Response curves:", len(curve_columns))

        for column in curve_columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            print(
                f"{column}: "
                f"valid={values.notna().sum()}, "
                f"missing={values.isna().sum()}"
            )

    return {
        "dataset": dataset_name,
        "path": str(csv_path),
        "rows": len(df),
        "columns": len(df.columns),
        "frequency_column": frequency_col,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "response_curves": (
            len(df.columns) - 1
            if frequency_col is not None else None
        )
    }


def inspect_all_datasets():
    """Inspect all three original datasets."""

    paths = default_dataset_paths()

    reports = []

    dataset_names = {
        "temperature": "Temperature",
        "water_volume": "Water Volume",
        "relative_humidity": "Relative Humidity"
    }

    for key, path in paths.items():

        report = inspect_csv(
            path,
            dataset_names.get(key, key)
        )

        if report is not None:
            reports.append(report)

    if reports:
        summary = pd.DataFrame(reports)

        print("\n\nOVERALL DATASET SUMMARY")
        print("=" * 65)
        print(summary.to_string(index=False))

        output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "data_quality_report.csv"

        summary.to_csv(output_file, index=False)

        print("\nReport saved to:", output_file)


if __name__ == "__main__":
    inspect_all_datasets()
