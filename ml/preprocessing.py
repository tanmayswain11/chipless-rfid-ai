
from pathlib import Path
import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty or df.shape[1] < 2:
        raise ValueError(
            "CSV must contain a frequency column "
            "and at least one RF response curve."
        )

    return df
